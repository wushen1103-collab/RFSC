#!/usr/bin/env python3
"""Faithful split/firewall adapter for the pinned official PrismNet model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import torch
import yaml
from rdkit import Chem
from torch.utils.data import DataLoader

from bps_gnn.io import freeze_prediction_file, read_labels_for_ids, read_structure_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def split_ids(target: str, fold: int) -> tuple[list[str], list[str], list[str]]:
    assignments = pd.read_parquet(Path("folds/scg") / target / "outer_folds.parquet")
    n_folds = int(assignments["outer_fold"].nunique())
    validation_fold = (fold + 1) % n_folds
    train_ids = assignments.loc[
        ~assignments["outer_fold"].isin([fold, validation_fold]), "mol_id"
    ].tolist()
    valid_ids = assignments.loc[assignments["outer_fold"] == validation_fold, "mol_id"].tolist()
    test_ids = assignments.loc[assignments["outer_fold"] == fold, "mol_id"].tolist()
    if set(train_ids) & set(valid_ids) or set(train_ids) & set(test_ids) or set(valid_ids) & set(test_ids):
        raise RuntimeError("SCG partition overlap")
    return train_ids, valid_ids, test_ids


def load_official_modules(root: Path):
    if not hasattr(scipy_stats, "gilbrat"):
        scipy_stats.gilbrat = scipy_stats.gibrat
    sys.path.insert(0, str(root.resolve()))
    from data_process.dataset_graph import construct_dataset, mol_collate_func
    from models.prismnet import make_model
    from scripts.finetune import RegressionLoss
    from scripts.utils import ScheduledOptim

    return construct_dataset, mol_collate_func, make_model, RegressionLoss, ScheduledOptim


def make_model_params(config: dict, max_atoms: int) -> dict:
    model = config["model"]
    return {
        "d_atom": int(model["d_atom"]),
        "d_edge": int(model["d_edge"]),
        "d_model": int(model["d_model"]),
        "N": int(model["layers"]),
        "h": int(model["heads"]),
        "N_dense": int(model["dense_layers"]),
        "n_generator_layers": int(model["generator_layers"]),
        "n_output": 1,
        "leaky_relu_slope": float(model["leaky_relu_slope"]),
        "dense_output_nonlinearity": model["dense_output_nonlinearity"],
        "distance_matrix_kernel": model["distance_matrix_kernel"],
        "aggregation_type": model["aggregation_type"],
        "scale_norm": bool(model["scale_norm"]),
        "dropout": float(model["dropout"]),
        "max_length": max_atoms + 1,
    }


def attach_dataset(construct_dataset, structures: pd.DataFrame, ids: list[str], labels: dict[str, float], params: dict):
    indexed = structures.set_index("mol_id")
    molecules = [Chem.MolFromSmiles(indexed.loc[mol_id, "canonical_isomeric_smiles"]) for mol_id in ids]
    if any(molecule is None for molecule in molecules):
        raise RuntimeError("PrismNet adapter encountered invalid standardized SMILES")
    values = [float(labels[mol_id]) for mol_id in ids]
    return construct_dataset(molecules, values, params["d_atom"], params["d_edge"], params["max_length"])


def validation_rmse(model, loader, device: torch.device) -> float:
    model.eval()
    squared, count = 0.0, 0
    with torch.no_grad():
        for _, adjacency, nodes, edges, truth in loader:
            adjacency, nodes, edges, truth = adjacency.to(device), nodes.to(device), edges.to(device), truth.to(device)
            mask = torch.sum(torch.abs(nodes), dim=-1) != 0
            prediction, _ = model(nodes, mask, adjacency, edges)
            squared += float(torch.square(prediction - truth).sum().cpu())
            count += int(truth.numel())
    if not count:
        raise RuntimeError("Empty complete-batch validation loader")
    return float(np.sqrt(squared / count))


def predict_frame(model, loader, ids: list[str], device: torch.device) -> pd.DataFrame:
    model.eval()
    rows, offset = [], 0
    with torch.no_grad():
        for _, adjacency, nodes, edges, _ in loader:
            adjacency, nodes, edges = adjacency.to(device), nodes.to(device), edges.to(device)
            mask = torch.sum(torch.abs(nodes), dim=-1) != 0
            prediction, _ = model(nodes, mask, adjacency, edges)
            values = prediction.view(-1).float().cpu().numpy().tolist()
            rows.extend(
                {"mol_id": mol_id, "prediction": float(value)}
                for mol_id, value in zip(ids[offset:offset + len(values)], values)
            )
            offset += len(values)
    if offset != len(ids):
        raise RuntimeError("PrismNet complete-batch prediction omitted molecules")
    return pd.DataFrame(rows)


def should_stop_early(epoch: int, best_epoch: int, patience: int | None, min_epochs: int) -> bool:
    if patience is None:
        return False
    return epoch >= min_epochs and epoch - best_epoch >= patience


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("Seed is not registered")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    construct_dataset, collate, make_model, RegressionLoss, ScheduledOptim = load_official_modules(Path("external/PrismNet"))

    structures = read_structure_table(Path(config["data_root"]), args.target).reset_index(drop=True)
    train_ids, valid_ids, test_ids = split_ids(args.target, args.fold)
    if args.smoke:
        train_ids, valid_ids = train_ids[:64], valid_ids[:32]
    authorized = [*train_ids, *valid_ids]
    labels = read_labels_for_ids(Path(config["data_root"]), args.target, authorized).set_index("mol_id")["y_pactivity"].to_dict()
    if set(labels) & set(test_ids):
        raise RuntimeError("Outer-test label entered PrismNet trainer")
    max_atoms = max(Chem.MolFromSmiles(value).GetNumAtoms() for value in structures["canonical_isomeric_smiles"])
    params = make_model_params(config, max_atoms)
    train_dataset = attach_dataset(construct_dataset, structures, train_ids, labels, params)
    valid_dataset = attach_dataset(construct_dataset, structures, valid_ids, labels, params)
    optimization = config["optimization"]
    batch_size = int(optimization["batch_size"])
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
        drop_last=bool(optimization["train_drop_last"]),
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        drop_last=False,
        num_workers=0,
    )
    model = make_model(**params).to(device)
    checkpoint_path = Path(config["initialization"]["checkpoint"])
    initialization = "random_init_architecture_audit"
    checkpoint_sha256 = None
    if checkpoint_path.exists():
        from bps_gnn.io import sha256_file
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = state.get("state_dict", state)
        state.pop("pos_embed.pe.weight", None)
        incompatible = model.load_state_dict(state, strict=False)
        initialization = "official_pretrained_checkpoint"
        checkpoint_sha256 = sha256_file(checkpoint_path)
        if len(incompatible.unexpected_keys) > 0:
            raise RuntimeError(f"Unexpected pretrained keys: {incompatible.unexpected_keys}")

    criterion = RegressionLoss(
        device=device,
        num_tasks=1,
        contrastive_weight=float(optimization["contrastive_weight"]),
    )
    warmup_steps = max(
        1,
        int(len(train_dataset) / batch_size) * int(optimization["warmup_epochs"]),
    )
    optimizer = ScheduledOptim(
        torch.optim.Adam(model.parameters(), lr=0),
        float(optimization["warmup_factor"]),
        int(params["d_model"]),
        warmup_steps,
    )
    max_epochs = 1 if args.smoke else int(optimization["max_epochs"])
    max_train_batches = 2 if args.smoke else None
    early_stopping_patience = None if args.smoke else optimization.get("early_stopping_patience")
    if early_stopping_patience is not None:
        early_stopping_patience = int(early_stopping_patience)
    early_stopping_min_epochs = int(optimization.get("early_stopping_min_epochs", max_epochs))
    best_state, best_rmse, best_epoch = None, float("inf"), 0
    history = []
    stopped_early = False
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch_index, (_, adjacency, nodes, edges, truth) in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            adjacency, nodes, edges, truth = adjacency.to(device), nodes.to(device), edges.to(device), truth.to(device)
            mask = torch.sum(torch.abs(nodes), dim=-1) != 0
            prediction, pooling = model(nodes, mask, adjacency, edges)
            loss = torch.sqrt(criterion.compute(prediction, truth, graph_features=pooling))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()
            losses.append(float(loss.detach().cpu()))
        rmse = validation_rmse(model, valid_loader, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_rmse": rmse})
        if rmse < best_rmse:
            best_rmse, best_epoch = rmse, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if should_stop_early(
            epoch, best_epoch, early_stopping_patience, early_stopping_min_epochs
        ):
            stopped_early = True
            break
    if best_state is None:
        raise RuntimeError("No finite PrismNet checkpoint")

    run_name = f"fold_{args.fold}/seed_{args.seed}"
    output = args.output_dir / args.target / run_name
    output.mkdir(parents=True, exist_ok=True)
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save({"state_dict": best_state, "config": config, "model_params": params}, output / "checkpoint.pt")
    metadata = {
        "target": args.target,
        "fold": args.fold,
        "seed": args.seed,
        "method": "prismnet",
        "initialization": initialization,
        "published_pretrained_reproduction": initialization == "official_pretrained_checkpoint",
        "pretrained_checkpoint_sha256": checkpoint_sha256,
        "smoke": args.smoke,
        "n_train": len(train_ids),
        "n_validation": len(valid_ids),
        "n_test": 0 if args.smoke else len(test_ids),
        "best_epoch": best_epoch,
        "best_validation_rmse": best_rmse,
        "epochs_completed": len(history),
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_epochs": early_stopping_min_epochs,
        "stopped_early": stopped_early,
        "validation_drop_last": False,
        "test_drop_last": False,
        "validation_labels_materialized": True,
        "outer_test_labels_materialized": False,
        "outer_test_predictions_generated": not args.smoke,
        "max_train_batches": max_train_batches,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "wall_time_seconds": time.perf_counter() - started,
    }
    model.load_state_dict(best_state)
    model.to(device).eval()
    metadata["validation_prediction_sha256"] = freeze_prediction_file(
        predict_frame(model, valid_loader, valid_ids, device),
        output / "validation_predictions.parquet",
        metadata,
    )
    if not args.smoke:
        dummy = {mol_id: 0.0 for mol_id in test_ids}
        test_dataset = attach_dataset(construct_dataset, structures, test_ids, dummy, params)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False, num_workers=0)
        metadata["prediction_sha256"] = freeze_prediction_file(
            predict_frame(model, test_loader, test_ids, device),
            output / "predictions.parquet",
            metadata,
        )
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
