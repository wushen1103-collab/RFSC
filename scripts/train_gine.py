#!/usr/bin/env python3
"""Train a label-firewalled GINE baseline on official or SCG splits."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupShuffleSplit
from torch_geometric.loader import DataLoader

from bps_gnn.io import (
    freeze_prediction_file,
    read_label_columns,
    read_labels_for_ids,
    read_structure_table,
)
from bps_gnn.external import load_acanet_loss
from bps_gnn.models import (
    AttentiveFPRegressor,
    GATv2Regressor,
    GCNRegressor,
    GINRegressor,
    GINERegressor,
    GraphCliffRegressor,
    GraphSAGERegressor,
    MPNNRegressor,
)
from bps_gnn.molecular_graphs import ATOM_DIM, BOND_DIM, load_or_build_graphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--protocol", choices=["official", "scg", "temporal"], required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def split_ids(config: dict, structures: pd.DataFrame, target: str, protocol: str, fold: int | None):
    data_root = Path(config["data_root"])
    if protocol == "official":
        metadata = read_label_columns(data_root, target, ["mol_id", "official_split"])
        frame = structures.merge(metadata, on="mol_id", validate="one_to_one")
        official_train = frame[frame["official_split"] == "train"].copy()
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=config["selection"]["validation_fraction"],
            random_state=config["selection"]["seed"],
        )
        train_index, valid_index = next(
            splitter.split(official_train, groups=official_train["scaffold_group"])
        )
        return (
            official_train.iloc[train_index]["mol_id"].tolist(),
            official_train.iloc[valid_index]["mol_id"].tolist(),
            frame[frame["official_split"] == "test"]["mol_id"].tolist(),
        )
    if protocol == "temporal":
        from bps_gnn.temporal import temporal_split_ids

        return temporal_split_ids(data_root, target)
    if fold is None:
        raise ValueError("SCG protocol requires --fold")
    assignments = pd.read_parquet(Path("folds/scg") / target / "outer_folds.parquet")
    n_folds = int(assignments["outer_fold"].nunique())
    if fold not in set(assignments["outer_fold"]):
        raise ValueError(f"Invalid outer fold {fold}")
    validation_fold = (fold + 1) % n_folds
    test_ids = assignments.loc[assignments["outer_fold"] == fold, "mol_id"].tolist()
    valid_ids = assignments.loc[assignments["outer_fold"] == validation_fold, "mol_id"].tolist()
    train_ids = assignments.loc[
        ~assignments["outer_fold"].isin([fold, validation_fold]), "mol_id"
    ].tolist()
    return train_ids, valid_ids, test_ids


def attach_labels(
    graphs: list,
    indices: list[int],
    scaled_values: dict[str, float],
    original_values: dict[str, float],
    mol_ids: list[str],
):
    result = []
    for index in indices:
        graph = copy.copy(graphs[index])
        graph.y = torch.tensor([scaled_values[mol_ids[index]]], dtype=torch.float32)
        graph.y_original = torch.tensor([original_values[mol_ids[index]]], dtype=torch.float32)
        result.append(graph)
    return result


def singleton_safe_batch_size(n_items: int, requested: int) -> int:
    """Avoid a final one-graph training batch for BatchNorm architectures."""
    if n_items <= 1:
        raise ValueError("GINE training requires at least two graphs")
    requested = min(int(requested), n_items)
    if n_items % requested == 1 and requested > 2:
        return requested - 1
    return requested


def predict(model, loader, device, center: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    indices, predictions = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch).float().cpu().numpy() * scale + center
            indices.extend(batch.row_id.cpu().numpy().tolist())
            predictions.extend(output.tolist())
    return np.asarray(indices, dtype=int), np.asarray(predictions, dtype=float)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError(f"Seed {args.seed} is not pre-registered")
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_root = Path(config["data_root"])
    structures = read_structure_table(data_root, args.target).reset_index(drop=True)
    mol_ids = structures["mol_id"].tolist()
    id_to_index = {value: index for index, value in enumerate(mol_ids)}
    train_ids, valid_ids, test_ids = split_ids(
        config, structures, args.target, args.protocol, args.fold
    )
    if set(train_ids) & set(valid_ids) or set(train_ids) & set(test_ids) or set(valid_ids) & set(test_ids):
        raise RuntimeError("Split ID overlap")
    authorized = [*train_ids, *valid_ids]
    labels = read_labels_for_ids(data_root, args.target, authorized)
    label_map = labels.set_index("mol_id")["y_pactivity"].to_dict()
    train_response = np.asarray([label_map[value] for value in train_ids])
    center = float(np.median(train_response))
    scale = float(np.median(np.abs(train_response - center)))
    scale = max(scale, 1e-3)
    scaled_labels = {key: (value - center) / scale for key, value in label_map.items()}
    cache_path = Path(config["cache_root"]) / f"{args.target}.pt"
    graphs = load_or_build_graphs(
        structures["canonical_isomeric_smiles"].tolist(), cache_path
    )
    train_graphs = attach_labels(
        graphs,
        [id_to_index[value] for value in train_ids],
        scaled_labels,
        label_map,
        mol_ids,
    )
    valid_graphs = attach_labels(
        graphs,
        [id_to_index[value] for value in valid_ids],
        scaled_labels,
        label_map,
        mol_ids,
    )
    test_graphs = [copy.copy(graphs[id_to_index[value]]) for value in test_ids]
    optimization = config["optimization"]
    generator = torch.Generator().manual_seed(args.seed)
    train_batch_size = singleton_safe_batch_size(
        len(train_graphs), int(optimization["batch_size"])
    )
    train_loader = DataLoader(
        train_graphs, batch_size=train_batch_size, shuffle=True, generator=generator
    )
    valid_loader = DataLoader(valid_graphs, batch_size=optimization["batch_size"], shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=optimization["batch_size"], shuffle=False)
    model_config = dict(config["model"])
    architecture = model_config.pop("architecture", "gine")
    model_classes = {
        "attentivefp": AttentiveFPRegressor,
        "gatv2": GATv2Regressor,
        "gcn": GCNRegressor,
        "gin": GINRegressor,
        "gine": GINERegressor,
        "graphcliff": GraphCliffRegressor,
        "graphsage": GraphSAGERegressor,
        "mpnn": MPNNRegressor,
    }
    if architecture not in model_classes:
        raise ValueError(f"Unsupported molecular graph architecture: {architecture}")
    model = model_classes[architecture](ATOM_DIM, BOND_DIM, **model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimization["learning_rate"],
        weight_decay=optimization["weight_decay"],
    )
    amp_enabled = bool(optimization["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    aca_loss = load_acanet_loss(config) if optimization["loss"] == "external_acaloss" else None
    best_state = None
    best_epoch, best_mae, patience = 0, float("inf"), 0
    if args.protocol == "scg":
        run_name = f"fold_{args.fold}"
    else:
        run_name = args.protocol
    output = args.output_dir / args.target / run_name / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    history_path.write_text("", encoding="utf-8")
    for epoch in range(1, optimization["max_epochs"] + 1):
        model.train()
        losses = []
        regression_losses, triplet_losses, active_triplets = [], [], []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                estimate, embedding = model.forward_with_embedding(batch)
                if aca_loss is not None:
                    with torch.amp.autocast("cuda", enabled=False):
                        original_estimate = estimate.float() * scale + center
                        aca_result = aca_loss(
                            batch.y_original.view(-1).float(),
                            original_estimate,
                            embedding.float(),
                        )
                    loss, reg_loss, triplet_loss, _, _, n_active, _ = aca_result
                    regression_losses.append(float(reg_loss.detach().cpu()))
                    triplet_losses.append(float(triplet_loss.detach().cpu()))
                    active_triplets.append(int(n_active))
                elif optimization["loss"] == "mae_on_robust_scaled_response":
                    loss = torch.nn.functional.l1_loss(estimate, batch.y.view(-1))
                else:
                    loss = torch.nn.functional.mse_loss(estimate, batch.y.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), optimization["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        valid_indices, valid_prediction = predict(model, valid_loader, device, center, scale)
        valid_truth = np.asarray([label_map[mol_ids[index]] for index in valid_indices])
        valid_mae = float(np.mean(np.abs(valid_prediction - valid_truth)))
        record = {
            "epoch": epoch,
            "train_objective": float(np.mean(losses)),
            "validation_mae": valid_mae,
        }
        if aca_loss is not None:
            record.update(
                {
                    "aca_regression_mae": float(np.mean(regression_losses)),
                    "aca_triplet_loss": float(np.mean(triplet_losses)),
                    "aca_active_triplets": int(np.sum(active_triplets)),
                }
            )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if valid_mae < best_mae - 1e-8:
            best_mae, best_epoch, patience = valid_mae, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
        if patience >= optimization["patience"]:
            break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    test_indices, test_prediction = predict(model, test_loader, device, center, scale)
    torch.save(
        {"model_state": best_state, "center": center, "scale": scale, "config": config},
        output / "checkpoint.pt",
    )
    metadata = {
        "target": args.target,
        "protocol": args.protocol,
        "fold": args.fold,
        "seed": args.seed,
        "method": config["method"],
        "architecture": architecture,
        "loss": optimization["loss"],
        "n_train": len(train_ids),
        "n_valid": len(valid_ids),
        "n_test": len(test_ids),
        "requested_batch_size": int(optimization["batch_size"]),
        "train_batch_size": train_batch_size,
        "best_epoch": best_epoch,
        "best_validation_mae": best_mae,
        "response_center": center,
        "response_mad": scale,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "test_labels_materialized": False,
    }
    digest = freeze_prediction_file(
        pd.DataFrame(
            {
                "mol_id": [mol_ids[index] for index in test_indices],
                "prediction": test_prediction,
            }
        ),
        output / "predictions.parquet",
        metadata,
    )
    metadata["prediction_sha256"] = digest
    metadata["wall_time_seconds"] = time.perf_counter() - started
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
