#!/usr/bin/env python3
"""Paired strict-SCG GINE/CliffLoss formula-adaptation smoke."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import torch
from torch_geometric.loader import DataLoader
import yaml

from bps_gnn.io import freeze_prediction_file, read_labels_for_ids, read_structure_table
from bps_gnn.models import GINERegressor
from bps_gnn.molecular_graphs import ATOM_DIM, BOND_DIM, load_or_build_graphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
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


def split_ids(target: str, fold: int) -> tuple[list[str], list[str], list[str]]:
    assignments = pd.read_parquet(Path("folds/scg") / target / "outer_folds.parquet")
    n_folds = int(assignments["outer_fold"].nunique())
    validation_fold = (fold + 1) % n_folds
    train = assignments.loc[
        ~assignments["outer_fold"].isin([fold, validation_fold]), "mol_id"
    ].tolist()
    validation = assignments.loc[
        assignments["outer_fold"] == validation_fold, "mol_id"
    ].tolist()
    test = assignments.loc[assignments["outer_fold"] == fold, "mol_id"].tolist()
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise RuntimeError("SCG partition overlap")
    return train, validation, test


def morgan_fingerprints(smiles: list[str], radius: int, n_bits: int) -> list:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprints = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid standardized SMILES at row {index}")
        fingerprints.append(generator.GetFingerprint(molecule))
    return fingerprints


def raw_severity_records(
    *,
    query_ids: list[str],
    train_ids: list[str],
    fingerprints: dict[str, object],
    query_truth: dict[str, float],
    train_truth: dict[str, float],
    similarity_threshold: float,
    neighbor_budget: int,
    aggregation_budget: int,
) -> tuple[np.ndarray, list[float], np.ndarray]:
    train_fingerprints = [fingerprints[value] for value in train_ids]
    raw_scores = []
    collected_delta = []
    neighbor_counts = []
    for query_id in query_ids:
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                fingerprints[query_id], train_fingerprints
            ),
            dtype=float,
        )
        candidates = []
        for index in np.argsort(-similarities, kind="stable"):
            reference_id = train_ids[int(index)]
            if reference_id == query_id:
                continue
            similarity = float(similarities[index])
            if similarity < similarity_threshold:
                break
            delta = abs(float(query_truth[query_id]) - float(train_truth[reference_id]))
            candidates.append((similarity, delta))
            if len(candidates) >= neighbor_budget:
                break
        neighbor_counts.append(len(candidates))
        collected_delta.extend(delta for _, delta in candidates)
        strengths = sorted(
            (similarity * delta for similarity, delta in candidates), reverse=True
        )[:aggregation_budget]
        raw_scores.append(float(np.mean(strengths)) if strengths else 0.0)
    return (
        np.asarray(raw_scores, dtype=float),
        collected_delta,
        np.asarray(neighbor_counts, dtype=int),
    )


def normalize_severity(raw_scores: np.ndarray, deltas: list[float], quantile: float) -> tuple[np.ndarray, float]:
    if not deltas:
        raise RuntimeError("No train-side neighbor deltas for CliffLoss severity")
    normalizer = max(float(np.quantile(np.asarray(deltas, dtype=float), quantile)), 1e-8)
    return raw_scores / normalizer, normalizer


def cliffloss_objective(
    error: torch.Tensor, severity: torch.Tensor, weight: float
) -> torch.Tensor:
    return error.mean() + float(weight) * (severity * error).mean()


def adaptive_weight(
    *,
    q1_mae: float,
    q4_mae: float,
    previous_ema: float,
    base_weight: float,
    ema_alpha: float,
    gamma: float,
    scale_min: float,
    scale_max: float,
) -> tuple[float, float, float]:
    gap = (q4_mae - q1_mae) / (0.5 * (abs(q4_mae) + abs(q1_mae)) + 1e-8)
    ema = ema_alpha * previous_ema + (1.0 - ema_alpha) * gap
    scale = float(np.clip(math.exp(gamma * ema), scale_min, scale_max))
    return base_weight * scale, ema, gap


def attach_graph_fields(
    graphs: list,
    ids: list[str],
    id_to_row: dict[str, int],
    labels: dict[str, float],
    severity: dict[str, float],
    center: float,
    scale: float,
) -> list:
    attached = []
    for value in ids:
        graph = copy.copy(graphs[id_to_row[value]])
        graph.y = torch.tensor([(labels[value] - center) / scale], dtype=torch.float32)
        graph.y_original = torch.tensor([labels[value]], dtype=torch.float32)
        graph.cliff_severity = torch.tensor([severity[value]], dtype=torch.float32)
        attached.append(graph)
    return attached


def predict(
    model: GINERegressor,
    loader: DataLoader,
    device: torch.device,
    center: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    rows, values = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            estimate = model(batch).float().cpu().numpy() * scale + center
            rows.extend(batch.row_id.cpu().numpy().tolist())
            values.extend(estimate.tolist())
    return np.asarray(rows, dtype=int), np.asarray(values, dtype=float)


def train_variant(
    *,
    variant: str,
    initial_state: dict,
    train_graphs: list,
    validation_graphs: list,
    validation_truth: dict[str, float],
    validation_ids: list[str],
    mol_ids: list[str],
    model_config: dict,
    optimization: dict,
    method_config: dict,
    seed: int,
    center: float,
    scale: float,
    device: torch.device,
    output: Path,
) -> tuple[GINERegressor, dict]:
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_graphs,
        batch_size=int(optimization["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_graphs,
        batch_size=int(optimization["batch_size"]),
        shuffle=False,
    )
    model = GINERegressor(ATOM_DIM, BOND_DIM, **model_config).to(device)
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    amp_enabled = bool(optimization["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    base_weight = float(method_config["base_weight"])
    current_weight = base_weight
    ema_gap = 0.0
    severity_values = np.asarray(
        [graph.cliff_severity.item() for graph in validation_graphs], dtype=float
    )
    order = np.argsort(severity_values, kind="stable")
    quartile = max(1, len(order) // 4)
    q1_rows = set(order[:quartile].tolist())
    q4_rows = set(order[-quartile:].tolist())
    best_state = None
    best_mae = float("inf")
    best_epoch = 0
    patience = 0
    history = []
    for epoch in range(1, int(optimization["max_epochs"]) + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                estimate = model(batch)
                error = torch.abs(estimate - batch.y.view(-1))
                if variant == "baseline":
                    loss = error.mean()
                elif variant == "cliffloss":
                    loss = cliffloss_objective(
                        error, batch.cliff_severity.view(-1), current_weight
                    )
                else:
                    raise ValueError(variant)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(optimization["gradient_clip"])
            )
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        rows, validation_prediction = predict(
            model, validation_loader, device, center, scale
        )
        row_to_prediction = {int(row): value for row, value in zip(rows, validation_prediction)}
        ordered_prediction = np.asarray(
            [row_to_prediction[index] for index in [graph.row_id.item() for graph in validation_graphs]]
        )
        ordered_truth = np.asarray([validation_truth[value] for value in validation_ids])
        absolute_error = np.abs(ordered_prediction - ordered_truth)
        validation_mae = float(absolute_error.mean())
        q1_mae = float(np.mean([absolute_error[index] for index in sorted(q1_rows)]))
        q4_mae = float(np.mean([absolute_error[index] for index in sorted(q4_rows)]))
        used_weight = current_weight if variant == "cliffloss" else 0.0
        gap = None
        if variant == "cliffloss":
            current_weight, ema_gap, gap = adaptive_weight(
                q1_mae=q1_mae,
                q4_mae=q4_mae,
                previous_ema=ema_gap,
                base_weight=base_weight,
                ema_alpha=float(method_config["ema_alpha"]),
                gamma=float(method_config["gamma"]),
                scale_min=float(method_config["scale_min"]),
                scale_max=float(method_config["scale_max"]),
            )
        history.append(
            {
                "epoch": epoch,
                "train_objective": float(np.mean(train_losses)),
                "validation_mae": validation_mae,
                "validation_q1_mae": q1_mae,
                "validation_q4_mae": q4_mae,
                "gap": gap,
                "weight_used": used_weight,
                "weight_next": current_weight if variant == "cliffloss" else 0.0,
            }
        )
        if validation_mae < best_mae - 1e-8:
            best_mae = validation_mae
            best_epoch = epoch
            patience = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            patience += 1
        if patience >= int(optimization["patience"]):
            break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    (output / variant).mkdir(parents=True, exist_ok=True)
    history_path = output / variant / "history.jsonl"
    history_path.write_text(
        "".join(json.dumps(row) + "\n" for row in history), encoding="utf-8"
    )
    return model, {
        "variant": variant,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_mae": best_mae,
        "initial_weight": base_weight if variant == "cliffloss" else None,
        "final_weight": current_weight if variant == "cliffloss" else None,
        "minimum_weight": (
            min(row["weight_used"] for row in history) if variant == "cliffloss" else None
        ),
        "maximum_weight": (
            max(row["weight_used"] for row in history) if variant == "cliffloss" else None
        ),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    smoke = config["smoke"]
    target, fold, seed = smoke["target"], int(smoke["fold"]), int(smoke["seed"])
    set_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    structures = read_structure_table(Path(config["data_root"]), target).reset_index(drop=True)
    mol_ids = structures["mol_id"].tolist()
    id_to_row = {value: index for index, value in enumerate(mol_ids)}
    train_ids, validation_ids, test_ids = split_ids(target, fold)
    authorized_ids = [*train_ids, *validation_ids]
    labels = read_labels_for_ids(Path(config["data_root"]), target, authorized_ids)
    label_map = labels.set_index("mol_id")["y_pactivity"].to_dict()
    train_truth = {value: label_map[value] for value in train_ids}
    validation_truth = {value: label_map[value] for value in validation_ids}
    fingerprint_config = config["severity"]["fingerprint"]
    all_fingerprints = morgan_fingerprints(
        structures["canonical_isomeric_smiles"].tolist(),
        int(fingerprint_config["radius"]),
        int(fingerprint_config["n_bits"]),
    )
    fingerprint_map = {
        value: all_fingerprints[index] for index, value in enumerate(mol_ids)
    }
    severity_config = config["severity"]
    common = {
        "train_ids": train_ids,
        "fingerprints": fingerprint_map,
        "train_truth": train_truth,
        "similarity_threshold": float(severity_config["similarity_threshold"]),
        "neighbor_budget": int(severity_config["neighbor_budget"]),
        "aggregation_budget": int(severity_config["aggregation_budget"]),
    }
    train_raw, train_deltas, train_neighbors = raw_severity_records(
        query_ids=train_ids, query_truth=train_truth, **common
    )
    train_severity, normalizer = normalize_severity(
        train_raw, train_deltas, float(severity_config["normalization_quantile"])
    )
    validation_raw, _, validation_neighbors = raw_severity_records(
        query_ids=validation_ids, query_truth=validation_truth, **common
    )
    validation_severity = validation_raw / normalizer
    train_severity_map = dict(zip(train_ids, train_severity.tolist()))
    validation_severity_map = dict(zip(validation_ids, validation_severity.tolist()))
    train_response = np.asarray([train_truth[value] for value in train_ids], dtype=float)
    center = float(np.median(train_response))
    scale = max(float(np.median(np.abs(train_response - center))), 1e-3)
    graphs = load_or_build_graphs(
        structures["canonical_isomeric_smiles"].tolist(),
        Path(config["cache_root"]) / f"{target}.pt",
    )
    train_graphs = attach_graph_fields(
        graphs,
        train_ids,
        id_to_row,
        train_truth,
        train_severity_map,
        center,
        scale,
    )
    validation_graphs = attach_graph_fields(
        graphs,
        validation_ids,
        id_to_row,
        validation_truth,
        validation_severity_map,
        center,
        scale,
    )
    model = GINERegressor(ATOM_DIM, BOND_DIM, **config["model"])
    initial_state = copy.deepcopy(model.state_dict())
    output = args.output_dir / target / f"fold_{fold}" / f"seed_{seed}"
    metadata = {
        "target": target,
        "fold": fold,
        "seed": seed,
        "method": config["name"],
        "adaptation_status": config["adaptation_status"],
        "n_train": len(train_ids),
        "n_validation": len(validation_ids),
        "n_test": len(test_ids),
        "validation_labels_materialized_for_adaptive_control": True,
        "outer_test_labels_materialized": False,
        "outer_test_predictions_generated": False,
        "severity": {
            "normalizer_q95": normalizer,
            "train_nonzero_fraction": float(np.mean(train_severity > 0)),
            "validation_nonzero_fraction": float(np.mean(validation_severity > 0)),
            "train_mean_neighbors": float(train_neighbors.mean()),
            "validation_mean_neighbors": float(validation_neighbors.mean()),
            "train_mean": float(train_severity.mean()),
            "validation_mean": float(validation_severity.mean()),
        },
        "config": config,
    }
    training = []
    for variant in ["baseline", "cliffloss"]:
        model, record = train_variant(
            variant=variant,
            initial_state=initial_state,
            train_graphs=train_graphs,
            validation_graphs=validation_graphs,
            validation_truth=validation_truth,
            validation_ids=validation_ids,
            mol_ids=mol_ids,
            model_config=config["model"],
            optimization=config["optimization"],
            method_config=config["cliffloss"],
            seed=seed,
            center=center,
            scale=scale,
            device=device,
            output=output,
        )
        training.append(record)
        loader = DataLoader(
            validation_graphs,
            batch_size=int(config["optimization"]["batch_size"]),
            shuffle=False,
        )
        rows, prediction = predict(model, loader, device, center, scale)
        row_to_prediction = {int(row): value for row, value in zip(rows, prediction)}
        ordered_prediction = [row_to_prediction[id_to_row[value]] for value in validation_ids]
        digest = freeze_prediction_file(
            pd.DataFrame({"mol_id": validation_ids, "prediction": ordered_prediction}),
            output / variant / "validation_predictions.parquet",
            {**metadata, "variant": variant, "training": record},
        )
        metadata[f"{variant}_validation_prediction_sha256"] = digest
    metadata["training"] = training
    metadata["wall_time_seconds"] = time.perf_counter() - started
    metadata["peak_vram_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
