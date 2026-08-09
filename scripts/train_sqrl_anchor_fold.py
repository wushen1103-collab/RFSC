#!/usr/bin/env python3
"""Freeze validation-only SQRL-style labeled-anchor baseline predictions."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.loader import DataLoader

from bps_gnn.dataset_graph import make_fingerprints, query_neighbors, stable_rng
from bps_gnn.io import read_labels_for_ids, read_structure_table, sha256_file
from bps_gnn.models import GINERegressor
from bps_gnn.molecular_graphs import ATOM_DIM, BOND_DIM, load_or_build_graphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--anchor-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="*", type=int)
    return parser.parse_args()


def partition_ids(target: str, fold: int) -> dict[str, list[str]]:
    assignments = pd.read_parquet(Path("folds/scg") / target / "outer_folds.parquet")
    n_folds = int(assignments["outer_fold"].nunique())
    validation_fold = (fold + 1) % n_folds
    result = {
        "train": assignments.loc[~assignments["outer_fold"].isin([fold, validation_fold]), "mol_id"].tolist(),
        "validation": assignments.loc[assignments["outer_fold"] == validation_fold, "mol_id"].tolist(),
        "test": assignments.loc[assignments["outer_fold"] == fold, "mol_id"].tolist(),
    }
    if set(result["train"]) & set(result["validation"]) or set(result["train"]) & set(result["test"]) or set(result["validation"]) & set(result["test"]):
        raise RuntimeError("SCG partition overlap")
    return result


def predict(
    checkpoint_path: Path,
    graphs: list,
    row_indices: list[int],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GINERegressor(ATOM_DIM, BOND_DIM, **checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loader = DataLoader([copy.copy(graphs[index]) for index in row_indices], batch_size=batch_size, shuffle=False)
    indices, values = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            scaled = model(batch)
            prediction = scaled.float().cpu().numpy() * checkpoint["scale"] + checkpoint["center"]
            indices.extend(batch.row_id.cpu().numpy().tolist())
            values.extend(prediction.tolist())
    return np.asarray(indices, dtype=int), np.asarray(values, dtype=float)


def anchor_predictions(
    query_base: np.ndarray,
    reference_base: np.ndarray,
    reference_truth: np.ndarray,
    neighbor_indices: list[np.ndarray],
    similarities: list[np.ndarray],
    permuted_residual: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {
        "base": query_base.copy(),
        "weighted_train_residual_anchor": query_base.copy(),
        "nearest_train_residual_anchor": query_base.copy(),
        "weighted_train_label_knn": query_base.copy(),
        "weighted_permuted_train_residual_anchor": query_base.copy(),
    }
    residual = reference_truth - reference_base
    for index, (neighbors, weights) in enumerate(zip(neighbor_indices, similarities)):
        if not len(neighbors):
            continue
        result["weighted_train_residual_anchor"][index] += float(np.average(residual[neighbors], weights=weights))
        result["nearest_train_residual_anchor"][index] += float(residual[neighbors[0]])
        result["weighted_train_label_knn"][index] = float(np.average(reference_truth[neighbors], weights=weights))
        result["weighted_permuted_train_residual_anchor"][index] += float(np.average(permuted_residual[neighbors], weights=weights))
    return result


def freeze(frame: pd.DataFrame, path: Path, metadata: dict) -> str:
    if any(column.lower().startswith("y_") for column in frame.columns):
        raise ValueError("Frozen anchor artifact contains a response column")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["method", "seed", "mol_id"], kind="stable").to_parquet(path, index=False)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256.json").write_text(
        json.dumps({"prediction_sha256": digest, "metadata": metadata}, indent=2), encoding="utf-8"
    )
    os.chmod(path, 0o444)
    return digest


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    started = time.perf_counter()
    base_config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.anchor_config.read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds else [int(value) for value in base_config["seeds"]]
    if not set(seeds).issubset(set(base_config["seeds"])):
        raise ValueError("Unregistered seed requested")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    structures = read_structure_table(Path(base_config["data_root"]), args.target).reset_index(drop=True)
    ids = structures["mol_id"].tolist()
    id_to_index = {mol_id: index for index, mol_id in enumerate(ids)}
    partitions = partition_ids(args.target, args.fold)
    train_truth = read_labels_for_ids(Path(base_config["data_root"]), args.target, partitions["train"]).set_index("mol_id")["y_pactivity"].loc[partitions["train"]].to_numpy(dtype=float)
    graph_config = config["graph"]
    fingerprints = make_fingerprints(structures["canonical_isomeric_smiles"].tolist(), int(graph_config["radius"]), int(graph_config["bits"]))
    indices = {name: [id_to_index[mol_id] for mol_id in values] for name, values in partitions.items()}
    train_fingerprints = [fingerprints[index] for index in indices["train"]]
    neighbors, similarities = query_neighbors(
        train_fingerprints,
        [fingerprints[index] for index in indices["validation"]],
        k=int(graph_config["query_to_train_k"]),
        floor=float(graph_config["similarity_floor"]),
    )
    graphs = load_or_build_graphs(structures["canonical_isomeric_smiles"].tolist(), Path(base_config["cache_root"]) / f"{args.target}.pt")
    rows, seed_metadata = [], []
    for seed in seeds:
        checkpoint = args.base_run_dir / args.target / f"fold_{args.fold}" / f"seed_{seed}" / "checkpoint.pt"
        combined = [*indices["train"], *indices["validation"]]
        row_id, prediction_values = predict(checkpoint, graphs, combined, device, int(base_config["optimization"]["batch_size"]))
        prediction_map = {int(key): value for key, value in zip(row_id, prediction_values)}
        train_base = np.asarray([prediction_map[index] for index in indices["train"]])
        validation_base = np.asarray([prediction_map[index] for index in indices["validation"]])
        residual = train_truth - train_base
        rng = stable_rng(int(config["controls"]["permutation_seed"]), args.target, args.fold, int(seed))
        candidates = anchor_predictions(validation_base, train_base, train_truth, neighbors, similarities, rng.permutation(residual))
        for method, values in candidates.items():
            rows.extend(
                {
                    "mol_id": mol_id,
                    "prediction": float(value),
                    "base_prediction": float(base),
                    "correction_value": float(value - base),
                    "method": method,
                    "seed": int(seed),
                    "neighbor_count": int(len(reference)),
                }
                for mol_id, value, base, reference in zip(partitions["validation"], values, validation_base, neighbors)
            )
        seed_metadata.append(
            {
                "seed": int(seed),
                "query_neighbor_coverage": float(np.mean([len(value) > 0 for value in neighbors])),
                "train_mean_residual": float(residual.mean()),
            }
        )
    metadata = {
        "target": args.target,
        "fold": args.fold,
        "method": config["name"],
        "train_labels_materialized": True,
        "individual_train_labels_used_at_query_inference": True,
        "validation_labels_materialized": False,
        "outer_test_predictions_generated": False,
        "outer_test_labels_materialized": False,
        "n_train": len(partitions["train"]),
        "n_validation": len(partitions["validation"]),
        "seeds": seeds,
        "seed_metadata": seed_metadata,
        "wall_time_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "config": config,
    }
    output = args.output_dir / args.target / f"fold_{args.fold}"
    metadata["prediction_sha256"] = freeze(pd.DataFrame(rows), output / "validation_predictions.parquet", metadata)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

