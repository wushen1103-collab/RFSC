#!/usr/bin/env python3
"""Freeze validation-only node and antisymmetric-edge residual predictions."""

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
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from bps_gnn.dataset_graph import make_fingerprints, query_neighbors, stable_rng, undirected_knn_pairs
from bps_gnn.io import read_labels_for_ids, read_structure_table, sha256_file
from bps_gnn.models import GINERegressor
from bps_gnn.molecular_graphs import ATOM_DIM, BOND_DIM, load_or_build_graphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--residual-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def partition_ids(target: str, fold: int) -> dict[str, list[str]]:
    assignments = pd.read_parquet(Path("folds/scg") / target / "outer_folds.parquet")
    n_folds = int(assignments["outer_fold"].nunique())
    validation_fold = (fold + 1) % n_folds
    result = {
        "train": assignments.loc[
            ~assignments["outer_fold"].isin([fold, validation_fold]), "mol_id"
        ].tolist(),
        "validation": assignments.loc[
            assignments["outer_fold"] == validation_fold, "mol_id"
        ].tolist(),
        "test": assignments.loc[assignments["outer_fold"] == fold, "mol_id"].tolist(),
    }
    if any(
        set(result[left]) & set(result[right])
        for left, right in [
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ]
    ):
        raise RuntimeError("SCG partition overlap")
    return result


def predict_embeddings(
    checkpoint_path: Path,
    graphs: list,
    row_indices: list[int],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GINERegressor(ATOM_DIM, BOND_DIM, **checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loader = DataLoader(
        [copy.copy(graphs[index]) for index in row_indices],
        batch_size=batch_size,
        shuffle=False,
    )
    indices, predictions, embeddings = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            scaled, embedding = model.forward_with_embedding(batch)
            prediction = scaled.float().cpu().numpy() * checkpoint["scale"] + checkpoint["center"]
            indices.extend(batch.row_id.cpu().numpy().tolist())
            predictions.extend(prediction.tolist())
            embeddings.extend(embedding.float().cpu().numpy().tolist())
    return (
        np.asarray(indices, dtype=int),
        np.asarray(predictions, dtype=float),
        np.asarray(embeddings, dtype=float),
    )


def directed_features(
    left_embedding: np.ndarray,
    right_embedding: np.ndarray,
    left_prediction: np.ndarray,
    right_prediction: np.ndarray,
) -> np.ndarray:
    """Return features whose sign reverses exactly when edge direction reverses."""
    return np.column_stack(
        [left_embedding - right_embedding, left_prediction - right_prediction]
    )


def aggregate_edge_correction(
    model,
    query_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    query_prediction: np.ndarray,
    reference_prediction: np.ndarray,
    neighbor_indices: list[np.ndarray],
    similarities: list[np.ndarray],
    train_mean_residual: float,
) -> tuple[np.ndarray, np.ndarray]:
    corrections = np.zeros(len(query_prediction), dtype=float)
    counts = np.zeros(len(query_prediction), dtype=int)
    for index, (neighbors, weights) in enumerate(zip(neighbor_indices, similarities)):
        counts[index] = len(neighbors)
        if not len(neighbors):
            continue
        features = directed_features(
            np.repeat(query_embedding[index][None, :], len(neighbors), axis=0),
            reference_embedding[neighbors],
            np.repeat(query_prediction[index], len(neighbors)),
            reference_prediction[neighbors],
        )
        delta = model.predict(features)
        corrections[index] = train_mean_residual + float(np.average(delta, weights=weights))
    return corrections, counts


def freeze(frame: pd.DataFrame, path: Path, metadata: dict) -> str:
    prohibited = {"y", "label", "truth", "response", "residual"}
    if any(any(token in column.lower() for token in prohibited) for column in frame.columns):
        raise ValueError("Frozen pseudo-query artifact contains a response-like column")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["method", "eta", "seed", "mol_id"], kind="stable").to_parquet(path, index=False)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256.json").write_text(
        json.dumps({"prediction_sha256": digest, "metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    os.chmod(path, 0o444)
    return digest


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    base_config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.residual_config.read_text(encoding="utf-8"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    structures = read_structure_table(Path(base_config["data_root"]), args.target).reset_index(drop=True)
    mol_ids = structures["mol_id"].tolist()
    id_to_index = {mol_id: index for index, mol_id in enumerate(mol_ids)}
    partitions = partition_ids(args.target, args.fold)

    # Only outer-train responses may enter this process.
    train_truth = read_labels_for_ids(
        Path(base_config["data_root"]), args.target, partitions["train"]
    ).set_index("mol_id")["y_pactivity"].loc[partitions["train"]]
    if set(train_truth.index) & (set(partitions["validation"]) | set(partitions["test"])):
        raise RuntimeError("Pseudo-query label entered residual trainer")

    graph_config = config["graph"]
    fingerprints = make_fingerprints(
        structures["canonical_isomeric_smiles"].tolist(),
        int(graph_config["radius"]),
        int(graph_config["bits"]),
    )
    partition_indices = {
        name: [id_to_index[value] for value in values] for name, values in partitions.items()
    }
    train_fingerprints = [fingerprints[index] for index in partition_indices["train"]]
    train_left, train_right, _ = undirected_knn_pairs(
        train_fingerprints,
        k=int(graph_config["query_to_train_k"]),
        floor=float(graph_config["similarity_floor"]),
    )
    neighbor_indices, similarities = query_neighbors(
        train_fingerprints,
        [fingerprints[index] for index in partition_indices["validation"]],
        k=int(graph_config["query_to_train_k"]),
        floor=float(graph_config["similarity_floor"]),
    )
    graphs = load_or_build_graphs(
        structures["canonical_isomeric_smiles"].tolist(),
        Path(base_config["cache_root"]) / f"{args.target}.pt",
    )

    alpha = float(config["models"]["ridge_alpha"])
    etas = [float(value) for value in config["models"]["correction_etas"]]
    rows, seed_metadata = [], []
    for seed in base_config["seeds"]:
        checkpoint = args.base_run_dir / args.target / f"fold_{args.fold}" / f"seed_{seed}" / "checkpoint.pt"
        combined = [*partition_indices["train"], *partition_indices["validation"]]
        row_id, prediction, embedding = predict_embeddings(
            checkpoint, graphs, combined, device, int(base_config["optimization"]["batch_size"])
        )
        pred_map = {int(key): value for key, value in zip(row_id, prediction)}
        emb_map = {int(key): value for key, value in zip(row_id, embedding)}
        part_prediction = {
            name: np.asarray([pred_map[index] for index in partition_indices[name]], dtype=float)
            for name in ["train", "validation"]
        }
        part_embedding = {
            name: np.asarray([emb_map[index] for index in partition_indices[name]], dtype=float)
            for name in ["train", "validation"]
        }
        residual = train_truth.to_numpy(dtype=float) - part_prediction["train"]
        node_features = np.column_stack([part_embedding["train"], part_prediction["train"]])
        node_model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        node_model.fit(node_features, residual)
        validation_node_features = np.column_stack(
            [part_embedding["validation"], part_prediction["validation"]]
        )
        node_correction = node_model.predict(validation_node_features)

        train_edge_features = directed_features(
            part_embedding["train"][train_left],
            part_embedding["train"][train_right],
            part_prediction["train"][train_left],
            part_prediction["train"][train_right],
        )
        train_edge_target = residual[train_left] - residual[train_right]
        augmented_features = np.vstack([train_edge_features, -train_edge_features])
        augmented_target = np.concatenate([train_edge_target, -train_edge_target])
        edge_model = make_pipeline(
            StandardScaler(with_mean=False),
            Ridge(alpha=alpha, fit_intercept=False),
        )
        edge_model.fit(augmented_features, augmented_target)
        edge_correction, neighbor_counts = aggregate_edge_correction(
            edge_model,
            part_embedding["validation"],
            part_embedding["train"],
            part_prediction["validation"],
            part_prediction["train"],
            neighbor_indices,
            similarities,
            float(residual.mean()),
        )

        rng = stable_rng(int(config["controls"]["permutation_seed"]), args.target, args.fold, int(seed))
        permuted_target = rng.permutation(train_edge_target)
        permuted_augmented_target = np.concatenate([permuted_target, -permuted_target])
        permuted_model = make_pipeline(
            StandardScaler(with_mean=False),
            Ridge(alpha=alpha, fit_intercept=False),
        )
        permuted_model.fit(augmented_features, permuted_augmented_target)
        permuted_correction, _ = aggregate_edge_correction(
            permuted_model,
            part_embedding["validation"],
            part_embedding["train"],
            part_prediction["validation"],
            part_prediction["train"],
            neighbor_indices,
            similarities,
            float(residual.mean()),
        )

        base = part_prediction["validation"]
        corrections = {
            "node_residual": node_correction,
            "edge_residual": edge_correction,
            "edge_permuted": permuted_correction,
        }
        rows.extend(
            {
                "mol_id": mol_id,
                "prediction": float(value),
                "correction_value": 0.0,
                "method": "base",
                "eta": 0.0,
                "seed": int(seed),
                "neighbor_count": int(count),
            }
            for mol_id, value, count in zip(partitions["validation"], base, neighbor_counts)
        )
        for method, correction in corrections.items():
            for eta in etas:
                rows.extend(
                    {
                        "mol_id": mol_id,
                        "prediction": float(value + eta * delta),
                        "correction_value": float(eta * delta),
                        "method": method,
                        "eta": eta,
                        "seed": int(seed),
                        "neighbor_count": int(count),
                    }
                    for mol_id, value, delta, count in zip(
                        partitions["validation"], base, correction, neighbor_counts
                    )
                )
        reverse_error = float(
            np.max(np.abs(edge_model.predict(train_edge_features[:128]) + edge_model.predict(-train_edge_features[:128])))
        ) if len(train_edge_features) else 0.0
        if reverse_error > 1e-8:
            raise RuntimeError(f"Edge model violates antisymmetry: {reverse_error}")
        seed_metadata.append(
            {
                "seed": int(seed),
                "train_edges": int(len(train_edge_target)),
                "train_mean_residual": float(residual.mean()),
                "validation_neighbor_coverage": float((neighbor_counts > 0).mean()),
                "maximum_reverse_error": reverse_error,
            }
        )

    output = args.output_dir / args.target / f"fold_{args.fold}"
    metadata = {
        "target": args.target,
        "fold": args.fold,
        "method": config["name"],
        "train_labels_materialized": True,
        "validation_labels_materialized": False,
        "outer_test_predictions_generated": False,
        "outer_test_labels_materialized": False,
        "n_train": len(partitions["train"]),
        "n_validation": len(partitions["validation"]),
        "seed_metadata": seed_metadata,
        "wall_time_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "config": config,
    }
    prediction_path = output / "validation_predictions.parquet"
    metadata["prediction_sha256"] = freeze(pd.DataFrame(rows), prediction_path, metadata)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
