#!/usr/bin/env python3
"""Fit a train-edge-only symmetric gate and freeze label-free query artifacts."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from bps_gnn.dataset_graph import (
    apply_dual_propagation,
    apply_mass_aware_soft_propagation,
    apply_soft_propagation,
    make_fingerprints,
    query_neighbors,
    shuffled_gate,
    similarity_matched_gate,
    stable_rng,
    symmetric_edge_features,
    undirected_knn_pairs,
)
from bps_gnn.io import read_labels_for_ids, read_structure_table, sha256_file
from bps_gnn.models import GINERegressor
from bps_gnn.molecular_graphs import ATOM_DIM, BOND_DIM, load_or_build_graphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
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
    if any(set(result[left]) & set(result[right]) for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]):
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


def query_gate_probabilities(
    classifier,
    query_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    query_prediction: np.ndarray,
    reference_prediction: np.ndarray,
    neighbor_indices: list[np.ndarray],
    similarities: list[np.ndarray],
) -> list[np.ndarray]:
    result = []
    for index, (neighbors, similarity) in enumerate(zip(neighbor_indices, similarities)):
        if not len(neighbors):
            result.append(np.empty(0, dtype=float))
            continue
        features = symmetric_edge_features(
            np.repeat(query_embedding[index][None, :], len(neighbors), axis=0),
            reference_embedding[neighbors],
            np.repeat(query_prediction[index], len(neighbors)),
            reference_prediction[neighbors],
            similarity,
        )
        result.append(classifier.predict_proba(features)[:, 1])
    return result


def candidate_rows(
    *,
    target: str,
    fold: int,
    seed: int,
    partition: str,
    query_ids: list[str],
    query_base: np.ndarray,
    reference_base: np.ndarray,
    neighbor_indices: list[np.ndarray],
    similarities: list[np.ndarray],
    learned_gate: list[np.ndarray],
    strength: float,
    mass_aware: bool,
    highpass_gammas: list[float],
    random_replicates: int,
    random_seed: int,
) -> list[dict]:
    frames = []

    def add(method: str, replicate: int, values: np.ndarray) -> None:
        frames.extend(
            {
                "mol_id": mol_id,
                "prediction": float(value),
                "method": method,
                "replicate": replicate,
                "seed": seed,
            }
            for mol_id, value in zip(query_ids, values)
        )

    zero_gate = [np.zeros_like(value) for value in learned_gate]
    add("base", -1, query_base)
    add(
        "global",
        -1,
        apply_soft_propagation(
            query_base, reference_base, neighbor_indices, similarities, zero_gate, strength
        ),
    )
    add(
        "learned",
        -1,
        apply_soft_propagation(
            query_base,
            reference_base,
            neighbor_indices,
            similarities,
            learned_gate,
            strength,
        ),
    )
    similarity_gate = similarity_matched_gate(learned_gate, similarities)
    add(
        "similarity",
        -1,
        apply_soft_propagation(
            query_base,
            reference_base,
            neighbor_indices,
            similarities,
            similarity_gate,
            strength,
        ),
    )
    if mass_aware:
        add(
            "learned_mass_aware",
            -1,
            apply_mass_aware_soft_propagation(
                query_base,
                reference_base,
                neighbor_indices,
                similarities,
                learned_gate,
                strength,
            ),
        )
        add(
            "similarity_mass_aware",
            -1,
            apply_mass_aware_soft_propagation(
                query_base,
                reference_base,
                neighbor_indices,
                similarities,
                similarity_gate,
                strength,
            ),
        )
    for gamma in highpass_gammas:
        suffix = str(gamma).replace(".", "p")
        add(
            f"learned_dual_g{suffix}",
            -1,
            apply_dual_propagation(
                query_base,
                reference_base,
                neighbor_indices,
                similarities,
                learned_gate,
                strength,
                gamma,
            ),
        )
        add(
            f"similarity_dual_g{suffix}",
            -1,
            apply_dual_propagation(
                query_base,
                reference_base,
                neighbor_indices,
                similarities,
                similarity_gate,
                strength,
                gamma,
            ),
        )
    for replicate in range(random_replicates):
        rng = stable_rng(random_seed, target, fold, seed, partition, replicate)
        random_gate = shuffled_gate(learned_gate, rng)
        add(
            "random",
            replicate,
            apply_soft_propagation(
                query_base,
                reference_base,
                neighbor_indices,
                similarities,
                random_gate,
                strength,
            ),
        )
        if mass_aware:
            add(
                "random_mass_aware",
                replicate,
                apply_mass_aware_soft_propagation(
                    query_base,
                    reference_base,
                    neighbor_indices,
                    similarities,
                    random_gate,
                    strength,
                ),
            )
        for gamma in highpass_gammas:
            suffix = str(gamma).replace(".", "p")
            add(
                f"random_dual_g{suffix}",
                replicate,
                apply_dual_propagation(
                    query_base,
                    reference_base,
                    neighbor_indices,
                    similarities,
                    random_gate,
                    strength,
                    gamma,
                ),
            )
    return frames


def freeze_table(frame: pd.DataFrame, path: Path, metadata: dict, sort_by: list[str]) -> str:
    if any(column.lower().startswith("y_") for column in frame.columns):
        raise ValueError("Frozen query artifact contains response columns")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(sort_by, kind="stable").to_parquet(path, index=False)
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
    gate_config = yaml.safe_load(args.gate_config.read_text(encoding="utf-8"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    structures = read_structure_table(Path(base_config["data_root"]), args.target).reset_index(drop=True)
    mol_ids = structures["mol_id"].tolist()
    id_to_index = {value: index for index, value in enumerate(mol_ids)}
    partitions = partition_ids(args.target, args.fold)

    # The trainer reads outer-train responses only. Validation/test remain private.
    train_truth = read_labels_for_ids(
        Path(base_config["data_root"]), args.target, partitions["train"]
    ).set_index("mol_id")["y_pactivity"].loc[partitions["train"]]
    if set(train_truth.index) & (set(partitions["validation"]) | set(partitions["test"])):
        raise RuntimeError("Query label entered learned-gate trainer")

    graph_config = gate_config["graph"]
    fingerprints = make_fingerprints(
        structures["canonical_isomeric_smiles"].tolist(),
        int(graph_config["radius"]),
        int(graph_config["bits"]),
    )
    partition_indices = {
        name: [id_to_index[value] for value in values] for name, values in partitions.items()
    }
    train_fingerprints = [fingerprints[index] for index in partition_indices["train"]]
    train_left, train_right, train_similarity = undirected_knn_pairs(
        train_fingerprints,
        k=int(graph_config["query_to_train_k"]),
        floor=float(graph_config["similarity_floor"]),
    )
    query_graphs = {}
    for partition in ["validation", "test"]:
        query_graphs[partition] = query_neighbors(
            train_fingerprints,
            [fingerprints[index] for index in partition_indices[partition]],
            k=int(graph_config["query_to_train_k"]),
            floor=float(graph_config["similarity_floor"]),
        )
    graphs = load_or_build_graphs(
        structures["canonical_isomeric_smiles"].tolist(),
        Path(base_config["cache_root"]) / f"{args.target}.pt",
    )
    prediction_rows = {"validation": [], "test": []}
    gate_rows = {"validation": [], "test": []}
    seed_metadata = []
    for seed in base_config["seeds"]:
        checkpoint = args.base_run_dir / args.target / f"fold_{args.fold}" / f"seed_{seed}" / "checkpoint.pt"
        combined_indices = [
            *partition_indices["train"],
            *partition_indices["validation"],
            *partition_indices["test"],
        ]
        row_id, prediction, embedding = predict_embeddings(
            checkpoint,
            graphs,
            combined_indices,
            device,
            int(base_config["optimization"]["batch_size"]),
        )
        pred_map = {int(key): value for key, value in zip(row_id, prediction)}
        emb_map = {int(key): value for key, value in zip(row_id, embedding)}
        part_prediction = {
            name: np.asarray([pred_map[index] for index in indices])
            for name, indices in partition_indices.items()
        }
        part_embedding = {
            name: np.asarray([emb_map[index] for index in indices])
            for name, indices in partition_indices.items()
        }
        train_features = symmetric_edge_features(
            part_embedding["train"][train_left],
            part_embedding["train"][train_right],
            part_prediction["train"][train_left],
            part_prediction["train"][train_right],
            train_similarity,
        )
        train_y = train_truth.to_numpy()
        train_boundary = (
            np.abs(train_y[train_left] - train_y[train_right])
            >= float(gate_config["gate"]["boundary_activity_difference"])
        ).astype(int)
        if len(np.unique(train_boundary)) != 2:
            raise RuntimeError("Train edge boundary labels contain only one class")
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(gate_config["gate"]["C"]),
                class_weight="balanced",
                max_iter=int(gate_config["gate"]["max_iter"]),
                solver=gate_config["gate"]["solver"],
                random_state=int(seed),
            ),
        )
        classifier.fit(train_features, train_boundary)
        seed_metadata.append(
            {
                "seed": int(seed),
                "train_edges": int(len(train_boundary)),
                "train_boundary_prevalence": float(train_boundary.mean()),
                "gate_features": int(train_features.shape[1]),
            }
        )
        for partition in ["validation", "test"]:
            neighbor_indices, similarities = query_graphs[partition]
            learned_gate = query_gate_probabilities(
                classifier,
                part_embedding[partition],
                part_embedding["train"],
                part_prediction[partition],
                part_prediction["train"],
                neighbor_indices,
                similarities,
            )
            prediction_rows[partition].extend(
                candidate_rows(
                    target=args.target,
                    fold=args.fold,
                    seed=int(seed),
                    partition=partition,
                    query_ids=partitions[partition],
                    query_base=part_prediction[partition],
                    reference_base=part_prediction["train"],
                    neighbor_indices=neighbor_indices,
                    similarities=similarities,
                    learned_gate=learned_gate,
                    strength=float(gate_config["propagation"]["lambda"]),
                    mass_aware=bool(
                        gate_config["propagation"].get("mass_aware_smoke", False)
                    ),
                    highpass_gammas=[
                        float(value)
                        for value in gate_config["propagation"].get(
                            "highpass_gammas", []
                        )
                    ],
                    random_replicates=int(gate_config["controls"]["random_soft_gate_permutations"]),
                    random_seed=int(gate_config["controls"]["seed"]),
                )
            )
            for query_id, neighbors, similarity, probability in zip(
                partitions[partition], neighbor_indices, similarities, learned_gate
            ):
                gate_rows[partition].extend(
                    {
                        "query_id": query_id,
                        "reference_id": partitions["train"][int(reference)],
                        "similarity": float(value),
                        "q_learned": float(q),
                        "seed": int(seed),
                    }
                    for reference, value, q in zip(neighbors, similarity, probability)
                )

    output = args.output_dir / args.target / f"fold_{args.fold}"
    metadata = {
        "target": args.target,
        "fold": args.fold,
        "method": gate_config["name"],
        "train_labels_materialized": True,
        "validation_labels_materialized": False,
        "outer_test_labels_materialized": False,
        "n_train": len(partitions["train"]),
        "n_validation": len(partitions["validation"]),
        "n_test": len(partitions["test"]),
        "seed_metadata": seed_metadata,
        "wall_time_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "config": gate_config,
    }
    hashes = {}
    for partition in ["validation", "test"]:
        hashes[f"{partition}_predictions"] = freeze_table(
            pd.DataFrame(prediction_rows[partition]),
            output / f"{partition}_predictions.parquet",
            metadata,
            ["method", "replicate", "seed", "mol_id"],
        )
        hashes[f"{partition}_gate"] = freeze_table(
            pd.DataFrame(gate_rows[partition]),
            output / f"{partition}_gate.parquet",
            metadata,
            ["seed", "query_id", "reference_id"],
        )
    metadata["artifact_hashes"] = hashes
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
