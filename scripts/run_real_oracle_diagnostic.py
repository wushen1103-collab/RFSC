#!/usr/bin/env python3
"""Run the authorized validation-only real boundary-cut diagnostic for one fold."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.loader import DataLoader

from bps_gnn.dataset_graph import apply_dual_propagation
from bps_gnn.io import read_labels_for_ids, read_structure_table
from bps_gnn.models import GINERegressor
from bps_gnn.molecular_graphs import ATOM_DIM, BOND_DIM, load_or_build_graphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--diagnostic-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def partition_ids(target: str, fold: int) -> tuple[list[str], list[str], list[str]]:
    assignments = pd.read_parquet(Path("folds/scg") / target / "outer_folds.parquet")
    n_folds = int(assignments["outer_fold"].nunique())
    if fold not in set(assignments["outer_fold"]):
        raise ValueError(f"Invalid outer fold {fold} for {target}")
    valid_fold = (fold + 1) % n_folds
    train = assignments.loc[
        ~assignments["outer_fold"].isin([fold, valid_fold]), "mol_id"
    ].tolist()
    valid = assignments.loc[assignments["outer_fold"] == valid_fold, "mol_id"].tolist()
    test = assignments.loc[assignments["outer_fold"] == fold, "mol_id"].tolist()
    return train, valid, test


def make_fingerprints(smiles: list[str], radius: int, bits: int):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    result = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid standardized SMILES at row {index}")
        result.append(generator.GetFingerprint(molecule))
    return result


def query_neighbors(
    train_fingerprints: list,
    query_fingerprints: list,
    *,
    k: int,
    floor: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    indices, similarities = [], []
    for fingerprint in query_fingerprints:
        values = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprint, train_fingerprints),
            dtype=float,
        )
        eligible = np.flatnonzero(values >= floor)
        if len(eligible) > k:
            order = np.argsort(-values[eligible], kind="stable")[:k]
            eligible = eligible[order]
        else:
            eligible = eligible[np.argsort(-values[eligible], kind="stable")]
        indices.append(eligible.astype(int))
        similarities.append(values[eligible])
    return indices, similarities


def predict_partition(
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
    loader = DataLoader(
        [copy.copy(graphs[index]) for index in row_indices],
        batch_size=batch_size,
        shuffle=False,
    )
    output_indices, output_values = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = (
                model(batch).float().cpu().numpy() * checkpoint["scale"]
                + checkpoint["center"]
            )
            output_indices.extend(batch.row_id.cpu().numpy().tolist())
            output_values.extend(prediction.tolist())
    return np.asarray(output_indices, dtype=int), np.asarray(output_values, dtype=float)


def apply_propagation(
    base_query: np.ndarray,
    base_reference: np.ndarray,
    neighbor_indices: list[np.ndarray],
    similarities: list[np.ndarray],
    removed: list[np.ndarray],
    strength: float,
) -> np.ndarray:
    result = base_query.copy()
    for query_index, (reference_index, weight, cut) in enumerate(
        zip(neighbor_indices, similarities, removed)
    ):
        keep = ~cut
        if not np.any(keep):
            continue
        message = np.average(base_reference[reference_index[keep]], weights=weight[keep])
        result[query_index] = (1.0 - strength) * base_query[query_index] + strength * message
    return result


def metric_row(
    truth: pd.Series,
    prediction: pd.Series,
    mmp_edges: pd.DataFrame,
) -> dict[str, float | int]:
    residual = prediction.loc[truth.index] - truth
    edges = mmp_edges[
        mmp_edges["mol_id_lo"].isin(truth.index)
        & mmp_edges["mol_id_hi"].isin(truth.index)
    ].copy()
    edges["delta_y"] = edges["mol_id_hi"].map(truth) - edges["mol_id_lo"].map(truth)
    cliff = edges[edges["delta_y"].abs() >= 1.0]
    endpoints = sorted(set(cliff["mol_id_lo"]) | set(cliff["mol_id_hi"]))
    if not endpoints:
        raise RuntimeError("Authorized validation fold has no MMP cliff endpoint")
    predicted_delta = cliff["mol_id_hi"].map(prediction) - cliff["mol_id_lo"].map(
        prediction
    )
    return {
        "overall_mae": float(residual.abs().mean()),
        "cliff_mae": float((prediction.loc[endpoints] - truth.loc[endpoints]).abs().mean()),
        "pair_delta_mae": float((predicted_delta - cliff["delta_y"]).abs().mean()),
        "n_query": int(len(truth)),
        "n_cliff_endpoints": int(len(endpoints)),
        "n_cliff_pairs": int(len(cliff)),
    }


def stable_rng(seed: int, target: str, fold: int, model_seed: int, replicate: int):
    payload = f"{seed}\0{target}\0{fold}\0{model_seed}\0{replicate}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(value)


def main() -> None:
    args = parse_args()
    diagnostic = yaml.safe_load(args.diagnostic_config.read_text(encoding="utf-8"))
    base_config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    structures = read_structure_table(Path(base_config["data_root"]), args.target).reset_index(
        drop=True
    )
    mol_ids = structures["mol_id"].tolist()
    id_to_index = {value: index for index, value in enumerate(mol_ids)}
    train_ids, valid_ids, test_ids = partition_ids(args.target, args.fold)
    if set(train_ids) & set(valid_ids) or (set(train_ids) | set(valid_ids)) & set(test_ids):
        raise RuntimeError("SCG partition overlap")

    # This is the sole label read: only authorized train and validation IDs.
    labels = read_labels_for_ids(
        Path(base_config["data_root"]), args.target, [*train_ids, *valid_ids]
    ).set_index("mol_id")["y_pactivity"]
    if set(labels.index) & set(test_ids):
        raise RuntimeError("Outer-test label entered authorized oracle diagnostic")
    train_truth = labels.loc[train_ids]
    valid_truth = labels.loc[valid_ids]

    graph_config = diagnostic["graph"]
    all_fingerprints = make_fingerprints(
        structures["canonical_isomeric_smiles"].tolist(),
        radius=int(graph_config["radius"]),
        bits=int(graph_config["bits"]),
    )
    train_indices = [id_to_index[value] for value in train_ids]
    valid_indices = [id_to_index[value] for value in valid_ids]
    neighbor_index, neighbor_similarity = query_neighbors(
        [all_fingerprints[index] for index in train_indices],
        [all_fingerprints[index] for index in valid_indices],
        k=int(graph_config["query_to_train_k"]),
        floor=float(graph_config["similarity_floor"]),
    )
    graphs = load_or_build_graphs(
        structures["canonical_isomeric_smiles"].tolist(),
        Path(base_config["cache_root"]) / f"{args.target}.pt",
    )
    mmp_edges = pd.read_parquet(
        Path("graphs/mmp") / args.target / "eligible_mmp_edges.parquet"
    )
    rows = []
    boundary_threshold = float(diagnostic["boundary"]["activity_difference"])
    lambdas = [float(value) for value in diagnostic["propagation"]["lambdas"]]
    random_replicates = int(diagnostic["random"]["replicates"])
    highpass_gammas = [
        float(value) for value in diagnostic["propagation"].get("highpass_gammas", [])
    ]
    for model_seed in base_config["seeds"]:
        checkpoint = (
            args.base_run_dir
            / args.target
            / f"fold_{args.fold}"
            / f"seed_{model_seed}"
            / "checkpoint.pt"
        )
        train_row_id, train_prediction = predict_partition(
            checkpoint,
            graphs,
            train_indices,
            device,
            int(base_config["optimization"]["batch_size"]),
        )
        valid_row_id, valid_prediction = predict_partition(
            checkpoint,
            graphs,
            valid_indices,
            device,
            int(base_config["optimization"]["batch_size"]),
        )
        train_prediction_map = dict(zip(train_row_id, train_prediction))
        valid_prediction_map = dict(zip(valid_row_id, valid_prediction))
        train_base = np.asarray(
            [train_prediction_map[id_to_index[value]] for value in train_ids]
        )
        valid_base = np.asarray(
            [valid_prediction_map[id_to_index[value]] for value in valid_ids]
        )
        base_series = pd.Series(valid_base, index=valid_ids)
        rows.append(
            {
                "target": args.target,
                "fold": args.fold,
                "seed": model_seed,
                "method": "base",
                "lambda": 0.0,
                "random_replicate": -1,
                **metric_row(valid_truth, base_series, mmp_edges),
            }
        )
        oracle_removed, similarity_removed = [], []
        oracle_cut_counts = []
        for query_id, references, similarity in zip(
            valid_ids, neighbor_index, neighbor_similarity
        ):
            delta = np.abs(train_truth.iloc[references].to_numpy() - valid_truth.loc[query_id])
            oracle_cut = delta >= boundary_threshold
            count = int(oracle_cut.sum())
            similarity_cut = np.zeros(len(references), dtype=bool)
            if count:
                similarity_cut[np.argsort(similarity, kind="stable")[:count]] = True
            oracle_removed.append(oracle_cut)
            similarity_removed.append(similarity_cut)
            oracle_cut_counts.append(count)
        global_removed = [np.zeros(len(value), dtype=bool) for value in neighbor_index]

        for strength in lambdas:
            for method, removed in [
                ("global", global_removed),
                ("oracle", oracle_removed),
                ("similarity", similarity_removed),
            ]:
                prediction = apply_propagation(
                    valid_base,
                    train_base,
                    neighbor_index,
                    neighbor_similarity,
                    removed,
                    strength,
                )
                rows.append(
                    {
                        "target": args.target,
                        "fold": args.fold,
                        "seed": model_seed,
                        "method": method,
                        "lambda": strength,
                        "random_replicate": -1,
                        **metric_row(
                            valid_truth, pd.Series(prediction, index=valid_ids), mmp_edges
                        ),
                    }
                )
                for gamma in highpass_gammas:
                    suffix = str(gamma).replace(".", "p")
                    gate = [value.astype(float) for value in removed]
                    dual_prediction = apply_dual_propagation(
                        valid_base,
                        train_base,
                        neighbor_index,
                        neighbor_similarity,
                        gate,
                        strength,
                        gamma,
                    )
                    rows.append(
                        {
                            "target": args.target,
                            "fold": args.fold,
                            "seed": model_seed,
                            "method": f"{method}_dual_g{suffix}",
                            "lambda": strength,
                            "random_replicate": -1,
                            **metric_row(
                                valid_truth,
                                pd.Series(dual_prediction, index=valid_ids),
                                mmp_edges,
                            ),
                        }
                    )
            for replicate in range(random_replicates):
                rng = stable_rng(
                    int(diagnostic["random"]["seed"]),
                    args.target,
                    args.fold,
                    int(model_seed),
                    replicate,
                )
                random_removed = []
                for references, count in zip(neighbor_index, oracle_cut_counts):
                    cut = np.zeros(len(references), dtype=bool)
                    if count:
                        cut[rng.choice(len(references), size=count, replace=False)] = True
                    random_removed.append(cut)
                prediction = apply_propagation(
                    valid_base,
                    train_base,
                    neighbor_index,
                    neighbor_similarity,
                    random_removed,
                    strength,
                )
                rows.append(
                    {
                        "target": args.target,
                        "fold": args.fold,
                        "seed": model_seed,
                        "method": "random",
                        "lambda": strength,
                        "random_replicate": replicate,
                        **metric_row(
                            valid_truth, pd.Series(prediction, index=valid_ids), mmp_edges
                        ),
                    }
                )
                for gamma in highpass_gammas:
                    suffix = str(gamma).replace(".", "p")
                    dual_prediction = apply_dual_propagation(
                        valid_base,
                        train_base,
                        neighbor_index,
                        neighbor_similarity,
                        [value.astype(float) for value in random_removed],
                        strength,
                        gamma,
                    )
                    rows.append(
                        {
                            "target": args.target,
                            "fold": args.fold,
                            "seed": model_seed,
                            "method": f"random_dual_g{suffix}",
                            "lambda": strength,
                            "random_replicate": replicate,
                            **metric_row(
                                valid_truth,
                                pd.Series(dual_prediction, index=valid_ids),
                                mmp_edges,
                            ),
                        }
                    )

    output = args.output_dir / args.target / f"fold_{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "diagnostic_seed_metrics.csv", index=False)
    edge_count = int(sum(map(len, neighbor_index)))
    metadata = {
        "target": args.target,
        "fold": args.fold,
        "partition": "outer_validation_pseudo_query",
        "n_train": len(train_ids),
        "n_validation": len(valid_ids),
        "n_outer_test_ids_held_private": len(test_ids),
        "outer_test_labels_materialized": False,
        "query_reference_edges": edge_count,
        "queries_without_neighbors": int(sum(len(value) == 0 for value in neighbor_index)),
        "oracle_cut_edges": int(
            sum(int(value.sum()) for value in oracle_removed)
        ),
        "diagnostic_config": diagnostic,
    }
    (output / "authorization.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
