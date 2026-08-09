#!/usr/bin/env python3
"""Evaluate frozen incomplete-graph Hodge candidates on unseen pair edges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_frozen_prediction

try:
    from build_deepdelta_delta_lgbm_predictions import (
        selected_response_values,
        structure_only_split,
    )
    from build_deepdelta_sparse_projection import density_key, is_connected, ordered_values
except ModuleNotFoundError:
    from scripts.build_deepdelta_delta_lgbm_predictions import (
        selected_response_values,
        structure_only_split,
    )
    from scripts.build_deepdelta_sparse_projection import (
        density_key,
        is_connected,
        ordered_values,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-prediction-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def masked_metrics(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if prediction.shape != truth.shape or mask.shape != truth.shape:
        raise ValueError("prediction, truth, and mask shapes must match")
    if not mask.any():
        raise ValueError("metric mask is empty")
    error = prediction[mask] - truth[mask]
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
    }


def bootstrap_relative(
    candidate: np.ndarray,
    reference: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    if len(candidate) != len(reference) or len(candidate) == 0:
        raise ValueError("paired non-empty arrays are required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(candidate), size=(iterations, len(candidate)))
    changes = candidate[indices].mean(axis=1) / reference[indices].mean(axis=1) - 1.0
    lower, upper = np.quantile(changes, [0.025, 0.975])
    return float(lower), float(upper)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_config_path = Path(str(config["source_config"]))
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    datasets = [str(value) for value in config["datasets"]]
    densities = [float(value) for value in config["sampling"]["undirected_density"]]
    seeds = [int(value) for value in config["sampling"]["random_states"]]

    # Verify every source/candidate hash, identifier, graph, and metadata field
    # before selecting a single held-out response value.
    candidates: dict[
        tuple[str, float, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    test_partitions: dict[str, np.ndarray] = {}
    for dataset_index, dataset in enumerate(datasets):
        _, _, test_index = structure_only_split(
            args.data_root / f"{dataset}.csv", source_config
        )
        test_partitions[dataset] = test_index
        n_nodes = len(test_index)
        edge_ids = [
            f"edge_{i:04d}_{j:04d}" for i in range(n_nodes) for j in range(n_nodes)
        ]
        node_ids = [f"position_{i:04d}" for i in range(n_nodes)]
        upper_i, upper_j = np.triu_indices(n_nodes, k=1)
        upper_ids = [
            f"undirected_{int(i):04d}_{int(j):04d}" for i, j in zip(upper_i, upper_j)
        ]
        source_dir = args.source_prediction_root / dataset
        source_metadata = json.loads((source_dir / "run.json").read_text(encoding="utf-8"))
        if source_metadata["held_out_response_materialized"] is not False:
            raise RuntimeError(f"{dataset}: source firewall failed")
        direct = ordered_values(
            read_frozen_prediction(source_dir / "direct_flow.parquet"),
            edge_ids,
            f"{dataset}/direct_flow",
        ).reshape(n_nodes, n_nodes)
        complete_potential = ordered_values(
            read_frozen_prediction(source_dir / "hodge_potential.parquet"),
            node_ids,
            f"{dataset}/complete_potential",
        )

        for density in densities:
            for seed in seeds:
                run_dir = args.candidate_root / dataset / density_key(density) / f"seed_{seed}"
                metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
                required = {
                    "dataset": dataset,
                    "dataset_index": dataset_index,
                    "target_undirected_density": density,
                    "random_state": seed,
                    "n_nodes": n_nodes,
                    "held_out_response_materialized": False,
                    "source_direct_flow_sha256": source_metadata["direct_flow_sha256"],
                }
                for key, expected in required.items():
                    if metadata[key] != expected:
                        raise RuntimeError(f"{dataset}/{density}/{seed}: metadata mismatch {key}")
                potential = ordered_values(
                    read_frozen_prediction(run_dir / "sparse_potential.parquet"),
                    node_ids,
                    f"{dataset}/{density}/{seed}/potential",
                )
                mask_values = ordered_values(
                    read_frozen_prediction(run_dir / "observed_mask.parquet"),
                    upper_ids,
                    f"{dataset}/{density}/{seed}/mask",
                )
                if not np.isin(mask_values, [0.0, 1.0]).all():
                    raise RuntimeError(f"{dataset}/{density}/{seed}: mask is not binary")
                upper_mask = mask_values.astype(bool)
                if int(upper_mask.sum()) != int(metadata["n_observed_undirected_edges"]):
                    raise RuntimeError(f"{dataset}/{density}/{seed}: edge count changed")
                if not is_connected(n_nodes, upper_i[upper_mask], upper_j[upper_mask]):
                    raise RuntimeError(f"{dataset}/{density}/{seed}: graph is disconnected")
                candidates[(dataset, density, seed)] = (
                    direct,
                    complete_potential,
                    potential,
                    upper_mask,
                )

    graph_rows: list[dict[str, object]] = []
    theorem_rows: list[dict[str, object]] = []
    tolerance = float(
        config["decision"]["observed_edge_pythagorean_relative_tolerance"]
    )
    # Held-out responses are selected only after the complete candidate preflight.
    for dataset in datasets:
        response = selected_response_values(
            args.data_root / f"{dataset}.csv", test_partitions[dataset]
        )
        truth = response[None, :] - response[:, None]
        n_nodes = len(response)
        upper_i, upper_j = np.triu_indices(n_nodes, k=1)
        off_diagonal = ~np.eye(n_nodes, dtype=bool)
        for density in densities:
            for seed in seeds:
                direct, complete_potential, sparse_potential, upper_mask = candidates[
                    (dataset, density, seed)
                ]
                observed_mask = np.zeros((n_nodes, n_nodes), dtype=bool)
                selected_i, selected_j = upper_i[upper_mask], upper_j[upper_mask]
                observed_mask[selected_i, selected_j] = True
                observed_mask[selected_j, selected_i] = True
                heldout_mask = off_diagonal & ~observed_mask
                sparse = sparse_potential[None, :] - sparse_potential[:, None]
                complete = complete_potential[None, :] - complete_potential[:, None]
                observed_raw = masked_metrics(direct, truth, observed_mask)
                observed_sparse = masked_metrics(sparse, truth, observed_mask)
                heldout_raw = masked_metrics(direct, truth, heldout_mask)
                heldout_sparse = masked_metrics(sparse, truth, heldout_mask)
                heldout_complete = masked_metrics(complete, truth, heldout_mask)
                graph_rows.append(
                    {
                        "dataset": dataset,
                        "density": density,
                        "seed": seed,
                        "n_nodes": n_nodes,
                        "observed_undirected_edges": int(upper_mask.sum()),
                        "effective_density": float(upper_mask.mean()),
                        "heldout_direct_rmse": heldout_raw["rmse"],
                        "heldout_sparse_hodge_rmse": heldout_sparse["rmse"],
                        "heldout_complete_hodge_rmse": heldout_complete["rmse"],
                        "heldout_direct_mae": heldout_raw["mae"],
                        "heldout_sparse_hodge_mae": heldout_sparse["mae"],
                        "heldout_complete_hodge_mae": heldout_complete["mae"],
                        "observed_direct_rmse": observed_raw["rmse"],
                        "observed_sparse_hodge_rmse": observed_sparse["rmse"],
                        "observed_direct_mae": observed_raw["mae"],
                        "observed_sparse_hodge_mae": observed_sparse["mae"],
                    }
                )
                raw_error = direct[observed_mask] - truth[observed_mask]
                sparse_error = sparse[observed_mask] - truth[observed_mask]
                removed = direct[observed_mask] - sparse[observed_mask]
                raw_sse = float(np.square(raw_error).sum())
                sparse_sse = float(np.square(sparse_error).sum())
                removed_energy = float(np.square(removed).sum())
                discrepancy = abs(raw_sse - sparse_sse - removed_energy) / max(
                    raw_sse, 1.0
                )
                theorem_rows.append(
                    {
                        "dataset": dataset,
                        "density": density,
                        "seed": seed,
                        "raw_observed_sse": raw_sse,
                        "sparse_hodge_observed_sse": sparse_sse,
                        "removed_energy": removed_energy,
                        "pythagorean_relative_discrepancy": discrepancy,
                        "passed": bool(discrepancy <= tolerance),
                    }
                )

    graph_frame = pd.DataFrame.from_records(graph_rows)
    value_columns = [
        column
        for column in graph_frame.columns
        if column not in {"dataset", "density", "seed", "n_nodes"}
    ]
    dataset_density = (
        graph_frame.groupby(["dataset", "density"], as_index=False)[value_columns]
        .mean()
        .sort_values(["density", "dataset"])
    )
    statistics: list[dict[str, object]] = []
    iterations = int(config["evaluation"]["bootstrap_iterations"])
    for density_index, density in enumerate(densities):
        frame = dataset_density[dataset_density["density"] == density]
        for metric_index, metric in enumerate(["rmse", "mae"]):
            reference = frame[f"heldout_direct_{metric}"].to_numpy(dtype=float)
            candidate = frame[f"heldout_sparse_hodge_{metric}"].to_numpy(dtype=float)
            lower, upper = bootstrap_relative(
                candidate,
                reference,
                iterations,
                297000 + 10 * density_index + metric_index,
            )
            statistics.append(
                {
                    "density": density,
                    "metric": f"heldout_pair_{metric}",
                    "datasets": len(frame),
                    "reference_macro": float(reference.mean()),
                    "candidate_macro": float(candidate.mean()),
                    "relative_change": float(candidate.mean() / reference.mean() - 1.0),
                    "bootstrap95_lower": lower,
                    "bootstrap95_upper": upper,
                    "favorable_datasets": int((candidate < reference).sum()),
                    "favorable_fraction": float((candidate < reference).mean()),
                }
            )
    statistics_frame = pd.DataFrame.from_records(statistics)
    primary_density = float(config["evaluation"]["primary_density"])

    def primary(metric: str) -> pd.Series:
        selected = statistics_frame[
            (statistics_frame["density"] == primary_density)
            & (statistics_frame["metric"] == metric)
        ]
        if len(selected) != 1:
            raise RuntimeError(f"missing primary statistic: {metric}")
        return selected.iloc[0]

    rmse = primary("heldout_pair_rmse")
    mae = primary("heldout_pair_mae")
    theorem_frame = pd.DataFrame.from_records(theorem_rows)
    decision_config = config["decision"]
    checks = {
        "primary_heldout_rmse_direction": bool(
            rmse["favorable_fraction"]
            >= float(
                decision_config[
                    "minimum_primary_heldout_rmse_improved_dataset_fraction"
                ]
            )
        ),
        "primary_heldout_rmse_bootstrap": bool(
            rmse["bootstrap95_upper"]
            < float(decision_config["maximum_primary_heldout_rmse_bootstrap_upper"])
        ),
        "primary_heldout_mae_direction": bool(
            mae["favorable_fraction"]
            >= float(
                decision_config[
                    "minimum_primary_heldout_mae_improved_dataset_fraction"
                ]
            )
        ),
        "primary_heldout_mae_bootstrap": bool(
            mae["bootstrap95_upper"]
            < float(decision_config["maximum_primary_heldout_mae_bootstrap_upper"])
        ),
        # Any disconnected candidate raises during the response-free preflight.
        "all_sampled_graphs_connected": bool(
            len(graph_frame) == len(datasets) * len(densities) * len(seeds)
        ),
        "all_observed_edge_identities": bool(theorem_frame["passed"].all()),
    }
    decision = {
        "dataset": config["name"],
        "primary_density": primary_density,
        "checks": checks,
        "strong_sparse_projection_go": bool(all(checks.values())),
        "candidate_hash_preflight_before_sparse_metrics": True,
        "evidence_status": "postprimary_generality_not_independent_confirmation",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    graph_frame.to_csv(args.output_dir / "graph_metrics.csv", index=False)
    dataset_density.to_csv(args.output_dir / "dataset_density_metrics.csv", index=False)
    statistics_frame.to_csv(args.output_dir / "statistics.csv", index=False)
    theorem_frame.to_csv(args.output_dir / "pythagorean_audit.csv", index=False)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "statistics": statistics}, indent=2))


if __name__ == "__main__":
    main()
