#!/usr/bin/env python3
"""Evaluate frozen direct Delta-LightGBM Hodge repair after held-out response release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_frozen_prediction
from bps_gnn.pair_flow import antisymmetric_pair_flow

try:
    from build_deepdelta_delta_lgbm_predictions import (
        selected_response_values,
        sha256_strings,
        structure_only_split,
    )
    from evaluate_deepdelta_cycle_projection import flow_metrics
except ModuleNotFoundError:
    from scripts.build_deepdelta_delta_lgbm_predictions import (
        selected_response_values,
        sha256_strings,
        structure_only_split,
    )
    from scripts.evaluate_deepdelta_cycle_projection import flow_metrics

try:
    from evaluate_chembl36_temporal import bootstrap_relative
except ModuleNotFoundError:
    from scripts.evaluate_chembl36_temporal import bootstrap_relative


LOWER_METRICS = ["pair_rmse", "pair_mae", "high_delta_pair_mae"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--held-out-fold", type=int)
    return parser.parse_args()


def ordered_values(series: pd.Series, ids: list[str], artifact: str) -> np.ndarray:
    if set(series.index) != set(ids):
        raise RuntimeError(f"{artifact}: frozen IDs do not match expected held-out partition")
    values = series.reindex(ids).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{artifact}: non-finite prediction")
    return values


def comparison_statistics(
    task_metrics: pd.DataFrame,
    candidate_method: str,
    reference_method: str,
    iterations: int,
    seed: int,
) -> list[dict[str, object]]:
    candidate = task_metrics[task_metrics["method"] == candidate_method].set_index("dataset")
    reference = task_metrics[task_metrics["method"] == reference_method].set_index("dataset")
    records: list[dict[str, object]] = []
    for metric_index, metric in enumerate(LOWER_METRICS):
        paired = pd.concat(
            [candidate[metric].rename("candidate"), reference[metric].rename("reference")],
            axis=1,
        ).dropna()
        lower, upper = bootstrap_relative(
            paired["candidate"].to_numpy(),
            paired["reference"].to_numpy(),
            iterations,
            seed + metric_index,
        )
        records.append(
            {
                "candidate": candidate_method,
                "reference": reference_method,
                "metric": metric,
                "datasets": int(len(paired)),
                "reference_macro": float(paired["reference"].mean()),
                "candidate_macro": float(paired["candidate"].mean()),
                "relative_change": float(
                    paired["candidate"].mean() / paired["reference"].mean() - 1.0
                ),
                "bootstrap95_lower": lower,
                "bootstrap95_upper": upper,
                "favorable_datasets": int((paired["candidate"] < paired["reference"]).sum()),
                "favorable_fraction": float((paired["candidate"] < paired["reference"]).mean()),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.held_out_fold is not None:
        config["split"]["held_out_fold"] = int(args.held_out_fold)
    datasets = [str(value) for value in config["datasets"]]

    # Hash, ID, and partition preflight before any held-out response is selected.
    candidates: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for dataset in datasets:
        path = args.data_root / f"{dataset}.csv"
        structures, train_index, test_index = structure_only_split(path, config)
        run_dir = args.prediction_root / dataset
        metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if metadata["held_out_response_materialized"] is not False:
            raise RuntimeError(f"{dataset}: candidate firewall metadata is false")
        if metadata["train_indices_sha256"] != sha256_strings(
            [str(int(value)) for value in train_index]
        ):
            raise RuntimeError(f"{dataset}: train partition changed")
        if metadata["test_indices_sha256"] != sha256_strings(
            [str(int(value)) for value in test_index]
        ):
            raise RuntimeError(f"{dataset}: test partition changed")
        if metadata["test_smiles_sha256"] != sha256_strings(
            [structures[int(value)] for value in test_index]
        ):
            raise RuntimeError(f"{dataset}: held-out structures changed")
        n_test = len(test_index)
        edge_ids = [f"edge_{i:04d}_{j:04d}" for i in range(n_test) for j in range(n_test)]
        node_ids = [f"position_{i:04d}" for i in range(n_test)]
        direct = ordered_values(
            read_frozen_prediction(run_dir / "direct_flow.parquet"), edge_ids, "direct_flow"
        ).reshape(n_test, n_test)
        hodge_potential = ordered_values(
            read_frozen_prediction(run_dir / "hodge_potential.parquet"),
            node_ids,
            "hodge_potential",
        )
        absolute_potential = ordered_values(
            read_frozen_prediction(run_dir / "absolute_lgbm_potential.parquet"),
            node_ids,
            "absolute_lgbm_potential",
        )
        candidates[dataset] = (test_index, direct, hodge_potential, absolute_potential)

    rows: list[dict[str, object]] = []
    theorem_rows: list[dict[str, object]] = []
    quantile = float(config["evaluation"]["high_delta_quantile"])
    tolerance = float(config["decision"]["pythagorean_relative_tolerance"])

    # Only after all prediction hashes/partitions pass do we select held-out responses.
    for dataset in datasets:
        path = args.data_root / f"{dataset}.csv"
        test_index, direct, hodge_potential, absolute_potential = candidates[dataset]
        response = selected_response_values(path, test_index)
        truth = response[None, :] - response[:, None]
        hodge = hodge_potential[None, :] - hodge_potential[:, None]
        anti = antisymmetric_pair_flow(direct)
        absolute = absolute_potential[None, :] - absolute_potential[:, None]
        off_diagonal = ~np.eye(len(truth), dtype=bool)
        cutoff = float(np.quantile(np.abs(truth[off_diagonal]), quantile))
        high_mask = off_diagonal & (np.abs(truth) >= cutoff)
        for method, prediction in {
            "direct_delta_lgbm": direct,
            "antisymmetric": anti,
            "hodge_cycle_consistent": hodge,
            "absolute_lgbm_difference": absolute,
        }.items():
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n_nodes": len(truth),
                    "n_off_diagonal_pairs": int(off_diagonal.sum()),
                    "high_delta_cutoff": cutoff,
                    "n_high_delta_pairs": int(high_mask.sum()),
                    **flow_metrics(prediction, truth, high_mask),
                }
            )
        raw_sse = float(np.square(direct - truth).sum())
        hodge_sse = float(np.square(hodge - truth).sum())
        removed_sse = float(np.square(direct - hodge).sum())
        discrepancy = abs(raw_sse - hodge_sse - removed_sse) / max(raw_sse, 1.0)
        theorem_rows.append(
            {
                "dataset": dataset,
                "raw_sse": raw_sse,
                "hodge_sse": hodge_sse,
                "removed_energy": removed_sse,
                "pythagorean_relative_discrepancy": discrepancy,
                "passed": bool(discrepancy <= tolerance),
            }
        )

    task_metrics = pd.DataFrame.from_records(rows)
    iterations = int(config["evaluation"]["bootstrap_iterations"])
    statistics = (
        comparison_statistics(
            task_metrics,
            "hodge_cycle_consistent",
            "direct_delta_lgbm",
            iterations,
            296001,
        )
        + comparison_statistics(
            task_metrics,
            "hodge_cycle_consistent",
            "antisymmetric",
            iterations,
            296101,
        )
        + comparison_statistics(
            task_metrics,
            "direct_delta_lgbm",
            "absolute_lgbm_difference",
            iterations,
            296201,
        )
    )
    statistics_frame = pd.DataFrame.from_records(statistics)

    def stat(candidate: str, reference: str, metric: str) -> pd.Series:
        selected = statistics_frame[
            (statistics_frame["candidate"] == candidate)
            & (statistics_frame["reference"] == reference)
            & (statistics_frame["metric"] == metric)
        ]
        if len(selected) != 1:
            raise RuntimeError(f"missing contrast {candidate}/{reference}/{metric}")
        return selected.iloc[0]

    primary_rmse = stat("hodge_cycle_consistent", "direct_delta_lgbm", "pair_rmse")
    primary_mae = stat("hodge_cycle_consistent", "direct_delta_lgbm", "pair_mae")
    primary_high = stat(
        "hodge_cycle_consistent", "direct_delta_lgbm", "high_delta_pair_mae"
    )
    anti_rmse = stat("hodge_cycle_consistent", "antisymmetric", "pair_rmse")
    quality_rmse = stat("direct_delta_lgbm", "absolute_lgbm_difference", "pair_rmse")
    decision_config = config["decision"]
    theorem_frame = pd.DataFrame.from_records(theorem_rows)
    checks = {
        "rmse_all_datasets_improve": bool(
            primary_rmse["favorable_fraction"]
            >= float(decision_config["minimum_pair_rmse_improved_dataset_fraction"])
        ),
        "rmse_bootstrap_upper_below_zero": bool(
            primary_rmse["bootstrap95_upper"]
            < float(decision_config["maximum_pair_rmse_bootstrap_upper"])
        ),
        "mae_at_least_8_of_10_improve": bool(
            primary_mae["favorable_fraction"]
            >= float(decision_config["minimum_pair_mae_improved_dataset_fraction"])
        ),
        "mae_bootstrap_upper_below_zero": bool(
            primary_mae["bootstrap95_upper"]
            < float(decision_config["maximum_pair_mae_bootstrap_upper"])
        ),
        "high_delta_mae_not_worse": bool(
            primary_high["relative_change"]
            <= float(decision_config["maximum_high_delta_pair_mae_relative_change"])
        ),
        "hodge_beats_antisymmetrization": bool(
            anti_rmse["bootstrap95_upper"]
            < float(decision_config["maximum_hodge_vs_antisymmetric_rmse_bootstrap_upper"])
        ),
        "pythagorean_identity_all_datasets": bool(theorem_frame["passed"].all()),
        "direct_model_quality_not_catastrophic": bool(
            quality_rmse["relative_change"]
            <= float(decision_config["maximum_direct_vs_absolute_rmse_relative_change"])
        ),
    }
    decision = {
        "dataset": config["name"],
        "held_out_fold": int(config["split"]["held_out_fold"]),
        "candidate": "hodge_cycle_consistent",
        "baseline": "direct_delta_lgbm",
        "checks": checks,
        "strong_second_architecture_go": bool(all(checks.values())),
        "prediction_hash_preflight_before_held_out_responses": True,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_metrics.to_csv(args.output_dir / "dataset_metrics.csv", index=False)
    statistics_frame.to_csv(args.output_dir / "statistics.csv", index=False)
    theorem_frame.to_csv(args.output_dir / "pythagorean_audit.csv", index=False)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "statistics": statistics}, indent=2))


if __name__ == "__main__":
    main()
