#!/usr/bin/env python3
"""One-shot evaluation of the fully hash-frozen ChEMBL temporal matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

from bps_gnn.io import read_frozen_prediction, read_labels_for_ids
from bps_gnn.temporal import temporal_split_ids


METHODS = ["svm", "rf", "gine", "rf_gine", "unscaled_reference", "bps"]
METRICS = ["overall_mae", "cliff_endpoint_mae", "pair_delta_mae"]


def metric_row(prediction: pd.Series, truth: pd.Series, edges: pd.DataFrame) -> dict[str, float | int]:
    aligned_truth = truth.loc[prediction.index]
    residual = prediction - aligned_truth
    edge_truth = edges["mol_id_hi"].map(truth) - edges["mol_id_lo"].map(truth)
    cliff_edges = edges.loc[edge_truth.abs() >= 1.0].copy()
    cliff_edges["delta_y"] = edge_truth.loc[cliff_edges.index]
    endpoints = set(cliff_edges["mol_id_lo"]) | set(cliff_edges["mol_id_hi"])
    cliff_residual = residual.loc[list(endpoints)] if endpoints else pd.Series(dtype=float)
    if len(cliff_edges):
        predicted_delta = (
            cliff_edges["mol_id_hi"].map(prediction) - cliff_edges["mol_id_lo"].map(prediction)
        )
        pair_mae = float(np.mean(np.abs(predicted_delta - cliff_edges["delta_y"])))
    else:
        pair_mae = float("nan")
    return {
        "n_test": int(len(prediction)),
        "n_mmp_pairs": int(len(edges)),
        "n_cliff_pairs": int(len(cliff_edges)),
        "n_cliff_endpoints": int(len(endpoints)),
        "overall_mae": float(np.mean(np.abs(residual))),
        "cliff_endpoint_mae": float(np.mean(np.abs(cliff_residual))) if endpoints else float("nan"),
        "pair_delta_mae": pair_mae,
    }


def bootstrap_relative(candidate: np.ndarray, reference: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        index = rng.integers(0, len(candidate), size=len(candidate))
        values[iteration] = candidate[index].mean() / reference[index].mean() - 1.0
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def preflight(
    prediction_root: Path, data_root: Path, targets: list[str], seeds: list[int]
) -> None:
    for target in targets:
        _, _, test_ids = temporal_split_ids(data_root, target)
        for seed in seeds:
            expected_index = None
            for method in METHODS:
                prediction = read_frozen_prediction(
                    prediction_root / target / f"seed_{seed}" / method / "predictions.parquet"
                )
                if expected_index is None:
                    expected_index = prediction.index
                elif not prediction.index.equals(expected_index):
                    raise RuntimeError(f"{target} seed {seed}: method partitions differ")
            if set(expected_index) != set(test_ids):
                raise RuntimeError(f"{target} seed {seed}: frozen predictions are not the test partition")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eligible-targets", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mmp-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    targets = json.loads(args.eligible_targets.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["seeds"]]
    preflight(args.prediction_root, args.data_root, targets, seeds)

    rows: list[dict[str, object]] = []
    for target in targets:
        _, _, test_ids = temporal_split_ids(args.data_root, target)
        truth = read_labels_for_ids(args.data_root, target, test_ids).set_index("mol_id")["y_pactivity"]
        all_edges = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        edges = all_edges[
            all_edges["mol_id_lo"].isin(truth.index) & all_edges["mol_id_hi"].isin(truth.index)
        ].copy()
        for seed in seeds:
            for method in METHODS:
                prediction = read_frozen_prediction(
                    args.prediction_root / target / f"seed_{seed}" / method / "predictions.parquet"
                )
                rows.append(
                    {"target": target, "seed": seed, "method": method, **metric_row(prediction, truth, edges)}
                )
    raw = pd.DataFrame.from_records(rows)
    target = raw.groupby(["target", "method"], as_index=False)[METRICS].mean()
    reference = target[target["method"] == "unscaled_reference"].set_index("target")
    candidate = target[target["method"] == "bps"].set_index("target")
    stats: list[dict[str, object]] = []
    raw_p: list[float] = []
    decision_config = config["decision"]
    for metric_index, metric in enumerate(METRICS):
        paired = pd.concat(
            [candidate[metric].rename("candidate"), reference[metric].rename("reference")], axis=1
        ).dropna()
        relative = float(paired["candidate"].mean() / paired["reference"].mean() - 1.0)
        lower, upper = bootstrap_relative(
            paired["candidate"].to_numpy(), paired["reference"].to_numpy(),
            int(config["evaluation"]["bootstrap_iterations"]), 2901 + metric_index,
        )
        margin = 0.0 if metric == "pair_delta_mae" else float(
            decision_config["maximum_overall_relative_cost"]
            if metric == "overall_mae" else decision_config["maximum_cliff_relative_cost"]
        )
        adjusted_difference = paired["candidate"] - (1.0 + margin) * paired["reference"]
        if np.allclose(adjusted_difference.to_numpy(), 0.0):
            p_value = 1.0
        else:
            p_value = float(wilcoxon(adjusted_difference, alternative="less").pvalue)
        raw_p.append(p_value)
        stats.append(
            {
                "metric": metric,
                "targets": int(len(paired)),
                "reference_macro": float(paired["reference"].mean()),
                "candidate_macro": float(paired["candidate"].mean()),
                "relative_change": relative,
                "bootstrap95_lower": lower,
                "bootstrap95_upper": upper,
                "favorable_targets": int((paired["candidate"] < paired["reference"]).sum()),
                "favorable_fraction": float((paired["candidate"] < paired["reference"]).mean()),
                "wilcoxon_margin": margin,
                "wilcoxon_p_raw": p_value,
            }
        )
    adjusted = holm_adjust(raw_p)
    for record, value in zip(stats, adjusted):
        record["wilcoxon_p_holm"] = value
    stats_frame = pd.DataFrame.from_records(stats)
    by_metric = stats_frame.set_index("metric")
    pair = by_metric.loc["pair_delta_mae"]
    overall = by_metric.loc["overall_mae"]
    cliff = by_metric.loc["cliff_endpoint_mae"]
    checks = {
        "pair_improvement_at_least_3pct": float(pair["relative_change"]) <= -float(decision_config["minimum_pair_relative_improvement"]),
        "pair_bootstrap_upper_below_zero": float(pair["bootstrap95_upper"]) < float(decision_config["maximum_pair_bootstrap_upper"]),
        "pair_direction_fraction": float(pair["favorable_fraction"]) >= float(decision_config["minimum_pair_improved_target_fraction"]),
        "overall_point_cost": float(overall["relative_change"]) <= float(decision_config["maximum_overall_relative_cost"]),
        "cliff_point_cost": float(cliff["relative_change"]) <= float(decision_config["maximum_cliff_relative_cost"]),
        "overall_ci_noninferiority": float(overall["bootstrap95_upper"]) <= float(decision_config["maximum_overall_bootstrap_upper"]),
        "cliff_ci_noninferiority": float(cliff["bootstrap95_upper"]) <= float(decision_config["maximum_cliff_bootstrap_upper"]),
        "pair_target_power": int(pair["targets"]) >= int(decision_config["minimum_pair_eligible_targets"]),
    }
    decision = {"confirmatory_go": bool(all(checks.values())), "checks": checks}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "raw_seed_metrics.csv", index=False)
    target.to_csv(args.output_dir / "target_metrics.csv", index=False)
    stats_frame.to_csv(args.output_dir / "primary_statistics.csv", index=False)
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps({"decision": decision, "statistics": stats}, indent=2))


if __name__ == "__main__":
    main()
