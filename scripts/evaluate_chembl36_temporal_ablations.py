#!/usr/bin/env python3
"""Evaluate the preregistered temporal ablations after the primary decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_frozen_prediction, read_labels_for_ids
from bps_gnn.temporal import temporal_split_ids
try:
    from evaluate_chembl36_temporal import METRICS, bootstrap_relative, metric_row
except ModuleNotFoundError:
    from scripts.evaluate_chembl36_temporal import METRICS, bootstrap_relative, metric_row


def baseline_for(method: str) -> str:
    if method == "svm_main_mmp":
        return "svm"
    if method == "rf_gine_main_mmp":
        return "rf_gine"
    return "identity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eligible-targets", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mmp-root", type=Path, required=True)
    parser.add_argument("--primary-prediction-root", type=Path, required=True)
    parser.add_argument("--primary-evaluation", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    targets = json.loads(args.eligible_targets.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["seeds"]]
    build_summary = pd.read_csv(args.ablation_root / "build_summary.csv")
    methods = sorted(build_summary["method"].unique())

    # Fail closed on every ablation hash and verify the duplicated main setting.
    for target in targets:
        _, _, test_ids = temporal_split_ids(args.data_root, target)
        for seed in seeds:
            for method in methods:
                prediction = read_frozen_prediction(
                    args.ablation_root / target / f"seed_{seed}" / method / "predictions.parquet"
                )
                if set(prediction.index) != set(test_ids):
                    raise RuntimeError(f"{target} seed {seed} {method}: partition mismatch")
            replicated_main = read_frozen_prediction(
                args.ablation_root
                / target
                / f"seed_{seed}"
                / "sensitivity_s1p5_t0p5"
                / "predictions.parquet"
            )
            primary = read_frozen_prediction(
                args.primary_prediction_root / target / f"seed_{seed}" / "bps" / "predictions.parquet"
            )
            if not np.array_equal(replicated_main.to_numpy(), primary.to_numpy()):
                raise RuntimeError(f"{target} seed {seed}: ablation main row differs from primary BPS")

    rows: list[dict[str, object]] = []
    for target in targets:
        _, _, test_ids = temporal_split_ids(args.data_root, target)
        truth = read_labels_for_ids(args.data_root, target, test_ids).set_index("mol_id")["y_pactivity"]
        all_edges = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        edges = all_edges[
            all_edges["mol_id_lo"].isin(truth.index) & all_edges["mol_id_hi"].isin(truth.index)
        ]
        for seed in seeds:
            for method in methods:
                prediction = read_frozen_prediction(
                    args.ablation_root / target / f"seed_{seed}" / method / "predictions.parquet"
                )
                rows.append(
                    {"target": target, "seed": seed, "method": method, **metric_row(prediction, truth, edges)}
                )
    raw = pd.DataFrame.from_records(rows)
    target_metrics = raw.groupby(["target", "method"], as_index=False)[METRICS].mean()
    primary_target = pd.read_csv(args.primary_evaluation / "target_metrics.csv")
    stats: list[dict[str, object]] = []
    for method in methods:
        candidate = target_metrics[target_metrics["method"] == method].set_index("target")
        baseline_method = baseline_for(method)
        if baseline_method == "identity":
            baseline = target_metrics[target_metrics["method"] == "identity"].set_index("target")
        else:
            baseline = primary_target[primary_target["method"] == baseline_method].set_index("target")
        for metric_index, metric in enumerate(METRICS):
            paired = pd.concat(
                [candidate[metric].rename("candidate"), baseline[metric].rename("baseline")], axis=1
            ).dropna()
            relative = float(paired["candidate"].mean() / paired["baseline"].mean() - 1.0)
            lower, upper = bootstrap_relative(
                paired["candidate"].to_numpy(), paired["baseline"].to_numpy(),
                int(config["evaluation"]["bootstrap_iterations"]),
                9100 + 100 * methods.index(method) + metric_index,
            )
            stats.append(
                {
                    "method": method,
                    "baseline": baseline_method,
                    "metric": metric,
                    "targets": int(len(paired)),
                    "baseline_macro": float(paired["baseline"].mean()),
                    "candidate_macro": float(paired["candidate"].mean()),
                    "relative_change": relative,
                    "bootstrap95_lower": lower,
                    "bootstrap95_upper": upper,
                    "favorable_targets": int((paired["candidate"] < paired["baseline"]).sum()),
                    "favorable_fraction": float((paired["candidate"] < paired["baseline"]).mean()),
                }
            )
    stats_frame = pd.DataFrame.from_records(stats)
    diagnostics = (
        build_summary.groupby("method", as_index=False)[
            ["active_edges", "active_nodes", "nontrivial_components", "largest_component", "correction_l2", "projection_seconds"]
        ]
        .mean()
        .sort_values("method")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "raw_seed_metrics.csv", index=False)
    target_metrics.to_csv(args.output_dir / "target_metrics.csv", index=False)
    stats_frame.to_csv(args.output_dir / "ablation_statistics.csv", index=False)
    diagnostics.to_csv(args.output_dir / "complexity_diagnostics.csv", index=False)
    pair = stats_frame[stats_frame["metric"] == "pair_delta_mae"].sort_values("relative_change")
    print(pair.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
