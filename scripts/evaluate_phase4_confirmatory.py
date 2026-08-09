#!/usr/bin/env python3
"""Evaluate the once-frozen Phase 4 confirmation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from evaluate_rf_gine_ensemble import metrics


METHODS = ["rf", "gine", "fixed_equal", "gated_component_scaled"]
METRICS = ["overall_mae", "cliff_mae", "pair_delta_mae"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = []
    run_paths = sorted(args.prediction_dir.glob("*/fold_*/seed_*/run.json"))
    for run_path in run_paths:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["test_labels_materialized"] or not run["confirmatory"]:
            raise RuntimeError("Invalid confirmatory prediction metadata")
        for method in METHODS:
            rows.append(
                {
                    "target": run["target"],
                    "fold": run["fold"],
                    "seed": run["seed"],
                    "method": method,
                    **metrics(run_path.parent / method / "predictions.parquet", run["target"], Path("data/processed/moleculeace"), Path("graphs/mmp")),
                }
            )
    expected_runs = 3 * sum(
        int(pd.read_parquet(Path("folds/scg") / target / "outer_folds.parquet").outer_fold.nunique())
        for target in config["confirmatory_targets"]
    )
    if len(run_paths) != expected_runs or len(rows) != expected_runs * len(METHODS):
        raise RuntimeError(f"Incomplete matrix: runs={len(run_paths)}, rows={len(rows)}, expected_runs={expected_runs}")
    frame = pd.DataFrame(rows)
    target = frame.groupby(["target", "method"], as_index=False).mean(numeric_only=True)
    indexed = {method: target[target.method == method].set_index("target") for method in METHODS}
    candidate = indexed["gated_component_scaled"]
    comparison = pd.DataFrame({"target": candidate.index})
    for baseline in ["fixed_equal", "gine", "rf"]:
        for metric in METRICS:
            comparison[f"{metric}_relative_change_vs_{baseline}"] = (
                candidate[metric] / indexed[baseline][metric] - 1.0
            ).to_numpy()
    for metric in METRICS:
        comparison[f"fixed_equal_{metric}_relative_change_vs_gine"] = (
            indexed["fixed_equal"][metric] / indexed["gine"][metric] - 1.0
        ).to_numpy()
    decision = config["decision"]
    summary = {
        column: float(comparison[column].mean())
        for column in comparison
        if column != "target"
    }
    summary["pair_improved_targets_vs_fixed_equal"] = int((comparison.pair_delta_mae_relative_change_vs_fixed_equal < 0).sum())
    summary["criteria"] = {
        "overall_vs_equal": summary["overall_mae_relative_change_vs_fixed_equal"] <= float(decision["maximum_overall_relative_change_vs_fixed_equal"]),
        "cliff_vs_equal": summary["cliff_mae_relative_change_vs_fixed_equal"] <= float(decision["maximum_cliff_relative_change_vs_fixed_equal"]),
        "pair_vs_equal": summary["pair_delta_mae_relative_change_vs_fixed_equal"] <= -float(decision["minimum_pair_relative_improvement_vs_fixed_equal"]),
        "target_directions": summary["pair_improved_targets_vs_fixed_equal"] >= int(decision["minimum_pair_improved_targets_vs_fixed_equal"]),
        "overall_vs_gine": summary["overall_mae_relative_change_vs_gine"] <= float(decision["maximum_overall_relative_change_vs_gine"]),
        "cliff_vs_gine": summary["cliff_mae_relative_change_vs_gine"] <= float(decision["maximum_cliff_relative_change_vs_gine"]),
        "pair_vs_gine": summary["pair_delta_mae_relative_change_vs_gine"] <= float(decision["maximum_pair_relative_change_vs_gine"]),
    }
    summary["confirmatory_go"] = all(summary["criteria"].values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_dir / "complete_results.parquet", index=False)
    target.to_csv(args.output_dir / "metrics_by_target.csv", index=False)
    comparison.to_csv(args.output_dir / "paired_comparison_by_target.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
