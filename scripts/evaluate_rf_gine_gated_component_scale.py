#!/usr/bin/env python3
"""Reveal outer-test metrics for frozen gated component scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from evaluate_rf_gine_ensemble import metrics


METRICS = ["overall_mae", "cliff_mae", "pair_delta_mae"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = []
    for run_path in sorted(args.ensemble_dir.glob("*/fold_*/seed_*/run.json")):
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["test_labels_materialized"]:
            raise RuntimeError("Test labels entered gated component-scale builder")
        for method in ["fixed_equal", "gated_component_scaled"]:
            rows.append(
                {
                    "target": run["target"],
                    "fold": run["fold"],
                    "seed": run["seed"],
                    "method": method,
                    **metrics(run_path.parent / method / "predictions.parquet", run["target"], Path("data/processed/moleculeace"), Path("graphs/mmp")),
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != 174:
        raise RuntimeError(f"Expected 174 metric rows, observed {len(frame)}")
    target = frame.groupby(["target", "method"], as_index=False).mean(numeric_only=True)
    base = target[target.method == "fixed_equal"].set_index("target")
    candidate = target[target.method == "gated_component_scaled"].set_index("target")
    gine = pd.read_csv("experiments/rf_gine_ensemble/results/ensemble_target_metrics.csv")
    gine = gine[gine.method == "gine"].set_index("target")
    comparison = pd.DataFrame(
        {
            "target": candidate.index,
            **{f"{metric}_relative_change_vs_fixed_equal": candidate[metric] / base[metric] - 1.0 for metric in METRICS},
            **{f"{metric}_relative_change_vs_gine": candidate[metric] / gine[metric] - 1.0 for metric in METRICS},
        }
    ).reset_index(drop=True)
    decision = config["decision"]
    summary = {
        column: float(comparison[column].mean())
        for column in comparison
        if column.endswith("_relative_change_vs_fixed_equal") or column.endswith("_relative_change_vs_gine")
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
    summary["promote"] = all(summary["criteria"].values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "metrics_by_run.csv", index=False)
    target.to_csv(args.output_dir / "metrics_by_target.csv", index=False)
    comparison.to_csv(args.output_dir / "comparison_by_target.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
