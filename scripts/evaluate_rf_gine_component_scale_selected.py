#!/usr/bin/env python3
"""Evaluate validation-selected component-preserving scales on outer test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from evaluate_rf_gine_ensemble import metrics


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
            raise RuntimeError("Test labels entered selected-scale builder")
        for method in ["fixed_equal", "component_scaled_selected"]:
            rows.append({"target": run["target"], "fold": run["fold"], "seed": run["seed"], "method": method, **metrics(run_path.parent / method / "predictions.parquet", run["target"], Path("data/processed/moleculeace"), Path("graphs/mmp"))})
    frame = pd.DataFrame(rows)
    target = frame.groupby(["target", "method"], as_index=False).mean(numeric_only=True)
    base = target[target.method == "fixed_equal"].set_index("target")
    candidate = target[target.method == "component_scaled_selected"].set_index("target")
    comparison = pd.DataFrame({"target": candidate.index, **{f"{metric}_relative_change_vs_fixed_equal": candidate[metric] / base[metric] - 1.0 for metric in ["overall_mae", "cliff_mae", "pair_delta_mae"]}}).reset_index(drop=True)
    decision = config["decision"]
    summary = {column: float(comparison[column].mean()) for column in comparison if column.endswith("_relative_change_vs_fixed_equal")}
    summary["pair_improved_targets"] = int((comparison.pair_delta_mae_relative_change_vs_fixed_equal < 0).sum())
    summary["criteria"] = {"overall": summary["overall_mae_relative_change_vs_fixed_equal"] <= float(decision["maximum_overall_relative_change_vs_fixed_equal"]), "cliff": summary["cliff_mae_relative_change_vs_fixed_equal"] <= float(decision["maximum_cliff_relative_change_vs_fixed_equal"]), "pair": summary["pair_delta_mae_relative_change_vs_fixed_equal"] <= -float(decision["minimum_pair_relative_improvement_vs_fixed_equal"]), "directions": summary["pair_improved_targets"] >= int(decision["minimum_pair_improved_targets_vs_fixed_equal"])}
    summary["promote"] = all(summary["criteria"].values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "metrics_by_run.csv", index=False)
    target.to_csv(args.output_dir / "metrics_by_target.csv", index=False)
    comparison.to_csv(args.output_dir / "comparison_by_target.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
