#!/usr/bin/env python3
"""Evaluate seed-mean RF/GINE predictions on the frozen outer-test partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from evaluate_rf_gine_ensemble import metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = []
    for run_path in sorted(args.ensemble_dir.glob("*/fold_*/run.json")):
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["test_labels_materialized"]:
            raise RuntimeError("Builder read test labels")
        for method in ["fixed_equal_seed_seed_mean" if False else "fixed_equal_seed_mean", "validation_selected_seed_mean", "gine_seed_mean", "rf_seed_mean"]:
            rows.append({"target": run["target"], "fold": run["fold"], "method": method, **metrics(run_path.parent / method / "predictions.parquet", run["target"], Path("data/processed/moleculeace"), Path("graphs/mmp"))})
    frame = pd.DataFrame(rows)
    target = frame.groupby(["target", "method"], as_index=False).mean(numeric_only=True)
    base = target[target["method"] == "gine_seed_mean"].set_index("target")
    decision = config["decision"]
    comparisons = []
    for method in ["fixed_equal_seed_mean", "validation_selected_seed_mean", "rf_seed_mean"]:
        candidate = target[target["method"] == method].set_index("target")
        comparisons.append(pd.DataFrame({
            "method": method,
            "target": candidate.index,
            **{f"{metric}_relative_change_vs_gine": candidate[metric] / base[metric] - 1.0 for metric in ["overall_mae", "cliff_mae", "pair_delta_mae"]},
        }).reset_index(drop=True))
    comparison = pd.concat(comparisons, ignore_index=True)
    summary = {}
    for method in ["fixed_equal_seed_mean", "validation_selected_seed_mean"]:
        part = comparison[comparison["method"] == method]
        prefix = method.replace("_seed_mean", "")
        summary[prefix] = {column: float(part[column].mean()) for column in part if column.endswith("_relative_change_vs_gine")}
        summary[prefix]["improved_targets"] = int((part["cliff_mae_relative_change_vs_gine"] < 0).sum())
    selected = summary["validation_selected"]
    summary["validation_selected"]["criteria"] = {
        "overall": selected["overall_mae_relative_change_vs_gine"] <= float(decision["maximum_overall_relative_change_vs_gine"]),
        "cliff": selected["cliff_mae_relative_change_vs_gine"] <= float(decision["maximum_cliff_relative_change_vs_gine"]),
        "pair": selected["pair_delta_mae_relative_change_vs_gine"] <= float(decision["maximum_pair_delta_relative_change_vs_gine"]),
        "directions": selected["improved_targets"] >= int(decision["minimum_improved_targets_vs_gine"]),
    }
    summary["validation_selected"]["promote"] = all(summary["validation_selected"]["criteria"].values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "metrics_by_fold.csv", index=False)
    target.to_csv(args.output_dir / "metrics_by_target.csv", index=False)
    comparison.to_csv(args.output_dir / "comparison_by_target.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
