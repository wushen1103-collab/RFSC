#!/usr/bin/env python3
"""Evaluate the once-frozen SVM-BPS candidate."""

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
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = []
    run_paths = sorted(args.prediction_dir.glob("*/fold_*/seed_*/run.json"))
    expected = int(config["scope"]["expected_runs"])
    if len(run_paths) != expected:
        raise RuntimeError(f"Expected {expected} runs, observed {len(run_paths)}")
    for run_path in run_paths:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["test_labels_materialized"]:
            raise RuntimeError("Builder materialized test labels")
        for method_dir in sorted(path for path in run_path.parent.iterdir() if path.is_dir()):
            prediction = method_dir / "predictions.parquet"
            if prediction.exists():
                rows.append({
                    "target": run["target"],
                    "fold": run["fold"],
                    "seed": run["seed"],
                    "method": method_dir.name,
                    **metrics(prediction, run["target"], args.data_root, args.mmp_root),
                })
    frame = pd.DataFrame(rows)
    target = frame.groupby(["target", "method"], as_index=False)[METRICS].mean()
    indexed = {method: part.set_index("target") for method, part in target.groupby("method")}
    candidate = indexed["svm_bps_selected"]
    comparison = pd.DataFrame({"target": candidate.index})
    for baseline in ["svm_ecfp", "svm_fixed_mixture", "fixed_equal", "gated_component_scaled"]:
        if baseline not in indexed:
            continue
        for metric in METRICS:
            comparison[f"{metric}_relative_change_vs_{baseline}"] = (
                candidate[metric] / indexed[baseline][metric] - 1.0
            ).to_numpy()
    decision = config["decision"]
    summary = {column: float(comparison[column].mean()) for column in comparison if column != "target"}
    for metric in METRICS:
        column = f"{metric}_relative_change_vs_svm_ecfp"
        summary[f"{metric}_improved_targets_vs_svm"] = int((comparison[column] < 0).sum())
    decision_names = {
        "overall_mae": "overall",
        "cliff_mae": "cliff",
        "pair_delta_mae": "pair",
    }
    summary["criteria"] = {
        metric: summary[f"{metric}_relative_change_vs_svm_ecfp"]
        <= float(decision[f"maximum_{decision_names[metric]}_relative_change_vs_svm"])
        and summary[f"{metric}_improved_targets_vs_svm"] >= int(decision["minimum_improved_targets_per_metric"])
        for metric in METRICS
    }
    summary["development_go"] = all(summary["criteria"].values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_dir / "complete_results.parquet", index=False)
    target.to_csv(args.output_dir / "metrics_by_target.csv", index=False)
    comparison.to_csv(args.output_dir / "paired_comparison_by_target.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
