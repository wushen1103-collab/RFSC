#!/usr/bin/env python3
"""Evaluate frozen deterministic SVM predictions and join Phase 4 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_rf_gine_ensemble import metrics


METRICS = ["overall_mae", "cliff_mae", "pair_delta_mae"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--svm-dir", type=Path, required=True)
    parser.add_argument("--phase4-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for run_path in sorted(args.svm_dir.glob("*/fold_*/seed_0/run.json")):
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["validation_labels_materialized"] or run["outer_test_labels_materialized"]:
            raise RuntimeError("SVM trainer materialized held-out labels")
        rows.append(
            {
                "target": run["target"],
                "fold": run["fold"],
                "seed": 0,
                "method": "svm_ecfp",
                **metrics(run_path.parent / "predictions.parquet", run["target"], Path("data/processed/moleculeace"), Path("graphs/mmp")),
            }
        )
    svm = pd.DataFrame(rows)
    if len(svm) != 114 or svm.target.nunique() != 23:
        raise RuntimeError(f"Incomplete SVM addendum: {len(svm)} folds, {svm.target.nunique()} targets")
    expanded = pd.concat([svm.assign(seed=seed) for seed in [0, 1, 2]], ignore_index=True)
    phase4 = pd.read_parquet(args.phase4_results)
    complete = pd.concat([phase4, expanded], ignore_index=True)
    target = complete.groupby(["target", "method"], as_index=False)[METRICS].mean()
    indexed = {method: target[target.method == method].set_index("target") for method in target.method.unique()}
    comparison = pd.DataFrame({"target": indexed["svm_ecfp"].index})
    for candidate in ["gated_component_scaled", "fixed_equal", "gine", "rf"]:
        for metric in METRICS:
            comparison[f"{candidate}_{metric}_relative_change_vs_svm"] = (
                indexed[candidate][metric] / indexed["svm_ecfp"][metric] - 1.0
            ).to_numpy()
    summary = {column: float(comparison[column].mean()) for column in comparison if column != "target"}
    for candidate in ["gated_component_scaled", "fixed_equal"]:
        for metric in METRICS:
            column = f"{candidate}_{metric}_relative_change_vs_svm"
            summary[f"{candidate}_{metric}_improved_targets_vs_svm"] = int((comparison[column] < 0).sum())
    summary["svm_deterministic_fits"] = len(svm)
    summary["svm_rows_repeated_only_for_seed_pairing"] = len(expanded)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    svm.to_csv(args.output_dir / "svm_metrics_by_fold.csv", index=False)
    complete.to_parquet(args.output_dir / "complete_results_with_svm.parquet", index=False)
    target.to_csv(args.output_dir / "metrics_by_target_with_svm.csv", index=False)
    comparison.to_csv(args.output_dir / "comparison_vs_svm_by_target.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

