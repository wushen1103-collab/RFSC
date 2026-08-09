#!/usr/bin/env python3
"""Aggregate ten-fold Delta-LightGBM robustness with dataset-level inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

try:
    from evaluate_chembl36_temporal import bootstrap_relative
except ModuleNotFoundError:
    from scripts.evaluate_chembl36_temporal import bootstrap_relative


METRICS = ["pair_rmse", "pair_mae", "high_delta_pair_mae"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_matrix(root: Path, folds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, theorem = [], []
    for fold in folds:
        fold_root = root / f"fold_{fold:02d}"
        metric_path = fold_root / "dataset_metrics.csv"
        theorem_path = fold_root / "pythagorean_audit.csv"
        if not metric_path.exists() or not theorem_path.exists():
            raise RuntimeError(f"Missing evaluated fold {fold}")
        metric_frame = pd.read_csv(metric_path)
        metric_frame.insert(0, "fold", fold)
        metrics.append(metric_frame)
        theorem_frame = pd.read_csv(theorem_path)
        theorem_frame.insert(0, "fold", fold)
        theorem.append(theorem_frame)
    return pd.concat(metrics, ignore_index=True), pd.concat(theorem, ignore_index=True)


def comparison_statistics(
    metrics: pd.DataFrame,
    candidate: str,
    reference: str,
    iterations: int,
    seed: int,
) -> list[dict[str, object]]:
    rows = []
    for metric_index, metric in enumerate(METRICS):
        selected = metrics[metrics.method.isin([candidate, reference])]
        paired = selected.pivot(index=["dataset", "fold"], columns="method", values=metric)
        if paired.isna().any().any():
            raise RuntimeError(f"Incomplete paired matrix for {candidate}/{reference}/{metric}")
        dataset_mean = paired.groupby(level="dataset").mean()
        lower, upper = bootstrap_relative(
            dataset_mean[candidate].to_numpy(),
            dataset_mean[reference].to_numpy(),
            iterations,
            seed + metric_index,
        )
        fold_macro = paired.groupby(level="fold").mean()
        fold_effect = fold_macro[candidate] / fold_macro[reference] - 1.0
        rows.append(
            {
                "candidate": candidate,
                "reference": reference,
                "metric": metric,
                "datasets": int(len(dataset_mean)),
                "dataset_folds": int(len(paired)),
                "reference_macro": float(dataset_mean[reference].mean()),
                "candidate_macro": float(dataset_mean[candidate].mean()),
                "relative_change": float(
                    dataset_mean[candidate].mean() / dataset_mean[reference].mean() - 1.0
                ),
                "bootstrap95_lower": lower,
                "bootstrap95_upper": upper,
                "favorable_datasets": int(
                    (dataset_mean[candidate] < dataset_mean[reference]).sum()
                ),
                "favorable_dataset_folds": int((paired[candidate] < paired[reference]).sum()),
                "fold_effect_mean": float(fold_effect.mean()),
                "fold_effect_std": float(fold_effect.std(ddof=1)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    folds = [int(value) for value in config["split"]["held_out_folds"]]
    metrics, theorem = load_matrix(args.evaluation_root, folds)
    expected_datasets = set(str(value) for value in config["datasets"])
    if set(metrics.dataset) != expected_datasets:
        raise RuntimeError("Dataset set mismatch")
    expected_metric_rows = len(folds) * len(expected_datasets) * 4
    if len(metrics) != expected_metric_rows:
        raise RuntimeError(f"Expected {expected_metric_rows} method rows, found {len(metrics)}")
    if len(theorem) != len(folds) * len(expected_datasets):
        raise RuntimeError("Theorem audit matrix is incomplete")
    statistics = pd.DataFrame(
        comparison_statistics(
            metrics, "hodge_cycle_consistent", "direct_delta_lgbm", 10000, 297001
        )
        + comparison_statistics(
            metrics, "hodge_cycle_consistent", "antisymmetric", 10000, 297101
        )
        + comparison_statistics(
            metrics, "direct_delta_lgbm", "absolute_lgbm_difference", 10000, 297201
        )
    )

    def stat(candidate: str, reference: str, metric: str) -> pd.Series:
        row = statistics[
            (statistics.candidate == candidate)
            & (statistics.reference == reference)
            & (statistics.metric == metric)
        ]
        if len(row) != 1:
            raise RuntimeError(f"Missing contrast {candidate}/{reference}/{metric}")
        return row.iloc[0]

    primary_rmse = stat("hodge_cycle_consistent", "direct_delta_lgbm", "pair_rmse")
    primary_mae = stat("hodge_cycle_consistent", "direct_delta_lgbm", "pair_mae")
    primary_high = stat(
        "hodge_cycle_consistent", "direct_delta_lgbm", "high_delta_pair_mae"
    )
    checks = {
        "all_100_dataset_folds_present": bool(len(theorem) == 100),
        "all_100_pythagorean_checks_pass": bool(theorem.passed.astype(bool).all()),
        "rmse_at_least_8_of_10_datasets": bool(primary_rmse.favorable_datasets >= 8),
        "rmse_bootstrap_upper_below_zero": bool(primary_rmse.bootstrap95_upper < 0),
        "mae_at_least_8_of_10_datasets": bool(primary_mae.favorable_datasets >= 8),
        "mae_bootstrap_upper_below_zero": bool(primary_mae.bootstrap95_upper < 0),
        "high_delta_mae_not_worse": bool(primary_high.relative_change <= 0),
    }
    decision = {
        "experiment": config["name"],
        "statistical_unit": "dataset_after_ten_fold_mean",
        "checks": checks,
        "robustness_go": bool(all(checks.values())),
    }
    dataset_fold_mean = (
        metrics.groupby(["dataset", "method"], as_index=False)[METRICS].mean()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "dataset_fold_metrics.csv", index=False)
    dataset_fold_mean.to_csv(args.output_dir / "dataset_tenfold_mean.csv", index=False)
    theorem.to_csv(args.output_dir / "pythagorean_100_audit.csv", index=False)
    statistics.to_csv(args.output_dir / "statistics.csv", index=False)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))
    print(statistics.to_string(index=False))


if __name__ == "__main__":
    main()

