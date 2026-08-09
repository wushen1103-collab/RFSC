#!/usr/bin/env python3
"""Aggregate the frozen SCG GINE development-panel anchor without plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = [
    "overall_mae",
    "overall_rmse",
    "cliff_mae",
    "cliff_rmse",
    "pair_delta_mae",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_runs(run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_dir.glob("*/fold_*/seed_*/evaluation.json")):
        evaluation = json.loads(path.read_text(encoding="utf-8"))
        run = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
        if evaluation["prediction_sha256"] != run["prediction_sha256"]:
            raise RuntimeError(f"Prediction/evaluation hash mismatch: {path.parent}")
        if run["test_labels_materialized"]:
            raise RuntimeError(f"Trainer materialized test labels: {path.parent}")
        rows.append(
            {
                **evaluation,
                "fold": int(path.parent.parent.name.split("_")[-1]),
                "seed": int(path.parent.name.split("_")[-1]),
                "best_epoch": run["best_epoch"],
                "wall_time_seconds": run["wall_time_seconds"],
                "peak_vram_bytes": run["peak_vram_bytes"],
                "n_train": run["n_train"],
                "n_valid": run["n_valid"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    audit = pd.read_csv(args.audit).set_index("target")
    targets = sorted(args.targets)
    seed = read_runs(args.run_dir)
    observed_targets = sorted(seed["target"].unique().tolist())
    if observed_targets != targets:
        raise RuntimeError(
            f"Observed targets differ from frozen panel: {observed_targets} != {targets}"
        )
    expected_folds = {target: int(audit.loc[target, "scg_folds"]) for target in targets}
    expected_runs = 3 * sum(expected_folds.values())
    fold_seed_counts = seed.groupby(["target", "fold"])["seed"].nunique()
    actual_folds = seed.groupby("target")["fold"].nunique().to_dict()
    fold = (
        seed.groupby(["target", "fold"], as_index=False)
        .agg(
            **{metric: (metric, "mean") for metric in METRICS},
            seeds=("seed", "nunique"),
            n_test=("n_test", "first"),
            n_cliff_endpoints=("n_cliff_endpoints", "first"),
            n_cliff_pairs=("n_cliff_pairs", "first"),
        )
    )
    target = (
        fold.groupby("target", as_index=False)
        .agg(
            **{metric: (metric, "mean") for metric in METRICS},
            folds=("fold", "nunique"),
            minimum_cliff_endpoints=("n_cliff_endpoints", "min"),
            total_cliff_endpoints_across_folds=("n_cliff_endpoints", "sum"),
            total_cliff_pairs_across_folds=("n_cliff_pairs", "sum"),
        )
    )
    target["cliff_to_overall_mae_ratio"] = target["cliff_mae"] / target["overall_mae"]
    checks = {
        "targets_expected": len(targets),
        "targets_observed": len(observed_targets),
        "folds_expected": int(sum(expected_folds.values())),
        "folds_observed": int(len(fold)),
        "runs_expected": expected_runs,
        "runs_observed": int(len(seed)),
        "all_folds_have_three_seeds": bool((fold_seed_counts == 3).all()),
        "actual_folds_match_audit": bool(actual_folds == expected_folds),
        "all_prediction_hashes_match": True,
        "all_trainers_kept_test_labels_private": True,
        "macro_target_overall_mae": float(target["overall_mae"].mean()),
        "macro_target_cliff_mae": float(target["cliff_mae"].mean()),
        "macro_target_pair_delta_mae": float(target["pair_delta_mae"].mean()),
        "mean_target_cliff_to_overall_mae_ratio": float(
            target["cliff_to_overall_mae_ratio"].mean()
        ),
        "total_gpu_job_hours": float(seed["wall_time_seconds"].sum() / 3600.0),
        "maximum_peak_vram_gib": float(seed["peak_vram_bytes"].max() / 2**30),
    }
    checks["complete"] = bool(
        checks["targets_expected"] == checks["targets_observed"]
        and checks["folds_expected"] == checks["folds_observed"]
        and checks["runs_expected"] == checks["runs_observed"]
        and checks["all_folds_have_three_seeds"]
        and checks["actual_folds_match_audit"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed.sort_values(["target", "fold", "seed"]).to_csv(
        args.output_dir / "gine_scg_seed_metrics.csv", index=False
    )
    fold.to_csv(args.output_dir / "gine_scg_fold_metrics.csv", index=False)
    target.to_csv(args.output_dir / "gine_scg_target_table.csv", index=False)
    (args.output_dir / "gine_scg_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
