#!/usr/bin/env python3
"""Aggregate authorized real oracle diagnostics at run/fold/target levels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


METRICS = ["overall_mae", "cliff_mae", "pair_delta_mae"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def bootstrap_mean_ci(values: pd.Series, seed: int = 29012026) -> list[float]:
    array = values.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(200_000, len(array)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frames = [
        pd.read_csv(path)
        for path in sorted(args.run_dir.glob("*/fold_*/diagnostic_seed_metrics.csv"))
    ]
    if not frames:
        raise RuntimeError("No oracle diagnostic metrics found")
    raw = pd.concat(frames, ignore_index=True)
    authorization = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.run_dir.glob("*/fold_*/authorization.json"))
    ]
    expected_folds = 29
    if len(authorization) != expected_folds:
        raise RuntimeError(f"Expected {expected_folds} authorizations, found {len(authorization)}")
    if any(item["outer_test_labels_materialized"] for item in authorization):
        raise RuntimeError("Outer-test labels entered an oracle diagnostic")

    # Average random gate replicates before any seed/fold/target aggregation.
    run = (
        raw.groupby(["target", "fold", "seed", "method", "lambda"], as_index=False)
        .agg(
            **{metric: (metric, "mean") for metric in METRICS},
            random_replicates=("random_replicate", lambda x: int((x >= 0).sum())),
            n_query=("n_query", "first"),
            n_cliff_endpoints=("n_cliff_endpoints", "first"),
            n_cliff_pairs=("n_cliff_pairs", "first"),
        )
    )
    random_rows = run["method"] == "random"
    expected_random = int(config["random"]["replicates"])
    if not (run.loc[random_rows, "random_replicates"] == expected_random).all():
        raise RuntimeError("Incomplete random diagnostic replicates")
    fold = (
        run.groupby(["target", "fold", "method", "lambda"], as_index=False)
        .agg(
            **{metric: (metric, "mean") for metric in METRICS},
            seeds=("seed", "nunique"),
        )
    )
    if not (fold["seeds"] == 3).all():
        raise RuntimeError("A diagnostic fold lacks three paired model seeds")
    target = (
        fold.groupby(["target", "method", "lambda"], as_index=False)
        .agg(
            **{metric: (metric, "mean") for metric in METRICS},
            folds=("fold", "nunique"),
        )
    )
    macro = (
        target.groupby(["method", "lambda"], as_index=False)
        .agg(
            **{metric: (metric, "mean") for metric in METRICS},
            targets=("target", "nunique"),
        )
        .sort_values(["lambda", "method"])
    )

    primary_lambda = float(config["propagation"]["primary_lambda"])
    primary = target[np.isclose(target["lambda"], primary_lambda)].copy()
    wide = primary.pivot(index="target", columns="method", values=METRICS)
    contrasts = pd.DataFrame(index=wide.index)
    for metric in METRICS:
        for method in ["oracle", "random", "similarity"]:
            contrasts[f"{method}_vs_global_{metric}_relative_change"] = (
                wide[(metric, method)] - wide[(metric, "global")]
            ) / wide[(metric, "global")]
        contrasts[f"oracle_vs_random_{metric}_relative_change"] = (
            wide[(metric, "oracle")] - wide[(metric, "random")]
        ) / wide[(metric, "random")]
    contrasts = contrasts.reset_index()
    cliff_change = contrasts["oracle_vs_global_cliff_mae_relative_change"]
    overall_change = contrasts["oracle_vs_global_overall_mae_relative_change"]
    oracle_vs_random = contrasts["oracle_vs_random_cliff_mae_relative_change"]
    checks = {
        "authorized_folds_expected": expected_folds,
        "authorized_folds_observed": len(authorization),
        "all_outer_test_labels_private": True,
        "primary_lambda": primary_lambda,
        "mean_target_oracle_cliff_relative_change_vs_global": float(cliff_change.mean()),
        "mean_target_oracle_overall_relative_change_vs_global": float(
            overall_change.mean()
        ),
        "mean_target_oracle_cliff_relative_change_vs_random": float(
            oracle_vs_random.mean()
        ),
        "target_bootstrap95_oracle_cliff_relative_change_vs_global": bootstrap_mean_ci(
            cliff_change
        ),
        "target_bootstrap95_oracle_overall_relative_change_vs_global": bootstrap_mean_ci(
            overall_change
        ),
        "targets_oracle_better_cliff_than_global": int((cliff_change < 0).sum()),
        "targets_oracle_better_cliff_than_random": int((oracle_vs_random < 0).sum()),
        "mean_query_reference_edges_per_fold": float(
            np.mean([item["query_reference_edges"] for item in authorization])
        ),
        "mean_queries_without_neighbors_per_fold": float(
            np.mean([item["queries_without_neighbors"] for item in authorization])
        ),
        "oracle_boundary_edge_fraction": float(
            sum(item["oracle_cut_edges"] for item in authorization)
            / sum(item["query_reference_edges"] for item in authorization)
        ),
    }
    decision = config["decision"]
    checks["go"] = bool(
        checks["mean_target_oracle_cliff_relative_change_vs_global"]
        <= -float(decision["minimum_oracle_cliff_improvement_vs_global"])
        and checks["mean_target_oracle_overall_relative_change_vs_global"]
        <= float(decision["maximum_oracle_overall_degradation_vs_global"])
        and checks["mean_target_oracle_cliff_relative_change_vs_random"] < 0
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "real_oracle_raw.csv", index=False)
    run.to_csv(args.output_dir / "real_oracle_run_metrics.csv", index=False)
    fold.to_csv(args.output_dir / "real_oracle_fold_metrics.csv", index=False)
    target.to_csv(args.output_dir / "real_oracle_target_metrics.csv", index=False)
    macro.to_csv(args.output_dir / "real_oracle_lambda_sensitivity.csv", index=False)
    contrasts.to_csv(args.output_dir / "real_oracle_primary_contrasts.csv", index=False)
    (args.output_dir / "real_oracle_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
