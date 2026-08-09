#!/usr/bin/env python3
"""Aggregate learned-gate development metrics and enforce the frozen H2 gate."""

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
    parser.add_argument("--oracle-target-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def bootstrap_mean_ci(values: pd.Series, seed: int = 29012026) -> list[float]:
    array = values.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sample = rng.choice(array, size=(200_000, len(array)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(sample, [0.025, 0.975])]


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    evaluations = []
    gates = []
    authorizations = []
    for path in sorted(args.run_dir.glob("*/fold_*/run.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        authorizations.append(run)
        if run["validation_labels_materialized"] or run["outer_test_labels_materialized"]:
            raise RuntimeError(f"Query labels entered trainer: {path}")
        for partition in ["validation", "test"]:
            evaluations.append(pd.read_csv(path.parent / f"{partition}_evaluation.csv"))
            gates.append(pd.read_csv(path.parent / f"{partition}_gate_metrics.csv"))
    if len(authorizations) != 29:
        raise RuntimeError(f"Expected 29 completed learned-gate folds, found {len(authorizations)}")
    raw = pd.concat(evaluations, ignore_index=True)
    gate_raw = pd.concat(gates, ignore_index=True)
    run = (
        raw.groupby(["target", "fold", "partition", "seed", "method"], as_index=False)
        .agg(
            **{metric: (metric, "mean") for metric in METRICS},
            replicates=("replicate", lambda x: int((x >= 0).sum())),
        )
    )
    random_rows = run["method"] == "random"
    expected_random = int(config["controls"]["random_soft_gate_permutations"])
    if not (run.loc[random_rows, "replicates"] == expected_random).all():
        raise RuntimeError("Incomplete random learned-gate controls")
    fold = (
        run.groupby(["target", "fold", "partition", "method"], as_index=False)
        .agg(**{metric: (metric, "mean") for metric in METRICS}, seeds=("seed", "nunique"))
    )
    if not (fold["seeds"] == 3).all():
        raise RuntimeError("A learned-gate fold lacks three seeds")
    target = (
        fold.groupby(["target", "partition", "method"], as_index=False)
        .agg(**{metric: (metric, "mean") for metric in METRICS}, folds=("fold", "nunique"))
    )
    macro = (
        target.groupby(["partition", "method"], as_index=False)
        .agg(**{metric: (metric, "mean") for metric in METRICS}, targets=("target", "nunique"))
    )
    gate_fold = (
        gate_raw.groupby(["target", "fold", "partition"], as_index=False)
        .agg(
            boundary_prevalence=("boundary_prevalence", "mean"),
            auprc=("auprc", "mean"),
            auroc=("auroc", "mean"),
            brier=("brier", "mean"),
            mean_q=("mean_q", "mean"),
            seeds=("seed", "nunique"),
        )
    )
    gate_target = (
        gate_fold.groupby(["target", "partition"], as_index=False)
        .agg(
            boundary_prevalence=("boundary_prevalence", "mean"),
            auprc=("auprc", "mean"),
            auroc=("auroc", "mean"),
            brier=("brier", "mean"),
            mean_q=("mean_q", "mean"),
        )
    )

    test = target[target["partition"] == "test"].pivot(
        index="target", columns="method", values=METRICS
    )
    contrasts = pd.DataFrame(index=test.index)
    for metric in METRICS:
        for reference in ["base", "global", "random", "similarity"]:
            contrasts[f"learned_vs_{reference}_{metric}_relative_change"] = (
                test[(metric, "learned")] - test[(metric, reference)]
            ) / test[(metric, reference)]
    contrasts = contrasts.reset_index()
    learned_cliff_vs_base = contrasts["learned_vs_base_cliff_mae_relative_change"]
    learned_overall_vs_base = contrasts["learned_vs_base_overall_mae_relative_change"]
    learned_cliff_vs_random = contrasts["learned_vs_random_cliff_mae_relative_change"]
    learned_cliff_vs_similarity = contrasts[
        "learned_vs_similarity_cliff_mae_relative_change"
    ]
    test_gate = gate_target[gate_target["partition"] == "test"].set_index("target")
    auprc_lift = test_gate["auprc"] - test_gate["boundary_prevalence"]

    validation_learned = target[
        (target["partition"] == "validation") & (target["method"] == "learned")
    ].set_index("target")
    oracle = pd.read_csv(args.oracle_target_table)
    oracle = oracle[
        (oracle["method"] == "oracle")
        & np.isclose(oracle["lambda"], float(config["propagation"]["lambda"]))
    ].set_index("target")
    common = validation_learned.index.intersection(oracle.index)
    oracle_vs_learned = (
        oracle.loc[common, "cliff_mae"] - validation_learned.loc[common, "cliff_mae"]
    ) / validation_learned.loc[common, "cliff_mae"]

    decision = config["decision"]
    checks = {
        "folds_expected": 29,
        "folds_observed": len(authorizations),
        "all_query_labels_private_during_training": True,
        "mean_target_learned_cliff_relative_change_vs_base": float(
            learned_cliff_vs_base.mean()
        ),
        "mean_target_learned_overall_relative_change_vs_base": float(
            learned_overall_vs_base.mean()
        ),
        "mean_target_learned_cliff_relative_change_vs_random": float(
            learned_cliff_vs_random.mean()
        ),
        "mean_target_learned_cliff_relative_change_vs_similarity": float(
            learned_cliff_vs_similarity.mean()
        ),
        "target_bootstrap95_learned_cliff_relative_change_vs_base": bootstrap_mean_ci(
            learned_cliff_vs_base
        ),
        "targets_learned_better_cliff_than_base": int((learned_cliff_vs_base < 0).sum()),
        "targets_learned_better_cliff_than_random": int(
            (learned_cliff_vs_random < 0).sum()
        ),
        "targets_learned_better_cliff_than_similarity": int(
            (learned_cliff_vs_similarity < 0).sum()
        ),
        "mean_target_test_gate_auprc": float(test_gate["auprc"].mean()),
        "mean_target_test_gate_boundary_prevalence": float(
            test_gate["boundary_prevalence"].mean()
        ),
        "mean_target_test_gate_auprc_lift": float(auprc_lift.mean()),
        "mean_validation_oracle_cliff_relative_change_vs_learned": float(
            oracle_vs_learned.mean()
        ),
        "targets_validation_oracle_better_than_learned": int(
            (oracle_vs_learned <= 0).sum()
        ),
        "total_gpu_job_hours": float(
            sum(item["wall_time_seconds"] for item in authorizations) / 3600.0
        ),
        "maximum_peak_vram_gib": float(
            max(item["peak_vram_bytes"] for item in authorizations) / 2**30
        ),
    }
    checks["go"] = bool(
        checks["mean_target_learned_cliff_relative_change_vs_base"]
        <= -float(decision["minimum_learned_cliff_improvement_vs_base"])
        and checks["mean_target_learned_overall_relative_change_vs_base"]
        <= float(decision["maximum_learned_overall_degradation_vs_base"])
        and checks["mean_target_learned_cliff_relative_change_vs_random"] < 0
        and checks["mean_target_learned_cliff_relative_change_vs_similarity"] < 0
        and checks["mean_target_test_gate_auprc_lift"]
        >= float(decision["minimum_gate_auprc_lift_over_prevalence"])
        and checks["mean_validation_oracle_cliff_relative_change_vs_learned"] <= 0
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "learned_gate_raw.csv", index=False)
    run.to_csv(args.output_dir / "learned_gate_run_metrics.csv", index=False)
    fold.to_csv(args.output_dir / "learned_gate_fold_metrics.csv", index=False)
    target.to_csv(args.output_dir / "learned_gate_target_metrics.csv", index=False)
    macro.to_csv(args.output_dir / "learned_gate_macro_table.csv", index=False)
    contrasts.to_csv(args.output_dir / "learned_gate_test_contrasts.csv", index=False)
    gate_target.to_csv(args.output_dir / "learned_gate_boundary_table.csv", index=False)
    (args.output_dir / "learned_gate_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
