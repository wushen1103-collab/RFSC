#!/usr/bin/env python3
"""Aggregate the frozen dual-channel H2 development panel and oracle bound."""

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
    parser.add_argument("--oracle-run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def method_suffix(gamma: float) -> str:
    return str(float(gamma)).replace(".", "p")


def bootstrap_mean_ci(values: pd.Series, seed: int = 29012026) -> list[float]:
    array = values.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sample = rng.choice(array, size=(200_000, len(array)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(sample, [0.025, 0.975])]


def aggregate_learned(raw: pd.DataFrame, expected_random: int):
    run = (
        raw.groupby(["target", "fold", "partition", "seed", "method"], as_index=False)
        .agg(
            **{metric: (metric, "mean") for metric in METRICS},
            replicates=("replicate", lambda x: int((x >= 0).sum())),
        )
    )
    random_rows = run["method"].str.startswith("random")
    if not (run.loc[random_rows, "replicates"] == expected_random).all():
        raise RuntimeError("Incomplete q-matched random dual controls")
    fold = (
        run.groupby(["target", "fold", "partition", "method"], as_index=False)
        .agg(**{metric: (metric, "mean") for metric in METRICS}, seeds=("seed", "nunique"))
    )
    if not (fold["seeds"] == 3).all():
        raise RuntimeError("A dual-channel fold lacks three model seeds")
    target = (
        fold.groupby(["target", "partition", "method"], as_index=False)
        .agg(**{metric: (metric, "mean") for metric in METRICS}, folds=("fold", "nunique"))
    )
    macro = (
        target.groupby(["partition", "method"], as_index=False)
        .agg(**{metric: (metric, "mean") for metric in METRICS}, targets=("target", "nunique"))
    )
    return run, fold, target, macro


def aggregate_oracle(raw: pd.DataFrame, expected_random: int, strength: float):
    raw = raw[np.isclose(raw["lambda"], strength)].copy()
    run = (
        raw.groupby(["target", "fold", "seed", "method", "lambda"], as_index=False)
        .agg(
            **{metric: (metric, "mean") for metric in METRICS},
            replicates=("random_replicate", lambda x: int((x >= 0).sum())),
        )
    )
    random_rows = run["method"].str.startswith("random")
    if not (run.loc[random_rows, "replicates"] == expected_random).all():
        raise RuntimeError("Incomplete oracle random dual controls")
    fold = (
        run.groupby(["target", "fold", "method", "lambda"], as_index=False)
        .agg(**{metric: (metric, "mean") for metric in METRICS}, seeds=("seed", "nunique"))
    )
    if not (fold["seeds"] == 3).all():
        raise RuntimeError("An oracle dual fold lacks three model seeds")
    target = (
        fold.groupby(["target", "method", "lambda"], as_index=False)
        .agg(**{metric: (metric, "mean") for metric in METRICS}, folds=("fold", "nunique"))
    )
    return run, fold, target


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    expected_folds = 29
    random_replicates = int(config["controls"]["random_soft_gate_permutations"])

    evaluation_frames, gate_frames, runs = [], [], []
    for path in sorted(args.run_dir.glob("*/fold_*/run.json")):
        run_metadata = json.loads(path.read_text(encoding="utf-8"))
        runs.append(run_metadata)
        if run_metadata["validation_labels_materialized"] or run_metadata["outer_test_labels_materialized"]:
            raise RuntimeError(f"Query label entered dual trainer: {path}")
        for partition in ["validation", "test"]:
            evaluation_frames.append(pd.read_csv(path.parent / f"{partition}_evaluation.csv"))
            gate_frames.append(pd.read_csv(path.parent / f"{partition}_gate_metrics.csv"))
    if len(runs) != expected_folds:
        raise RuntimeError(f"Expected {expected_folds} dual folds, found {len(runs)}")
    learned_raw = pd.concat(evaluation_frames, ignore_index=True)
    gate_raw = pd.concat(gate_frames, ignore_index=True)
    learned_run, learned_fold, learned_target, learned_macro = aggregate_learned(
        learned_raw, random_replicates
    )

    gate_target = (
        gate_raw.groupby(["target", "partition"], as_index=False)
        .agg(
            boundary_prevalence=("boundary_prevalence", "mean"),
            auprc=("auprc", "mean"),
            auroc=("auroc", "mean"),
            brier=("brier", "mean"),
            mean_q=("mean_q", "mean"),
        )
    )

    oracle_paths = sorted(args.oracle_run_dir.glob("*/fold_*/diagnostic_seed_metrics.csv"))
    oracle_auth = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.oracle_run_dir.glob("*/fold_*/authorization.json"))
    ]
    if len(oracle_paths) != expected_folds or len(oracle_auth) != expected_folds:
        raise RuntimeError("Incomplete authorized dual-oracle panel")
    if any(item["outer_test_labels_materialized"] for item in oracle_auth):
        raise RuntimeError("Outer-test label entered dual-oracle diagnostic")
    oracle_raw = pd.concat([pd.read_csv(path) for path in oracle_paths], ignore_index=True)
    oracle_run, oracle_fold, oracle_target = aggregate_oracle(
        oracle_raw,
        random_replicates,
        float(config["propagation"]["lambda"]),
    )

    test = learned_target[learned_target["partition"] == "test"].pivot(
        index="target", columns="method", values=METRICS
    )
    sensitivity_rows = []
    for gamma in config["propagation"]["highpass_gammas"]:
        gamma = float(gamma)
        suffix = method_suffix(gamma)
        learned_method = f"learned_dual_g{suffix}"
        random_method = f"random_dual_g{suffix}"
        similarity_method = f"similarity_dual_g{suffix}"
        for target_name in test.index:
            row = {"target": target_name, "gamma": gamma}
            for metric in METRICS:
                value = test.loc[target_name, (metric, learned_method)]
                row[metric] = float(value)
                for reference_name, reference_method in [
                    ("base", "base"),
                    ("cut_only", "learned"),
                    ("random", random_method),
                    ("similarity", similarity_method),
                ]:
                    reference = test.loc[target_name, (metric, reference_method)]
                    row[f"vs_{reference_name}_{metric}_relative_change"] = float(
                        (value - reference) / reference
                    )
            sensitivity_rows.append(row)
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity_macro = (
        sensitivity.groupby("gamma", as_index=False)
        .agg(
            **{
                column: (column, "mean")
                for column in sensitivity.columns
                if column not in {"target", "gamma"}
            },
            targets=("target", "nunique"),
        )
    )

    primary_gamma = float(config["propagation"]["primary_highpass_gamma"])
    primary = sensitivity[np.isclose(sensitivity["gamma"], primary_gamma)].set_index("target")
    primary_suffix = method_suffix(primary_gamma)
    learned_validation = learned_target[
        (learned_target["partition"] == "validation")
        & (learned_target["method"] == f"learned_dual_g{primary_suffix}")
    ].set_index("target")
    oracle_validation = oracle_target[
        oracle_target["method"] == f"oracle_dual_g{primary_suffix}"
    ].set_index("target")
    common = learned_validation.index.intersection(oracle_validation.index)
    oracle_vs_learned = (
        oracle_validation.loc[common, "cliff_mae"]
        - learned_validation.loc[common, "cliff_mae"]
    ) / learned_validation.loc[common, "cliff_mae"]

    test_gate = gate_target[gate_target["partition"] == "test"].set_index("target")
    auprc_lift = test_gate["auprc"] - test_gate["boundary_prevalence"]
    cliff_vs_base = primary["vs_base_cliff_mae_relative_change"]
    overall_vs_base = primary["vs_base_overall_mae_relative_change"]
    cliff_vs_random = primary["vs_random_cliff_mae_relative_change"]
    cliff_vs_similarity = primary["vs_similarity_cliff_mae_relative_change"]
    pair_vs_cut = primary["vs_cut_only_pair_delta_mae_relative_change"]
    pair_vs_base = primary["vs_base_pair_delta_mae_relative_change"]
    checks = {
        "folds_expected": expected_folds,
        "folds_observed": len(runs),
        "all_query_labels_private_during_training": True,
        "all_outer_test_labels_private_in_oracle": True,
        "primary_gamma": primary_gamma,
        "mean_target_primary_cliff_relative_change_vs_base": float(cliff_vs_base.mean()),
        "mean_target_primary_overall_relative_change_vs_base": float(overall_vs_base.mean()),
        "mean_target_primary_cliff_relative_change_vs_random": float(cliff_vs_random.mean()),
        "mean_target_primary_cliff_relative_change_vs_similarity": float(cliff_vs_similarity.mean()),
        "mean_target_primary_pair_delta_relative_change_vs_cut_only": float(pair_vs_cut.mean()),
        "mean_target_primary_pair_delta_relative_change_vs_base": float(pair_vs_base.mean()),
        "target_bootstrap95_primary_cliff_relative_change_vs_base": bootstrap_mean_ci(cliff_vs_base),
        "targets_primary_better_cliff_than_base": int((cliff_vs_base < 0).sum()),
        "targets_primary_better_cliff_than_random": int((cliff_vs_random < 0).sum()),
        "targets_primary_better_cliff_than_similarity": int((cliff_vs_similarity < 0).sum()),
        "targets_primary_better_pair_delta_than_cut_only": int((pair_vs_cut < 0).sum()),
        "targets_primary_no_worse_pair_delta_than_base": int((pair_vs_base <= 0).sum()),
        "mean_target_test_gate_auprc": float(test_gate["auprc"].mean()),
        "mean_target_test_gate_boundary_prevalence": float(test_gate["boundary_prevalence"].mean()),
        "mean_target_test_gate_auprc_lift": float(auprc_lift.mean()),
        "mean_validation_oracle_dual_cliff_relative_change_vs_learned": float(oracle_vs_learned.mean()),
        "targets_validation_oracle_dual_better_than_learned": int((oracle_vs_learned <= 0).sum()),
        "total_gpu_job_hours_learned": float(sum(item["wall_time_seconds"] for item in runs) / 3600.0),
        "maximum_peak_vram_gib": float(max(item["peak_vram_bytes"] for item in runs) / 2**30),
    }
    decision = config["decision"]
    checks["go"] = bool(
        checks["mean_target_primary_cliff_relative_change_vs_base"]
        <= -float(decision["minimum_primary_cliff_improvement_vs_base"])
        and checks["mean_target_primary_overall_relative_change_vs_base"]
        <= float(decision["maximum_primary_overall_degradation_vs_base"])
        and checks["mean_target_primary_cliff_relative_change_vs_random"] < 0
        and checks["mean_target_primary_cliff_relative_change_vs_similarity"] < 0
        and checks["mean_target_primary_pair_delta_relative_change_vs_cut_only"] < 0
        and checks["mean_target_primary_pair_delta_relative_change_vs_base"] <= 0
        and checks["mean_target_test_gate_auprc_lift"]
        >= float(decision["minimum_gate_auprc_lift_over_prevalence"])
        and checks["mean_validation_oracle_dual_cliff_relative_change_vs_learned"] <= 0
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    learned_raw.to_csv(args.output_dir / "dual_gate_raw.csv", index=False)
    learned_run.to_csv(args.output_dir / "dual_gate_run_metrics.csv", index=False)
    learned_fold.to_csv(args.output_dir / "dual_gate_fold_metrics.csv", index=False)
    learned_target.to_csv(args.output_dir / "dual_gate_target_metrics.csv", index=False)
    learned_macro.to_csv(args.output_dir / "dual_gate_macro_table.csv", index=False)
    sensitivity.to_csv(args.output_dir / "dual_gate_gamma_target_contrasts.csv", index=False)
    sensitivity_macro.to_csv(args.output_dir / "dual_gate_gamma_sensitivity.csv", index=False)
    gate_target.to_csv(args.output_dir / "dual_gate_boundary_table.csv", index=False)
    oracle_run.to_csv(args.output_dir / "dual_oracle_run_metrics.csv", index=False)
    oracle_fold.to_csv(args.output_dir / "dual_oracle_fold_metrics.csv", index=False)
    oracle_target.to_csv(args.output_dir / "dual_oracle_target_metrics.csv", index=False)
    (args.output_dir / "dual_gate_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
