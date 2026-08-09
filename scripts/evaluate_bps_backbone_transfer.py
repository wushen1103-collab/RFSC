#!/usr/bin/env python3
"""Evaluate frozen BPS backbone transfer with target-level inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_labels_for_ids

try:
    from audit_gated_component_scale_validation import read_prediction
    from evaluate_chembl36_temporal import bootstrap_relative
except ModuleNotFoundError:
    from scripts.audit_gated_component_scale_validation import read_prediction
    from scripts.evaluate_chembl36_temporal import bootstrap_relative


METRICS = [
    "overall_mae",
    "overall_rmse",
    "cliff_mae",
    "cliff_rmse",
    "pair_delta_mae",
]


def prediction_metrics(
    prediction: pd.Series,
    truth: pd.Series,
    edges: pd.DataFrame,
) -> dict[str, float]:
    delta_y = edges.mol_id_hi.map(truth) - edges.mol_id_lo.map(truth)
    cliffs = edges[delta_y.abs() >= 1.0].copy()
    cliffs["delta_y"] = delta_y.loc[cliffs.index]
    endpoints = set(cliffs.mol_id_lo) | set(cliffs.mol_id_hi)
    endpoint_ids = [value for value in prediction.index if value in endpoints]
    residual = prediction - truth.reindex(prediction.index)
    cliff_residual = prediction.reindex(endpoint_ids) - truth.reindex(endpoint_ids)
    predicted_delta = (
        cliffs.mol_id_hi.map(prediction) - cliffs.mol_id_lo.map(prediction)
    )
    return {
        "overall_mae": float(residual.abs().mean()),
        "overall_rmse": float(np.sqrt(np.square(residual).mean())),
        "cliff_mae": float(cliff_residual.abs().mean()),
        "cliff_rmse": float(np.sqrt(np.square(cliff_residual).mean())),
        "pair_delta_mae": float((predicted_delta - cliffs.delta_y).abs().mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    input_root = Path(config["input_root"])
    data_root = Path(config["data_root"])
    mmp_root = Path(config["mmp_root"])
    run_paths = sorted(args.prediction_dir.glob("*/*/fold_*/seed_*/run.json"))
    expected = int(config["decision"]["expected_predictions"])
    if len(run_paths) != expected:
        raise RuntimeError(f"Expected {expected} frozen transfers, found {len(run_paths)}")
    rows = []
    for run_path in run_paths:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["test_labels_materialized"] is not False:
            raise RuntimeError(f"Invalid firewall metadata: {run_path}")
        architecture = run["architecture"]
        target, fold, seed = run["target"], int(run["fold"]), int(run["seed"])
        raw_path = input_root / architecture / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet"
        if run["raw_prediction_sha256"] != json.loads(
            raw_path.with_suffix(raw_path.suffix + ".sha256.json").read_text(encoding="utf-8")
        )["prediction_sha256"]:
            raise RuntimeError(f"Raw provenance changed: {raw_path}")
        candidates = {
            "raw": read_prediction(raw_path),
            "bps_fixed_transfer": read_prediction(run_path.parent / "predictions.parquet"),
        }
        if not candidates["raw"].index.equals(candidates["bps_fixed_transfer"].index):
            raise RuntimeError(f"Prediction IDs changed: {run_path}")
        truth = read_labels_for_ids(data_root, target, candidates["raw"].index).set_index(
            "mol_id"
        )["y_pactivity"]
        edges = pd.read_parquet(mmp_root / target / "eligible_mmp_edges.parquet")
        edges = edges[
            edges.mol_id_lo.isin(truth.index) & edges.mol_id_hi.isin(truth.index)
        ].copy()
        for method, prediction in candidates.items():
            rows.append(
                {
                    "architecture": architecture,
                    "target": target,
                    "fold": fold,
                    "seed": seed,
                    "method": method,
                    **prediction_metrics(prediction, truth, edges),
                }
            )
    frame = pd.DataFrame(rows)
    seed_target = frame.groupby(
        ["architecture", "target", "seed", "method"], as_index=False
    )[METRICS].mean()
    target_method = seed_target.groupby(
        ["architecture", "target", "method"], as_index=False
    )[METRICS].agg(["mean", "std"])
    target_method.columns = [
        "_".join(value).rstrip("_") if isinstance(value, tuple) else value
        for value in target_method.columns
    ]

    averaged = frame.groupby(["architecture", "target", "method"], as_index=False)[METRICS].mean()
    comparisons = []
    for architecture, group in averaged.groupby("architecture", sort=True):
        raw = group[group.method == "raw"].set_index("target")
        bps = group[group.method == "bps_fixed_transfer"].set_index("target")
        for target in raw.index:
            row = {"architecture": architecture, "target": target}
            for metric in METRICS:
                row[f"{metric}_relative_change"] = float(
                    bps.loc[target, metric] / raw.loc[target, metric] - 1.0
                )
            comparisons.append(row)
    comparison = pd.DataFrame(comparisons)

    architecture_rows = []
    for architecture, group in averaged.groupby("architecture", sort=True):
        raw = group[group.method == "raw"].set_index("target")
        bps = group[group.method == "bps_fixed_transfer"].set_index("target")
        row = {"architecture": architecture, "targets": int(len(raw))}
        for metric in METRICS:
            row[f"{metric}_relative_change"] = float(
                bps[metric].mean() / raw[metric].mean() - 1.0
            )
            row[f"{metric}_favorable_targets"] = int((bps[metric] < raw[metric]).sum())
        architecture_rows.append(row)
    architecture_summary = pd.DataFrame(architecture_rows)

    target_family = averaged.groupby(["target", "method"], as_index=False)[METRICS].mean()
    raw_family = target_family[target_family.method == "raw"].set_index("target")
    bps_family = target_family[target_family.method == "bps_fixed_transfer"].set_index("target")
    family = {"targets": int(len(raw_family)), "architectures": int(len(config["architectures"]))}
    for metric_index, metric in enumerate(METRICS):
        lower, upper = bootstrap_relative(
            bps_family[metric].to_numpy(),
            raw_family[metric].to_numpy(),
            int(config["evaluation"]["bootstrap_iterations"]),
            298001 + metric_index,
        )
        family[f"{metric}_relative_change"] = float(
            bps_family[metric].mean() / raw_family[metric].mean() - 1.0
        )
        family[f"{metric}_bootstrap95"] = [lower, upper]
        family[f"{metric}_favorable_target_architectures"] = int(
            (comparison[f"{metric}_relative_change"] < 0).sum()
        )
    decision_config = config["decision"]
    checks = {
        "all_predictions_complete": bool(len(run_paths) == expected),
        "family_pair_gain_at_least_1pct": bool(
            family["pair_delta_mae_relative_change"]
            <= -float(decision_config["minimum_pair_relative_improvement"])
        ),
        "pair_favorable_fraction_at_least_70pct": bool(
            family["pair_delta_mae_favorable_target_architectures"] / len(comparison)
            >= float(decision_config["minimum_pair_favorable_target_architecture_fraction"])
        ),
        "overall_cost_within_1pct": bool(
            family["overall_mae_relative_change"]
            <= float(decision_config["maximum_overall_relative_increase"])
        ),
        "cliff_cost_within_1pct": bool(
            family["cliff_mae_relative_change"]
            <= float(decision_config["maximum_cliff_relative_increase"])
        ),
    }
    decision = {"checks": checks, "broad_transfer_go": bool(all(checks.values()))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_dir / "run_metrics.parquet", index=False)
    seed_target.to_csv(args.output_dir / "seed_target_metrics.csv", index=False)
    target_method.to_csv(args.output_dir / "target_mean_std.csv", index=False)
    comparison.to_csv(args.output_dir / "target_architecture_comparison.csv", index=False)
    architecture_summary.to_csv(args.output_dir / "architecture_summary.csv", index=False)
    (args.output_dir / "family_summary.json").write_text(
        json.dumps(family, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"family": family, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()

