#!/usr/bin/env python3
"""Evaluate the non-confirmatory validation-selected patent follow-up once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from bps_gnn.io import read_frozen_prediction, read_labels_for_ids
from bps_gnn.temporal import temporal_split_ids

try:
    from evaluate_chembl36_temporal import METRICS, bootstrap_relative, metric_row
except ModuleNotFoundError:
    from scripts.evaluate_chembl36_temporal import METRICS, bootstrap_relative, metric_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eligible-targets", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mmp-root", type=Path, required=True)
    parser.add_argument("--primary-prediction-root", type=Path, required=True)
    parser.add_argument("--candidate-prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    targets = json.loads(args.eligible_targets.read_text(encoding="utf-8"))
    seeds = [int(value) for value in config["seeds"]]
    for target in targets:
        _, _, test_ids = temporal_split_ids(args.data_root, target)
        expected = set(test_ids)
        for seed in seeds:
            paths = [
                args.primary_prediction_root
                / target
                / f"seed_{seed}"
                / "unscaled_reference"
                / "predictions.parquet",
                args.candidate_prediction_root / target / f"seed_{seed}" / "predictions.parquet",
            ]
            for path in paths:
                if set(read_frozen_prediction(path).index) != expected:
                    raise RuntimeError(f"{target} seed {seed}: prediction partition mismatch")

    rows: list[dict[str, object]] = []
    for target in targets:
        _, _, test_ids = temporal_split_ids(args.data_root, target)
        truth = read_labels_for_ids(args.data_root, target, test_ids).set_index("mol_id")[
            "y_pactivity"
        ]
        all_edges = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        edges = all_edges[
            all_edges["mol_id_lo"].isin(truth.index)
            & all_edges["mol_id_hi"].isin(truth.index)
        ].copy()
        for seed in seeds:
            predictions = {
                "unscaled_reference": read_frozen_prediction(
                    args.primary_prediction_root
                    / target
                    / f"seed_{seed}"
                    / "unscaled_reference"
                    / "predictions.parquet"
                ),
                "validation_selected_bps": read_frozen_prediction(
                    args.candidate_prediction_root
                    / target
                    / f"seed_{seed}"
                    / "predictions.parquet"
                ),
            }
            for method, prediction in predictions.items():
                rows.append(
                    {"target": target, "seed": seed, "method": method, **metric_row(prediction, truth, edges)}
                )
    raw = pd.DataFrame.from_records(rows)
    target = raw.groupby(["target", "method"], as_index=False)[METRICS].mean()
    reference = target[target["method"] == "unscaled_reference"].set_index("target")
    candidate = target[target["method"] == "validation_selected_bps"].set_index("target")
    statistics: list[dict[str, object]] = []
    for metric_index, metric in enumerate(METRICS):
        paired = pd.concat(
            [candidate[metric].rename("candidate"), reference[metric].rename("reference")], axis=1
        ).dropna()
        lower, upper = bootstrap_relative(
            paired["candidate"].to_numpy(),
            paired["reference"].to_numpy(),
            int(config["evaluation"]["bootstrap_iterations"]),
            29190 + metric_index,
        )
        statistics.append(
            {
                "metric": metric,
                "targets": int(len(paired)),
                "reference_macro": float(paired["reference"].mean()),
                "candidate_macro": float(paired["candidate"].mean()),
                "relative_change": float(
                    paired["candidate"].mean() / paired["reference"].mean() - 1.0
                ),
                "bootstrap95_lower": lower,
                "bootstrap95_upper": upper,
                "favorable_targets": int((paired["candidate"] < paired["reference"]).sum()),
                "favorable_fraction": float((paired["candidate"] < paired["reference"]).mean()),
            }
        )
    frame = pd.DataFrame.from_records(statistics).set_index("metric")
    decision_config = config["decision"]
    checks = {
        "pair_improvement_at_least_3pct": bool(
            frame.loc["pair_delta_mae", "relative_change"]
            <= -float(decision_config["minimum_pair_relative_improvement"])
        ),
        "pair_bootstrap_upper_below_zero": bool(
            frame.loc["pair_delta_mae", "bootstrap95_upper"]
            < float(decision_config["maximum_pair_bootstrap_upper"])
        ),
        "pair_direction_fraction": bool(
            frame.loc["pair_delta_mae", "favorable_fraction"]
            >= float(decision_config["minimum_pair_improved_target_fraction"])
        ),
        "overall_point_cost": bool(
            frame.loc["overall_mae", "relative_change"]
            <= float(decision_config["maximum_overall_relative_cost"])
        ),
        "cliff_point_cost": bool(
            frame.loc["cliff_endpoint_mae", "relative_change"]
            <= float(decision_config["maximum_cliff_relative_cost"])
        ),
        "overall_ci_noninferiority": bool(
            frame.loc["overall_mae", "bootstrap95_upper"]
            <= float(decision_config["maximum_overall_bootstrap_upper"])
        ),
        "cliff_ci_noninferiority": bool(
            frame.loc["cliff_endpoint_mae", "bootstrap95_upper"]
            <= float(decision_config["maximum_cliff_bootstrap_upper"])
        ),
        "pair_target_power": bool(
            frame.loc["pair_delta_mae", "targets"]
            >= int(decision_config["minimum_pair_eligible_targets"])
        ),
    }
    decision = {
        "status": config["status"],
        "confirmatory_eligible": False,
        "numerical_gates_pass": bool(all(checks.values())),
        "checks": checks,
        "prediction_hash_preflight_before_test_response": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "raw_seed_metrics.csv", index=False)
    target.to_csv(args.output_dir / "target_metrics.csv", index=False)
    frame.reset_index().to_csv(args.output_dir / "statistics.csv", index=False)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "statistics": statistics}, indent=2))


if __name__ == "__main__":
    main()
