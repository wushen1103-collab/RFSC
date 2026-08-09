#!/usr/bin/env python3
"""Evaluate the once-frozen BindingDB-patent external confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

from bps_gnn.io import read_frozen_prediction, read_labels_for_ids
from bps_gnn.temporal import temporal_split_ids

try:
    from evaluate_chembl36_temporal import METRICS, bootstrap_relative, holm_adjust, metric_row
except ModuleNotFoundError:
    from scripts.evaluate_chembl36_temporal import (
        METRICS,
        bootstrap_relative,
        holm_adjust,
        metric_row,
    )


METHODS = ["unscaled_reference", "bps", "confidence_weighted_bps"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eligible-targets", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mmp-root", type=Path, required=True)
    parser.add_argument("--primary-prediction-root", type=Path, required=True)
    parser.add_argument("--weighted-prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    targets = json.loads(args.eligible_targets.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["seeds"]]

    # Every required file is hash-checked and matched to its response-free test
    # partition before any test response is materialized.
    for target in targets:
        _, _, test_ids = temporal_split_ids(args.data_root, target)
        expected = set(test_ids)
        for seed in seeds:
            primary = args.primary_prediction_root / target / f"seed_{seed}"
            paths = {
                "unscaled_reference": primary / "unscaled_reference" / "predictions.parquet",
                "bps": primary / "bps" / "predictions.parquet",
                "confidence_weighted_bps": args.weighted_prediction_root
                / target
                / f"seed_{seed}"
                / "predictions.parquet",
            }
            for method, path in paths.items():
                prediction = read_frozen_prediction(path)
                if set(prediction.index) != expected:
                    raise RuntimeError(f"{target} seed {seed} {method}: test partition mismatch")

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
        ]
        for seed in seeds:
            primary = args.primary_prediction_root / target / f"seed_{seed}"
            predictions = {
                "unscaled_reference": read_frozen_prediction(
                    primary / "unscaled_reference" / "predictions.parquet"
                ),
                "bps": read_frozen_prediction(primary / "bps" / "predictions.parquet"),
                "confidence_weighted_bps": read_frozen_prediction(
                    args.weighted_prediction_root
                    / target
                    / f"seed_{seed}"
                    / "predictions.parquet"
                ),
            }
            for method, prediction in predictions.items():
                rows.append(
                    {
                        "target": target,
                        "seed": seed,
                        "method": method,
                        **metric_row(prediction, truth, edges),
                    }
                )
    raw = pd.DataFrame.from_records(rows)
    target_metrics = raw.groupby(["target", "method"], as_index=False)[METRICS].mean()
    reference = target_metrics[target_metrics["method"] == "unscaled_reference"].set_index(
        "target"
    )
    decision_config = config["decision"]
    statistics: list[dict[str, object]] = []
    primary_p_values: list[float] = []
    primary_record_indices: list[int] = []
    for method_index, method in enumerate(["bps", "confidence_weighted_bps"]):
        candidate = target_metrics[target_metrics["method"] == method].set_index("target")
        for metric_index, metric in enumerate(METRICS):
            paired = pd.concat(
                [candidate[metric].rename("candidate"), reference[metric].rename("reference")],
                axis=1,
            ).dropna()
            relative = float(paired["candidate"].mean() / paired["reference"].mean() - 1.0)
            lower, upper = bootstrap_relative(
                paired["candidate"].to_numpy(),
                paired["reference"].to_numpy(),
                int(config["evaluation"]["bootstrap_iterations"]),
                29010 + 100 * method_index + metric_index,
            )
            margin = 0.0 if metric == "pair_delta_mae" else float(
                decision_config["maximum_overall_relative_cost"]
                if metric == "overall_mae"
                else decision_config["maximum_cliff_relative_cost"]
            )
            adjusted_difference = paired["candidate"] - (1.0 + margin) * paired["reference"]
            if np.allclose(adjusted_difference.to_numpy(), 0.0):
                p_value = 1.0
            else:
                p_value = float(wilcoxon(adjusted_difference, alternative="less").pvalue)
            record = {
                "method": method,
                "metric": metric,
                "targets": int(len(paired)),
                "reference_macro": float(paired["reference"].mean()),
                "candidate_macro": float(paired["candidate"].mean()),
                "relative_change": relative,
                "bootstrap95_lower": lower,
                "bootstrap95_upper": upper,
                "favorable_targets": int((paired["candidate"] < paired["reference"]).sum()),
                "favorable_fraction": float((paired["candidate"] < paired["reference"]).mean()),
                "wilcoxon_margin": margin,
                "wilcoxon_p_raw": p_value,
                "wilcoxon_p_holm": float("nan"),
            }
            statistics.append(record)
            if method == "confidence_weighted_bps":
                primary_record_indices.append(len(statistics) - 1)
                primary_p_values.append(p_value)
    for index, adjusted in zip(primary_record_indices, holm_adjust(primary_p_values)):
        statistics[index]["wilcoxon_p_holm"] = adjusted
    statistics_frame = pd.DataFrame.from_records(statistics)
    primary = statistics_frame[
        statistics_frame["method"] == "confidence_weighted_bps"
    ].set_index("metric")
    pair = primary.loc["pair_delta_mae"]
    overall = primary.loc["overall_mae"]
    cliff = primary.loc["cliff_endpoint_mae"]
    checks = {
        "pair_improvement_at_least_3pct": bool(
            pair["relative_change"] <= -float(decision_config["minimum_pair_relative_improvement"])
        ),
        "pair_bootstrap_upper_below_zero": bool(
            pair["bootstrap95_upper"] < float(decision_config["maximum_pair_bootstrap_upper"])
        ),
        "pair_direction_fraction": bool(
            pair["favorable_fraction"]
            >= float(decision_config["minimum_pair_improved_target_fraction"])
        ),
        "overall_point_cost": bool(
            overall["relative_change"] <= float(decision_config["maximum_overall_relative_cost"])
        ),
        "cliff_point_cost": bool(
            cliff["relative_change"] <= float(decision_config["maximum_cliff_relative_cost"])
        ),
        "overall_ci_noninferiority": bool(
            overall["bootstrap95_upper"]
            <= float(decision_config["maximum_overall_bootstrap_upper"])
        ),
        "cliff_ci_noninferiority": bool(
            cliff["bootstrap95_upper"] <= float(decision_config["maximum_cliff_bootstrap_upper"])
        ),
        "pair_target_power": bool(
            pair["targets"] >= int(decision_config["minimum_pair_eligible_targets"])
        ),
    }
    decision = {
        "dataset": config["name"],
        "candidate": "confidence_weighted_bps",
        "baseline": "unscaled_reference",
        "checks": checks,
        "confirmatory_go": all(checks.values()),
        "prediction_hash_preflight_before_test_response": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "raw_seed_metrics.csv", index=False)
    target_metrics.to_csv(args.output_dir / "target_metrics.csv", index=False)
    statistics_frame.to_csv(args.output_dir / "statistics.csv", index=False)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "statistics": statistics}, indent=2))


if __name__ == "__main__":
    main()
