#!/usr/bin/env python3
"""Evaluate post-primary CW-BPS on ChEMBL as explicitly exploratory evidence."""

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
    seeds = [0, 1, 2]
    # Verify all new artifacts and partitions before materializing responses.
    for target in targets:
        _, _, test_ids = temporal_split_ids(args.data_root, target)
        for seed in seeds:
            prediction = read_frozen_prediction(
                args.weighted_prediction_root / target / f"seed_{seed}" / "predictions.parquet"
            )
            if set(prediction.index) != set(test_ids):
                raise RuntimeError(f"{target} seed {seed}: CW-BPS partition mismatch")

    rows: list[dict[str, object]] = []
    for target in targets:
        _, _, test_ids = temporal_split_ids(args.data_root, target)
        truth = read_labels_for_ids(args.data_root, target, test_ids).set_index("mol_id")["y_pactivity"]
        all_edges = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        edges = all_edges[
            all_edges["mol_id_lo"].isin(truth.index) & all_edges["mol_id_hi"].isin(truth.index)
        ]
        for seed in seeds:
            primary_dir = args.primary_prediction_root / target / f"seed_{seed}"
            predictions = {
                "unscaled_reference": read_frozen_prediction(
                    primary_dir / "unscaled_reference" / "predictions.parquet"
                ),
                "bps": read_frozen_prediction(primary_dir / "bps" / "predictions.parquet"),
                "confidence_weighted_bps": read_frozen_prediction(
                    args.weighted_prediction_root / target / f"seed_{seed}" / "predictions.parquet"
                ),
            }
            for method, prediction in predictions.items():
                rows.append(
                    {"target": target, "seed": seed, "method": method, **metric_row(prediction, truth, edges)}
                )
    raw = pd.DataFrame.from_records(rows)
    target_metrics = raw.groupby(["target", "method"], as_index=False)[METRICS].mean()
    stats: list[dict[str, object]] = []
    for candidate_method in ["bps", "confidence_weighted_bps"]:
        candidate = target_metrics[target_metrics["method"] == candidate_method].set_index("target")
        reference = target_metrics[target_metrics["method"] == "unscaled_reference"].set_index("target")
        for metric_index, metric in enumerate(METRICS):
            paired = pd.concat(
                [candidate[metric].rename("candidate"), reference[metric].rename("reference")], axis=1
            ).dropna()
            relative = float(paired["candidate"].mean() / paired["reference"].mean() - 1.0)
            lower, upper = bootstrap_relative(
                paired["candidate"].to_numpy(), paired["reference"].to_numpy(), 10000,
                12000 + 100 * ["bps", "confidence_weighted_bps"].index(candidate_method) + metric_index,
            )
            stats.append(
                {
                    "method": candidate_method,
                    "metric": metric,
                    "targets": int(len(paired)),
                    "relative_change": relative,
                    "bootstrap95_lower": lower,
                    "bootstrap95_upper": upper,
                    "favorable_targets": int((paired["candidate"] < paired["reference"]).sum()),
                }
            )
    stats_frame = pd.DataFrame.from_records(stats)
    table = stats_frame.pivot(index="method", columns="metric", values="relative_change")
    main, weighted = table.loc["bps"], table.loc["confidence_weighted_bps"]
    checks = {
        "pair_improvement_at_least_3pct": float(weighted["pair_delta_mae"])
        <= -float(config["decision"]["minimum_pair_relative_improvement_vs_unscaled"]),
        "overall_cost_no_larger_than_ordinary_bps": float(weighted["overall_mae"])
        <= float(main["overall_mae"]),
        "cliff_cost_no_larger_than_ordinary_bps": float(weighted["cliff_endpoint_mae"])
        <= float(main["cliff_endpoint_mae"]),
    }
    decision = {"exploratory_go_to_fresh_confirmation": bool(all(checks.values())), "checks": checks}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "raw_seed_metrics.csv", index=False)
    target_metrics.to_csv(args.output_dir / "target_metrics.csv", index=False)
    stats_frame.to_csv(args.output_dir / "statistics.csv", index=False)
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps({"decision": decision, "statistics": stats}, indent=2))


if __name__ == "__main__":
    main()
