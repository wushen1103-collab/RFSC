#!/usr/bin/env python3
"""Evaluate the pre-registered validation smoke after predictions are frozen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from evaluate_rf_gine_ensemble import metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = []
    metadata = []
    for run_path in sorted(args.run_dir.glob("*/fold_*/seed_*/run.json")):
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["validation_labels_materialized"] or run["outer_test_labels_or_predictions_materialized"]:
            raise RuntimeError("Query label entered prediction builder")
        metadata.append(run)
        for method in ["fixed_equal", "context_expert"]:
            rows.append(
                {
                    "target": run["target"],
                    "fold": run["fold"],
                    "seed": run["seed"],
                    "method": method,
                    **metrics(
                        run_path.parent / method / "predictions.parquet",
                        run["target"],
                        args.data_root,
                        args.mmp_root,
                    ),
                }
            )
    table = pd.DataFrame(rows)
    comparisons = []
    for keys, group in table.groupby(["target", "fold", "seed"]):
        group = group.set_index("method")
        row = {"target": keys[0], "fold": keys[1], "seed": keys[2]}
        for metric_name in ["overall_mae", "cliff_mae", "pair_delta_mae"]:
            row[f"{metric_name}_relative_change_vs_fixed_equal"] = float(
                group.loc["context_expert", metric_name] / group.loc["fixed_equal", metric_name] - 1.0
            )
        comparisons.append(row)
    comparison = pd.DataFrame(comparisons)
    decision = config["decision"]
    macro = {
        column: float(comparison[column].mean())
        for column in comparison
        if column.endswith("_relative_change_vs_fixed_equal")
    }
    macro["supported_query_edges"] = int(metadata[0]["supported_query_edges"])
    macro["pair_improved_seeds"] = int(
        (comparison["pair_delta_mae_relative_change_vs_fixed_equal"] < 0).sum()
    )
    criteria = {
        "support": macro["supported_query_edges"] >= int(decision["minimum_supported_query_edges"]),
        "pair": macro["pair_delta_mae_relative_change_vs_fixed_equal"] <= -float(
            decision["minimum_pair_delta_improvement_vs_fixed_equal"]
        ),
        "overall": macro["overall_mae_relative_change_vs_fixed_equal"] <= float(
            decision["maximum_overall_relative_change_vs_fixed_equal"]
        ),
        "cliff": macro["cliff_mae_relative_change_vs_fixed_equal"] <= float(
            decision["maximum_cliff_relative_change_vs_fixed_equal"]
        ),
        "directions": macro["pair_improved_seeds"] >= int(decision["minimum_pair_improved_seeds"]),
    }
    macro["criteria"] = criteria
    macro["authorize_full_validation"] = all(criteria.values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "metrics.csv", index=False)
    comparison.to_csv(args.output_dir / "comparison.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(macro, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(macro, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

