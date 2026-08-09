#!/usr/bin/env python3
"""Compare locked RF reproduction metrics with the official result table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.run_dir.glob("*/seed_*/evaluation.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["seed"] = int(path.parent.name.split("_")[-1])
        rows.append(payload)
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("No RF evaluation files found")
    aggregate = raw.groupby("target").agg(
        reproduced_rmse=("overall_rmse", "mean"),
        reproduced_rmse_std=("overall_rmse", "std"),
        reproduced_cliff_rmse=("cliff_rmse", "mean"),
        reproduced_cliff_rmse_std=("cliff_rmse", "std"),
        seeds=("seed", "nunique"),
    ).reset_index()
    reference = pd.read_csv(args.reference)
    reference = reference[(reference["algorithm"] == "RF") & (reference["descriptor"] == "ECFP")][
        ["dataset", "rmse", "cliff_rmse"]
    ].rename(columns={"dataset": "target", "rmse": "official_rmse", "cliff_rmse": "official_cliff_rmse"})
    table = aggregate.merge(reference, on="target", how="outer", validate="one_to_one")
    table["overall_relative_deviation"] = (table["reproduced_rmse"] - table["official_rmse"]) / table["official_rmse"]
    table["cliff_relative_deviation"] = (table["reproduced_cliff_rmse"] - table["official_cliff_rmse"]) / table["official_cliff_rmse"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "rf_reproduction_seed_metrics.csv", index=False)
    table.to_csv(args.output_dir / "reproduction_table.csv", index=False)
    checks = {
        "targets_expected": 30,
        "targets_observed": int(len(table)),
        "all_targets_have_three_seeds": bool((table["seeds"] == 3).all()),
        "mean_absolute_overall_relative_deviation": float(table["overall_relative_deviation"].abs().mean()),
        "max_absolute_overall_relative_deviation": float(table["overall_relative_deviation"].abs().max()),
        "mean_absolute_cliff_relative_deviation": float(table["cliff_relative_deviation"].abs().mean()),
    }
    checks["go"] = bool(
        checks["targets_observed"] == 30
        and checks["all_targets_have_three_seeds"]
        and checks["mean_absolute_overall_relative_deviation"] <= 0.10
        and checks["max_absolute_overall_relative_deviation"] <= 0.25
        and checks["mean_absolute_cliff_relative_deviation"] <= 0.15
    )
    (args.output_dir / "metric_unit_tests.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
