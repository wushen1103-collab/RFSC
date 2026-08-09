#!/usr/bin/env python3
"""Evaluate fixed-half PrismNet pair-expert predictions on unseen folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from scripts.evaluate_rf_gine_scale_calibration import metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation-dir", type=Path, required=True)
    parser.add_argument("--fixed-equal-dir", type=Path, required=True)
    parser.add_argument("--gine-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    return parser.parse_args()


def confirmation_gate(comparison: pd.DataFrame, decision: dict) -> tuple[dict, bool]:
    macro = {column: float(comparison[column].mean()) for column in comparison.columns if column.endswith("_vs_fixed_equal") or column.endswith("_vs_gine")}
    pair_directions = int((comparison["pair_delta_vs_fixed_equal"] < 0).sum())
    criteria = {
        "overall_vs_fixed_equal": macro["overall_vs_fixed_equal"] <= float(decision["maximum_overall_relative_change_vs_fixed_equal"]),
        "cliff_vs_fixed_equal": macro["cliff_vs_fixed_equal"] <= float(decision["maximum_cliff_relative_change_vs_fixed_equal"]),
        "pair_vs_fixed_equal": macro["pair_delta_vs_fixed_equal"] <= -float(decision["minimum_pair_delta_improvement_vs_fixed_equal"]),
        "pair_both_directions": pair_directions == len(comparison) if decision["require_pair_improvement_in_both_runs"] else True,
        "overall_vs_gine": macro["overall_vs_gine"] <= float(decision["maximum_overall_relative_change_vs_gine"]),
        "cliff_vs_gine": macro["cliff_vs_gine"] <= float(decision["maximum_cliff_relative_change_vs_gine"]),
        "pair_vs_gine": macro["pair_delta_vs_gine"] <= float(decision["maximum_pair_delta_relative_change_vs_gine"]),
    }
    return {**macro, "pair_improved_runs": pair_directions, "criteria": criteria}, bool(all(criteria.values()))


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows, comparisons = [], []
    for run_path in sorted(args.confirmation_dir.glob("*/fold_*/seed_*/run.json")):
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["outer_test_labels_materialized"] or run["weight_selected_on_confirmation_validation"]:
            raise RuntimeError("Confirmation contract violation")
        target, fold, seed = run["target"], int(run["fold"]), int(run["seed"])
        paths = {
            "fixed_half_prism_expert": run_path.parent / "predictions.parquet",
            "fixed_equal": args.fixed_equal_dir / target / f"fold_{fold}" / f"seed_{seed}" / "fixed_equal" / "predictions.parquet",
            "gine": args.gine_dir / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet",
        }
        local = []
        for method, path in paths.items():
            local.append({"target": target, "fold": fold, "seed": seed, "method": method,
                          **metrics(path, target, args.data_root, args.mmp_root)})
        rows.extend(local)
        table = pd.DataFrame(local).set_index("method")
        row = {"target": target, "fold": fold, "seed": seed}
        for short, metric in [("overall", "overall_mae"), ("cliff", "cliff_mae"), ("pair_delta", "pair_delta_mae")]:
            row[f"{short}_vs_fixed_equal"] = float(table.loc["fixed_half_prism_expert", metric] / table.loc["fixed_equal", metric] - 1)
            row[f"{short}_vs_gine"] = float(table.loc["fixed_half_prism_expert", metric] / table.loc["gine", metric] - 1)
        comparisons.append(row)
    comparison = pd.DataFrame(comparisons)
    if len(comparison) != 2:
        raise RuntimeError(f"Expected two unseen confirmation folds, observed {len(comparison)}")
    summary, passed = confirmation_gate(comparison, config["decision"])
    summary["confirmation_passed"] = passed
    summary["runs"] = len(comparison)
    summary["published_pretrained_prismnet"] = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "confirmation_metrics.csv", index=False)
    comparison.to_csv(args.output_dir / "confirmation_comparison.csv", index=False)
    (args.output_dir / "confirmation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

