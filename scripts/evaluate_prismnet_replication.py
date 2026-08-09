#!/usr/bin/env python3
"""Evaluate the frozen two-point PrismNet random-init replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from scripts.evaluate_rf_gine_scale_calibration import metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gine-dir", type=Path, required=True)
    parser.add_argument("--rf-dir", type=Path, required=True)
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    return parser.parse_args()


def replication_gate(comparison: pd.DataFrame, decision: dict) -> tuple[dict, bool]:
    limit = float(decision["maximum_endpoint_relative_change"])
    passed_cells = int(
        (comparison[["overall", "cliff", "pair_delta"]] <= limit).to_numpy().sum()
    )
    all_overall = bool((comparison["overall"] <= limit).all())
    pair_improved = int((comparison["pair_delta"] < 0).sum())
    criteria = {
        "all_runs_overall": all_overall
        if decision["all_runs_overall_must_pass"]
        else True,
        "passed_endpoint_cells": passed_cells
        >= int(decision["minimum_passed_endpoint_cells"]),
        "pair_improved_runs": pair_improved >= int(decision["minimum_pair_improved_runs"]),
    }
    return {
        "passed_endpoint_cells": passed_cells,
        "pair_improved_runs": pair_improved,
        "criteria": criteria,
    }, bool(all(criteria.values()))


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    metric_rows, comparison_rows, run_rows = [], [], []
    for task in config["replication"]["tasks"]:
        target, fold, seed = task["target"], int(task["fold"]), int(task["seed"])
        run_path = args.run_dir / target / f"fold_{fold}" / f"seed_{seed}" / "run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["outer_test_labels_materialized"] or run["published_pretrained_reproduction"]:
            raise RuntimeError(f"Invalid random-init replication contract: {run_path}")
        paths = {
            "prismnet_random_init": run_path.parent / "predictions.parquet",
            "gine": args.gine_dir / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet",
            "rf_ecfp": args.rf_dir / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet",
            "fixed_equal": args.ensemble_dir / target / f"fold_{fold}" / f"seed_{seed}" / "fixed_equal" / "predictions.parquet",
        }
        rows = [
            {"target": target, "fold": fold, "seed": seed, "method": method,
             **metrics(path, target, args.data_root, args.mmp_root)}
            for method, path in paths.items()
        ]
        metric_rows.extend(rows)
        table = pd.DataFrame(rows).set_index("method")
        comparison_rows.append(
            {"target": target, "fold": fold, "seed": seed,
             **{name: float(table.loc["prismnet_random_init", f"{name}_mae"] / table.loc["gine", f"{name}_mae"] - 1)
                for name in ["overall", "cliff", "pair_delta"]}}
        )
        run_rows.append(run)
    comparison = pd.DataFrame(comparison_rows)
    gate, advance = replication_gate(comparison, config["decision"])
    summary = {
        "runs": len(comparison),
        "random_init_only": True,
        "mean_relative_change_vs_gine": {
            name: float(comparison[name].mean()) for name in ["overall", "cliff", "pair_delta"]
        },
        **gate,
        "advance_random_init_matrix": advance,
        "total_wall_time_seconds": float(sum(row["wall_time_seconds"] for row in run_rows)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "prismnet_replication_metrics.csv", index=False)
    comparison.to_csv(args.output_dir / "prismnet_replication_comparison.csv", index=False)
    pd.DataFrame(run_rows).to_csv(args.output_dir / "prismnet_replication_runs.csv", index=False)
    (args.output_dir / "prismnet_replication_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

