#!/usr/bin/env python3
"""Evaluate one frozen PrismNet SCG pilot against paired frozen anchors."""

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


def pilot_gate(relative: dict[str, float], decision: dict) -> tuple[dict, bool]:
    criteria = {
        name: relative[name] <= float(decision[f"maximum_{name}_relative_change"])
        for name in ["overall", "cliff", "pair_delta"]
    }
    passed = sum(criteria.values())
    advance = (
        (criteria["overall"] if decision["overall_must_pass"] else True)
        and passed >= int(decision["minimum_passed_endpoints"])
    )
    return criteria, bool(advance)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pilot = config["pilot"]
    target, fold, seed = pilot["target"], int(pilot["fold"]), int(pilot["seed"])
    run_path = args.run_dir / target / f"fold_{fold}" / f"seed_{seed}" / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run["outer_test_labels_materialized"] or not run["outer_test_predictions_generated"]:
        raise RuntimeError("PrismNet pilot violated the label-private prediction contract")
    if run["published_pretrained_reproduction"]:
        raise RuntimeError("Random-init pilot unexpectedly used a pretrained checkpoint")
    if (run["target"], int(run["fold"]), int(run["seed"])) != (target, fold, seed):
        raise RuntimeError("PrismNet pilot identity does not match preregistration")

    paths = {
        "prismnet_random_init": run_path.parent / "predictions.parquet",
        "gine": args.gine_dir / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet",
        "rf_ecfp": args.rf_dir / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet",
        "fixed_equal": args.ensemble_dir / target / f"fold_{fold}" / f"seed_{seed}" / "fixed_equal" / "predictions.parquet",
    }
    rows = [
        {
            "target": target,
            "fold": fold,
            "seed": seed,
            "method": method,
            **metrics(path, target, args.data_root, args.mmp_root),
        }
        for method, path in paths.items()
    ]
    table = pd.DataFrame(rows)
    base = table[table["method"] == "gine"].iloc[0]
    prism = table[table["method"] == "prismnet_random_init"].iloc[0]
    relative = {
        name: float(prism[f"{name}_mae"] / base[f"{name}_mae"] - 1)
        for name in ["overall", "cliff", "pair_delta"]
    }
    criteria, advance = pilot_gate(relative, config["decision"])
    summary = {
        "target": target,
        "fold": fold,
        "seed": seed,
        "initialization": run["initialization"],
        "published_pretrained_reproduction": False,
        "best_epoch": run["best_epoch"],
        "best_validation_rmse": run["best_validation_rmse"],
        "wall_time_seconds": run["wall_time_seconds"],
        "peak_vram_bytes": run["peak_vram_bytes"],
        "relative_change_vs_gine": relative,
        "criteria": criteria,
        "advance_random_init_matrix": advance,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "prismnet_pilot_metrics.csv", index=False)
    (args.output_dir / "prismnet_pilot_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

