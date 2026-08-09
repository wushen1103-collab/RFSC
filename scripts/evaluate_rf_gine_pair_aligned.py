#!/usr/bin/env python3
"""Evaluate pair-aligned RF/GINE test predictions with fixed outer-test labels."""

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
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--gine-dir", type=Path, required=True)
    parser.add_argument("--rf-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    selected_method = str(config["selection"].get("output_method", "pair_selected"))
    rows, metadata = [], []
    for run_path in sorted(args.ensemble_dir.glob("*/fold_*/seed_*/run.json")):
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["outer_test_labels_materialized"]:
            raise RuntimeError("Test labels entered pair-aligned builder")
        metadata.append(run)
        target, fold, seed = run["target"], int(run["fold"]), int(run["seed"])
        paths = {
            selected_method: run_path.parent / selected_method / "predictions.parquet",
            "fixed_equal": run_path.parent / "fixed_equal" / "predictions.parquet",
            "gine": args.gine_dir / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet",
            "rf_ecfp": args.rf_dir / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet",
        }
        for method, path in paths.items():
            rows.append({"target": target, "fold": fold, "seed": seed, "method": method, **metrics(path, target, Path("data/processed/moleculeace"), Path("graphs/mmp"))})
    frame = pd.DataFrame(rows)
    if len(metadata) != 87 or len(frame) != 87 * 4:
        raise RuntimeError(f"Expected 87 runs and 348 rows, got {len(metadata)} and {len(frame)}")
    fold = frame.groupby(["target", "fold", "method"], as_index=False).mean(numeric_only=True)
    target = fold.groupby(["target", "method"], as_index=False).mean(numeric_only=True)
    comparisons = []
    for target_name in sorted(target["target"].unique()):
        group = target[target["target"] == target_name].set_index("method")
        for method in [selected_method, "fixed_equal"]:
            comparisons.append({
                "target": target_name,
                "method": method,
                **{f"{metric}_relative_change_vs_gine": float(group.loc[method, metric] / group.loc["gine", metric] - 1.0) for metric in ["overall_mae", "cliff_mae", "pair_delta_mae"]},
                **{f"{metric}_relative_change_vs_fixed_equal": float(group.loc[method, metric] / group.loc["fixed_equal", metric] - 1.0) for metric in ["overall_mae", "cliff_mae", "pair_delta_mae"]},
            })
    comparison = pd.DataFrame(comparisons)
    primary = comparison[comparison["method"] == selected_method]
    decision = config["decision"]
    macro = {
        column: float(primary[column].mean())
        for column in primary
        if column.endswith("_relative_change_vs_gine") or column.endswith("_relative_change_vs_fixed_equal")
    }
    macro["pair_improved_runs_vs_fixed_equal"] = int(
        (frame[frame["method"] == selected_method].set_index(["target", "fold", "seed"])["pair_delta_mae"]
         < frame[frame["method"] == "fixed_equal"].set_index(["target", "fold", "seed"])["pair_delta_mae"]).sum()
    )
    macro["improved_targets_vs_gine"] = int((primary["pair_delta_mae_relative_change_vs_gine"] < 0).sum())
    macro["criteria"] = {
        "overall_vs_gine": macro["overall_mae_relative_change_vs_gine"] <= float(decision["maximum_overall_relative_change_vs_gine"]),
        "cliff_vs_gine": macro["cliff_mae_relative_change_vs_gine"] <= float(decision["maximum_cliff_relative_change_vs_gine"]),
        "pair_vs_gine": macro["pair_delta_mae_relative_change_vs_gine"] <= float(decision["maximum_pair_delta_relative_change_vs_gine"]),
        "overall_vs_equal": macro["overall_mae_relative_change_vs_fixed_equal"] <= float(decision["maximum_overall_relative_change_vs_fixed_equal"]),
        "cliff_vs_equal": macro["cliff_mae_relative_change_vs_fixed_equal"] <= float(decision["maximum_cliff_relative_change_vs_fixed_equal"]),
        "pair_vs_equal": macro["pair_delta_mae_relative_change_vs_fixed_equal"] <= float(decision["maximum_pair_delta_relative_change_vs_fixed_equal"]),
        "pair_directions": macro["pair_improved_runs_vs_fixed_equal"] >= int(decision["minimum_pair_improved_runs_vs_fixed_equal"]),
        "target_directions": macro["improved_targets_vs_gine"] >= int(decision["minimum_improved_targets_vs_gine"]),
    }
    macro["promote"] = all(macro["criteria"].values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "metrics.csv", index=False)
    comparison.to_csv(args.output_dir / "comparison.csv", index=False)
    pd.DataFrame(metadata).to_csv(args.output_dir / "selection.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(macro, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(macro, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
