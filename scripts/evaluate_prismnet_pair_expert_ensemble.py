#!/usr/bin/env python3
"""Evaluate the exploratory three-run PrismNet pair-expert ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from scripts.build_prismnet_pair_expert_ensemble import weight_name
from scripts.evaluate_rf_gine_scale_calibration import metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--fixed-equal-dir", type=Path, required=True)
    parser.add_argument("--gine-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    weights = [float(x) for x in config["selection"]["prism_weight_candidates"]]
    methods = ["validation_selected", "fixed_equal", "gine", *[weight_name(x) for x in weights]]
    rows, metadata = [], []
    for run_path in sorted(args.ensemble_dir.glob("*/fold_*/seed_*/run.json")):
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["outer_test_labels_materialized"]:
            raise RuntimeError("Test label entered pair-expert builder")
        target, fold, seed = run["target"], int(run["fold"]), int(run["seed"])
        paths = {
            "validation_selected": run_path.parent / "validation_selected" / "predictions.parquet",
            "fixed_equal": args.fixed_equal_dir / target / f"fold_{fold}" / f"seed_{seed}" / "fixed_equal" / "predictions.parquet",
            "gine": args.gine_dir / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet",
            **{weight_name(w): run_path.parent / weight_name(w) / "predictions.parquet" for w in weights},
        }
        for method in methods:
            rows.append({"target": target, "fold": fold, "seed": seed, "method": method,
                         **metrics(paths[method], target, args.data_root, args.mmp_root)})
        metadata.append(run)
    table = pd.DataFrame(rows)
    if len(metadata) != 3:
        raise RuntimeError(f"Expected three ensemble runs, observed {len(metadata)}")
    comparisons = []
    for keys, group in table.groupby(["target", "fold", "seed"]):
        group = group.set_index("method")
        for method in ["validation_selected", *[weight_name(x) for x in weights]]:
            row = {"target": keys[0], "fold": keys[1], "seed": keys[2], "method": method}
            for metric in ["overall_mae", "cliff_mae", "pair_delta_mae"]:
                row[f"{metric}_relative_change_vs_fixed_equal"] = float(group.loc[method, metric] / group.loc["fixed_equal", metric] - 1)
                row[f"{metric}_relative_change_vs_gine"] = float(group.loc[method, metric] / group.loc["gine", metric] - 1)
            comparisons.append(row)
    comparison = pd.DataFrame(comparisons)
    primary = comparison[comparison["method"] == "validation_selected"]
    decision = config["decision"]
    macro = {column: float(primary[column].mean()) for column in primary.columns if column.endswith("_relative_change_vs_fixed_equal") or column.endswith("_relative_change_vs_gine")}
    macro["pair_improved_runs_vs_fixed_equal"] = int((primary["pair_delta_mae_relative_change_vs_fixed_equal"] < 0).sum())
    criteria = {
        "overall": macro["overall_mae_relative_change_vs_fixed_equal"] <= float(decision["maximum_overall_relative_change_vs_fixed_equal"]),
        "cliff": macro["cliff_mae_relative_change_vs_fixed_equal"] <= float(decision["maximum_cliff_relative_change_vs_fixed_equal"]),
        "pair": macro["pair_delta_mae_relative_change_vs_fixed_equal"] <= -float(decision["minimum_pair_delta_improvement_vs_fixed_equal"]),
        "pair_directions": macro["pair_improved_runs_vs_fixed_equal"] >= int(decision["minimum_pair_improved_runs"]),
    }
    macro["criteria"] = criteria
    macro["authorize_unseen_confirmation"] = bool(all(criteria.values()))
    macro["confirmatory_result"] = False
    macro["selected_weight_counts"] = {str(k): int(v) for k, v in pd.DataFrame(metadata)["selected_prism_weight"].value_counts().sort_index().items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "pair_expert_metrics.csv", index=False)
    comparison.to_csv(args.output_dir / "pair_expert_comparison.csv", index=False)
    pd.DataFrame(metadata).to_csv(args.output_dir / "pair_expert_selection.csv", index=False)
    (args.output_dir / "pair_expert_macro.json").write_text(json.dumps(macro, indent=2), encoding="utf-8")
    print(json.dumps(macro, indent=2))
    print(primary.to_string(index=False))


if __name__ == "__main__":
    main()

