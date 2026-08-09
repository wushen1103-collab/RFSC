#!/usr/bin/env python3
"""Compare paired RF and GINE strict-SCG metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf-dir", type=Path, required=True)
    parser.add_argument("--gine-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def collect(root: Path, method: str) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*/fold_*/seed_*/evaluation.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        parts = path.parts
        rows.append({"target": value["target"], "fold": int(parts[-3].split("_")[-1]), "seed": int(parts[-2].split("_")[-1]), "method": method, "overall_mae": value["overall_mae"], "cliff_mae": value["cliff_mae"], "pair_delta_mae": value["pair_delta_mae"]})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = pd.concat([collect(args.rf_dir, "rf_ecfp"), collect(args.gine_dir, "gine")], ignore_index=True)
    counts = seed.groupby("method").size().to_dict()
    if counts != {"gine": 87, "rf_ecfp": 87}:
        raise RuntimeError(f"Expected 87 paired runs per method, got {counts}")
    fold = seed.groupby(["target", "fold", "method"], as_index=False).mean(numeric_only=True)
    target = fold.groupby(["target", "method"], as_index=False).mean(numeric_only=True)
    rf = target[target["method"] == "rf_ecfp"].set_index("target")
    gine = target[target["method"] == "gine"].set_index("target")
    comparison = pd.DataFrame(
        {
            "target": rf.index,
            "rf_overall_mae": rf["overall_mae"],
            "gine_overall_mae": gine["overall_mae"],
            "overall_relative_change": rf["overall_mae"] / gine["overall_mae"] - 1,
            "rf_cliff_mae": rf["cliff_mae"],
            "gine_cliff_mae": gine["cliff_mae"],
            "cliff_relative_change": rf["cliff_mae"] / gine["cliff_mae"] - 1,
            "rf_pair_delta_mae": rf["pair_delta_mae"],
            "gine_pair_delta_mae": gine["pair_delta_mae"],
            "pair_delta_relative_change": rf["pair_delta_mae"] / gine["pair_delta_mae"] - 1,
        }
    ).reset_index(drop=True)
    decision = config["decision"]
    macro = {
        "paired_runs_per_method": 87,
        "targets": len(comparison),
        "overall_relative_change": float(comparison["overall_relative_change"].mean()),
        "cliff_relative_change": float(comparison["cliff_relative_change"].mean()),
        "pair_delta_relative_change": float(comparison["pair_delta_relative_change"].mean()),
        "overall_improved_targets": int((comparison["overall_relative_change"] < 0).sum()),
        "cliff_improved_targets": int((comparison["cliff_relative_change"] < 0).sum()),
        "pair_delta_improved_targets": int((comparison["pair_delta_relative_change"] < 0).sum()),
    }
    criteria = {
        "cliff_gain": macro["cliff_relative_change"] <= -float(decision["minimum_cliff_improvement_vs_gine"]),
        "overall": macro["overall_relative_change"] <= float(decision["maximum_overall_relative_change_vs_gine"]),
        "pair_delta": macro["pair_delta_relative_change"] <= float(decision["maximum_pair_delta_degradation_vs_gine"]),
        "directions": macro["cliff_improved_targets"] >= int(decision["minimum_improved_targets"]),
    }
    macro["criteria"], macro["switch_backbone"] = criteria, bool(all(criteria.values()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed.to_csv(args.output_dir / "rf_gine_seed_metrics.csv", index=False)
    fold.to_csv(args.output_dir / "rf_gine_fold_metrics.csv", index=False)
    comparison.to_csv(args.output_dir / "rf_gine_target_comparison.csv", index=False)
    (args.output_dir / "rf_gine_macro.json").write_text(json.dumps(macro, indent=2), encoding="utf-8")
    print(json.dumps(macro, indent=2))


if __name__ == "__main__":
    main()

