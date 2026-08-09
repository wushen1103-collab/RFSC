#!/usr/bin/env python3
"""Evaluate frozen RF/GINE ensemble predictions and apply the backbone gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_labels_for_ids, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--rf-dir", type=Path, required=True)
    parser.add_argument("--gine-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    return parser.parse_args()


def metrics(path: Path, target: str, data_root: Path, mmp_root: Path) -> dict:
    manifest = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Frozen prediction hash mismatch: {path}")
    prediction = pd.read_parquet(path).set_index("mol_id")["prediction"]
    truth = read_labels_for_ids(data_root, target, prediction.index).set_index("mol_id")["y_pactivity"]
    edges = pd.read_parquet(mmp_root / target / "eligible_mmp_edges.parquet")
    edges = edges[edges["mol_id_lo"].isin(truth.index) & edges["mol_id_hi"].isin(truth.index)].copy()
    edges["delta_y"] = edges["mol_id_hi"].map(truth) - edges["mol_id_lo"].map(truth)
    cliffs = edges[edges["delta_y"].abs() >= 1.0]
    endpoints = set(cliffs["mol_id_lo"]) | set(cliffs["mol_id_hi"])
    endpoint_ids = [mol_id for mol_id in prediction.index if mol_id in endpoints]
    predicted_delta = prediction.reindex(cliffs["mol_id_hi"]).to_numpy() - prediction.reindex(cliffs["mol_id_lo"]).to_numpy()
    return {
        "overall_mae": float(np.mean(np.abs(prediction - truth.loc[prediction.index]))),
        "cliff_mae": float(np.mean(np.abs(prediction.loc[endpoint_ids] - truth.loc[endpoint_ids]))),
        "pair_delta_mae": float(np.mean(np.abs(predicted_delta - cliffs["delta_y"].to_numpy()))),
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = []
    for run_path in sorted(args.ensemble_dir.glob("*/fold_*/seed_*/run.json")):
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["outer_test_labels_materialized"]:
            raise RuntimeError(f"Test label entered ensemble builder: {run_path}")
        target, fold, seed = run["target"], int(run["fold"]), int(run["seed"])
        paths = {
            "validation_selected": run_path.parent / "validation_selected" / "predictions.parquet",
            "fixed_equal": run_path.parent / "fixed_equal" / "predictions.parquet",
            "rf_ecfp": args.rf_dir / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet",
            "gine": args.gine_dir / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet",
        }
        for method, path in paths.items():
            rows.append({"target": target, "fold": fold, "seed": seed, "method": method, **metrics(path, target, args.data_root, args.mmp_root)})
    seed = pd.DataFrame(rows)
    if len(seed) != 87 * 4:
        raise RuntimeError(f"Expected 348 method-runs, observed {len(seed)}")
    fold = seed.groupby(["target", "fold", "method"], as_index=False).mean(numeric_only=True)
    target = fold.groupby(["target", "method"], as_index=False).mean(numeric_only=True)
    base = target[target["method"] == "gine"].set_index("target")
    comparisons = []
    for method in ["rf_ecfp", "fixed_equal", "validation_selected"]:
        table = target[target["method"] == method].set_index("target")
        for name in table.index:
            comparisons.append({"target": name, "method": method, "overall_relative_change_vs_gine": float(table.loc[name, "overall_mae"] / base.loc[name, "overall_mae"] - 1), "cliff_relative_change_vs_gine": float(table.loc[name, "cliff_mae"] / base.loc[name, "cliff_mae"] - 1), "pair_delta_relative_change_vs_gine": float(table.loc[name, "pair_delta_mae"] / base.loc[name, "pair_delta_mae"] - 1)})
    comparison = pd.DataFrame(comparisons)
    primary = comparison[comparison["method"] == "validation_selected"].set_index("target")
    equal = target[target["method"] == "fixed_equal"].set_index("target")
    selected = target[target["method"] == "validation_selected"].set_index("target")
    decision = config["decision"]
    macro = {
        "paired_runs": 87,
        "targets": len(primary),
        "overall_relative_change_vs_gine": float(primary["overall_relative_change_vs_gine"].mean()),
        "cliff_relative_change_vs_gine": float(primary["cliff_relative_change_vs_gine"].mean()),
        "pair_delta_relative_change_vs_gine": float(primary["pair_delta_relative_change_vs_gine"].mean()),
        "overall_improved_targets": int((primary["overall_relative_change_vs_gine"] < 0).sum()),
        "cliff_improved_targets": int((primary["cliff_relative_change_vs_gine"] < 0).sum()),
        "pair_delta_improved_targets": int((primary["pair_delta_relative_change_vs_gine"] < 0).sum()),
        "selected_minus_equal_cliff_mae": float((selected["cliff_mae"] - equal["cliff_mae"]).mean()),
    }
    criteria = {
        "cliff_gain": macro["cliff_relative_change_vs_gine"] <= -float(decision["minimum_cliff_improvement_vs_gine"]),
        "overall": macro["overall_relative_change_vs_gine"] <= float(decision["maximum_overall_relative_change_vs_gine"]),
        "pair_delta": macro["pair_delta_relative_change_vs_gine"] <= float(decision["maximum_pair_delta_relative_change_vs_gine"]),
        "directions": macro["cliff_improved_targets"] >= int(decision["minimum_improved_targets"]),
        "beats_equal": macro["selected_minus_equal_cliff_mae"] < 0,
    }
    macro["criteria"], macro["promote_backbone"] = criteria, bool(all(criteria.values()))
    weights = pd.DataFrame([json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.ensemble_dir.glob("*/fold_*/seed_*/run.json"))])
    weight_counts = weights["selected_rf_weight"].value_counts().sort_index().to_dict()
    macro["selected_weight_counts"] = {str(key): int(value) for key, value in weight_counts.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed.to_csv(args.output_dir / "ensemble_seed_metrics.csv", index=False)
    fold.to_csv(args.output_dir / "ensemble_fold_metrics.csv", index=False)
    target.to_csv(args.output_dir / "ensemble_target_metrics.csv", index=False)
    comparison.to_csv(args.output_dir / "ensemble_comparison.csv", index=False)
    weights.to_csv(args.output_dir / "ensemble_weight_selection.csv", index=False)
    (args.output_dir / "ensemble_macro.json").write_text(json.dumps(macro, indent=2), encoding="utf-8")
    print(json.dumps(macro, indent=2))


if __name__ == "__main__":
    main()

