#!/usr/bin/env python3
"""Evaluate frozen scale calibration and apply its preregistered gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_labels_for_ids, sha256_file
from scripts.build_rf_gine_scale_calibration import scale_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--gine-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    return parser.parse_args()


def metrics(path: Path, target: str, data_root: Path, mmp_root: Path) -> dict:
    manifest = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text())
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Frozen prediction hash mismatch: {path}")
    prediction = pd.read_parquet(path).set_index("mol_id")["prediction"]
    truth = read_labels_for_ids(data_root, target, prediction.index).set_index("mol_id")[
        "y_pactivity"
    ]
    edges = pd.read_parquet(mmp_root / target / "eligible_mmp_edges.parquet")
    edges = edges[
        edges["mol_id_lo"].isin(truth.index) & edges["mol_id_hi"].isin(truth.index)
    ].copy()
    edges["delta_y"] = edges["mol_id_hi"].map(truth) - edges["mol_id_lo"].map(truth)
    cliffs = edges[edges["delta_y"].abs() >= 1.0]
    endpoints = set(cliffs["mol_id_lo"]) | set(cliffs["mol_id_hi"])
    endpoint_ids = [mol_id for mol_id in prediction.index if mol_id in endpoints]
    predicted_delta = (
        prediction.reindex(cliffs["mol_id_hi"]).to_numpy()
        - prediction.reindex(cliffs["mol_id_lo"]).to_numpy()
    )
    return {
        "overall_mae": float(np.mean(np.abs(prediction - truth.loc[prediction.index]))),
        "cliff_mae": float(
            np.mean(np.abs(prediction.loc[endpoint_ids] - truth.loc[endpoint_ids]))
        ),
        "pair_delta_mae": float(
            np.mean(np.abs(predicted_delta - cliffs["delta_y"].to_numpy()))
        ),
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    candidates = [float(value) for value in config["selection"]["scale_candidates"]]
    methods = ["validation_selected", "fixed_equal", "gine", *[scale_name(x) for x in candidates]]
    rows = []
    metadata_rows = []
    for run_path in sorted(args.calibration_dir.glob("*/fold_*/seed_*/run.json")):
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["outer_test_labels_materialized"]:
            raise RuntimeError(f"Test label entered calibration builder: {run_path}")
        target, fold, seed = run["target"], int(run["fold"]), int(run["seed"])
        paths = {
            "validation_selected": run_path.parent / "validation_selected" / "predictions.parquet",
            "fixed_equal": args.ensemble_dir / target / f"fold_{fold}" / f"seed_{seed}" / "fixed_equal" / "predictions.parquet",
            "gine": args.gine_dir / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet",
            **{
                scale_name(scale): run_path.parent / scale_name(scale) / "predictions.parquet"
                for scale in candidates
            },
        }
        for method in methods:
            rows.append(
                {
                    "target": target,
                    "fold": fold,
                    "seed": seed,
                    "method": method,
                    **metrics(paths[method], target, args.data_root, args.mmp_root),
                }
            )
        metadata_rows.append(run)

    seed_frame = pd.DataFrame(rows)
    expected = 87 * len(methods)
    if len(seed_frame) != expected:
        raise RuntimeError(f"Expected {expected} method-runs, observed {len(seed_frame)}")
    fold_frame = seed_frame.groupby(["target", "fold", "method"], as_index=False).mean(
        numeric_only=True
    )
    target_frame = fold_frame.groupby(["target", "method"], as_index=False).mean(
        numeric_only=True
    )
    base = target_frame[target_frame["method"] == "gine"].set_index("target")
    equal = target_frame[target_frame["method"] == "fixed_equal"].set_index("target")
    selected = target_frame[target_frame["method"] == "validation_selected"].set_index(
        "target"
    )
    comparisons = []
    for method in ["validation_selected", *[scale_name(x) for x in candidates]]:
        table = target_frame[target_frame["method"] == method].set_index("target")
        for target in table.index:
            comparisons.append(
                {
                    "target": target,
                    "method": method,
                    **{
                        f"{metric}_relative_change_vs_gine": float(
                            table.loc[target, metric] / base.loc[target, metric] - 1
                        )
                        for metric in ["overall_mae", "cliff_mae", "pair_delta_mae"]
                    },
                    **{
                        f"{metric}_relative_change_vs_fixed_equal": float(
                            table.loc[target, metric] / equal.loc[target, metric] - 1
                        )
                        for metric in ["overall_mae", "cliff_mae", "pair_delta_mae"]
                    },
                }
            )
    comparison = pd.DataFrame(comparisons)
    primary = comparison[comparison["method"] == "validation_selected"].set_index("target")
    decision = config["decision"]
    macro = {
        "paired_runs": 87,
        "targets": len(primary),
        **{
            column: float(primary[column].mean())
            for column in primary.columns
            if column.endswith("_relative_change_vs_gine")
            or column.endswith("_relative_change_vs_fixed_equal")
        },
        "cliff_improved_targets_vs_gine": int(
            (primary["cliff_mae_relative_change_vs_gine"] < 0).sum()
        ),
        "pair_improved_targets_vs_fixed_equal": int(
            (primary["pair_delta_mae_relative_change_vs_fixed_equal"] < 0).sum()
        ),
    }
    criteria = {
        "cliff_gain_vs_gine": macro["cliff_mae_relative_change_vs_gine"]
        <= -float(decision["minimum_cliff_improvement_vs_gine"]),
        "overall_vs_gine": macro["overall_mae_relative_change_vs_gine"]
        <= float(decision["maximum_overall_relative_change_vs_gine"]),
        "pair_vs_gine": macro["pair_delta_mae_relative_change_vs_gine"]
        <= float(decision["maximum_pair_delta_relative_change_vs_gine"]),
        "cliff_directions_vs_gine": macro["cliff_improved_targets_vs_gine"]
        >= int(decision["minimum_cliff_improved_targets_vs_gine"]),
        "cliff_vs_fixed_equal": macro["cliff_mae_relative_change_vs_fixed_equal"]
        <= float(decision["maximum_cliff_relative_change_vs_fixed_equal"]),
        "overall_vs_fixed_equal": macro["overall_mae_relative_change_vs_fixed_equal"]
        <= float(decision["maximum_overall_relative_change_vs_fixed_equal"]),
        "pair_vs_fixed_equal": macro["pair_delta_mae_relative_change_vs_fixed_equal"]
        <= float(decision["maximum_pair_delta_relative_change_vs_fixed_equal"]),
        "pair_directions_vs_fixed_equal": macro["pair_improved_targets_vs_fixed_equal"]
        >= int(decision["minimum_pair_improved_targets_vs_fixed_equal"]),
    }
    macro["criteria"] = criteria
    macro["promote_calibration"] = bool(all(criteria.values()))
    metadata = pd.DataFrame(metadata_rows)
    macro["selected_scale_counts"] = {
        str(key): int(value)
        for key, value in metadata["selected_scale"].value_counts().sort_index().items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_frame.to_csv(args.output_dir / "calibration_seed_metrics.csv", index=False)
    fold_frame.to_csv(args.output_dir / "calibration_fold_metrics.csv", index=False)
    target_frame.to_csv(args.output_dir / "calibration_target_metrics.csv", index=False)
    comparison.to_csv(args.output_dir / "calibration_comparison.csv", index=False)
    metadata.to_csv(args.output_dir / "calibration_selection.csv", index=False)
    (args.output_dir / "calibration_macro.json").write_text(
        json.dumps(macro, indent=2), encoding="utf-8"
    )
    print(json.dumps(macro, indent=2))


if __name__ == "__main__":
    main()

