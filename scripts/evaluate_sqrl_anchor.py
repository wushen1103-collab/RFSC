#!/usr/bin/env python3
"""Evaluate frozen validation-only labeled-anchor predictions."""

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
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    return parser.parse_args()


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows, completed = [], 0
    for run_path in sorted(args.run_dir.glob("*/fold_*/run.json")):
        metadata = json.loads(run_path.read_text(encoding="utf-8"))
        if not metadata["individual_train_labels_used_at_query_inference"]:
            raise RuntimeError("Anchor baseline did not disclose train-label inference")
        if metadata["validation_labels_materialized"] or metadata["outer_test_predictions_generated"] or metadata["outer_test_labels_materialized"]:
            raise RuntimeError(f"Anchor firewall violation: {run_path}")
        path = run_path.parent / "validation_predictions.parquet"
        manifest = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8"))
        if sha256_file(path) != manifest["prediction_sha256"]:
            raise RuntimeError(f"Frozen hash mismatch: {path}")
        predictions = pd.read_parquet(path)
        target, fold = metadata["target"], int(metadata["fold"])
        query_ids = predictions["mol_id"].unique().tolist()
        truth = read_labels_for_ids(args.data_root, target, query_ids).set_index("mol_id")["y_pactivity"]
        mmp = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        internal = mmp[mmp["mol_id_lo"].isin(query_ids) & mmp["mol_id_hi"].isin(query_ids)].copy()
        internal["delta_y"] = internal["mol_id_hi"].map(truth) - internal["mol_id_lo"].map(truth)
        cliffs = internal[internal["delta_y"].abs() >= 1.0]
        endpoints = set(cliffs["mol_id_lo"]) | set(cliffs["mol_id_hi"])
        for (seed, method), frame in predictions.groupby(["seed", "method"]):
            series = frame.set_index("mol_id")["prediction"]
            correction = frame.set_index("mol_id")["correction_value"]
            base = frame.set_index("mol_id")["base_prediction"]
            endpoint_ids = [mol_id for mol_id in series.index if mol_id in endpoints]
            endpoint_correction = correction.loc[endpoint_ids].to_numpy()
            endpoint_required = (truth - base).loc[endpoint_ids].to_numpy()
            active = np.abs(endpoint_correction) > 1e-12
            predicted_delta = series.reindex(cliffs["mol_id_hi"]).to_numpy() - series.reindex(cliffs["mol_id_lo"]).to_numpy()
            rows.append(
                {
                    "target": target,
                    "fold": fold,
                    "seed": int(seed),
                    "method": method,
                    "overall_mae": float(np.mean(np.abs(series - truth.loc[series.index]))),
                    "cliff_endpoint_mae": float(np.mean(np.abs(series.loc[endpoint_ids] - truth.loc[endpoint_ids]))),
                    "pair_delta_mae": float(np.mean(np.abs(predicted_delta - cliffs["delta_y"].to_numpy()))),
                    "endpoint_correction_residual_correlation": safe_corr(endpoint_correction, endpoint_required),
                    "endpoint_correction_sign_accuracy": float(np.mean(np.sign(endpoint_correction[active]) == np.sign(endpoint_required[active]))) if active.any() else float("nan"),
                }
            )
        completed += 1
    if completed != 29:
        raise RuntimeError(f"Expected 29 folds, observed {completed}")
    seed = pd.DataFrame(rows)
    fold = seed.groupby(["target", "fold", "method"], as_index=False).mean(numeric_only=True)
    target = fold.groupby(["target", "method"], as_index=False).mean(numeric_only=True)
    base = target[target["method"] == "base"].set_index("target")
    comparisons = []
    for method in sorted(set(target["method"]) - {"base"}):
        table = target[target["method"] == method].set_index("target")
        for name in table.index:
            comparisons.append(
                {
                    "target": name,
                    "method": method,
                    "overall_relative_change_vs_base": float(table.loc[name, "overall_mae"] / base.loc[name, "overall_mae"] - 1),
                    "cliff_relative_change_vs_base": float(table.loc[name, "cliff_endpoint_mae"] / base.loc[name, "cliff_endpoint_mae"] - 1),
                    "pair_delta_relative_change_vs_base": float(table.loc[name, "pair_delta_mae"] / base.loc[name, "pair_delta_mae"] - 1),
                    "endpoint_correction_residual_correlation": float(table.loc[name, "endpoint_correction_residual_correlation"]),
                    "endpoint_correction_sign_accuracy": float(table.loc[name, "endpoint_correction_sign_accuracy"]),
                }
            )
    comparison = pd.DataFrame(comparisons)
    primary_name = config["methods"]["primary"]
    primary = comparison[comparison["method"] == primary_name].set_index("target")
    label_knn = target[target["method"] == "weighted_train_label_knn"].set_index("target")
    permuted = target[target["method"] == "weighted_permuted_train_residual_anchor"].set_index("target")
    primary_metric = target[target["method"] == primary_name].set_index("target")
    decision = config["decision"]
    macro = {
        "folds": completed,
        "targets": int(len(primary)),
        "primary": primary_name,
        "overall_relative_change_vs_base": float(primary["overall_relative_change_vs_base"].mean()),
        "cliff_relative_change_vs_base": float(primary["cliff_relative_change_vs_base"].mean()),
        "pair_delta_relative_change_vs_base": float(primary["pair_delta_relative_change_vs_base"].mean()),
        "endpoint_correction_residual_correlation": float(primary["endpoint_correction_residual_correlation"].mean()),
        "endpoint_correction_sign_accuracy": float(primary["endpoint_correction_sign_accuracy"].mean()),
        "improved_targets": int((primary["cliff_relative_change_vs_base"] < 0).sum()),
        "primary_minus_label_knn_cliff_mae": float((primary_metric["cliff_endpoint_mae"] - label_knn["cliff_endpoint_mae"]).mean()),
        "primary_minus_permuted_cliff_mae": float((primary_metric["cliff_endpoint_mae"] - permuted["cliff_endpoint_mae"]).mean()),
    }
    criteria = {
        "cliff_improvement": macro["cliff_relative_change_vs_base"] <= -float(decision["minimum_primary_cliff_improvement_vs_base"]),
        "overall_constraint": macro["overall_relative_change_vs_base"] <= float(decision["maximum_primary_overall_degradation_vs_base"]),
        "correlation": macro["endpoint_correction_residual_correlation"] >= float(decision["minimum_endpoint_correction_residual_correlation"]),
        "sign_accuracy": macro["endpoint_correction_sign_accuracy"] >= float(decision["minimum_endpoint_correction_sign_accuracy"]),
        "beats_label_knn": macro["primary_minus_label_knn_cliff_mae"] < 0,
        "beats_permuted": macro["primary_minus_permuted_cliff_mae"] < 0,
        "target_directions": macro["improved_targets"] >= int(decision["minimum_improved_targets"]),
    }
    macro["criteria"], macro["go"] = criteria, bool(all(criteria.values()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed.to_csv(args.output_dir / "anchor_seed_metrics.csv", index=False)
    fold.to_csv(args.output_dir / "anchor_fold_metrics.csv", index=False)
    target.to_csv(args.output_dir / "anchor_target_metrics.csv", index=False)
    comparison.to_csv(args.output_dir / "anchor_comparison.csv", index=False)
    (args.output_dir / "anchor_macro.json").write_text(json.dumps(macro, indent=2), encoding="utf-8")
    print(json.dumps(macro, indent=2))


if __name__ == "__main__":
    main()

