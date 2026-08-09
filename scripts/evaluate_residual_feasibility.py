#!/usr/bin/env python3
"""Evaluate frozen residual predictions using validation labels only."""

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
    metric_rows = []
    completed = 0
    for run_path in sorted(args.run_dir.glob("*/fold_*/run.json")):
        metadata = json.loads(run_path.read_text(encoding="utf-8"))
        if metadata["validation_labels_materialized"] or metadata["outer_test_predictions_generated"] or metadata["outer_test_labels_materialized"]:
            raise RuntimeError(f"Query/test firewall violation: {run_path}")
        prediction_path = run_path.parent / "validation_predictions.parquet"
        manifest = json.loads(
            prediction_path.with_suffix(prediction_path.suffix + ".sha256.json").read_text(encoding="utf-8")
        )
        if sha256_file(prediction_path) != manifest["prediction_sha256"]:
            raise RuntimeError(f"Frozen prediction hash mismatch: {prediction_path}")
        predictions = pd.read_parquet(prediction_path)
        target, fold = metadata["target"], int(metadata["fold"])
        query_ids = predictions["mol_id"].unique().tolist()
        labels = read_labels_for_ids(args.data_root, target, query_ids).set_index("mol_id")["y_pactivity"]
        mmp = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        internal = mmp[mmp["mol_id_lo"].isin(query_ids) & mmp["mol_id_hi"].isin(query_ids)].copy()
        internal["delta_y"] = internal["mol_id_hi"].map(labels) - internal["mol_id_lo"].map(labels)
        cliff_pairs = internal[internal["delta_y"].abs() >= 1.0]
        endpoints = set(cliff_pairs["mol_id_lo"]) | set(cliff_pairs["mol_id_hi"])
        if not endpoints:
            raise RuntimeError(f"No validation cliff endpoint: {target} fold {fold}")
        for (seed, method, eta), frame in predictions.groupby(["seed", "method", "eta"]):
            series = frame.set_index("mol_id")["prediction"]
            correction = frame.set_index("mol_id")["correction_value"]
            if set(series.index) != set(query_ids):
                raise RuntimeError(f"Incomplete prediction: {target} fold {fold}")
            truth = labels.loc[series.index]
            endpoint_ids = [value for value in series.index if value in endpoints]
            endpoint_correction = correction.loc[endpoint_ids].to_numpy()
            endpoint_required = (truth - (series - correction)).loc[endpoint_ids].to_numpy()
            active = np.abs(endpoint_correction) > 1e-12
            pair_prediction = series.reindex(cliff_pairs["mol_id_hi"]).to_numpy() - series.reindex(cliff_pairs["mol_id_lo"]).to_numpy()
            metric_rows.append(
                {
                    "target": target,
                    "fold": fold,
                    "seed": int(seed),
                    "method": method,
                    "eta": float(eta),
                    "overall_mae": float(np.mean(np.abs(series - truth))),
                    "cliff_endpoint_mae": float(np.mean(np.abs(series.loc[endpoint_ids] - truth.loc[endpoint_ids]))),
                    "pair_delta_mae": float(np.mean(np.abs(pair_prediction - cliff_pairs["delta_y"].to_numpy()))),
                    "endpoint_correction_residual_correlation": safe_corr(endpoint_correction, endpoint_required),
                    "endpoint_correction_sign_accuracy": float(np.mean(np.sign(endpoint_correction[active]) == np.sign(endpoint_required[active]))) if active.any() else float("nan"),
                    "endpoint_count": len(endpoint_ids),
                    "cliff_pair_count": len(cliff_pairs),
                }
            )
        completed += 1
    if completed != 29:
        raise RuntimeError(f"Expected 29 folds, observed {completed}")

    seed_metrics = pd.DataFrame(metric_rows)
    fold_metrics = seed_metrics.groupby(["target", "fold", "method", "eta"], as_index=False).mean(numeric_only=True)
    target_metrics = fold_metrics.groupby(["target", "method", "eta"], as_index=False).mean(numeric_only=True)
    primary_eta = float(config["models"]["primary_eta"])
    primary = target_metrics[(target_metrics["eta"] == primary_eta) | (target_metrics["method"] == "base")].copy()
    base = primary[primary["method"] == "base"].set_index("target")
    comparisons = []
    for method in ["node_residual", "edge_residual", "edge_permuted"]:
        method_table = primary[primary["method"] == method].set_index("target")
        for target in method_table.index:
            comparisons.append(
                {
                    "target": target,
                    "method": method,
                    "eta": primary_eta,
                    "overall_relative_change_vs_base": float(method_table.loc[target, "overall_mae"] / base.loc[target, "overall_mae"] - 1.0),
                    "cliff_relative_change_vs_base": float(method_table.loc[target, "cliff_endpoint_mae"] / base.loc[target, "cliff_endpoint_mae"] - 1.0),
                    "pair_delta_relative_change_vs_base": float(method_table.loc[target, "pair_delta_mae"] / base.loc[target, "pair_delta_mae"] - 1.0),
                    "endpoint_correction_residual_correlation": float(method_table.loc[target, "endpoint_correction_residual_correlation"]),
                    "endpoint_correction_sign_accuracy": float(method_table.loc[target, "endpoint_correction_sign_accuracy"]),
                }
            )
    comparison = pd.DataFrame(comparisons)
    edge = comparison[comparison["method"] == "edge_residual"].set_index("target")
    node = comparison[comparison["method"] == "node_residual"].set_index("target")
    permuted = comparison[comparison["method"] == "edge_permuted"].set_index("target")
    edge_target = primary[primary["method"] == "edge_residual"].set_index("target")
    node_target = primary[primary["method"] == "node_residual"].set_index("target")
    permuted_target = primary[primary["method"] == "edge_permuted"].set_index("target")
    decision = config["decision"]
    macro = {
        "folds": completed,
        "targets": int(len(edge)),
        "primary_eta": primary_eta,
        "edge_overall_relative_change_vs_base": float(edge["overall_relative_change_vs_base"].mean()),
        "edge_cliff_relative_change_vs_base": float(edge["cliff_relative_change_vs_base"].mean()),
        "edge_pair_delta_relative_change_vs_base": float(edge["pair_delta_relative_change_vs_base"].mean()),
        "edge_endpoint_correction_residual_correlation": float(edge["endpoint_correction_residual_correlation"].mean()),
        "edge_endpoint_correction_sign_accuracy": float(edge["endpoint_correction_sign_accuracy"].mean()),
        "edge_improved_targets": int((edge["cliff_relative_change_vs_base"] < 0).sum()),
        "node_cliff_relative_change_vs_base": float(node["cliff_relative_change_vs_base"].mean()),
        "permuted_cliff_relative_change_vs_base": float(permuted["cliff_relative_change_vs_base"].mean()),
        "edge_minus_node_cliff_mae": float((edge_target["cliff_endpoint_mae"] - node_target["cliff_endpoint_mae"]).mean()),
        "edge_minus_permuted_cliff_mae": float((edge_target["cliff_endpoint_mae"] - permuted_target["cliff_endpoint_mae"]).mean()),
    }
    criteria = {
        "cliff_improvement": macro["edge_cliff_relative_change_vs_base"] <= -float(decision["minimum_edge_cliff_improvement_vs_base"]),
        "overall_constraint": macro["edge_overall_relative_change_vs_base"] <= float(decision["maximum_edge_overall_degradation_vs_base"]),
        "correlation": macro["edge_endpoint_correction_residual_correlation"] >= float(decision["minimum_endpoint_correction_residual_correlation"]),
        "sign_accuracy": macro["edge_endpoint_correction_sign_accuracy"] >= float(decision["minimum_endpoint_correction_sign_accuracy"]),
        "beats_node": macro["edge_minus_node_cliff_mae"] < 0,
        "beats_permuted": macro["edge_minus_permuted_cliff_mae"] < 0,
        "target_directions": macro["edge_improved_targets"] >= int(decision["minimum_improved_targets"]),
    }
    macro["criteria"] = criteria
    macro["go"] = bool(all(criteria.values()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(args.output_dir / "residual_seed_metrics.csv", index=False)
    fold_metrics.to_csv(args.output_dir / "residual_fold_metrics.csv", index=False)
    target_metrics.to_csv(args.output_dir / "residual_target_metrics.csv", index=False)
    comparison.to_csv(args.output_dir / "residual_primary_comparison.csv", index=False)
    (args.output_dir / "residual_macro.json").write_text(json.dumps(macro, indent=2), encoding="utf-8")
    print(json.dumps(macro, indent=2))


if __name__ == "__main__":
    main()
