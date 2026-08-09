#!/usr/bin/env python3
"""Evaluate the once-frozen AqSolDB external SCG confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_frozen_prediction, read_labels_for_ids


METHODS = ["svm", "rf", "gine", "unscaled_reference", "bps"]
METRICS = ["overall_mae", "cliff_endpoint_mae", "pair_delta_mae"]


def paired_bootstrap(candidate: np.ndarray, reference: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(iterations)
    for index in range(iterations):
        chosen = rng.integers(0, len(candidate), len(candidate))
        values[index] = candidate[chosen].mean() / reference[chosen].mean() - 1.0
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def metrics(prediction: pd.Series, truth: pd.Series, edges: pd.DataFrame, cliff_threshold: float) -> dict[str, float | int]:
    edges = edges.copy()
    edges["truth_delta"] = edges["mol_id_hi"].map(truth) - edges["mol_id_lo"].map(truth)
    cliffs = edges[edges["truth_delta"].abs() >= cliff_threshold]
    endpoints = set(cliffs["mol_id_lo"]) | set(cliffs["mol_id_hi"])
    predicted_delta = prediction.reindex(cliffs["mol_id_hi"]).to_numpy() - prediction.reindex(cliffs["mol_id_lo"]).to_numpy()
    return {
        "overall_mae": float(np.mean(np.abs(prediction - truth.reindex(prediction.index)))),
        "cliff_endpoint_mae": float(np.mean(np.abs(prediction.reindex(sorted(endpoints)) - truth.reindex(sorted(endpoints))))) if endpoints else float("nan"),
        "pair_delta_mae": float(np.mean(np.abs(predicted_delta - cliffs["truth_delta"].to_numpy()))) if len(cliffs) else float("nan"),
        "true_cliff_pairs": int(len(cliffs)), "true_cliff_endpoints": int(len(endpoints)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    target = str(config["target"])
    data_root, mmp_root = Path(config["data_root"]), Path(config["mmp_root"])
    all_edges = pd.read_parquet(mmp_root / target / "eligible_mmp_edges.parquet")
    rows = []
    run_paths = sorted(args.prediction_dir.glob(f"{target}/fold_*/seed_*/run.json"))
    for run_path in run_paths:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["test_labels_materialized"] is not False:
            raise RuntimeError("Invalid response firewall metadata")
        predictions = {method: read_frozen_prediction(run_path.parent / method / "predictions.parquet") for method in METHODS}
        ids = predictions["bps"].index
        if any(set(value.index) != set(ids) for value in predictions.values()):
            raise RuntimeError("Method partition mismatch")
        truth = read_labels_for_ids(data_root, target, ids).set_index("mol_id")["y_pactivity"]
        edges = all_edges[all_edges["mol_id_lo"].isin(ids) & all_edges["mol_id_hi"].isin(ids)]
        for method, prediction in predictions.items():
            rows.append({"fold": int(run["fold"]), "seed": int(run["seed"]), "method": method, **metrics(prediction, truth, edges, float(config["evaluation"]["cliff_threshold"]))})
    expected = len(config["seeds"]) * pd.read_parquet(Path(config["fold_root"]) / target / "outer_folds.parquet")["outer_fold"].nunique()
    if len(run_paths) != expected:
        raise RuntimeError(f"Incomplete frozen matrix: {len(run_paths)} versus {expected}")
    seed_metrics = pd.DataFrame(rows)
    fold_metrics = seed_metrics.groupby(["fold", "method"], as_index=False).mean(numeric_only=True)
    summary = fold_metrics.groupby("method")[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = ["method" if column[0] == "method" else f"{column[0]}_{column[1]}" for column in summary.columns]
    candidate = fold_metrics[fold_metrics["method"] == "bps"].set_index("fold")
    reference = fold_metrics[fold_metrics["method"] == "unscaled_reference"].set_index("fold")
    comparisons = []
    for metric_index, metric in enumerate(METRICS):
        paired = pd.concat([candidate[metric].rename("candidate"), reference[metric].rename("reference")], axis=1).dropna()
        lower, upper = paired_bootstrap(paired["candidate"].to_numpy(), paired["reference"].to_numpy(), int(config["evaluation"]["bootstrap_iterations"]), int(config["evaluation"]["bootstrap_seed"]) + metric_index)
        comparisons.append({"metric": metric, "folds": len(paired), "reference_mean": paired["reference"].mean(), "candidate_mean": paired["candidate"].mean(), "relative_change": paired["candidate"].mean() / paired["reference"].mean() - 1.0, "bootstrap95_lower": lower, "bootstrap95_upper": upper, "favorable_folds": int((paired["candidate"] < paired["reference"]).sum())})
    contrast = pd.DataFrame(comparisons).set_index("metric")
    decision_config = config["decision"]
    checks = {
        "pair_point_improves": float(contrast.loc["pair_delta_mae", "relative_change"]) < 0.0,
        "pair_fold_directions": int(contrast.loc["pair_delta_mae", "favorable_folds"]) / int(contrast.loc["pair_delta_mae", "folds"]) >= float(decision_config["minimum_pair_improved_fold_fraction"]),
        "pair_bootstrap_upper": float(contrast.loc["pair_delta_mae", "bootstrap95_upper"]) < float(decision_config["maximum_pair_bootstrap_upper"]),
        "overall_cost": float(contrast.loc["overall_mae", "relative_change"]) <= float(decision_config["maximum_overall_relative_cost"]),
        "cliff_cost": float(contrast.loc["cliff_endpoint_mae", "relative_change"]) <= float(decision_config["maximum_cliff_relative_cost"]),
    }
    decision = {"checks": checks, "strong_external_confirmation": bool(all(checks.values())), "no_external_tuning": True}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(args.output_dir / "seed_fold_metrics.csv", index=False)
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "mean_std.csv", index=False)
    contrast.reset_index().to_csv(args.output_dir / "bps_vs_reference.csv", index=False)
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"decision": decision, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
