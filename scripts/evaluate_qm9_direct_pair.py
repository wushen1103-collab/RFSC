#!/usr/bin/env python3
"""Evaluate frozen QM9 flow repairs and measured-anchor scalar recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_frozen_prediction, sha256_file

try:
    from build_qm9_direct_pair_predictions import selected_column
except ModuleNotFoundError:
    from scripts.build_qm9_direct_pair_predictions import selected_column


FLOW_METHODS = ["raw_flow", "diagonal_repaired_flow", "antisymmetric_flow", "hodge_flow", "absolute_difference"]
LOWER_METRICS = ["pair_rmse", "pair_mae", "high_delta_pair_mae"]


def ordered(series: pd.Series, ids: list[str]) -> np.ndarray:
    if set(series.index) != set(ids):
        raise RuntimeError("Frozen artifact IDs do not match the expected cohort")
    return series.reindex(ids).to_numpy(dtype=float)


def flow_metrics(prediction: np.ndarray, truth: np.ndarray, high_mask: np.ndarray) -> dict[str, float]:
    off = ~np.eye(len(truth), dtype=bool)
    error = prediction[off] - truth[off]
    return {
        "pair_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "pair_mae": float(np.mean(np.abs(error))),
        "high_delta_pair_mae": float(np.mean(np.abs(prediction[high_mask] - truth[high_mask]))),
        "pearson_r": float(np.corrcoef(prediction[off], truth[off])[0, 1]),
        "self_pair_mae": float(np.mean(np.abs(np.diag(prediction)))),
        "symmetric_energy_fraction": float(np.square(0.5 * (prediction + prediction.T)).sum() / max(np.square(prediction).sum(), 1e-15)),
    }


def bootstrap_relative(candidate: np.ndarray, reference: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    result = np.empty(iterations)
    for index in range(iterations):
        chosen = rng.integers(0, len(candidate), len(candidate))
        result[index] = candidate[chosen].mean() / reference[chosen].mean() - 1.0
    return tuple(float(x) for x in np.quantile(result, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if sha256_file(args.source) != config["source"]["expected_sha256"]:
        raise RuntimeError("QM9 source changed")

    frozen = {}
    expected_runs = len(config["tasks"]) * int(config["split"]["folds"])
    run_paths = sorted(args.prediction_root.glob("*/fold_*/run.json"))
    if len(run_paths) != expected_runs:
        raise RuntimeError(f"Incomplete QM9 matrix: {len(run_paths)} versus {expected_runs}")
    # Complete hash/partition preflight before selecting any held-out response.
    for run_path in run_paths:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["held_out_response_materialized"] is not False or run["smoke"]:
            raise RuntimeError("Invalid full-run firewall metadata")
        n = int(run["test_nodes"])
        edge_ids = [f"edge_{i:04d}_{j:04d}" for i in range(n) for j in range(n)]
        node_ids = [f"position_{i:04d}" for i in range(n)]
        raw = ordered(read_frozen_prediction(run_path.parent / "raw_flow.parquet"), edge_ids).reshape(n, n)
        diagonal = ordered(read_frozen_prediction(run_path.parent / "diagonal_repaired_flow.parquet"), edge_ids).reshape(n, n)
        anti = ordered(read_frozen_prediction(run_path.parent / "antisymmetric_flow.parquet"), edge_ids).reshape(n, n)
        potential = ordered(read_frozen_prediction(run_path.parent / "hodge_potential.parquet"), node_ids)
        absolute = ordered(read_frozen_prediction(run_path.parent / "absolute_potential.parquet"), node_ids)
        frozen[(run["task"], int(run["fold"]))] = (run, raw, diagonal, anti, potential, absolute)

    flow_rows, anchor_rows, energy_rows = [], [], []
    quantile = float(config["evaluation"]["high_delta_quantile"])
    for (task, fold), (run, raw, diagonal, anti, potential, absolute) in frozen.items():
        test_indices = np.asarray(run["test_indices"], dtype=int)
        truth_nodes = selected_column(args.source, task, test_indices)
        truth = truth_nodes[None, :] - truth_nodes[:, None]
        hodge = potential[None, :] - potential[:, None]
        absolute_flow = absolute[None, :] - absolute[:, None]
        off = ~np.eye(len(truth), dtype=bool)
        cutoff = float(np.quantile(np.abs(truth[off]), quantile))
        high = off & (np.abs(truth) >= cutoff)
        methods = {
            "raw_flow": raw, "diagonal_repaired_flow": diagonal,
            "antisymmetric_flow": anti, "hodge_flow": hodge,
            "absolute_difference": absolute_flow,
        }
        for method, prediction in methods.items():
            flow_rows.append({"task": task, "fold": fold, "method": method, "n_nodes": len(truth_nodes), "high_delta_cutoff": cutoff, **flow_metrics(prediction, truth, high)})
        anti_residual = anti - hodge
        energy_rows.append({
            "task": task, "fold": fold,
            "antisymmetric_energy": float(np.square(anti).sum()),
            "cyclic_residual_energy": float(np.square(anti_residual).sum()),
            "cyclic_energy_fraction": float(np.square(anti_residual).sum() / max(np.square(anti).sum(), 1e-15)),
            "hodge_orthogonality_inner_product": float(np.sum(hodge * anti_residual)),
        })
        anchor_positions = list(map(int, run["anchor_positions"]))
        for count in map(int, config["evaluation"]["anchors"]):
            anchors = anchor_positions[:count]
            offset = float(np.mean(truth_nodes[anchors] - potential[anchors]))
            prediction = potential + offset
            mask = np.ones(len(truth_nodes), dtype=bool)
            mask[anchors] = False
            anchor_rows.append({"task": task, "fold": fold, "method": "hodge_measured_anchor", "anchors": count, "evaluation_nodes": int(mask.sum()), "node_mae": float(np.mean(np.abs(prediction[mask] - truth_nodes[mask])))})
            anchor_rows.append({"task": task, "fold": fold, "method": "absolute_lgbm", "anchors": count, "evaluation_nodes": int(mask.sum()), "node_mae": float(np.mean(np.abs(absolute[mask] - truth_nodes[mask])))})

    flow = pd.DataFrame(flow_rows)
    anchors = pd.DataFrame(anchor_rows)
    energy = pd.DataFrame(energy_rows)
    fold_summary = flow.groupby(["task", "method"], as_index=False).agg({metric: ["mean", "std"] for metric in LOWER_METRICS + ["pearson_r", "self_pair_mae", "symmetric_energy_fraction"]})
    contrasts = []
    for reference_name in ["raw_flow", "diagonal_repaired_flow", "antisymmetric_flow", "absolute_difference"]:
        candidate = flow[flow["method"] == "hodge_flow"].set_index(["task", "fold"])
        reference = flow[flow["method"] == reference_name].set_index(["task", "fold"])
        for metric_index, metric in enumerate(LOWER_METRICS):
            paired = pd.concat([candidate[metric].rename("candidate"), reference[metric].rename("reference")], axis=1).dropna()
            lower, upper = bootstrap_relative(paired["candidate"].to_numpy(), paired["reference"].to_numpy(), int(config["evaluation"]["bootstrap_iterations"]), int(config["evaluation"]["bootstrap_seed"]) + metric_index + 100 * len(contrasts))
            by_task = paired.reset_index().groupby("task")[["candidate", "reference"]].mean()
            contrasts.append({"candidate": "hodge_flow", "reference": reference_name, "metric": metric, "task_folds": len(paired), "relative_change": paired["candidate"].mean() / paired["reference"].mean() - 1.0, "bootstrap95_lower": lower, "bootstrap95_upper": upper, "favorable_task_folds": int((paired["candidate"] < paired["reference"]).sum()), "favorable_tasks": int((by_task["candidate"] < by_task["reference"]).sum())})
    contrast = pd.DataFrame(contrasts)
    primary = contrast[(contrast["reference"] == "antisymmetric_flow") & (contrast["metric"] == "pair_mae")].iloc[0]
    repaired_self = flow[flow["method"] == "diagonal_repaired_flow"]["self_pair_mae"]
    checks = {
        "hodge_vs_antisymmetric_task_directions": int(primary["favorable_tasks"]) / len(config["tasks"]) >= float(config["decision"]["minimum_hodge_vs_antisymmetric_improved_task_fraction"]),
        "hodge_vs_antisymmetric_bootstrap_upper": float(primary["bootstrap95_upper"]) < float(config["decision"]["maximum_hodge_vs_antisymmetric_pair_mae_bootstrap_upper"]),
        "diagonal_repair_zero": float(repaired_self.max()) <= float(config["decision"]["diagonal_zero_tolerance"]),
    }
    decision = {"checks": checks, "strong_mechanism_support": bool(all(checks.values())), "held_out_response_opened_after_complete_hash_preflight": True}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    flow.to_csv(args.output_dir / "task_fold_flow_metrics.csv", index=False)
    fold_summary.to_csv(args.output_dir / "fivefold_mean_std.csv", index=False)
    contrast.to_csv(args.output_dir / "paired_contrasts.csv", index=False)
    anchors.to_csv(args.output_dir / "measured_anchor_recovery.csv", index=False)
    energy.to_csv(args.output_dir / "cyclic_energy.csv", index=False)
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"decision": decision, "primary": primary.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
