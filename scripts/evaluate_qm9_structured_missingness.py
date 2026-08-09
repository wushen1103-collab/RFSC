#!/usr/bin/env python3
"""Evaluate frozen structured-missingness Hodge potentials after preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_frozen_prediction

try:
    from build_qm9_direct_pair_predictions import selected_column
except ModuleNotFoundError:
    from scripts.build_qm9_direct_pair_predictions import selected_column


def ordered(path: Path, ids: list[str]) -> np.ndarray:
    series = read_frozen_prediction(path)
    if set(series.index) != set(ids): raise RuntimeError(f"Artifact ID mismatch: {path}")
    return series.reindex(ids).to_numpy(dtype=float)


def bootstrap_relative(candidate: np.ndarray, reference: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed); result = np.empty(iterations)
    for i in range(iterations):
        chosen = rng.integers(0, len(candidate), len(candidate))
        result[i] = candidate[chosen].mean() / reference[chosen].mean() - 1.0
    return tuple(float(x) for x in np.quantile(result, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--flow-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    expected = len(config["tasks"]) * int(config["folds"]) * len(config["schemes"]) * len(config["densities"])
    run_paths = sorted(args.prediction_root.glob("*/fold_*/*/density_*/run.json"))
    if len(run_paths) != expected: raise RuntimeError(f"Incomplete structured matrix: {len(run_paths)} versus {expected}")
    frozen = []
    for run_path in run_paths:
        run = json.loads(run_path.read_text(encoding="utf-8")); n = int(run["n_nodes"])
        node_ids = [f"position_{i:04d}" for i in range(n)]
        edge_ids = [f"edge_{i:04d}_{j:04d}" for i in range(n) for j in range(n)]
        potential = ordered(run_path.parent / "potential.parquet", node_ids)
        observed = ordered(run_path.parent / "observed_mask.parquet", edge_ids).reshape(n, n).astype(bool)
        evaluable = ordered(run_path.parent / "evaluable_mask.parquet", edge_ids).reshape(n, n).astype(bool)
        flow_run = json.loads((args.flow_root / run["task"] / f"fold_{run['fold']}" / "run.json").read_text(encoding="utf-8"))
        all_n = int(flow_run["test_nodes"])
        all_edge_ids = [f"edge_{i:04d}_{j:04d}" for i in range(all_n) for j in range(all_n)]
        raw = ordered(args.flow_root / run["task"] / f"fold_{run['fold']}" / "raw_flow.parquet", all_edge_ids).reshape(all_n, all_n)[:n, :n]
        if flow_run["test_indices"][:n] != run["test_indices"]: raise RuntimeError("Structured cohort changed")
        frozen.append((run, potential, observed, evaluable, raw))

    rows = []
    for run, potential, observed, evaluable, raw in frozen:
        indices = np.asarray(run["test_indices"], dtype=int)
        truth_nodes = selected_column(args.source, run["task"], indices)
        truth = truth_nodes[None, :] - truth_nodes[:, None]
        projected = potential[None, :] - potential[:, None]
        held = evaluable & ~observed & ~np.eye(len(truth), dtype=bool)
        if not held.any(): raise RuntimeError("No identifiable held edges")
        raw_error, projected_error = raw[held] - truth[held], projected[held] - truth[held]
        rows.append({
            "task": run["task"], "fold": int(run["fold"]), "scheme": run["scheme"], "density": float(run["density"]),
            "n_nodes": len(indices), "observed_directed_edges": int(observed.sum()), "held_directed_edges": int(held.sum()),
            "held_edge_coverage": float(held.sum() / (len(indices) * (len(indices) - 1))),
            "raw_held_rmse": float(np.sqrt(np.mean(np.square(raw_error)))), "hodge_held_rmse": float(np.sqrt(np.mean(np.square(projected_error)))),
            "rmse_relative_change": float(np.sqrt(np.mean(np.square(projected_error))) / np.sqrt(np.mean(np.square(raw_error))) - 1.0),
            "raw_held_mae": float(np.mean(np.abs(raw_error))), "hodge_held_mae": float(np.mean(np.abs(projected_error))),
            "mae_relative_change": float(np.mean(np.abs(projected_error)) / np.mean(np.abs(raw_error)) - 1.0),
            "observed_residual_energy": float(run["observed_residual_energy"]),
        })
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["scheme", "density"], as_index=False).agg({column: ["mean", "std"] for column in ["held_edge_coverage", "raw_held_rmse", "hodge_held_rmse", "rmse_relative_change", "raw_held_mae", "hodge_held_mae", "mae_relative_change"]})
    contrasts = []
    for (scheme, density), group in frame.groupby(["scheme", "density"]):
        for metric_index, (candidate_name, reference_name, label) in enumerate([("hodge_held_rmse", "raw_held_rmse", "rmse"), ("hodge_held_mae", "raw_held_mae", "mae")]):
            lower, upper = bootstrap_relative(group[candidate_name].to_numpy(), group[reference_name].to_numpy(), int(config["evaluation"]["bootstrap_iterations"]), int(config["evaluation"]["bootstrap_seed"]) + metric_index)
            by_task = group.groupby("task")[[candidate_name, reference_name]].mean()
            contrasts.append({"scheme": scheme, "density": density, "metric": label, "task_folds": len(group), "relative_change": group[candidate_name].mean() / group[reference_name].mean() - 1.0, "bootstrap95_lower": lower, "bootstrap95_upper": upper, "favorable_task_folds": int((group[candidate_name] < group[reference_name]).sum()), "favorable_tasks": int((by_task[candidate_name] < by_task[reference_name]).sum())})
    contrast = pd.DataFrame(contrasts)
    primary = config["primary"]
    selected = contrast[(contrast["scheme"] == primary["scheme"]) & np.isclose(contrast["density"], float(primary["density"])) & (contrast["metric"] == "rmse")].iloc[0]
    decision = {"primary": primary, "primary_rmse_relative_change": float(selected["relative_change"]), "primary_rmse_bootstrap95_upper": float(selected["bootstrap95_upper"]), "primary_favorable_tasks": int(selected["favorable_tasks"]), "responses_opened_after_complete_hash_preflight": True}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "task_fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "mean_std.csv", index=False)
    contrast.to_csv(args.output_dir / "paired_contrasts.csv", index=False)
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
