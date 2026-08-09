#!/usr/bin/env python3
"""Run the pre-registered synthetic boundary-protection mechanism gate."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors


def build_problem(seed: int, n: int, k: int, jump: float, noise: float) -> dict:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 1.0, size=(n, 2))
    neighbors = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(coords)
    distances, indices = neighbors.kneighbors(coords)
    edge_set: set[tuple[int, int]] = set()
    for node in range(n):
        for other in indices[node, 1:]:
            edge_set.add((min(node, int(other)), max(node, int(other))))
    edges = np.asarray(sorted(edge_set), dtype=np.int64)
    edge_dist = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
    scale = np.median(edge_dist)
    weights = np.exp(-((edge_dist / max(scale, 1e-12)) ** 2))
    regions = coords[:, 0] >= 0.5
    smooth = np.sin(2 * np.pi * coords[:, 0]) + 0.5 * np.cos(2 * np.pi * coords[:, 1])
    truth = smooth + jump * regions.astype(float)
    observed = truth + rng.normal(0.0, noise, size=n)
    true_boundary = (regions[edges[:, 0]] != regions[edges[:, 1]]) if jump > 0 else np.zeros(len(edges), dtype=bool)
    return {
        "rng": rng,
        "coords": coords,
        "edges": edges,
        "edge_dist": edge_dist,
        "weights": weights,
        "truth": truth,
        "observed": observed,
        "true_boundary": true_boundary,
    }


def gate_scores(problem: dict, method: str) -> tuple[np.ndarray, np.ndarray]:
    boundary = problem["true_boundary"]
    n_boundary = int(boundary.sum())
    n_edges = len(boundary)
    rng = problem["rng"]
    score = np.zeros(n_edges, dtype=float)
    if method == "global" or method == "observation":
        cut = np.zeros(n_edges, dtype=bool)
    elif method == "oracle":
        score = boundary.astype(float)
        cut = boundary.copy()
    elif method == "random":
        score = rng.random(n_edges)
        chosen = np.argsort(score, kind="stable")[-n_boundary:] if n_boundary else []
        cut = np.zeros(n_edges, dtype=bool)
        cut[chosen] = True
    elif method == "edge_length":
        score = problem["edge_dist"].copy()
        chosen = np.argsort(score, kind="stable")[-n_boundary:] if n_boundary else []
        cut = np.zeros(n_edges, dtype=bool)
        cut[chosen] = True
    elif method == "anti_oracle":
        score = (~boundary).astype(float)
        smooth_indices = np.flatnonzero(~boundary)
        chosen = rng.choice(smooth_indices, size=min(n_boundary, len(smooth_indices)), replace=False) if n_boundary else []
        cut = np.zeros(n_edges, dtype=bool)
        cut[chosen] = True
    else:
        raise ValueError(method)
    return cut, score


def smooth_signal(problem: dict, cut: np.ndarray, lam: float) -> np.ndarray:
    edges, weights = problem["edges"], problem["weights"] * (~cut)
    n = len(problem["truth"])
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([weights, weights])
    adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    laplacian = sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    system = sparse.eye(n, format="csr") + lam * laplacian
    return np.asarray(spsolve(system, problem["observed"]), dtype=float)


def measure(problem: dict, prediction: np.ndarray, method: str, score: np.ndarray) -> dict:
    truth, edges, boundary = problem["truth"], problem["edges"], problem["true_boundary"]
    error2 = (prediction - truth) ** 2
    boundary_nodes = np.unique(edges[boundary].ravel()) if boundary.any() else np.asarray([], dtype=int)
    smooth_nodes = np.setdiff1d(np.arange(len(truth)), boundary_nodes, assume_unique=False)
    if boundary.any():
        predicted_jump = prediction[edges[boundary, 1]] - prediction[edges[boundary, 0]]
        true_jump = truth[edges[boundary, 1]] - truth[edges[boundary, 0]]
        jump_mae = float(np.mean(np.abs(predicted_jump - true_jump)))
        gate_auprc = float(average_precision_score(boundary.astype(int), score))
    else:
        jump_mae = np.nan
        gate_auprc = np.nan
    return {
        "method": method,
        "overall_mse": float(error2.mean()),
        "boundary_mse": float(error2[boundary_nodes].mean()) if len(boundary_nodes) else float(error2.mean()),
        "smooth_mse": float(error2[smooth_nodes].mean()) if len(smooth_nodes) else np.nan,
        "jump_delta_mae": jump_mae,
        "gate_auprc": gate_auprc,
        "boundary_edge_prevalence": float(boundary.mean()),
        "n_boundary_edges": int(boundary.sum()),
        "n_boundary_nodes": int(len(boundary_nodes)),
    }


def run_case(seed: int, cfg: dict, jump: float, noise: float, methods: list[str], lambdas: list[float]) -> list[dict]:
    problem = build_problem(seed, cfg["n_nodes"], cfg["k"], jump, noise)
    rows = []
    for method in methods:
        cut, score = gate_scores(problem, method)
        method_lambdas = [0.0] if method == "observation" else lambdas
        for lam in method_lambdas:
            prediction = problem["observed"] if method == "observation" else smooth_signal(problem, cut, lam)
            rows.append({"seed": seed, "jump": jump, "noise_std": noise, "lambda": lam, **measure(problem, prediction, method, score)})
    return rows


def execute_cases(cases, cfg, methods, lambdas, workers):
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_case, seed, cfg, jump, noise, methods, lambdas): (seed, jump, noise) for seed, jump, noise in cases}
        for future in as_completed(futures):
            rows.extend(future.result())
            seed, jump, noise = futures[future]
            print(f"done seed={seed} jump={jump} noise={noise}", flush=True)
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    graph = config["graph"]
    jumps, noises = config["signal"]["jump"], config["signal"]["noise_std"]
    methods, lambdas = config["methods"], config["smoother"]["lambda"]
    validation_cases = [(seed, jump, noise) for seed in config["seed"]["validation"] for jump in jumps for noise in noises]
    validation = execute_cases(validation_cases, graph, methods, lambdas, args.workers)
    selected = (
        validation.groupby(["jump", "noise_std", "method", "lambda"], dropna=False)["overall_mse"].mean().reset_index().sort_values(["jump", "noise_std", "method", "overall_mse", "lambda"]).drop_duplicates(["jump", "noise_std", "method"])
    )
    lambda_map = {
        (row["jump"], row["noise_std"], row["method"]): row["lambda"]
        for _, row in selected.iterrows()
    }

    evaluation_cases = [(seed, jump, noise) for seed in config["seed"]["evaluation"] for jump in jumps for noise in noises]
    evaluation_all = execute_cases(evaluation_cases, graph, methods, lambdas, args.workers)
    mask = evaluation_all.apply(lambda row: np.isclose(row["lambda"], lambda_map[(row["jump"], row["noise_std"], row["method"])]), axis=1)
    evaluation = evaluation_all[mask].copy()
    aggregate = evaluation.groupby(["jump", "noise_std", "method"]).agg(
        overall_mse_mean=("overall_mse", "mean"), overall_mse_std=("overall_mse", "std"),
        boundary_mse_mean=("boundary_mse", "mean"), boundary_mse_std=("boundary_mse", "std"),
        smooth_mse_mean=("smooth_mse", "mean"), jump_delta_mae_mean=("jump_delta_mae", "mean"),
        gate_auprc_mean=("gate_auprc", "mean"), n_boundary_edges_mean=("n_boundary_edges", "mean")
    ).reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation.to_csv(args.output_dir / "validation_raw.csv", index=False)
    selected.to_csv(args.output_dir / "selected_lambdas.csv", index=False)
    evaluation.to_csv(args.output_dir / "evaluation_raw.csv", index=False)
    aggregate.to_csv(args.output_dir / "table_synthetic_oracle.csv", index=False)

    pivot = aggregate.pivot_table(index=["jump", "noise_std"], columns="method", values=["overall_mse_mean", "boundary_mse_mean"])
    checks = []
    for (jump, noise), row in pivot.iterrows():
        global_boundary = row[("boundary_mse_mean", "global")]
        oracle_boundary = row[("boundary_mse_mean", "oracle")]
        global_overall = row[("overall_mse_mean", "global")]
        oracle_overall = row[("overall_mse_mean", "oracle")]
        checks.append({
            "jump": float(jump), "noise_std": float(noise),
            "oracle_boundary_relative_improvement": float((global_boundary-oracle_boundary)/global_boundary),
            "oracle_overall_relative_change": float((oracle_overall-global_overall)/global_overall),
            "oracle_better_than_anti_oracle_boundary": bool(oracle_boundary < row[("boundary_mse_mean", "anti_oracle")]),
        })
    (args.output_dir / "go_no_go_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
