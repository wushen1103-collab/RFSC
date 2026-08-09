#!/usr/bin/env python3
"""Build response-free incomplete-graph Hodge candidates from frozen pair flows."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import freeze_prediction_file, read_frozen_prediction
from bps_gnn.pair_flow import hodge_project_observed_pair_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-datasets", type=int)
    parser.add_argument("--maximum-densities", type=int)
    parser.add_argument("--maximum-seeds", type=int)
    return parser.parse_args()


def ordered_values(series: pd.Series, ids: list[str], artifact: str) -> np.ndarray:
    if set(series.index) != set(ids):
        raise RuntimeError(f"{artifact}: frozen IDs do not match expected order")
    values = series.reindex(ids).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{artifact}: non-finite value")
    return values


def sample_connected_undirected_mask(
    n_nodes: int, density: float, random_state: int, dataset_index: int
) -> np.ndarray:
    """Sample a random spanning tree, then uniformly fill to target density."""
    if n_nodes < 2:
        raise ValueError("at least two nodes are required")
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    upper_i, upper_j = np.triu_indices(n_nodes, k=1)
    edge_to_position = {
        (int(first), int(second)): position
        for position, (first, second) in enumerate(zip(upper_i, upper_j))
    }
    rng = np.random.default_rng(np.random.SeedSequence([random_state, dataset_index]))
    permutation = rng.permutation(n_nodes)
    selected: set[int] = set()
    for child_position in range(1, n_nodes):
        child = int(permutation[child_position])
        parent = int(permutation[int(rng.integers(0, child_position))])
        edge = (min(child, parent), max(child, parent))
        selected.add(edge_to_position[edge])
    target_count = max(n_nodes - 1, int(math.ceil(density * len(upper_i))))
    remaining = np.asarray(
        [position for position in range(len(upper_i)) if position not in selected],
        dtype=int,
    )
    needed = target_count - len(selected)
    if needed > 0:
        selected.update(int(value) for value in rng.choice(remaining, needed, replace=False))
    mask = np.zeros(len(upper_i), dtype=bool)
    mask[np.asarray(sorted(selected), dtype=int)] = True
    return mask


def is_connected(n_nodes: int, source: np.ndarray, target: np.ndarray) -> bool:
    adjacency: list[list[int]] = [[] for _ in range(n_nodes)]
    for first, second in zip(source, target):
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))
    seen = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return len(seen) == n_nodes


def density_key(density: float) -> str:
    return f"density_{int(round(100.0 * density)):02d}"


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    datasets = [str(value) for value in config["datasets"]]
    densities = [float(value) for value in config["sampling"]["undirected_density"]]
    seeds = [int(value) for value in config["sampling"]["random_states"]]
    if args.maximum_datasets is not None:
        datasets = datasets[: args.maximum_datasets]
    if args.maximum_densities is not None:
        densities = densities[: args.maximum_densities]
    if args.maximum_seeds is not None:
        seeds = seeds[: args.maximum_seeds]

    rows: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(datasets):
        source_dir = args.source_prediction_root / dataset
        source_metadata = json.loads((source_dir / "run.json").read_text(encoding="utf-8"))
        if source_metadata["held_out_response_materialized"] is not False:
            raise RuntimeError(f"{dataset}: source prediction firewall failed")
        n_nodes = int(source_metadata["test_molecules"])
        edge_ids = [
            f"edge_{i:04d}_{j:04d}" for i in range(n_nodes) for j in range(n_nodes)
        ]
        direct = ordered_values(
            read_frozen_prediction(source_dir / "direct_flow.parquet"),
            edge_ids,
            f"{dataset}/direct_flow",
        ).reshape(n_nodes, n_nodes)
        upper_i, upper_j = np.triu_indices(n_nodes, k=1)
        upper_ids = [
            f"undirected_{int(i):04d}_{int(j):04d}" for i, j in zip(upper_i, upper_j)
        ]
        node_ids = [f"position_{i:04d}" for i in range(n_nodes)]

        for density in densities:
            for seed in seeds:
                started = time.perf_counter()
                mask = sample_connected_undirected_mask(
                    n_nodes, density, seed, dataset_index
                )
                selected_i, selected_j = upper_i[mask], upper_j[mask]
                connected = is_connected(n_nodes, selected_i, selected_j)
                if not connected:
                    raise RuntimeError(f"{dataset}/{density}/{seed}: sampled graph disconnected")
                source = np.concatenate([selected_i, selected_j])
                target = np.concatenate([selected_j, selected_i])
                observed = np.concatenate(
                    [direct[selected_i, selected_j], direct[selected_j, selected_i]]
                )
                potential, _ = hodge_project_observed_pair_flow(
                    n_nodes, source, target, observed
                )
                output = args.output_dir / dataset / density_key(density) / f"seed_{seed}"
                common: dict[str, object] = {
                    "dataset": dataset,
                    "dataset_index": dataset_index,
                    "target_undirected_density": density,
                    "effective_undirected_density": float(mask.mean()),
                    "random_state": seed,
                    "n_nodes": n_nodes,
                    "n_observed_undirected_edges": int(mask.sum()),
                    "n_total_undirected_edges": int(len(mask)),
                    "connected": connected,
                    "observe_both_directions": True,
                    "source_direct_flow_sha256": source_metadata["direct_flow_sha256"],
                    "held_out_response_materialized": False,
                }
                potential_digest = freeze_prediction_file(
                    pd.DataFrame({"mol_id": node_ids, "prediction": potential}),
                    output / "sparse_potential.parquet",
                    {**common, "artifact": "sparse_potential"},
                )
                mask_digest = freeze_prediction_file(
                    pd.DataFrame(
                        {"mol_id": upper_ids, "prediction": mask.astype(float)}
                    ),
                    output / "observed_mask.parquet",
                    {**common, "artifact": "observed_undirected_mask"},
                )
                record = {
                    **common,
                    "sparse_potential_sha256": potential_digest,
                    "observed_mask_sha256": mask_digest,
                    "wall_seconds": time.perf_counter() - started,
                }
                output.mkdir(parents=True, exist_ok=True)
                (output / "run.json").write_text(
                    json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
                )
                rows.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame.from_records(rows)
    summary.to_csv(args.output_dir / "build_summary.csv", index=False)
    print(
        json.dumps(
            {
                "candidates": len(rows),
                "datasets": len(datasets),
                "held_out_responses_materialized": False,
                "all_connected": bool(summary["connected"].all()),
                "total_seconds": float(summary["wall_seconds"].sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

