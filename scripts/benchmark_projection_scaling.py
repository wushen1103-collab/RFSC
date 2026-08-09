#!/usr/bin/env python3
"""CPU runtime and traced-allocation benchmark for the incidence projectors."""

from __future__ import annotations

import argparse
import gc
import json
import time
import tracemalloc
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from bps_gnn.pair_flow import (
    hodge_project_complete_pair_flow,
    hodge_project_observed_pair_flow,
)
from bps_gnn.projection import boundary_projection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def measure(call: Callable[[], object], repeats: int) -> tuple[float, float, int]:
    durations: list[float] = []
    peaks: list[int] = []
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        result = call()
        durations.append(time.perf_counter() - started)
        _, peak = tracemalloc.get_traced_memory()
        peaks.append(peak)
        tracemalloc.stop()
        if isinstance(result, tuple):
            arrays = [np.asarray(value) for value in result]
        else:
            arrays = [np.asarray(result)]
        if not all(np.isfinite(value).all() for value in arrays):
            raise RuntimeError("benchmark operator produced a non-finite value")
        del result
    return (
        1000.0 * float(np.median(durations)),
        1000.0 * float(np.quantile(durations, 0.95)),
        int(max(peaks)),
    )


def node_case(n_nodes: int, repeats: int) -> dict[str, object]:
    rng = np.random.default_rng(29001 + n_nodes)
    n_edges = 4 * n_nodes
    source = rng.integers(0, n_nodes, size=n_edges)
    offset = rng.integers(1, n_nodes, size=n_edges)
    target = (source + offset) % n_nodes
    signal = pd.Series(rng.normal(size=n_nodes), index=np.arange(n_nodes))
    edges = pd.DataFrame({"mol_id_lo": source, "mol_id_hi": target})
    median, p95, peak = measure(
        lambda: boundary_projection(signal, edges, 1.5), repeats
    )
    input_bytes = int(signal.memory_usage(deep=True) + edges.memory_usage(deep=True).sum())
    return {
        "operator": "node_boundary_projection",
        "n_nodes": n_nodes,
        "observed_directed_edges": n_edges,
        "density": np.nan,
        "input_mib": input_bytes / 2**20,
        "median_ms": median,
        "p95_ms": p95,
        "peak_traced_allocation_mib": peak / 2**20,
        "repeats": repeats,
        "asymptotic_time": "O(n+m)",
        "additional_memory": "O(n+m) implementation objects",
    }


def complete_edge_case(n_nodes: int, repeats: int) -> dict[str, object]:
    rng = np.random.default_rng(29101 + n_nodes)
    flow = rng.normal(size=(n_nodes, n_nodes))
    median, p95, peak = measure(
        lambda: hodge_project_complete_pair_flow(flow), repeats
    )
    return {
        "operator": "complete_edge_hodge",
        "n_nodes": n_nodes,
        "observed_directed_edges": n_nodes * n_nodes,
        "density": 1.0,
        "input_mib": flow.nbytes / 2**20,
        "median_ms": median,
        "p95_ms": p95,
        "peak_traced_allocation_mib": peak / 2**20,
        "repeats": repeats,
        "asymptotic_time": "O(n^2)",
        "additional_memory": "O(n^2) returned flow; O(n) beyond input/output",
    }


def sparse_edge_case(n_nodes: int, density: float, repeats: int) -> dict[str, object]:
    rng = np.random.default_rng(29201 + n_nodes)
    upper_i, upper_j = np.triu_indices(n_nodes, k=1)
    count = max(n_nodes - 1, int(np.ceil(density * len(upper_i))))
    selected = rng.choice(len(upper_i), size=count, replace=False)
    first, second = upper_i[selected], upper_j[selected]
    source = np.concatenate([first, second])
    target = np.concatenate([second, first])
    flow = rng.normal(size=len(source))
    median, p95, peak = measure(
        lambda: hodge_project_observed_pair_flow(
            n_nodes, source, target, flow
        ),
        repeats,
    )
    input_bytes = source.nbytes + target.nbytes + flow.nbytes
    return {
        "operator": "incomplete_edge_hodge_dense_lstsq",
        "n_nodes": n_nodes,
        "observed_directed_edges": len(source),
        "density": density,
        "input_mib": input_bytes / 2**20,
        "median_ms": median,
        "p95_ms": p95,
        "peak_traced_allocation_mib": peak / 2**20,
        "repeats": repeats,
        "asymptotic_time": "dense least-squares backend",
        "additional_memory": "O(nm) dense incidence; benchmark-scale only",
    }


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.smoke:
        node_sizes = [100]
        complete_sizes = [25]
        sparse_sizes = [25]
    else:
        node_sizes = [100, 1000, 5000, 10000, 25000]
        complete_sizes = [25, 50, 100, 250, 500, 1000, 2000]
        sparse_sizes = [25, 50, 100, 133]
    rows = [node_case(value, args.repeats) for value in node_sizes]
    rows.extend(complete_edge_case(value, args.repeats) for value in complete_sizes)
    rows.extend(sparse_edge_case(value, 0.5, args.repeats) for value in sparse_sizes)
    frame = pd.DataFrame.from_records(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "projection_scaling.csv", index=False)
    summary = {
        "rows": len(frame),
        "repeats": args.repeats,
        "smoke": args.smoke,
        "labels_read": False,
        "timing": "single-process wall clock",
        "memory": "peak Python/NumPy allocation traced during operator call",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

