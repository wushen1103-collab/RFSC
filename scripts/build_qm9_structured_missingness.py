#!/usr/bin/env python3
"""Freeze Hodge potentials under three structured QM9 observation mechanisms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from bps_gnn.io import freeze_prediction_file, read_frozen_prediction
from bps_gnn.pair_flow import hodge_project_observed_pair_flow


class DSU:
    def __init__(self, n: int): self.p = list(range(n))
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a: int, b: int) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b: return False
        self.p[b] = a; return True


def hashes(smiles: list[str], seed: int) -> list[int]:
    return [int.from_bytes(hashlib.sha256(f"{seed}\0{i}\0{s}".encode()).digest()[:8], "little") for i, s in enumerate(smiles)]


def similarities(smiles: list[str], radius: int, bits: int) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(value)) for value in smiles]
    matrix = np.eye(len(fps), dtype=float)
    for i in range(len(fps)):
        matrix[i, i + 1:] = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        matrix[i + 1:, i] = matrix[i, i + 1:]
    return matrix


def connected_prefix(order: list[tuple[int, int]], n: int, allowed: np.ndarray | None = None) -> list[tuple[int, int]]:
    dsu, tree = DSU(n), []
    for a, b in order:
        if allowed is not None and not allowed[a, b]: continue
        if dsu.union(a, b): tree.append((a, b))
    return tree


def select_pairs(scheme: str, n: int, count: int, similarity: np.ndarray, key: list[int]) -> tuple[list[tuple[int, int]], np.ndarray]:
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if scheme == "chemistry_local":
        order = sorted(pairs, key=lambda pair: (-similarity[pair], key[pair[0]] ^ key[pair[1]], pair))
        prefix = connected_prefix(order, n)
        ordered = prefix + [pair for pair in order if pair not in set(prefix)]
        components = np.zeros(n, dtype=int)
    elif scheme == "hub_biased":
        hubs = sorted(range(n), key=lambda i: (key[i], i))
        star = [(hubs[0], node) if hubs[0] < node else (node, hubs[0]) for node in range(n) if node != hubs[0]]
        star_set = set(star)
        ordered = star + sorted([pair for pair in pairs if pair not in star_set], key=lambda pair: (min(hubs.index(pair[0]), hubs.index(pair[1])), key[pair[0]] ^ key[pair[1]], pair))
        components = np.zeros(n, dtype=int)
    elif scheme == "disconnected_blocks":
        ranked = sorted(range(n), key=lambda i: (key[i], i))
        components = np.empty(n, dtype=int)
        for position, node in enumerate(ranked): components[node] = position % 2
        allowed = np.equal.outer(components, components)
        stars = []
        for block in [0, 1]:
            members = [i for i in ranked if components[i] == block]
            stars.extend([(min(members[0], node), max(members[0], node)) for node in members[1:]])
        star_set = set(stars)
        remaining = [pair for pair in pairs if allowed[pair] and pair not in star_set]
        ordered = stars + sorted(remaining, key=lambda pair: (key[pair[0]] ^ key[pair[1]], pair))
    else:
        raise ValueError(scheme)
    if count < len(connected_prefix(ordered, n, np.equal.outer(components, components) if scheme == "disconnected_blocks" else None)):
        raise RuntimeError("Density is too low for the frozen connectivity rule")
    return ordered[:count], components


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--flow-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-runs", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_smiles = pd.read_csv(args.source, usecols=["smiles"])["smiles"].astype(str).tolist()
    records = []
    run_paths = sorted(args.flow_root.glob("*/fold_*/run.json"))
    if args.maximum_runs is not None: run_paths = run_paths[:args.maximum_runs]
    for run_path in run_paths:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        n = min(int(config["nodes"]), int(run["test_nodes"]))
        test_indices = np.asarray(run["test_indices"][:n], dtype=int)
        smiles = [source_smiles[index] for index in test_indices]
        edge_ids_full = [f"edge_{i:04d}_{j:04d}" for i in range(int(run["test_nodes"])) for j in range(int(run["test_nodes"]))]
        raw_full = read_frozen_prediction(run_path.parent / "raw_flow.parquet").reindex(edge_ids_full).to_numpy().reshape(int(run["test_nodes"]), -1)
        raw = raw_full[:n, :n]
        similarity = similarities(smiles, int(config["fingerprint"]["radius"]), int(config["fingerprint"]["bits"]))
        key = hashes(smiles, int(config["hash_seed"]) + int(run["fold"]))
        total = n * (n - 1) // 2
        for scheme in config["schemes"]:
            for density in map(float, config["densities"]):
                selected, components = select_pairs(scheme, n, int(round(total * density)), similarity, key)
                source = np.asarray([a for a, b in selected] + [b for a, b in selected], dtype=int)
                target = np.asarray([b for a, b in selected] + [a for a, b in selected], dtype=int)
                observed = raw[source, target]
                potential, fitted = hodge_project_observed_pair_flow(n, source, target, observed)
                observed_mask = np.zeros((n, n), dtype=float); observed_mask[source, target] = 1.0
                evaluable = np.ones((n, n), dtype=float) if scheme != "disconnected_blocks" else np.equal.outer(components, components).astype(float)
                np.fill_diagonal(evaluable, 0.0)
                out = args.output_dir / run["task"] / f"fold_{run['fold']}" / scheme / f"density_{density:.2f}"
                node_ids = [f"position_{i:04d}" for i in range(n)]
                edge_ids = [f"edge_{i:04d}_{j:04d}" for i in range(n) for j in range(n)]
                common = {"task": run["task"], "fold": int(run["fold"]), "scheme": scheme, "density": density, "n_nodes": n, "observed_directed_edges": len(observed), "held_out_response_materialized": False}
                for name, values, ids in [("potential", potential, node_ids), ("observed_mask", observed_mask.ravel(), edge_ids), ("evaluable_mask", evaluable.ravel(), edge_ids)]:
                    freeze_prediction_file(pd.DataFrame({"mol_id": ids, "prediction": values}), out / f"{name}.parquet", {**common, "artifact": name})
                residual = observed - fitted
                record = {**common, "observed_residual_energy": float(np.square(residual).sum()), "test_indices": test_indices.tolist()}
                out.mkdir(parents=True, exist_ok=True); (out / "run.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
                records.append({key: value for key, value in record.items() if key != "test_indices"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(json.dumps({"structured_runs": len(records), "held_out_responses_materialized": False}, indent=2))


if __name__ == "__main__":
    main()
