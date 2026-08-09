#!/usr/bin/env python3
"""Fit QM9 direct-pair controls and freeze all response-free repairs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import KFold

from bps_gnn.io import freeze_prediction_file, sha256_file
from bps_gnn.pair_flow import antisymmetric_pair_flow, hodge_project_complete_pair_flow


def hash_order(values: list[str], indices: np.ndarray, seed: int) -> np.ndarray:
    keyed = []
    for index in indices:
        digest = hashlib.sha256(f"{seed}\0{int(index)}\0{values[int(index)]}".encode()).digest()
        keyed.append((digest, int(index)))
    return np.asarray([index for _, index in sorted(keyed)], dtype=int)


def split_indices(smiles: list[str], config: dict, fold: int, smoke: bool) -> tuple[np.ndarray, np.ndarray]:
    splitter = KFold(n_splits=int(config["split"]["folds"]), shuffle=bool(config["split"]["shuffle"]), random_state=int(config["split"]["random_state"]))
    train, test = list(splitter.split(np.arange(len(smiles))))[fold]
    cohort = config["cohorts"]
    train_cap = 2048 if smoke else int(cohort["maximum_train_nodes"])
    test_cap = 64 if smoke else int(cohort["evaluation_nodes"])
    train = hash_order(smiles, train, int(cohort["hash_seed"]))[:train_cap]
    test = hash_order(smiles, test, int(cohort["hash_seed"]) + 1)[:test_cap]
    return np.sort(train), test


def selected_column(path: Path, column: str, indices: np.ndarray) -> np.ndarray:
    selected = set(map(int, indices))
    values = pd.read_csv(path, usecols=[column], skiprows=lambda line: line > 0 and (line - 1) not in selected)[column].to_numpy(dtype=float)
    if len(values) != len(indices) or not np.isfinite(values).all():
        raise RuntimeError(f"Selected {column} response partition mismatch")
    # pandas returns selected source rows in ascending file order. Restore the
    # caller's response-free hash order without ever loading skipped responses.
    order = np.argsort(indices)
    restored = np.empty_like(values)
    restored[order] = values
    return restored


def fingerprints(smiles: list[str], radius: int, bits: int) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    result = np.empty((len(smiles), bits), dtype=np.uint8)
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise RuntimeError(f"Invalid QM9 SMILES at row {index}")
        result[index] = generator.GetFingerprintAsNumPy(molecule).astype(np.uint8)
    return result


def balanced_pairs(n_nodes: int, maximum: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    diagonal = np.arange(n_nodes, dtype=int)
    capacity = max(0, (maximum - n_nodes) // 2)
    if capacity == 0:
        return diagonal, diagonal
    rng, pairs = np.random.default_rng(seed), set()
    while len(pairs) < capacity:
        batch = max(4096, (capacity - len(pairs)) * 2)
        left = rng.integers(0, n_nodes, batch)
        right = rng.integers(0, n_nodes, batch)
        for a, b in zip(left, right):
            if a == b:
                continue
            pairs.add((min(int(a), int(b)), max(int(a), int(b))))
            if len(pairs) == capacity:
                break
    ordered = sorted(pairs)
    first = np.asarray([a for a, b in ordered] + [b for a, b in ordered] + diagonal.tolist(), dtype=int)
    second = np.asarray([b for a, b in ordered] + [a for a, b in ordered] + diagonal.tolist(), dtype=int)
    order = rng.permutation(len(first))
    return first[order], second[order]


def pair_features(fp: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.concatenate([fp[first], fp[second]], axis=1)


def params(config: dict, smoke: bool) -> dict:
    model = config["lightgbm"]
    return {
        "n_estimators": 10 if smoke else int(model["n_estimators"]),
        "subsample": float(model["subsample"]), "subsample_freq": int(model["subsample_freq"]),
        "random_state": int(model["random_state"]), "n_jobs": int(model["n_jobs"]),
        "deterministic": bool(model["deterministic"]), "force_col_wise": bool(model["force_col_wise"]),
        "verbosity": -1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=["homo", "lumo", "gap"])
    parser.add_argument("--fold", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    digest = sha256_file(args.source)
    if digest != config["source"]["expected_sha256"]:
        raise RuntimeError(f"QM9 source hash mismatch: {digest}")
    smiles = pd.read_csv(args.source, usecols=[config["source"]["smiles_column"]])[config["source"]["smiles_column"]].astype(str).tolist()
    fp_config = config["fingerprint"]
    all_fp = fingerprints(smiles, int(fp_config["radius"]), int(fp_config["bits"]))
    tasks = [args.task] if args.task else list(config["tasks"])
    folds = [args.fold] if args.fold is not None else list(range(int(config["split"]["folds"])))
    records = []
    for task in tasks:
        for fold in folds:
            started = time.perf_counter()
            train_index, test_index = split_indices(smiles, config, int(fold), args.smoke)
            train_y = selected_column(args.source, task, train_index)
            train_fp, test_fp = all_fp[train_index], all_fp[test_index]
            maximum_pairs = 10000 if args.smoke else int(config["training_pairs"]["maximum_ordered_pairs"])
            first, second = balanced_pairs(len(train_index), maximum_pairs, int(config["training_pairs"]["sampling_random_state"]) + int(fold))
            features = pair_features(train_fp, first, second)
            model = lgb.LGBMRegressor(**params(config, args.smoke))
            model.fit(features, train_y[second] - train_y[first])
            del features
            gc.collect()
            n_test = len(test_index)
            test_first = np.repeat(np.arange(n_test), n_test)
            test_second = np.tile(np.arange(n_test), n_test)
            raw = model.predict(pair_features(test_fp, test_first, test_second)).reshape(n_test, n_test)
            diagonal = raw.copy()
            np.fill_diagonal(diagonal, 0.0)
            anti = antisymmetric_pair_flow(diagonal)
            hodge_potential, _ = hodge_project_complete_pair_flow(anti)
            absolute_model = lgb.LGBMRegressor(**params(config, args.smoke))
            absolute_model.fit(train_fp, train_y)
            absolute = absolute_model.predict(test_fp)
            del model, absolute_model
            gc.collect()

            run_dir = args.output_dir / task / f"fold_{fold}"
            common = {
                "dataset": "QM9", "task": task, "fold": int(fold), "smoke": bool(args.smoke),
                "source_sha256": digest, "train_nodes": int(len(train_index)), "test_nodes": int(n_test),
                "train_ordered_pairs": int(len(first)), "held_out_response_materialized": False,
                "train_indices": train_index.tolist(), "test_indices": test_index.tolist(),
                "anchor_positions": hash_order([smiles[int(i)] for i in test_index], np.arange(n_test), int(config["cohorts"]["hash_seed"]) + 2)[:max(config["evaluation"]["anchors"])].tolist(),
            }
            edge_ids = [f"edge_{i:04d}_{j:04d}" for i in range(n_test) for j in range(n_test)]
            node_ids = [f"position_{i:04d}" for i in range(n_test)]
            artifacts = {
                "raw_flow": (raw.ravel(), edge_ids), "diagonal_repaired_flow": (diagonal.ravel(), edge_ids),
                "antisymmetric_flow": (anti.ravel(), edge_ids), "hodge_potential": (hodge_potential, node_ids),
                "absolute_potential": (absolute, node_ids),
            }
            hashes = {}
            for name, (values, ids) in artifacts.items():
                hashes[name] = freeze_prediction_file(pd.DataFrame({"mol_id": ids, "prediction": values}), run_dir / f"{name}.parquet", {**common, "artifact": name})
            record = {**common, "artifact_sha256": hashes, "wall_seconds": time.perf_counter() - started}
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
            records.append({key: value for key, value in record.items() if key not in {"train_indices", "test_indices", "anchor_positions", "artifact_sha256"}})
            print(json.dumps({"task": task, "fold": fold, "seconds": record["wall_seconds"], "smoke": args.smoke}), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "build_summary.csv", mode="a" if (args.output_dir / "build_summary.csv").exists() else "w", header=not (args.output_dir / "build_summary.csv").exists(), index=False)


if __name__ == "__main__":
    main()
