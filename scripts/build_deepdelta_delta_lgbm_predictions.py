#!/usr/bin/env python3
"""Train direct/absolute LightGBM controls and freeze response-firewalled predictions."""

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

from bps_gnn.io import freeze_prediction_file
from bps_gnn.pair_flow import hodge_project_complete_pair_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-datasets", type=int)
    parser.add_argument("--held-out-fold", type=int)
    parser.add_argument("--n-jobs", type=int)
    return parser.parse_args()


def sha256_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def selected_response_values(path: Path, selected_indices: np.ndarray) -> np.ndarray:
    """Parse only selected response rows; skipped held-out rows never enter the frame."""
    selected = {int(value) for value in selected_indices}
    frame = pd.read_csv(
        path,
        usecols=["Y"],
        skiprows=lambda line: line > 0 and (line - 1) not in selected,
    )
    values = frame["Y"].to_numpy(dtype=float)
    if len(values) != len(selected_indices) or not np.isfinite(values).all():
        raise RuntimeError(f"{path}: selected response partition mismatch")
    return values


def structure_only_split(path: Path, config: dict[str, object]) -> tuple[list[str], np.ndarray, np.ndarray]:
    structures = pd.read_csv(path, usecols=["SMILES"])["SMILES"].astype(str).tolist()
    split = config["split"]
    splitter = KFold(
        n_splits=int(split["folds"]),
        shuffle=bool(split["shuffle"]),
        random_state=int(split["random_state"]),
    )
    fold = int(split["held_out_fold"])
    for current, (train_index, test_index) in enumerate(
        splitter.split(np.arange(len(structures))), start=1
    ):
        if current == fold:
            return structures, train_index, test_index
    raise RuntimeError(f"held-out fold {fold} does not exist")


def morgan_fingerprints(smiles: list[str], radius: int, bits: int) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    fingerprints = np.empty((len(smiles), bits), dtype=np.uint8)
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise RuntimeError(f"invalid SMILES at row {index}: {value}")
        fingerprints[index] = generator.GetFingerprintAsNumPy(molecule).astype(np.uint8)
    return fingerprints


def balanced_ordered_pairs(n_items: int, maximum: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if maximum < n_items:
        raise ValueError("maximum ordered pairs must retain every diagonal pair")
    upper_i, upper_j = np.triu_indices(n_items, k=1)
    capacity = (maximum - n_items) // 2
    rng = np.random.default_rng(seed)
    if len(upper_i) > capacity:
        selected = rng.choice(len(upper_i), size=capacity, replace=False)
        upper_i, upper_j = upper_i[selected], upper_j[selected]
    diagonal = np.arange(n_items, dtype=int)
    first = np.concatenate([upper_i, upper_j, diagonal])
    second = np.concatenate([upper_j, upper_i, diagonal])
    order = rng.permutation(len(first))
    return first[order], second[order]


def pair_features(fingerprints: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    bits = fingerprints.shape[1]
    result = np.empty((len(first), bits * 2), dtype=np.uint8)
    result[:, :bits] = fingerprints[first]
    result[:, bits:] = fingerprints[second]
    return result


def lgbm_parameters(config: dict[str, object]) -> dict[str, object]:
    return {
        "n_estimators": int(config["n_estimators"]),
        "subsample": float(config["subsample"]),
        "subsample_freq": int(config["subsample_freq"]),
        "random_state": int(config["random_state"]),
        "n_jobs": int(config["n_jobs"]),
        "deterministic": bool(config["deterministic"]),
        "force_col_wise": bool(config["force_col_wise"]),
        "verbosity": -1,
    }


def freeze_vector(values: np.ndarray, ids: list[str], path: Path, metadata: dict[str, object]) -> str:
    return freeze_prediction_file(
        pd.DataFrame({"mol_id": ids, "prediction": values}), path, metadata
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.held_out_fold is not None:
        config["split"]["held_out_fold"] = int(args.held_out_fold)
    if args.n_jobs is not None:
        config["direct_lgbm"]["n_jobs"] = int(args.n_jobs)
        config["absolute_lgbm_control"]["n_jobs"] = int(args.n_jobs)
    datasets = [str(value) for value in config["datasets"]]
    if args.maximum_datasets is not None:
        datasets = datasets[: args.maximum_datasets]
    records: list[dict[str, object]] = []

    for dataset in datasets:
        started = time.perf_counter()
        path = args.data_root / f"{dataset}.csv"
        structures, train_index, test_index = structure_only_split(path, config)
        train_response = selected_response_values(path, train_index)
        fingerprint_config = config["fingerprint"]
        fingerprints = morgan_fingerprints(
            structures, int(fingerprint_config["radius"]), int(fingerprint_config["bits"])
        )
        train_fingerprints = fingerprints[train_index]
        test_fingerprints = fingerprints[test_index]

        pair_config = config["training_pairs"]
        first, second = balanced_ordered_pairs(
            len(train_index),
            int(pair_config["maximum_ordered_pairs"]),
            int(pair_config["sampling_random_state"]),
        )
        train_pair_features = pair_features(train_fingerprints, first, second)
        train_pair_response = train_response[second] - train_response[first]
        direct_model = lgb.LGBMRegressor(**lgbm_parameters(config["direct_lgbm"]))
        direct_model.fit(train_pair_features, train_pair_response)
        del train_pair_features, train_pair_response
        gc.collect()

        test_first = np.repeat(np.arange(len(test_index)), len(test_index))
        test_second = np.tile(np.arange(len(test_index)), len(test_index))
        test_pair_features = pair_features(test_fingerprints, test_first, test_second)
        direct_flow = direct_model.predict(test_pair_features).reshape(
            len(test_index), len(test_index)
        )
        del test_pair_features, direct_model
        gc.collect()

        absolute_model = lgb.LGBMRegressor(**lgbm_parameters(config["absolute_lgbm_control"]))
        absolute_model.fit(train_fingerprints, train_response)
        absolute_potential = absolute_model.predict(test_fingerprints)
        hodge_potential, _ = hodge_project_complete_pair_flow(direct_flow)

        output = args.output_dir / dataset
        common: dict[str, object] = {
            "dataset": dataset,
            "method_family": "DeepDelta_source_derived_Delta_LGBM",
            "held_out_fold": int(config["split"]["held_out_fold"]),
            "train_molecules": int(len(train_index)),
            "test_molecules": int(len(test_index)),
            "train_ordered_pairs": int(len(first)),
            "train_indices_sha256": sha256_strings([str(int(value)) for value in train_index]),
            "test_indices_sha256": sha256_strings([str(int(value)) for value in test_index]),
            "test_smiles_sha256": sha256_strings([structures[int(value)] for value in test_index]),
            "held_out_response_materialized": False,
            "lightgbm_version": lgb.__version__,
        }
        edge_ids = [
            f"edge_{i:04d}_{j:04d}"
            for i in range(len(test_index))
            for j in range(len(test_index))
        ]
        node_ids = [f"position_{i:04d}" for i in range(len(test_index))]
        direct_digest = freeze_vector(
            direct_flow.ravel(), edge_ids, output / "direct_flow.parquet", {**common, "artifact": "direct_flow"}
        )
        hodge_digest = freeze_vector(
            hodge_potential,
            node_ids,
            output / "hodge_potential.parquet",
            {**common, "artifact": "hodge_potential"},
        )
        absolute_digest = freeze_vector(
            absolute_potential,
            node_ids,
            output / "absolute_lgbm_potential.parquet",
            {**common, "artifact": "absolute_lgbm_potential"},
        )
        record = {
            **common,
            "direct_flow_sha256": direct_digest,
            "hodge_potential_sha256": hodge_digest,
            "absolute_lgbm_potential_sha256": absolute_digest,
            "wall_seconds": time.perf_counter() - started,
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "run.json").write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )
        records.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(
        json.dumps(
            {
                "datasets": len(records),
                "held_out_responses_materialized": False,
                "total_seconds": sum(float(row["wall_seconds"]) for row in records),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
