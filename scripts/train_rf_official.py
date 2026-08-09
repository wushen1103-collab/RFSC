#!/usr/bin/env python3
"""Train an official-split ECFP-RF without reading test responses."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

from bps_gnn.fingerprints import ecfp_matrix
from bps_gnn.io import (
    freeze_prediction_file,
    read_label_columns,
    read_labels_for_ids,
    read_structure_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path)
    return parser.parse_args()


def build_model(config: dict, n_estimators: int, seed: int) -> RandomForestRegressor:
    model = config["model"]
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=model["n_jobs"],
        max_features=model["max_features"],
        min_samples_leaf=model["min_samples_leaf"],
    )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError(f"Seed {args.seed} is not pre-registered")
    data_root = Path(config["data_root"])
    structures = read_structure_table(data_root, args.target)
    split = read_label_columns(data_root, args.target, ["mol_id", "official_split"])
    structures = structures.merge(split, on="mol_id", how="inner", validate="one_to_one")
    train_frame = structures[structures["official_split"] == "train"].copy()
    test_frame = structures[structures["official_split"] == "test"].copy()
    if len(train_frame) + len(test_frame) != len(structures):
        raise RuntimeError("Unexpected official split value")
    train_labels = read_labels_for_ids(data_root, args.target, train_frame["mol_id"])
    train_frame = train_frame.merge(train_labels, on="mol_id", validate="one_to_one")
    fingerprint = config["fingerprint"]
    all_features = ecfp_matrix(
        structures["canonical_isomeric_smiles"].tolist(),
        radius=fingerprint["radius"],
        n_bits=fingerprint["n_bits"],
    )
    feature_map = {mol_id: index for index, mol_id in enumerate(structures["mol_id"])}
    train_indices = np.asarray([feature_map[mol_id] for mol_id in train_frame["mol_id"]])
    test_indices = np.asarray([feature_map[mol_id] for mol_id in test_frame["mol_id"]])
    x_train, y_train = all_features[train_indices], train_frame["y_pactivity"].to_numpy()
    groups = train_frame[config["selection"]["group_column"]].to_numpy()
    if args.selection_csv:
        selection = pd.read_csv(args.selection_csv)
        if not set(selection["n_estimators"]).issubset(config["selection"]["n_estimators"]):
            raise RuntimeError("Reused selection contains a non-preregistered candidate")
        selection = selection.sort_values(["cv_rmse_mean", "n_estimators"])
    else:
        folds = GroupKFold(n_splits=config["selection"]["folds"])
        selection_rows = []
        for n_estimators in config["selection"]["n_estimators"]:
            fold_rmse = []
            for train_index, valid_index in folds.split(x_train, y_train, groups):
                candidate = build_model(config, n_estimators, config["selection"]["seed"])
                candidate.fit(x_train[train_index], y_train[train_index])
                prediction = candidate.predict(x_train[valid_index])
                fold_rmse.append(float(mean_squared_error(y_train[valid_index], prediction) ** 0.5))
            selection_rows.append(
                {
                    "n_estimators": n_estimators,
                    "cv_rmse_mean": float(np.mean(fold_rmse)),
                    "cv_rmse_std": float(np.std(fold_rmse, ddof=1)),
                }
            )
        selection = pd.DataFrame(selection_rows).sort_values(["cv_rmse_mean", "n_estimators"])
    chosen = int(selection.iloc[0]["n_estimators"])
    model = build_model(config, chosen, args.seed)
    model.fit(x_train, y_train)
    prediction = model.predict(all_features[test_indices])
    output = args.output_dir / args.target / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output / "model_selection.csv", index=False)
    with (output / "checkpoint.pkl").open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "target": args.target,
        "seed": args.seed,
        "method": config["method"],
        "protocol": config["protocol"],
        "n_train": len(train_frame),
        "n_test": len(test_frame),
        "n_estimators": chosen,
        "selection_reused": bool(args.selection_csv),
        "test_labels_materialized": False,
    }
    digest = freeze_prediction_file(
        pd.DataFrame({"mol_id": test_frame["mol_id"], "prediction": prediction}),
        output / "predictions.parquet",
        metadata,
    )
    metadata["prediction_sha256"] = digest
    metadata["wall_time_seconds"] = time.perf_counter() - started
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
