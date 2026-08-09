#!/usr/bin/env python3
"""Train frozen SVM or RF on ChEMBL temporal partitions without test labels."""

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
from sklearn.svm import SVR

from bps_gnn.fingerprints import ecfp_matrix
from bps_gnn.io import freeze_prediction_file, read_labels_for_ids, read_structure_table
from bps_gnn.temporal import temporal_split_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", choices=["svm", "rf"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def fit_svm(features: np.ndarray, train_index: np.ndarray, train_y: np.ndarray, config: dict):
    model_config = config["model"]
    model = SVR(
        kernel=str(model_config["kernel"]),
        C=float(model_config["C"]),
        gamma=float(model_config["gamma"]),
        epsilon=float(model_config["epsilon"]),
        cache_size=float(model_config["cache_size_mb"]),
    )
    model.fit(features[train_index], train_y)
    return model, [], None


def fit_rf(
    features: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    config: dict,
    seed: int,
    smoke: bool,
):
    candidates = [int(value) for value in config["selection"]["n_estimators"]]
    if smoke:
        candidates = candidates[:1]
    model_config = config["model"]
    model = RandomForestRegressor(
        n_estimators=candidates[0],
        random_state=seed,
        n_jobs=int(model_config["n_jobs"]),
        max_features=float(model_config["max_features"]),
        min_samples_leaf=int(model_config["min_samples_leaf"]),
        criterion=str(model_config.get("criterion", "squared_error")),
        warm_start=True,
    )
    rows: list[dict[str, float | int]] = []
    best_key, best_blob, chosen = (float("inf"), float("inf")), None, None
    for n_estimators in candidates:
        model.set_params(n_estimators=n_estimators)
        model.fit(features[train_index], train_y)
        prediction = model.predict(features[validation_index])
        rmse = float(mean_squared_error(validation_y, prediction) ** 0.5)
        mae = float(np.mean(np.abs(validation_y - prediction)))
        rows.append({"n_estimators": n_estimators, "validation_rmse": rmse, "validation_mae": mae})
        if (rmse, n_estimators) < best_key:
            best_key = (rmse, n_estimators)
            best_blob = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            chosen = n_estimators
    if best_blob is None:
        raise RuntimeError("No RF candidate completed")
    return pickle.loads(best_blob), rows, chosen


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("Seed is not frozen in the model config")
    if args.model == "svm" and args.seed != 0:
        raise ValueError("The deterministic SVM is frozen at seed 0")
    data_root = Path(config["data_root"])
    structures = read_structure_table(data_root, args.target).reset_index(drop=True)
    train_ids, validation_ids, test_ids = temporal_split_ids(data_root, args.target)
    if args.smoke:
        train_ids = train_ids[: min(128, len(train_ids))]
        validation_ids = validation_ids[: min(64, len(validation_ids))]
    authorized = train_ids if args.model == "svm" else [*train_ids, *validation_ids]
    labels = read_labels_for_ids(data_root, args.target, authorized).set_index("mol_id")["y_pactivity"]
    if set(labels.index) & set(test_ids):
        raise RuntimeError("Temporal test response entered classical trainer")
    fingerprint = config["fingerprint"]
    features = ecfp_matrix(
        structures["canonical_isomeric_smiles"].tolist(),
        radius=int(fingerprint["radius"]),
        n_bits=int(fingerprint["n_bits"]),
    )
    id_to_index = {mol_id: index for index, mol_id in enumerate(structures["mol_id"])}
    index = lambda ids: np.asarray([id_to_index[mol_id] for mol_id in ids], dtype=int)
    train_index, validation_index, test_index = index(train_ids), index(validation_ids), index(test_ids)
    train_y = labels.loc[train_ids].to_numpy(dtype=float)
    if args.model == "svm":
        model, selection_rows, chosen = fit_svm(features, train_index, train_y, config)
    else:
        validation_y = labels.loc[validation_ids].to_numpy(dtype=float)
        model, selection_rows, chosen = fit_rf(
            features, train_index, validation_index, train_y, validation_y, config, args.seed, args.smoke
        )

    output = args.output_dir / args.target / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "checkpoint.pkl").open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if selection_rows:
        pd.DataFrame(selection_rows).to_csv(output / "model_selection.csv", index=False)
    metadata = {
        "target": args.target,
        "protocol": "chembl36_temporal",
        "method": args.model,
        "seed": args.seed,
        "smoke": args.smoke,
        "n_train": len(train_ids),
        "n_validation": len(validation_ids),
        "n_test": len(test_ids),
        "chosen_n_estimators": chosen,
        "validation_labels_materialized": args.model == "rf",
        "test_labels_materialized": False,
        "test_predictions_generated": not args.smoke,
    }
    metadata["validation_prediction_sha256"] = freeze_prediction_file(
        pd.DataFrame(
            {"mol_id": validation_ids, "prediction": model.predict(features[validation_index])}
        ),
        output / "validation_predictions.parquet",
        metadata,
    )
    if not args.smoke:
        metadata["prediction_sha256"] = freeze_prediction_file(
            pd.DataFrame({"mol_id": test_ids, "prediction": model.predict(features[test_index])}),
            output / "predictions.parquet",
            metadata,
        )
    metadata["wall_time_seconds"] = time.perf_counter() - started
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
