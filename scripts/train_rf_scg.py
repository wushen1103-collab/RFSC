#!/usr/bin/env python3
"""Train ECFP-RF on frozen SCG folds without outer-test responses."""

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

from bps_gnn.fingerprints import ecfp_matrix, multifingerprint_4263_matrix
from bps_gnn.io import freeze_prediction_file, read_labels_for_ids, read_structure_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    return parser.parse_args()


def split_ids(target: str, fold: int) -> tuple[list[str], list[str], list[str]]:
    assignments = pd.read_parquet(Path("folds/scg") / target / "outer_folds.parquet")
    n_folds = int(assignments["outer_fold"].nunique())
    validation_fold = (fold + 1) % n_folds
    train_ids = assignments.loc[~assignments["outer_fold"].isin([fold, validation_fold]), "mol_id"].tolist()
    valid_ids = assignments.loc[assignments["outer_fold"] == validation_fold, "mol_id"].tolist()
    test_ids = assignments.loc[assignments["outer_fold"] == fold, "mol_id"].tolist()
    if set(train_ids) & set(valid_ids) or set(train_ids) & set(test_ids) or set(valid_ids) & set(test_ids):
        raise RuntimeError("SCG partition overlap")
    return train_ids, valid_ids, test_ids


def build_model(config: dict, n_estimators: int, seed: int) -> RandomForestRegressor:
    model = config["model"]
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=int(model["n_jobs"]),
        max_features=float(model["max_features"]),
        min_samples_leaf=int(model["min_samples_leaf"]),
        criterion=str(model.get("criterion", "squared_error")),
    )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("Seed is not registered")
    structures = read_structure_table(Path(config["data_root"]), args.target).reset_index(drop=True)
    train_ids, valid_ids, test_ids = split_ids(args.target, args.fold)
    if args.smoke:
        train_ids, valid_ids = train_ids[:128], valid_ids[:64]
    authorized = [*train_ids, *valid_ids]
    labels = read_labels_for_ids(Path(config["data_root"]), args.target, authorized).set_index("mol_id")["y_pactivity"]
    if set(labels.index) & set(test_ids):
        raise RuntimeError("Outer-test response entered RF trainer")
    fingerprint = config["fingerprint"]
    fingerprint_kind = fingerprint.get("kind", "ecfp")
    if fingerprint_kind == "ecfp":
        features = ecfp_matrix(
            structures["canonical_isomeric_smiles"].tolist(),
            radius=int(fingerprint["radius"]),
            n_bits=int(fingerprint["n_bits"]),
        )
    elif fingerprint_kind == "multifingerprint_4263":
        features = multifingerprint_4263_matrix(
            structures["canonical_isomeric_smiles"].tolist()
        )
    else:
        raise ValueError(f"Unsupported fingerprint kind: {fingerprint_kind}")
    id_to_index = {mol_id: index for index, mol_id in enumerate(structures["mol_id"])}
    train_index = np.asarray([id_to_index[mol_id] for mol_id in train_ids])
    valid_index = np.asarray([id_to_index[mol_id] for mol_id in valid_ids])
    test_index = np.asarray([id_to_index[mol_id] for mol_id in test_ids])
    x_train, x_valid = features[train_index], features[valid_index]
    y_train = labels.loc[train_ids].to_numpy(dtype=float)
    y_valid = labels.loc[valid_ids].to_numpy(dtype=float)
    candidates = [int(config["selection"]["n_estimators"][0])] if args.smoke else [int(value) for value in config["selection"]["n_estimators"]]
    selection_rows, best_model, best_rmse, chosen = [], None, float("inf"), None
    for n_estimators in candidates:
        model = build_model(config, n_estimators, args.seed)
        model.fit(x_train, y_train)
        validation_prediction = model.predict(x_valid)
        rmse = float(mean_squared_error(y_valid, validation_prediction) ** 0.5)
        mae = float(np.mean(np.abs(validation_prediction - y_valid)))
        selection_rows.append({"n_estimators": n_estimators, "validation_rmse": rmse, "validation_mae": mae})
        if (rmse, n_estimators) < (best_rmse, chosen if chosen is not None else float("inf")):
            best_model, best_rmse, chosen = model, rmse, n_estimators
    if best_model is None:
        raise RuntimeError("No RF candidate completed")
    output = args.output_dir / args.target / f"fold_{args.fold}" / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selection_rows).sort_values(["validation_rmse", "n_estimators"]).to_csv(output / "model_selection.csv", index=False)
    with (output / "checkpoint.pkl").open("wb") as handle:
        pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    valid_prediction = best_model.predict(x_valid)
    metadata = {
        "target": args.target,
        "fold": args.fold,
        "seed": args.seed,
        "method": config["method"],
        "protocol": config["protocol"],
        "smoke": args.smoke,
        "n_train": len(train_ids),
        "n_validation": len(valid_ids),
        "n_test": len(test_ids),
        "n_estimators": chosen,
        "fingerprint_kind": fingerprint_kind,
        "feature_dimension": int(features.shape[1]),
        "best_validation_rmse": best_rmse,
        "validation_labels_materialized": True,
        "outer_test_labels_materialized": False,
        "outer_test_predictions_generated": not args.smoke and not args.validation_only,
        "validation_only": args.validation_only,
    }
    metadata["validation_prediction_sha256"] = freeze_prediction_file(
        pd.DataFrame({"mol_id": valid_ids, "prediction": valid_prediction}),
        output / "validation_predictions.parquet",
        metadata,
    )
    if not args.smoke and not args.validation_only:
        metadata["prediction_sha256"] = freeze_prediction_file(
            pd.DataFrame({"mol_id": test_ids, "prediction": best_model.predict(features[test_index])}),
            output / "predictions.parquet",
            metadata,
        )
    metadata["wall_time_seconds"] = time.perf_counter() - started
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
