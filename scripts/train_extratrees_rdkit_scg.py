#!/usr/bin/env python3
"""Validation-only RDKit-descriptor ExtraTrees strict-SCG smoke."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

from bps_gnn.fingerprints import fit_clean_rdkit_descriptors, rdkit_descriptor_matrix
from bps_gnn.io import freeze_prediction_file, read_labels_for_ids, read_structure_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def split_ids(target: str, fold: int) -> tuple[list[str], list[str], list[str]]:
    assignments = pd.read_parquet(Path("folds/scg") / target / "outer_folds.parquet")
    n_folds = int(assignments["outer_fold"].nunique())
    validation_fold = (fold + 1) % n_folds
    train_ids = assignments.loc[
        ~assignments["outer_fold"].isin([fold, validation_fold]), "mol_id"
    ].tolist()
    valid_ids = assignments.loc[assignments["outer_fold"] == validation_fold, "mol_id"].tolist()
    test_ids = assignments.loc[assignments["outer_fold"] == fold, "mol_id"].tolist()
    if set(train_ids) & set(valid_ids) or set(train_ids) & set(test_ids) or set(valid_ids) & set(test_ids):
        raise RuntimeError("SCG partition overlap")
    return train_ids, valid_ids, test_ids


def task_is_authorized(evaluation: dict, target: str, fold: int, seed: int) -> bool:
    tasks = evaluation.get("tasks")
    if tasks is None:
        tasks = [evaluation]
    return any(
        target == task["target"]
        and fold == int(task["fold"])
        and seed == int(task["seed"])
        for task in tasks
    )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    evaluation = config["evaluation"]
    if not task_is_authorized(evaluation, args.target, args.fold, args.seed):
        raise ValueError("Run is outside the preregistered smoke point")
    structures = read_structure_table(Path(config["data_root"]), args.target).set_index("mol_id")
    train_ids, valid_ids, test_ids = split_ids(args.target, args.fold)
    authorized = [*train_ids, *valid_ids]
    labels = read_labels_for_ids(Path(config["data_root"]), args.target, authorized).set_index("mol_id")[
        "y_pactivity"
    ]
    if set(labels.index) & set(test_ids):
        raise RuntimeError("Outer-test response entered ExtraTrees trainer")

    ordered_ids = [*train_ids, *valid_ids]
    raw, descriptor_names = rdkit_descriptor_matrix(
        structures.loc[ordered_ids, "canonical_isomeric_smiles"].tolist()
    )
    raw_train = raw[: len(train_ids)]
    raw_valid = raw[len(train_ids) :]
    x_train, x_valid, medians, usable = fit_clean_rdkit_descriptors(raw_train, raw_valid)
    y_train = labels.loc[train_ids].to_numpy(dtype=float)
    y_valid = labels.loc[valid_ids].to_numpy(dtype=float)

    model_config = config["model"]
    model = ExtraTreesRegressor(
        n_estimators=int(model_config["n_estimators"]),
        criterion=str(model_config["criterion"]),
        max_features=float(model_config["max_features"]),
        min_samples_leaf=int(model_config["min_samples_leaf"]),
        bootstrap=bool(model_config["bootstrap"]),
        random_state=args.seed,
        n_jobs=int(model_config["n_jobs"]),
    )
    model.fit(x_train, y_train)
    prediction = model.predict(x_valid)
    output = args.output_dir / args.target / f"fold_{args.fold}" / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "checkpoint.pkl").open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "descriptor_names": descriptor_names,
                "train_medians": medians,
                "usable_mask": usable,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    metadata = {
        "target": args.target,
        "fold": args.fold,
        "seed": args.seed,
        "method": config["method"],
        "protocol": config["protocol"],
        "n_train": len(train_ids),
        "n_validation": len(valid_ids),
        "n_test": len(test_ids),
        "n_estimators": int(model_config["n_estimators"]),
        "raw_descriptor_dimension": int(raw.shape[1]),
        "usable_descriptor_dimension": int(usable.sum()),
        "n_nonfinite_train_values": int((~np.isfinite(raw_train)).sum()),
        "n_float32_clipped_train_values": int(
            (np.abs(raw_train) > np.finfo(np.float32).max).sum()
        ),
        "n_float32_clipped_validation_values": int(
            (np.abs(raw_valid) > np.finfo(np.float32).max).sum()
        ),
        "validation_rmse": float(mean_squared_error(y_valid, prediction) ** 0.5),
        "validation_mae": float(np.mean(np.abs(y_valid - prediction))),
        "validation_labels_materialized": True,
        "outer_test_labels_materialized": False,
        "outer_test_predictions_generated": False,
        "validation_only": True,
    }
    metadata["validation_prediction_sha256"] = freeze_prediction_file(
        pd.DataFrame({"mol_id": valid_ids, "prediction": prediction}),
        output / "validation_predictions.parquet",
        metadata,
    )
    metadata["wall_time_seconds"] = time.perf_counter() - started
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
