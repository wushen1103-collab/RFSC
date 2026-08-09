#!/usr/bin/env python3
"""Fit the public-global-mode ECFP-SVM without reading validation/test labels."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.svm import SVR

from bps_gnn.fingerprints import ecfp_matrix
from bps_gnn.io import freeze_prediction_file, read_labels_for_ids, read_structure_table
try:
    from train_rf_scg import split_ids
except ModuleNotFoundError:
    from scripts.train_rf_scg import split_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    structures = read_structure_table(Path(config["data_root"]), args.target).reset_index(drop=True)
    train_ids, valid_ids, test_ids = split_ids(args.target, args.fold)
    if args.smoke:
        train_ids, valid_ids = train_ids[:128], valid_ids[:64]
    labels = read_labels_for_ids(Path(config["data_root"]), args.target, train_ids).set_index("mol_id")["y_pactivity"]
    if set(labels.index) & (set(valid_ids) | set(test_ids)):
        raise RuntimeError("Validation/test response entered SVM trainer")
    fingerprint = config["fingerprint"]
    features = ecfp_matrix(
        structures.canonical_isomeric_smiles.tolist(),
        radius=int(fingerprint["radius"]),
        n_bits=int(fingerprint["n_bits"]),
    )
    id_to_index = {mol_id: index for index, mol_id in enumerate(structures.mol_id)}
    index = lambda ids: np.asarray([id_to_index[mol_id] for mol_id in ids])
    model_config = config["model"]
    model = SVR(
        kernel=str(model_config["kernel"]),
        C=float(model_config["C"]),
        gamma=float(model_config["gamma"]),
        epsilon=float(model_config["epsilon"]),
        cache_size=float(model_config["cache_size_mb"]),
    )
    model.fit(features[index(train_ids)], labels.loc[train_ids].to_numpy(dtype=float))
    output = args.output_dir / args.target / f"fold_{args.fold}" / "seed_0"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "checkpoint.pkl").open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "target": args.target,
        "fold": args.fold,
        "seed": 0,
        "method": config["method"],
        "protocol": config["protocol"],
        "smoke": bool(args.smoke),
        "deterministic": True,
        "n_train": len(train_ids),
        "n_validation": len(valid_ids),
        "n_test": len(test_ids),
        "feature_dimension": int(features.shape[1]),
        "validation_labels_materialized": False,
        "outer_test_labels_materialized": False,
        "outer_test_predictions_generated": not args.smoke,
        "parameters": {key: model.get_params()[key] for key in ["kernel", "C", "gamma", "epsilon"]},
    }
    metadata["validation_prediction_sha256"] = freeze_prediction_file(
        pd.DataFrame({"mol_id": valid_ids, "prediction": model.predict(features[index(valid_ids)])}),
        output / "validation_predictions.parquet",
        metadata,
    )
    if not args.smoke:
        metadata["prediction_sha256"] = freeze_prediction_file(
            pd.DataFrame({"mol_id": test_ids, "prediction": model.predict(features[index(test_ids)])}),
            output / "predictions.parquet",
            metadata,
        )
    metadata["wall_time_seconds"] = time.perf_counter() - started
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
