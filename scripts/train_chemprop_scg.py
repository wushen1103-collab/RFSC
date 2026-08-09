#!/usr/bin/env python3
"""Run official Chemprop D-MPNN with an explicit SCG label firewall."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import (
    freeze_prediction_file,
    read_labels_for_ids,
    read_structure_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def split_ids(target: str, fold: int, fold_root: Path) -> tuple[list[str], list[str], list[str]]:
    assignments = pd.read_parquet(fold_root / target / "outer_folds.parquet")
    n_folds = int(assignments.outer_fold.nunique())
    if fold not in set(assignments.outer_fold):
        raise ValueError(f"Invalid fold {fold} for {target}")
    validation_fold = (fold + 1) % n_folds
    train = assignments.loc[
        ~assignments.outer_fold.isin([fold, validation_fold]), "mol_id"
    ].tolist()
    valid = assignments.loc[assignments.outer_fold == validation_fold, "mol_id"].tolist()
    test = assignments.loc[assignments.outer_fold == fold, "mol_id"].tolist()
    return train, valid, test


def select_prediction_column(frame: pd.DataFrame) -> str:
    candidates = [
        column
        for column in frame.columns
        if column not in {"mol_id", "smiles"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one numeric Chemprop prediction column, found {candidates}")
    return candidates[0]


def select_checkpoint(model_dir: Path) -> Path:
    """Select Chemprop's validation-best checkpoint, never its last checkpoint."""
    checkpoints = sorted(model_dir.rglob("*.ckpt"))
    best = [path for path in checkpoints if path.name.startswith("best-epoch=")]
    if len(best) == 1:
        return best[0]
    non_last = [path for path in checkpoints if path.name != "last.ckpt"]
    if len(non_last) == 1:
        return non_last[0]
    raise RuntimeError(
        f"Expected exactly one validation-best Chemprop checkpoint, found {checkpoints}"
    )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError(f"Seed {args.seed} is not pre-registered")
    data_root = Path(config["data_root"])
    fold_root = Path(config["fold_root"])
    structures = read_structure_table(data_root, args.target)
    structure_map = structures.set_index("mol_id")["canonical_isomeric_smiles"]
    train_ids, valid_ids, test_ids = split_ids(args.target, args.fold, fold_root)
    authorized = [*train_ids, *valid_ids]
    labels = read_labels_for_ids(data_root, args.target, authorized).set_index("mol_id")
    run_dir = args.output_dir / args.target / f"fold_{args.fold}" / f"seed_{args.seed}"
    input_dir = run_dir / "firewall_input"
    model_dir = run_dir / "chemprop_model"
    input_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, ids in [("train", train_ids), ("val", valid_ids), ("test", test_ids)]:
        for mol_id in ids:
            rows.append(
                {
                    "mol_id": mol_id,
                    "smiles": structure_map.loc[mol_id],
                    "y": float(labels.loc[mol_id, "y_pactivity"])
                    if split != "test"
                    else np.nan,
                    "split": split,
                }
            )
    training_csv = input_dir / "train_valid_test_blank.csv"
    pd.DataFrame(rows).to_csv(training_csv, index=False)
    test_csv = input_dir / "test_structures_only.csv"
    pd.DataFrame(
        {
            "mol_id": test_ids,
            "smiles": [structure_map.loc[value] for value in test_ids],
        }
    ).to_csv(test_csv, index=False)

    chemprop = Path(sys.executable).with_name("chemprop")
    if not chemprop.exists():
        raise RuntimeError(f"Chemprop console entry point missing beside {sys.executable}")
    training = config["training"]
    command = [
        str(chemprop),
        "train",
        "-i",
        str(training_csv),
        "--smiles-columns",
        "smiles",
        "--target-columns",
        "y",
        "--ignore-columns",
        "mol_id",
        "--splits-column",
        "split",
        "--task-type",
        str(training["task_type"]),
        "--loss-function",
        str(training["loss_function"]),
        "--metrics",
        *[str(value) for value in training["metrics"]],
        "--epochs",
        str(training["epochs"]),
        "--patience",
        str(training["patience"]),
        "--batch-size",
        str(training["batch_size"]),
        "--num-workers",
        str(training["num_workers"]),
        "--pytorch-seed",
        str(args.seed),
        "--data-seed",
        str(args.seed),
        "--accelerator",
        "gpu" if args.device == "cuda" else "cpu",
        "--devices",
        "1",
        "--output-dir",
        str(model_dir),
    ]
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    trained = subprocess.run(command, text=True, capture_output=True, env=environment)
    (run_dir / "chemprop_train.stdout").write_text(trained.stdout, encoding="utf-8")
    (run_dir / "chemprop_train.stderr").write_text(trained.stderr, encoding="utf-8")
    if trained.returncode:
        raise RuntimeError(f"Chemprop training failed with code {trained.returncode}")
    checkpoint = select_checkpoint(model_dir)
    raw_prediction_path = run_dir / "chemprop_predictions.csv"
    predicted = subprocess.run(
        [
            str(chemprop),
            "predict",
            "-i",
            str(test_csv),
            "--smiles-columns",
            "smiles",
            "--model-path",
            str(checkpoint),
            "--accelerator",
            "gpu" if args.device == "cuda" else "cpu",
            "--devices",
            "1",
            "-o",
            str(raw_prediction_path),
        ],
        text=True,
        capture_output=True,
        env=environment,
    )
    (run_dir / "chemprop_predict.stdout").write_text(predicted.stdout, encoding="utf-8")
    (run_dir / "chemprop_predict.stderr").write_text(predicted.stderr, encoding="utf-8")
    if predicted.returncode:
        raise RuntimeError(f"Chemprop prediction failed with code {predicted.returncode}")
    prediction_frame = pd.read_csv(raw_prediction_path)
    prediction_column = select_prediction_column(prediction_frame)
    if prediction_frame.mol_id.tolist() != test_ids:
        raise RuntimeError("Chemprop prediction IDs or order changed")
    values = prediction_frame[prediction_column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("Chemprop emitted non-finite predictions")
    metadata = {
        "target": args.target,
        "fold": args.fold,
        "seed": args.seed,
        "method": config["method"],
        "architecture": "chemprop_dmpnn_v2",
        "chemprop_version": config["chemprop_version"],
        "n_train": len(train_ids),
        "n_valid": len(valid_ids),
        "n_test": len(test_ids),
        "test_labels_materialized": False,
        "checkpoint": str(checkpoint.relative_to(run_dir)),
        "wall_time_seconds": time.perf_counter() - started,
    }
    digest = freeze_prediction_file(
        pd.DataFrame({"mol_id": test_ids, "prediction": values}),
        run_dir / "predictions.parquet",
        metadata,
    )
    metadata["prediction_sha256"] = digest
    (run_dir / "run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
