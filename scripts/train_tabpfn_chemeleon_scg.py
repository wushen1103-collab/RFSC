#!/usr/bin/env python3
"""Validation-only TabPFN on frozen CheMeleon fingerprints.

This adapter follows the public TabPFN-CheMeleonFP recipe: frozen 2048-d
CheMeleon fingerprints, train-only standard scaling for features and targets,
feature clipping to [-6, 6], and ``TabPFNRegressor`` with the preregistered
seed.  Outer-test IDs are used only for partition checks and never for labels
or prediction materialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN
from fastprop.data import inverse_standard_scale, standard_scale
from rdkit.Chem import MolFromSmiles
from tabpfn import TabPFNRegressor

from bps_gnn.io import freeze_prediction_file, read_labels_for_ids, read_structure_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full-validation", action="store_true")
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


class CheMeleonFingerprinter:
    """Small explicit-path wrapper around the authors' reference implementation."""

    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
        message_passing = nn.BondMessagePassing(**payload["hyper_parameters"])
        message_passing.load_state_dict(payload["state_dict"])
        self.model = MPNN(
            message_passing=message_passing,
            agg=nn.MeanAggregation(),
            predictor=RegressionFFN(input_dim=message_passing.output_dim),
        ).eval().to(device=device)
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.device = device

    def __call__(self, smiles: list[str]) -> np.ndarray:
        graph = BatchMolGraph([self.featurizer(MolFromSmiles(smi)) for smi in smiles])
        graph.to(device=self.device)
        with torch.no_grad():
            return self.model.fingerprint(graph).detach().cpu().numpy().astype(np.float32)


def smiles_to_features(smiles: list[str], fingerprinter: CheMeleonFingerprinter, batch_size: int) -> np.ndarray:
    valid_positions: list[int] = []
    valid_smiles: list[str] = []
    for index, value in enumerate(smiles):
        if isinstance(value, str) and MolFromSmiles(value) is not None:
            valid_positions.append(index)
            valid_smiles.append(value)
    if not valid_smiles:
        raise RuntimeError("No valid SMILES available for CheMeleon")
    batches = [
        fingerprinter(valid_smiles[start : start + batch_size])
        for start in range(0, len(valid_smiles), batch_size)
    ]
    valid_features = np.vstack(batches).astype(np.float32)
    features = np.zeros((len(smiles), valid_features.shape[1]), dtype=np.float32)
    features[np.asarray(valid_positions)] = valid_features
    return features


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    evaluation = config["evaluation"]
    if not (args.smoke ^ args.full_validation):
        raise ValueError("Select exactly one of --smoke or --full-validation")
    authorized_tasks = (
        evaluation["smoke_tasks"]
        if args.smoke
        else [
            *evaluation["full_validation_tasks"],
            *evaluation["replication_tasks"],
            *evaluation["support_gate_tasks"],
        ]
    )
    if not any(
        args.target == task["target"]
        and args.fold == int(task["fold"])
        and args.seed == int(task["seed"])
        for task in authorized_tasks
    ):
        raise ValueError("Run is outside the preregistered TabPFN task list")

    structures = read_structure_table(Path(config["data_root"]), args.target).set_index("mol_id")
    train_ids, valid_ids, test_ids = split_ids(args.target, args.fold)
    if args.smoke:
        fit_ids = train_ids[: int(evaluation["smoke_train_size"])]
        query_ids = train_ids[
            int(evaluation["smoke_train_size"])
            : int(evaluation["smoke_train_size"]) + int(evaluation["smoke_query_size"])
        ]
    else:
        fit_ids, query_ids = train_ids, valid_ids
    labels = read_labels_for_ids(Path(config["data_root"]), args.target, fit_ids).set_index("mol_id")
    if set(labels.index) & set(valid_ids + test_ids):
        raise RuntimeError("Smoke labels escaped the SCG training partition")

    ordered_ids = [*fit_ids, *query_ids]
    smiles = structures.loc[ordered_ids, "canonical_isomeric_smiles"].tolist()
    device = torch.device(config["chemeleon"]["device"])
    checkpoint = Path(config["chemeleon"]["checkpoint"]).expanduser().resolve()
    if checkpoint.stat().st_size != int(config["chemeleon"]["checkpoint_bytes"]):
        raise RuntimeError("CheMeleon checkpoint size mismatch")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha256 != config["chemeleon"]["checkpoint_sha256"]:
        raise RuntimeError("CheMeleon checkpoint hash mismatch")
    fingerprinter = CheMeleonFingerprinter(checkpoint, device)
    features = smiles_to_features(smiles, fingerprinter, int(config["chemeleon"]["batch_size"]))
    x_fit = torch.tensor(features[: len(fit_ids)], dtype=torch.float32)
    x_query = torch.tensor(features[len(fit_ids) :], dtype=torch.float32)
    y_fit = torch.tensor(labels.loc[fit_ids, "y_pactivity"].to_numpy(), dtype=torch.float32).reshape(-1, 1)

    _, target_means, target_vars = standard_scale(y_fit)
    y_scaled = standard_scale(y_fit, target_means, target_vars).numpy().ravel()
    _, feature_means, feature_vars = standard_scale(x_fit)
    x_fit_scaled = standard_scale(x_fit, feature_means, feature_vars).clamp(-6, 6).numpy()
    x_query_scaled = standard_scale(x_query, feature_means, feature_vars).clamp(-6, 6).numpy()

    tabpfn_config = config["tabpfn"]
    model = TabPFNRegressor(
        n_estimators=int(tabpfn_config["n_estimators_smoke"] if args.smoke else tabpfn_config["n_estimators_full"]),
        random_state=int(args.seed),
        ignore_pretraining_limits=True,
        device=tabpfn_config["device"],
        n_preprocessing_jobs=1,
    )
    model.fit(x_fit_scaled, y_scaled)
    prediction_scaled = np.asarray(model.predict(x_query_scaled), dtype=np.float32)
    prediction = inverse_standard_scale(
        torch.tensor(prediction_scaled).reshape(-1, 1), target_means, target_vars
    ).numpy().ravel()
    if prediction.shape != (len(query_ids),) or not np.isfinite(prediction).all():
        raise RuntimeError("TabPFN smoke prediction is invalid")

    output = args.output_dir / args.target / f"fold_{args.fold}" / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "target": args.target,
        "fold": args.fold,
        "seed": args.seed,
        "method": config["method"],
        "protocol": config["protocol"],
        "smoke": bool(args.smoke),
        "n_train": len(train_ids),
        "n_validation": len(valid_ids),
        "n_test": len(test_ids),
        "n_fit_used": len(fit_ids),
        "n_query_used": len(query_ids),
        "feature_dimension": int(features.shape[1]),
        "feature_preprocessing": "train_only_standard_scale_then_clip[-6,6]",
        "target_preprocessing": "train_only_standard_scale_then_inverse",
        "chemeleon_checkpoint": str(checkpoint),
        "chemeleon_checkpoint_sha256": checkpoint_sha256,
        "tabpfn_n_estimators": int(tabpfn_config["n_estimators_smoke"] if args.smoke else tabpfn_config["n_estimators_full"]),
        "tabpfn_ignore_pretraining_limits": True,
        "validation_labels_materialized": False,
        "outer_test_labels_materialized": False,
        "outer_test_predictions_generated": False,
        "validation_only": True,
    }
    prediction_name = "smoke_predictions.parquet" if args.smoke else "validation_predictions.parquet"
    metadata["prediction_sha256"] = freeze_prediction_file(
        pd.DataFrame({"mol_id": query_ids, "prediction": prediction}),
        output / prediction_name,
        metadata,
    )
    metadata["wall_time_seconds"] = time.perf_counter() - started
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
