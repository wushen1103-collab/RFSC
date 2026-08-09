#!/usr/bin/env python3
"""Freeze validation-selected train-centered calibration of fixed RF/GINE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import freeze_prediction_file, read_labels_for_ids, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, default=Path("folds/scg"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def verify(path: Path) -> str:
    manifest_path = path.with_suffix(path.suffix + ".sha256.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = sha256_file(path)
    if digest != manifest["prediction_sha256"]:
        raise RuntimeError(f"Frozen component hash mismatch: {path}")
    return digest


def read_series(path: Path) -> tuple[pd.Series, str]:
    digest = verify(path)
    frame = pd.read_parquet(path)
    if set(frame.columns) != {"mol_id", "prediction"}:
        raise RuntimeError(f"Unexpected component prediction schema: {path}")
    if frame["mol_id"].duplicated().any():
        raise RuntimeError(f"Duplicate component molecule: {path}")
    return frame.set_index("mol_id")["prediction"].sort_index(), digest


def gine_validation(path: Path, seed: int) -> tuple[pd.Series, str]:
    digest = verify(path)
    frame = pd.read_parquet(path)
    frame = frame[(frame["method"] == "base") & (frame["seed"] == seed)]
    if frame.empty or frame["mol_id"].duplicated().any():
        raise RuntimeError(f"Invalid GINE validation predictions: {path}")
    return frame.set_index("mol_id")["prediction"].sort_index(), digest


def split_ids(fold_path: Path, fold: int) -> tuple[list[str], list[str], list[str]]:
    assignments = pd.read_parquet(fold_path)
    n_folds = int(assignments["outer_fold"].nunique())
    validation_fold = (fold + 1) % n_folds
    train = assignments.loc[
        ~assignments["outer_fold"].isin([fold, validation_fold]), "mol_id"
    ].tolist()
    validation = assignments.loc[
        assignments["outer_fold"] == validation_fold, "mol_id"
    ].tolist()
    test = assignments.loc[assignments["outer_fold"] == fold, "mol_id"].tolist()
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise RuntimeError("SCG partition overlap")
    return train, validation, test


def scale_prediction(prediction: pd.Series, center: float, scale: float) -> pd.Series:
    return center + scale * (prediction - center)


def validation_metrics(
    prediction: pd.Series, truth: pd.Series, cliffs: pd.DataFrame
) -> tuple[float, float, float]:
    endpoints = set(cliffs["mol_id_lo"]) | set(cliffs["mol_id_hi"])
    endpoint_ids = [mol_id for mol_id in prediction.index if mol_id in endpoints]
    if not endpoint_ids or cliffs.empty:
        raise RuntimeError("Validation fold contains no internal cliff pairs")
    expected = truth.loc[prediction.index]
    predicted_delta = (
        prediction.reindex(cliffs["mol_id_hi"]).to_numpy()
        - prediction.reindex(cliffs["mol_id_lo"]).to_numpy()
    )
    return (
        float(np.mean(np.abs(prediction - expected))),
        float(np.mean(np.abs(prediction.loc[endpoint_ids] - truth.loc[endpoint_ids]))),
        float(np.mean(np.abs(predicted_delta - cliffs["delta_y"].to_numpy()))),
    )


def scale_name(scale: float) -> str:
    return f"scale_{scale:.2f}".replace(".", "p")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ensemble_root = Path(config["components"]["fixed_equal_ensemble"])
    rf_root = Path(config["components"]["rf"])
    gine_validation_root = Path(config["components"]["gine_validation_source"])
    candidates = [float(value) for value in config["selection"]["scale_candidates"]]
    identity = float(config["controls"]["identity_scale"])
    if identity not in candidates or min(candidates) < 1.0 or max(candidates) > 2.0:
        raise RuntimeError("Scale grid violates the preregistered bounded identity safeguard")
    alpha = float(config["selection"]["overall_alpha"])
    beta = float(config["selection"]["pair_delta_beta"])

    tasks = []
    for path in sorted(ensemble_root.glob("*/fold_*/seed_*/fixed_equal/predictions.parquet")):
        target = path.parts[-5]
        fold = int(path.parts[-4].split("_")[-1])
        seed = int(path.parts[-3].split("_")[-1])
        tasks.append((target, fold, seed, path))
    if len(tasks) != 87:
        raise RuntimeError(f"Expected 87 fixed-equal runs, observed {len(tasks)}")
    if args.smoke:
        tasks = tasks[:1]

    records = []
    for target, fold, seed, fixed_test_path in tasks:
        train_ids, validation_ids, test_ids = split_ids(
            args.fold_root / target / "outer_folds.parquet", fold
        )
        fixed_test, fixed_test_hash = read_series(fixed_test_path)
        rf_validation, rf_validation_hash = read_series(
            rf_root / target / f"fold_{fold}" / f"seed_{seed}" / "validation_predictions.parquet"
        )
        gine_valid, gine_validation_hash = gine_validation(
            gine_validation_root / target / f"fold_{fold}" / "validation_predictions.parquet",
            seed,
        )
        if not rf_validation.index.equals(gine_valid.index):
            raise RuntimeError(f"Validation component mismatch: {target} fold {fold} seed {seed}")
        if set(rf_validation.index) != set(validation_ids) or set(fixed_test.index) != set(test_ids):
            raise RuntimeError(f"Prediction/SCG partition mismatch: {target} fold {fold} seed {seed}")

        train_truth = read_labels_for_ids(args.data_root, target, train_ids).set_index("mol_id")[
            "y_pactivity"
        ]
        validation_truth = read_labels_for_ids(
            args.data_root, target, validation_ids
        ).set_index("mol_id")["y_pactivity"]
        center = float(train_truth.median())
        fixed_validation = 0.5 * rf_validation + 0.5 * gine_valid
        mmp = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        internal = mmp[
            mmp["mol_id_lo"].isin(validation_truth.index)
            & mmp["mol_id_hi"].isin(validation_truth.index)
        ].copy()
        internal["delta_y"] = (
            internal["mol_id_hi"].map(validation_truth)
            - internal["mol_id_lo"].map(validation_truth)
        )
        cliffs = internal[internal["delta_y"].abs() >= 1.0]

        selection_rows = []
        for scale in candidates:
            prediction = scale_prediction(fixed_validation, center, scale)
            overall, cliff, pair_delta = validation_metrics(
                prediction, validation_truth, cliffs
            )
            selection_rows.append(
                {
                    "scale": scale,
                    "validation_overall_mae": overall,
                    "validation_cliff_mae": cliff,
                    "validation_pair_delta_mae": pair_delta,
                    "selection_objective": cliff + alpha * overall + beta * pair_delta,
                }
            )
        selection = pd.DataFrame(selection_rows)
        selection["identity_distance"] = (selection["scale"] - identity).abs()
        selection = selection.sort_values(
            ["selection_objective", "identity_distance", "scale"], kind="stable"
        )
        selected_scale = float(selection.iloc[0]["scale"])
        output = args.output_dir / target / f"fold_{fold}" / f"seed_{seed}"
        output.mkdir(parents=True, exist_ok=True)
        selection.drop(columns="identity_distance").to_csv(
            output / "scale_selection.csv", index=False
        )
        metadata = {
            "target": target,
            "fold": fold,
            "seed": seed,
            "method": config["method"],
            "train_labels_materialized_for_center": True,
            "validation_labels_materialized_for_selection": True,
            "outer_test_labels_materialized": False,
            "selected_scale": selected_scale,
            "identity_scale": identity,
            "train_center": center,
            "n_train_center": len(train_truth),
            "component_prediction_hashes_verified": True,
            "fixed_equal_test_prediction_sha256": fixed_test_hash,
            "rf_validation_prediction_sha256": rf_validation_hash,
            "gine_validation_prediction_sha256": gine_validation_hash,
            "smoke": args.smoke,
        }
        for label, scale in [
            *( (scale_name(value), value) for value in candidates ),
            ("validation_selected", selected_scale),
        ]:
            prediction = scale_prediction(fixed_test, center, scale)
            path = output / label / "predictions.parquet"
            digest = freeze_prediction_file(
                pd.DataFrame(
                    {"mol_id": prediction.index, "prediction": prediction.to_numpy()}
                ),
                path,
                {
                    **metadata,
                    "calibration_output": label,
                    "scale": scale,
                    "outer_test_labels_materialized": False,
                },
            )
            metadata[f"{label}_prediction_sha256"] = digest
        (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        records.append(metadata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(
        json.dumps(
            {
                "runs": len(records),
                "smoke": args.smoke,
                "outer_test_labels_materialized": False,
                "selected_scales": pd.Series(
                    [row["selected_scale"] for row in records]
                ).value_counts().sort_index().to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

