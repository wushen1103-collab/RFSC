#!/usr/bin/env python3
"""Select component-preserving scale on outer validation, then freeze outer-test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import freeze_prediction_file, read_labels_for_ids, sha256_file
try:
    from build_rf_gine_component_scale import scale_components
except ModuleNotFoundError:
    from scripts.build_rf_gine_component_scale import scale_components


def read_prediction(path: Path) -> pd.Series:
    manifest = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Prediction hash mismatch: {path}")
    return pd.read_parquet(path).set_index("mol_id")["prediction"].sort_index()


def metrics(prediction: pd.Series, truth: pd.Series, cliffs: pd.DataFrame) -> dict:
    endpoints = set(cliffs.mol_id_lo) | set(cliffs.mol_id_hi)
    endpoint_ids = [value for value in prediction.index if value in endpoints]
    delta = prediction.reindex(cliffs.mol_id_hi).to_numpy() - prediction.reindex(cliffs.mol_id_lo).to_numpy()
    return {
        "overall_mae": float(np.mean(abs(prediction - truth.loc[prediction.index]))),
        "cliff_mae": float(np.mean(abs(prediction.loc[endpoint_ids] - truth.loc[endpoint_ids]))),
        "pair_delta_mae": float(np.mean(abs(delta - cliffs.delta_y.to_numpy()))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ensemble-dir", type=Path, default=Path("experiments/rf_gine_ensemble/full"))
    parser.add_argument("--rf-dir", type=Path, default=Path("experiments/rf_scg/full"))
    parser.add_argument("--gine-validation-source", type=Path, default=Path("experiments/sqrl_anchor/full"))
    parser.add_argument("--fold-root", type=Path, default=Path("folds/scg"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    scales = [float(value) for value in config["operator"]["scale_candidates"]]
    paths = sorted(args.ensemble_dir.glob("*/fold_*/seed_*/fixed_equal/predictions.parquet"))
    if len(paths) != 87:
        raise RuntimeError(f"Expected 87 fixed-equal runs, observed {len(paths)}")
    records = []
    for test_path in paths:
        target, fold, seed = test_path.parts[-5], int(test_path.parts[-4].split("_")[-1]), int(test_path.parts[-3].split("_")[-1])
        assignments = pd.read_parquet(args.fold_root / target / "outer_folds.parquet")
        n_folds = int(assignments["outer_fold"].nunique())
        validation_fold = (fold + 1) % n_folds
        validation_ids = set(assignments.loc[assignments.outer_fold == validation_fold, "mol_id"])
        test_ids = set(assignments.loc[assignments.outer_fold == fold, "mol_id"])
        rf_valid = read_prediction(args.rf_dir / target / f"fold_{fold}" / f"seed_{seed}" / "validation_predictions.parquet")
        gine_frame = pd.read_parquet(args.gine_validation_source / target / f"fold_{fold}" / "validation_predictions.parquet")
        gine_valid = gine_frame[(gine_frame.method == "base") & (gine_frame.seed == seed)].set_index("mol_id")["prediction"].sort_index()
        validation_base = 0.5 * rf_valid + 0.5 * gine_valid
        if set(validation_base.index) != validation_ids:
            raise RuntimeError(
                f"Validation partition mismatch: {target} fold {fold} seed {seed} "
                f"missing={len(validation_ids - set(validation_base.index))} "
                f"extra={len(set(validation_base.index) - validation_ids)}"
            )
        truth = read_labels_for_ids(args.data_root, target, validation_ids).set_index("mol_id")["y_pactivity"]
        edges = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        validation_edges = edges[edges.mol_id_lo.isin(validation_ids) & edges.mol_id_hi.isin(validation_ids)].copy()
        validation_edges["delta_y"] = validation_edges.mol_id_hi.map(truth) - validation_edges.mol_id_lo.map(truth)
        cliffs = validation_edges[validation_edges.delta_y.abs() >= 1.0]
        identity = metrics(validation_base, truth, cliffs)
        selection = []
        for scale in scales:
            prediction = scale_components(validation_base, validation_edges, scale)
            measures = metrics(prediction, truth, cliffs)
            objective = measures["pair_delta_mae"] + 0.25 * measures["cliff_mae"] + 0.25 * measures["overall_mae"]
            selection.append({"scale": scale, **measures, "selection_objective": objective, "safe": measures["overall_mae"] <= identity["overall_mae"] and measures["cliff_mae"] <= identity["cliff_mae"]})
        selected_table = pd.DataFrame(selection)
        safe = selected_table[selected_table.safe]
        pool = safe if not safe.empty else selected_table[selected_table.scale == 1.0]
        selected = pool.assign(identity_distance=(pool.scale - 1.0).abs()).sort_values(["selection_objective", "identity_distance", "scale"], kind="stable").iloc[0]
        selected_scale = float(selected.scale)
        base_test = read_prediction(test_path)
        if set(base_test.index) != test_ids:
            raise RuntimeError("Test partition mismatch")
        test_edges = edges[edges.mol_id_lo.isin(test_ids) & edges.mol_id_hi.isin(test_ids)]
        candidate = scale_components(base_test, test_edges, selected_scale)
        run_dir = args.output_dir / target / f"fold_{fold}" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        selected_table.to_csv(run_dir / "scale_selection.csv", index=False)
        metadata = {"target": target, "fold": fold, "seed": seed, "selected_scale": selected_scale, "test_labels_materialized": False, "validation_labels_materialized_for_selection": True}
        for method, prediction in [("fixed_equal", base_test), ("component_scaled_selected", candidate)]:
            freeze_prediction_file(pd.DataFrame({"mol_id": prediction.index, "prediction": prediction.to_numpy()}), run_dir / method / "predictions.parquet", {**metadata, "method": method})
        (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        records.append(metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(json.dumps({"runs": len(records), "selected_scales": pd.Series([row["selected_scale"] for row in records]).value_counts().sort_index().to_dict()}, indent=2))


if __name__ == "__main__":
    main()
