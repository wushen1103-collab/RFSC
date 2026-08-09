#!/usr/bin/env python3
"""Select RF/GINE weights using validation pair-cliff error, then freeze test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import freeze_prediction_file, read_labels_for_ids, sha256_file


def verify(path: Path) -> None:
    manifest = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Frozen component hash mismatch: {path}")


def read_series(path: Path) -> pd.Series:
    verify(path)
    frame = pd.read_parquet(path)
    if set(frame.columns) != {"mol_id", "prediction"}:
        raise RuntimeError(f"Unexpected component prediction schema: {path}")
    return frame.set_index("mol_id")["prediction"].sort_index()


def read_gine_validation(path: Path, seed: int) -> pd.Series:
    verify(path)
    frame = pd.read_parquet(path)
    frame = frame[(frame["method"] == "base") & (frame["seed"] == seed)]
    return frame.set_index("mol_id")["prediction"].sort_index()


def validation_metrics(prediction: pd.Series, truth: pd.Series, edges: pd.DataFrame) -> dict:
    endpoints = set(edges["mol_id_lo"]) | set(edges["mol_id_hi"])
    endpoint_ids = [value for value in prediction.index if value in endpoints]
    predicted_delta = prediction.reindex(edges["mol_id_hi"]).to_numpy() - prediction.reindex(
        edges["mol_id_lo"]
    ).to_numpy()
    return {
        "overall_mae": float(np.mean(np.abs(prediction - truth.loc[prediction.index]))),
        "cliff_mae": float(np.mean(np.abs(prediction.loc[endpoint_ids] - truth.loc[endpoint_ids]))),
        "pair_delta_mae": float(np.mean(np.abs(predicted_delta - edges["delta_y"].to_numpy()))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    components = config["components"]
    candidates = [float(value) for value in config["selection"]["rf_weight_candidates"]]
    tasks = []
    for path in sorted(Path(components["rf"]).glob("*/fold_*/seed_*/predictions.parquet")):
        tasks.append((path.parts[-4], int(path.parts[-3].split("_")[-1]), int(path.parts[-2].split("_")[-1]), path))
    if len(tasks) != 87:
        raise RuntimeError(f"Expected 87 RF component runs, observed {len(tasks)}")
    if args.smoke:
        tasks = tasks[:1]
    records = []
    for target, fold, seed, rf_test_path in tasks:
        run_root = rf_test_path.parent
        rf_validation = read_series(run_root / "validation_predictions.parquet")
        gine_validation = read_gine_validation(
            Path(components["gine_validation_source"]) / target / f"fold_{fold}" / "validation_predictions.parquet",
            seed,
        )
        rf_test = read_series(rf_test_path)
        gine_test = read_series(
            Path(components["gine"]) / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet"
        )
        if not rf_validation.index.equals(gine_validation.index) or not rf_test.index.equals(gine_test.index):
            raise RuntimeError(f"RF/GINE partition mismatch: {target} fold {fold} seed {seed}")
        truth = read_labels_for_ids(args.data_root, target, rf_validation.index).set_index("mol_id")["y_pactivity"]
        mmp = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        edges = mmp[mmp["mol_id_lo"].isin(truth.index) & mmp["mol_id_hi"].isin(truth.index)].copy()
        edges["delta_y"] = edges["mol_id_hi"].map(truth) - edges["mol_id_lo"].map(truth)
        cliffs = edges[edges["delta_y"].abs() >= 1.0]
        selections = []
        for weight in candidates:
            prediction = weight * rf_validation + (1.0 - weight) * gine_validation
            measures = validation_metrics(prediction, truth, cliffs)
            objective = (
                float(config["selection"]["pair_alpha"]) * measures["pair_delta_mae"]
                + float(config["selection"]["endpoint_alpha"]) * measures["cliff_mae"]
                + float(config["selection"]["overall_alpha"]) * measures["overall_mae"]
            )
            selections.append({"rf_weight": weight, **measures, "selection_objective": objective})
        selection = pd.DataFrame(selections)
        selection["equal_distance"] = (selection["rf_weight"] - 0.5).abs()
        candidate_selection = selection
        if bool(config["selection"].get("guarded_by_fixed_equal", False)):
            equal_row = selection.loc[selection["rf_weight"] == 0.5].iloc[0]
            guarded = selection[
                (selection["overall_mae"] <= equal_row["overall_mae"])
                & (selection["cliff_mae"] <= equal_row["cliff_mae"])
            ]
            candidate_selection = guarded if not guarded.empty else selection.loc[
                selection["rf_weight"] == 0.5
            ]
        candidate_selection = candidate_selection.sort_values(
            ["selection_objective", "equal_distance", "rf_weight"], kind="stable"
        )
        selected_weight = float(candidate_selection.iloc[0]["rf_weight"])
        output = args.output_dir / target / f"fold_{fold}" / f"seed_{seed}"
        output.mkdir(parents=True, exist_ok=True)
        selection.drop(columns="equal_distance").to_csv(output / "weight_selection.csv", index=False)
        metadata = {
            "target": target,
            "fold": fold,
            "seed": seed,
            "method": config["method"],
            "validation_labels_materialized": True,
            "outer_test_labels_materialized": False,
            "selected_rf_weight": selected_weight,
            "fixed_equal_rf_weight": float(config["controls"]["fixed_equal_rf_weight"]),
            "selection_objective": config["selection"]["objective"],
            "guarded_by_fixed_equal": bool(config["selection"].get("guarded_by_fixed_equal", False)),
            "smoke": args.smoke,
        }
        for method, weight in [
            (str(config["selection"].get("output_method", "pair_selected")), selected_weight),
            ("fixed_equal", float(config["controls"]["fixed_equal_rf_weight"])),
        ]:
            prediction = weight * rf_test + (1.0 - weight) * gine_test
            digest = freeze_prediction_file(
                pd.DataFrame({"mol_id": prediction.index, "prediction": prediction.to_numpy()}),
                output / method / "predictions.parquet",
                {**metadata, "ensemble_method": method, "rf_weight": weight},
            )
            metadata[f"{method}_prediction_sha256"] = digest
        (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        records.append(metadata)
    pd.DataFrame(records).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(json.dumps({"runs": len(records), "smoke": args.smoke}, indent=2))


if __name__ == "__main__":
    main()
