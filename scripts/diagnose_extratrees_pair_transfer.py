#!/usr/bin/env python3
"""Post-hoc direction/amplitude diagnosis for frozen ExtraTrees replications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_labels_for_ids, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--extra-root", type=Path, required=True)
    parser.add_argument("--rf-root", type=Path, default=Path("experiments/rf_scg/full"))
    parser.add_argument("--gine-root", type=Path, default=Path("experiments/sqrl_anchor/full"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--output-table", type=Path, required=True)
    return parser.parse_args()


def verified(path: Path, method: str | None = None, seed: int | None = None) -> pd.Series:
    manifest = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Prediction hash mismatch: {path}")
    frame = pd.read_parquet(path)
    if method is not None:
        frame = frame[(frame["method"] == method) & (frame["seed"] == seed)]
    if frame["mol_id"].duplicated().any():
        raise RuntimeError(f"Duplicate prediction ID: {path}")
    return frame.set_index("mol_id")["prediction"].sort_index()


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = []
    for task in config["evaluation"]["tasks"]:
        target, fold, seed = task["target"], int(task["fold"]), int(task["seed"])
        extra = verified(
            args.extra_root / target / f"fold_{fold}" / f"seed_{seed}" / "validation_predictions.parquet"
        )
        rf = verified(
            args.rf_root / target / f"fold_{fold}" / f"seed_{seed}" / "validation_predictions.parquet"
        )
        gine = verified(
            args.gine_root / target / f"fold_{fold}" / "validation_predictions.parquet",
            method="base",
            seed=seed,
        )
        if not extra.index.equals(rf.index) or not extra.index.equals(gine.index):
            raise RuntimeError(f"Validation molecule mismatch: {target}")
        truth = read_labels_for_ids(Path(config["data_root"]), target, extra.index).set_index("mol_id")[
            "y_pactivity"
        ]
        edges = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        edges = edges[
            edges["mol_id_lo"].isin(truth.index) & edges["mol_id_hi"].isin(truth.index)
        ].copy()
        true_delta = edges["mol_id_hi"].map(truth) - edges["mol_id_lo"].map(truth)
        cliffs = edges[true_delta.abs() >= 1.0].copy()
        true_delta = (
            cliffs["mol_id_hi"].map(truth) - cliffs["mol_id_lo"].map(truth)
        ).to_numpy(dtype=float)
        methods = {
            "rf": rf,
            "gine": gine,
            "rdkit_extratrees": extra,
            "reference_rf_gine_equal": 0.5 * rf + 0.5 * gine,
            "preregistered_diversified": 0.25 * rf + 0.25 * extra + 0.5 * gine,
        }
        for method, prediction in methods.items():
            predicted_delta = (
                cliffs["mol_id_hi"].map(prediction) - cliffs["mol_id_lo"].map(prediction)
            ).to_numpy(dtype=float)
            rows.append(
                {
                    "target": target,
                    "fold": fold,
                    "seed": seed,
                    "method": method,
                    "n_cliff_pairs": len(cliffs),
                    "true_mean_abs_delta": float(np.mean(np.abs(true_delta))),
                    "predicted_mean_abs_delta": float(np.mean(np.abs(predicted_delta))),
                    "pair_delta_pearson": safe_correlation(predicted_delta, true_delta),
                    "pair_delta_sign_accuracy": float(np.mean(np.sign(predicted_delta) == np.sign(true_delta))),
                    "pair_delta_mean_bias": float(np.mean(predicted_delta - true_delta)),
                }
            )
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_table, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
