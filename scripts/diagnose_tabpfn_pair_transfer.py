#!/usr/bin/env python3
"""Explain TabPFN cliff-pair failures without selecting any parameter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bps_gnn.io import read_labels_for_ids, sha256_file


def verified(path: Path, *, method: str | None = None, seed: int | None = None) -> pd.Series:
    manifest = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Prediction hash mismatch: {path}")
    frame = pd.read_parquet(path)
    if method is not None:
        frame = frame[(frame["method"] == method) & (frame["seed"] == seed)]
    if frame["mol_id"].duplicated().any():
        raise RuntimeError(f"Duplicate prediction ID: {path}")
    return frame.set_index("mol_id")["prediction"].sort_index()


def corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rf-root", type=Path, default=Path("experiments/rf_scg/full"))
    parser.add_argument("--gine-root", type=Path, default=Path("experiments/sqrl_anchor/full"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stem = Path(args.target) / f"fold_{args.fold}" / f"seed_{args.seed}"
    candidate = verified(args.candidate_root / stem / "validation_predictions.parquet")
    rf = verified(
        args.rf_root / args.target / f"fold_{args.fold}" / "seed_0" / "validation_predictions.parquet"
    )
    gine = verified(
        args.gine_root / args.target / f"fold_{args.fold}" / "validation_predictions.parquet",
        method="base",
        seed=0,
    )
    if not candidate.index.equals(rf.index) or not candidate.index.equals(gine.index):
        raise RuntimeError("Validation molecule mismatch")
    truth = read_labels_for_ids(args.data_root, args.target, candidate.index).set_index("mol_id")["y_pactivity"]
    edges = pd.read_parquet(args.mmp_root / args.target / "eligible_mmp_edges.parquet")
    edges = edges[edges["mol_id_lo"].isin(truth.index) & edges["mol_id_hi"].isin(truth.index)].copy()
    delta = edges["mol_id_hi"].map(truth) - edges["mol_id_lo"].map(truth)
    cliffs = edges[delta.abs() >= 1.0]
    true_delta = (cliffs["mol_id_hi"].map(truth) - cliffs["mol_id_lo"].map(truth)).to_numpy(float)
    methods = {
        "gine": gine,
        "candidate_tabpfn": candidate,
        "fixed_rf_gine": 0.5 * rf + 0.5 * gine,
        "equal_rf_gine_tabpfn": (rf + gine + candidate) / 3.0,
    }
    rows = []
    for name, prediction in methods.items():
        pred_delta = (cliffs["mol_id_hi"].map(prediction) - cliffs["mol_id_lo"].map(prediction)).to_numpy(float)
        rows.append(
            {
                "target": args.target,
                "fold": args.fold,
                "seed": args.seed,
                "method": name,
                "n_cliff_pairs": len(cliffs),
                "true_mean_abs_delta": float(np.mean(np.abs(true_delta))),
                "predicted_mean_abs_delta": float(np.mean(np.abs(pred_delta))),
                "pair_delta_pearson": corr(pred_delta, true_delta),
                "pair_delta_sign_accuracy": float(np.mean(np.sign(pred_delta) == np.sign(true_delta))),
                "pair_delta_mean_bias": float(np.mean(pred_delta - true_delta)),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
