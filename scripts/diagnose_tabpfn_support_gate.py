#!/usr/bin/env python3
"""Post-hoc fixed support gate diagnostic; no parameter is selected."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bps_gnn.io import read_labels_for_ids, sha256_file


def read_pred(path: Path, *, method: str | None = None, seed: int | None = None) -> pd.Series:
    manifest = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Prediction hash mismatch: {path}")
    frame = pd.read_parquet(path)
    if method is not None:
        frame = frame[(frame["method"] == method) & (frame["seed"] == seed)]
    return frame.set_index("mol_id")["prediction"].sort_index()


def split_ids(target: str, fold: int) -> tuple[list[str], list[str]]:
    assignments = pd.read_parquet(Path("folds/scg") / target / "outer_folds.parquet")
    n_folds = int(assignments["outer_fold"].nunique())
    validation_fold = (fold + 1) % n_folds
    train = assignments.loc[~assignments["outer_fold"].isin([fold, validation_fold]), "mol_id"].tolist()
    valid = assignments.loc[assignments["outer_fold"] == validation_fold, "mol_id"].tolist()
    return train, valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--gine", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = read_pred(args.candidate)
    gine = read_pred(args.gine, method="base", seed=0)
    train_ids, valid_ids = split_ids(args.target, args.fold)
    if not candidate.index.equals(gine.index) or set(candidate.index) != set(valid_ids):
        raise RuntimeError("Validation molecule mismatch")
    edges = pd.read_parquet(Path("graphs/mmp") / args.target / "eligible_mmp_edges.parquet")
    train_set, valid_set = set(train_ids), set(valid_ids)
    support = {mol_id: 0 for mol_id in valid_ids}
    query_degree = {mol_id: 0 for mol_id in valid_ids}
    for left, right in zip(edges["mol_id_lo"], edges["mol_id_hi"]):
        if left in valid_set and right in train_set:
            support[left] += 1
        elif right in valid_set and left in train_set:
            support[right] += 1
        elif left in valid_set and right in valid_set:
            query_degree[left] += 1
            query_degree[right] += 1
    gated = candidate.copy()
    supported_ids = [mol_id for mol_id, count in query_degree.items() if count > 0]
    gated.loc[supported_ids] = gine.loc[supported_ids]
    truth = read_labels_for_ids(Path("data/processed/moleculeace"), args.target, valid_ids).set_index("mol_id")["y_pactivity"]
    cliffs = edges[edges["mol_id_lo"].isin(valid_set) & edges["mol_id_hi"].isin(valid_set)].copy()
    delta = cliffs["mol_id_hi"].map(truth) - cliffs["mol_id_lo"].map(truth)
    cliffs = cliffs[delta.abs() >= 1.0]
    rows = []
    for name, prediction in {"gine": gine, "candidate_tabpfn": candidate, "support_gated": gated}.items():
        err = prediction - truth.loc[prediction.index]
        pair = prediction.reindex(cliffs["mol_id_hi"]).to_numpy() - prediction.reindex(cliffs["mol_id_lo"]).to_numpy()
        true_pair = truth.reindex(cliffs["mol_id_hi"]).to_numpy() - truth.reindex(cliffs["mol_id_lo"]).to_numpy()
        rows.append(
            {
                "target": args.target,
                "fold": args.fold,
                "method": name,
                "supported_query_fraction": float(np.mean(np.asarray(list(query_degree.values())) > 0)),
                "overall_mae": float(np.mean(np.abs(err))),
                "cliff_endpoint_mae": float(np.mean(np.abs(prediction.loc[sorted(set(cliffs["mol_id_lo"]) | set(cliffs["mol_id_hi"]))] - truth.loc[sorted(set(cliffs["mol_id_lo"]) | set(cliffs["mol_id_hi"]))]))),
                "pair_delta_mae": float(np.mean(np.abs(pair - true_pair))),
                "n_cliff_pairs": int(len(cliffs)),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
