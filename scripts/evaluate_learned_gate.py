#!/usr/bin/env python3
"""Evaluate frozen learned-gate query artifacts after SHA256 verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from bps_gnn.io import read_labels_for_ids, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--partition", choices=["validation", "test"], required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def verify(path: Path) -> None:
    manifest = json.loads(
        path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8")
    )
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Frozen artifact hash mismatch: {path}")


def partition_ids(target: str, fold: int, partition: str) -> tuple[list[str], list[str]]:
    assignments = pd.read_parquet(Path("folds/scg") / target / "outer_folds.parquet")
    n_folds = int(assignments["outer_fold"].nunique())
    validation_fold = (fold + 1) % n_folds
    train = assignments.loc[
        ~assignments["outer_fold"].isin([fold, validation_fold]), "mol_id"
    ].tolist()
    query_fold = validation_fold if partition == "validation" else fold
    query = assignments.loc[assignments["outer_fold"] == query_fold, "mol_id"].tolist()
    return train, query


def evaluate_predictions(
    frame: pd.DataFrame,
    truth: pd.Series,
    mmp_edges: pd.DataFrame,
) -> list[dict]:
    edges = mmp_edges[
        mmp_edges["mol_id_lo"].isin(truth.index)
        & mmp_edges["mol_id_hi"].isin(truth.index)
    ].copy()
    edges["delta_y"] = edges["mol_id_hi"].map(truth) - edges["mol_id_lo"].map(truth)
    cliff = edges[edges["delta_y"].abs() >= 1.0]
    endpoints = sorted(set(cliff["mol_id_lo"]) | set(cliff["mol_id_hi"]))
    rows = []
    for keys, group in frame.groupby(["method", "replicate", "seed"], sort=True):
        prediction = group.set_index("mol_id")["prediction"]
        if set(prediction.index) != set(truth.index):
            raise RuntimeError(f"Incomplete query predictions for {keys}")
        predicted_delta = cliff["mol_id_hi"].map(prediction) - cliff["mol_id_lo"].map(
            prediction
        )
        rows.append(
            {
                "method": keys[0],
                "replicate": int(keys[1]),
                "seed": int(keys[2]),
                "overall_mae": float((prediction.loc[truth.index] - truth).abs().mean()),
                "cliff_mae": float(
                    (prediction.loc[endpoints] - truth.loc[endpoints]).abs().mean()
                ),
                "pair_delta_mae": float(
                    (predicted_delta - cliff["delta_y"]).abs().mean()
                ),
                "n_query": int(len(truth)),
                "n_cliff_endpoints": int(len(endpoints)),
                "n_cliff_pairs": int(len(cliff)),
            }
        )
    return rows


def evaluate_gate(frame: pd.DataFrame, labels: pd.Series) -> list[dict]:
    rows = []
    for seed, group in frame.groupby("seed"):
        query_y = group["query_id"].map(labels).to_numpy()
        reference_y = group["reference_id"].map(labels).to_numpy()
        boundary = (np.abs(query_y - reference_y) >= 1.0).astype(int)
        probability = group["q_learned"].to_numpy()
        rows.append(
            {
                "seed": int(seed),
                "edges": int(len(group)),
                "boundary_prevalence": float(boundary.mean()) if len(boundary) else float("nan"),
                "auprc": float(average_precision_score(boundary, probability)) if len(boundary) else float("nan"),
                "auroc": float(roc_auc_score(boundary, probability)) if len(np.unique(boundary)) == 2 else float("nan"),
                "brier": float(brier_score_loss(boundary, probability)) if len(boundary) else float("nan"),
                "mean_q": float(probability.mean()) if len(probability) else float("nan"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    verify(args.predictions)
    verify(args.gate)
    predictions = pd.read_parquet(args.predictions)
    gate = pd.read_parquet(args.gate)
    if any(column.lower().startswith("y_") for column in [*predictions.columns, *gate.columns]):
        raise RuntimeError("Frozen learned-gate artifact contains a response")
    train_ids, query_ids = partition_ids(args.target, args.fold, args.partition)
    labels = read_labels_for_ids(args.data_root, args.target, [*train_ids, *query_ids]).set_index(
        "mol_id"
    )["y_pactivity"]
    truth = labels.loc[query_ids]
    edges = pd.read_parquet(args.mmp_root / args.target / "eligible_mmp_edges.parquet")
    prediction_rows = evaluate_predictions(predictions, truth, edges)
    gate_rows = evaluate_gate(gate, labels)
    for row in prediction_rows:
        row.update({"target": args.target, "fold": args.fold, "partition": args.partition})
    for row in gate_rows:
        row.update({"target": args.target, "fold": args.fold, "partition": args.partition})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(
        args.output_dir / f"{args.partition}_evaluation.csv", index=False
    )
    pd.DataFrame(gate_rows).to_csv(
        args.output_dir / f"{args.partition}_gate_metrics.csv", index=False
    )
    print(json.dumps({"partition": args.partition, "prediction_rows": len(prediction_rows), "gate_rows": len(gate_rows)}, indent=2))


if __name__ == "__main__":
    main()
