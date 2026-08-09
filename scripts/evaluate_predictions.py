#!/usr/bin/env python3
"""Evaluate frozen predictions after verifying the test-label firewall hash."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bps_gnn.io import read_labels_for_ids, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--protocol", choices=["official", "scg"], required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def errors(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - y
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.predictions.with_suffix(args.predictions.suffix + ".sha256.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(args.predictions)
    if actual_hash != manifest["prediction_sha256"]:
        raise RuntimeError("Prediction hash mismatch; evaluator refuses test-label access")
    prediction = pd.read_parquet(args.predictions)
    if set(prediction.columns) != {"mol_id", "prediction"}:
        raise RuntimeError("Frozen prediction schema is not label-free")
    label_columns = ["mol_id", "y_pactivity"]
    if args.protocol == "official":
        label_columns.append("official_cliff_mol")
    labels = read_labels_for_ids(
        args.data_root, args.target, prediction["mol_id"], label_columns
    )
    evaluated = prediction.merge(labels, on="mol_id", validate="one_to_one")
    result = {
        "target": args.target,
        "protocol": args.protocol,
        "prediction_sha256": actual_hash,
        "n_test": len(evaluated),
        **{f"overall_{key}": value for key, value in errors(evaluated["y_pactivity"].to_numpy(), evaluated["prediction"].to_numpy()).items()},
    }
    if args.protocol == "official":
        cliff = evaluated[evaluated["official_cliff_mol"].astype(bool)]
        result["n_cliff_endpoints"] = len(cliff)
        result.update(
            {
                f"cliff_{key}": value
                for key, value in errors(
                    cliff["y_pactivity"].to_numpy(), cliff["prediction"].to_numpy()
                ).items()
            }
        )
    else:
        edges = pd.read_parquet(args.mmp_root / args.target / "eligible_mmp_edges.parquet")
        truth = evaluated.set_index("mol_id")["y_pactivity"]
        edges = edges[
            edges["mol_id_lo"].isin(truth.index) & edges["mol_id_hi"].isin(truth.index)
        ].copy()
        edges["delta_y"] = edges["mol_id_hi"].map(truth) - edges["mol_id_lo"].map(truth)
        cliff_edges = edges[edges["delta_y"].abs() >= 1.0]
        endpoint_ids = set(cliff_edges["mol_id_lo"]) | set(cliff_edges["mol_id_hi"])
        cliff = evaluated[evaluated["mol_id"].isin(endpoint_ids)]
        pred = evaluated.set_index("mol_id")["prediction"]
        predicted_delta = cliff_edges["mol_id_hi"].map(pred) - cliff_edges["mol_id_lo"].map(pred)
        result.update(
            {
                "n_mmp_pairs": int(len(edges)),
                "n_cliff_pairs": int(len(cliff_edges)),
                "n_cliff_endpoints": int(len(cliff)),
                **{
                    f"cliff_{key}": value
                    for key, value in errors(
                        cliff["y_pactivity"].to_numpy(), cliff["prediction"].to_numpy()
                    ).items()
                },
                "pair_delta_mae": float(np.mean(np.abs(predicted_delta - cliff_edges["delta_y"]))),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

