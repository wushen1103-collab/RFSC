#!/usr/bin/env python3
"""Build and hash-freeze response-free Hodge projections of DeepDelta flows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import freeze_prediction_file
from bps_gnn.pair_flow import hodge_project_complete_pair_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-datasets", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    datasets = [str(value) for value in config["datasets"]]
    if args.maximum_datasets is not None:
        datasets = datasets[: args.maximum_datasets]
    records: list[dict[str, object]] = []
    for dataset in datasets:
        for fold in range(1, int(config["cross_validation"]["folds"]) + 1):
            source = args.source_root / f"{dataset}_DeepDelta_Individual_Fold_{fold}.csv"
            frame = pd.read_csv(source)
            if len(frame) != 1:
                raise RuntimeError(f"{source}: expected one response-free prediction row")
            flat = frame.iloc[0].to_numpy(dtype=float)
            n_nodes = int(round(np.sqrt(len(flat))))
            if n_nodes * n_nodes != len(flat):
                raise RuntimeError(f"{source}: prediction count is not a square")
            flow = flat.reshape(n_nodes, n_nodes)
            potential, projected = hodge_project_complete_pair_flow(flow)
            if not np.allclose(projected, -projected.T, atol=1e-12):
                raise RuntimeError(f"{source}: projected flow is not antisymmetric")
            metadata = {
                "dataset": dataset,
                "fold": fold,
                "method": "hodge_cycle_consistent_pair_flow",
                "source_prediction": str(source),
                "source_prediction_sha256": sha256_file(source),
                "n_nodes": n_nodes,
                "test_labels_materialized": False,
                "hyperparameters": 0,
            }
            output = args.output_dir / dataset / f"fold_{fold}"
            digest = freeze_prediction_file(
                pd.DataFrame(
                    {
                        "mol_id": [f"position_{index}" for index in range(n_nodes)],
                        "prediction": potential,
                    }
                ),
                output / "potential.parquet",
                metadata,
            )
            metadata["potential_sha256"] = digest
            output.mkdir(parents=True, exist_ok=True)
            (output / "run.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
            )
            records.append(metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(
        json.dumps(
            {
                "datasets": len(datasets),
                "folds": len(records),
                "test_labels_materialized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

