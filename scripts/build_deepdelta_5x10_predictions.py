#!/usr/bin/env python3
"""Freeze response-free Hodge potentials for official DeepDelta 5x10 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
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
    parser.add_argument("--maximum-repeats", type=int)
    return parser.parse_args()


def last_nonempty_line(path: Path, chunk_size: int = 64 * 1024) -> bytes:
    """Return only the final nonempty line; earlier CSV rows are never parsed."""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        cursor = handle.tell()
        suffix = b""
        while cursor > 0:
            start = max(0, cursor - chunk_size)
            handle.seek(start)
            suffix = handle.read(cursor - start) + suffix
            stripped = suffix.rstrip(b"\r\n")
            boundary = stripped.rfind(b"\n")
            if boundary >= 0:
                return stripped[boundary + 1 :].rstrip(b"\r")
            cursor = start
    if suffix.strip():
        return suffix.strip()
    raise RuntimeError(f"{path}: no nonempty line")


def load_prediction_row_only(path: Path) -> tuple[np.ndarray, str]:
    line = last_nonempty_line(path)
    values = np.fromstring(line.decode("utf-8"), sep=",")
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError(f"{path}: invalid final prediction row")
    return values, hashlib.sha256(line).hexdigest()


def infer_kfold_sizes(n_pair_predictions: int, n_folds: int) -> list[int]:
    upper = int(math.ceil(math.sqrt(n_pair_predictions * n_folds))) + n_folds
    matches: list[list[int]] = []
    for n_samples in range(n_folds, upper + 1):
        quotient, remainder = divmod(n_samples, n_folds)
        sizes = [quotient + 1] * remainder + [quotient] * (n_folds - remainder)
        if sum(size * size for size in sizes) == n_pair_predictions:
            matches.append(sizes)
    if len(matches) != 1:
        raise RuntimeError(
            f"cannot uniquely infer {n_folds}-fold sizes from {n_pair_predictions} values: "
            f"{matches}"
        )
    return matches[0]


def expected_ids(sizes: list[int]) -> list[str]:
    return [
        f"fold_{fold:02d}_position_{position:06d}"
        for fold, size in enumerate(sizes, start=1)
        for position in range(size)
    ]


def split_square_blocks(values: np.ndarray, sizes: list[int]) -> list[np.ndarray]:
    blocks: list[np.ndarray] = []
    offset = 0
    for size in sizes:
        stop = offset + size * size
        blocks.append(values[offset:stop].reshape(size, size))
        offset = stop
    if offset != len(values):
        raise RuntimeError(f"block partition consumed {offset} of {len(values)} values")
    return blocks


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    datasets = [str(value) for value in config["datasets"]]
    repeats = [int(value) for value in config["cross_validation"]["repeats"]]
    if args.maximum_datasets is not None:
        datasets = datasets[: args.maximum_datasets]
    if args.maximum_repeats is not None:
        repeats = repeats[: args.maximum_repeats]
    n_folds = int(config["cross_validation"]["folds"])
    records: list[dict[str, object]] = []

    for dataset in datasets:
        for repeat in repeats:
            source = args.source_root / f"{dataset}_DeepDelta5_{repeat}.csv"
            started = time.perf_counter()
            flat, prediction_row_sha256 = load_prediction_row_only(source)
            sizes = infer_kfold_sizes(len(flat), n_folds)
            potentials: list[np.ndarray] = []
            for flow in split_square_blocks(flat, sizes):
                potential, projected = hodge_project_complete_pair_flow(flow)
                if not np.allclose(projected, -projected.T, atol=1e-12):
                    raise RuntimeError(f"{source}: projected flow is not antisymmetric")
                potentials.append(potential)
            elapsed = time.perf_counter() - started
            metadata: dict[str, object] = {
                "dataset": dataset,
                "repeat": repeat,
                "fold_sizes": sizes,
                "method": "hodge_cycle_consistent_pair_flow",
                "source_prediction": str(source),
                "source_prediction_row_sha256": prediction_row_sha256,
                "n_pair_predictions": int(len(flat)),
                "n_potentials": int(sum(sizes)),
                "candidate_numeric_parser_received_only_last_prediction_row": True,
                "response_row_parsed": False,
                "hyperparameters": 0,
                "projection_seconds": elapsed,
            }
            output = args.output_dir / dataset / f"repeat_{repeat}"
            digest = freeze_prediction_file(
                pd.DataFrame(
                    {
                        "mol_id": expected_ids(sizes),
                        "prediction": np.concatenate(potentials),
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
                "repeats": len(records),
                "folds": len(records) * n_folds,
                "response_rows_parsed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

