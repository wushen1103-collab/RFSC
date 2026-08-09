#!/usr/bin/env python3
"""Build the frozen validation-selected BPS follow-up without response access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from bps_gnn.io import freeze_prediction_file, read_frozen_prediction
from bps_gnn.projection import boundary_projection, select_active_edges
from bps_gnn.temporal import temporal_split_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eligible-targets", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mmp-root", type=Path, required=True)
    parser.add_argument("--primary-prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-targets", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    targets = json.loads(args.eligible_targets.read_text(encoding="utf-8"))
    if args.maximum_targets is not None:
        targets = targets[: args.maximum_targets]
    threshold = float(config["operator"]["predicted_delta_threshold"])
    scale = float(config["operator"]["scale"])
    rows: list[dict[str, object]] = []
    for target in targets:
        _, _, test_ids = temporal_split_ids(args.data_root, target)
        expected = set(test_ids)
        all_edges = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        edges = all_edges[
            all_edges["mol_id_lo"].isin(expected) & all_edges["mol_id_hi"].isin(expected)
        ].copy()
        for seed in [int(value) for value in config["seeds"]]:
            source = (
                args.primary_prediction_root
                / target
                / f"seed_{seed}"
                / "unscaled_reference"
                / "predictions.parquet"
            )
            unscaled = read_frozen_prediction(source)
            if set(unscaled.index) != expected:
                raise RuntimeError(f"{target} seed {seed}: test partition mismatch")
            active = select_active_edges(unscaled, edges, threshold)
            prediction = boundary_projection(unscaled, active, scale)
            metadata = {
                "target": target,
                "seed": seed,
                "method": "validation_selected_bps",
                "status": config["status"],
                "selection_source": config["selection_source"],
                "source_unscaled_prediction": str(source),
                "predicted_delta_threshold": threshold,
                "scale": scale,
                "internal_test_mmp_edges": int(len(edges)),
                "active_test_mmp_edges": int(len(active)),
                "test_labels_materialized": False,
            }
            output = args.output_dir / target / f"seed_{seed}"
            freeze_prediction_file(
                pd.DataFrame({"mol_id": prediction.index, "prediction": prediction.to_numpy()}),
                output / "predictions.parquet",
                metadata,
            )
            output.mkdir(parents=True, exist_ok=True)
            (output / "run.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
            )
            rows.append(metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(
        json.dumps(
            {"targets": len(targets), "runs": len(rows), "test_labels_materialized": False},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

