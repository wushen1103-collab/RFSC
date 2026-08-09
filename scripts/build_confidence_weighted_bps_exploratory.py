#!/usr/bin/env python3
"""Build the frozen post-primary confidence-weighted BPS predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import freeze_prediction_file, read_frozen_prediction
from bps_gnn.projection import confidence_weighted_boundary_projection, select_active_edges
from bps_gnn.temporal import temporal_split_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eligible-targets", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mmp-root", type=Path, required=True)
    parser.add_argument("--primary-prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    targets = json.loads(args.eligible_targets.read_text(encoding="utf-8"))
    base_weights = config["base_weights"]
    weights = {name: float(value) for name, value in base_weights.items()}
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise RuntimeError("Base uncertainty weights must sum to one")
    floor_additive = float(config["uncertainty"]["median_floor_additive"])
    operator = config["operator"]
    scale, threshold = float(operator["scale"]), float(operator["predicted_delta_threshold"])
    tolerance = float(config["decision"]["active_edge_difference_tolerance"])
    records: list[dict[str, object]] = []
    for target in targets:
        _, _, test_ids = temporal_split_ids(args.data_root, target)
        test_set = set(test_ids)
        all_edges = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        edges = all_edges[
            all_edges["mol_id_lo"].isin(test_set) & all_edges["mol_id_hi"].isin(test_set)
        ].copy()
        for seed in [0, 1, 2]:
            run_dir = args.primary_prediction_root / target / f"seed_{seed}"
            bases = {
                "svm": read_frozen_prediction(run_dir / "svm" / "predictions.parquet"),
                "rf": read_frozen_prediction(run_dir / "rf" / "predictions.parquet"),
                "gine": read_frozen_prediction(run_dir / "gine" / "predictions.parquet"),
            }
            unscaled = read_frozen_prediction(
                run_dir / "unscaled_reference" / "predictions.parquet"
            )
            ordinary = read_frozen_prediction(run_dir / "bps" / "predictions.parquet")
            uncertainty = sum(
                weights[name] * np.square(prediction - unscaled)
                for name, prediction in bases.items()
            )
            uncertainty_floor = float(np.median(uncertainty.to_numpy())) + floor_additive
            precision = 1.0 / (uncertainty + uncertainty_floor)
            active = select_active_edges(unscaled, edges, threshold)
            prediction = confidence_weighted_boundary_projection(
                unscaled, active, scale, precision
            )
            if len(active):
                weighted_delta = active["mol_id_hi"].map(prediction) - active["mol_id_lo"].map(prediction)
                ordinary_delta = active["mol_id_hi"].map(ordinary) - active["mol_id_lo"].map(ordinary)
                max_active_difference = float(np.max(np.abs(weighted_delta - ordinary_delta)))
            else:
                max_active_difference = 0.0
            if max_active_difference > tolerance:
                raise RuntimeError(f"{target} seed {seed}: active-edge invariant failed")
            metadata = {
                "target": target,
                "seed": seed,
                "method": "confidence_weighted_bps",
                "status": str(config.get("candidate_status", "post_primary_exploratory")),
                "test_labels_materialized": False,
                "uncertainty_floor": uncertainty_floor,
                "mean_uncertainty": float(uncertainty.mean()),
                "mean_precision": float(precision.mean()),
                "active_edges": int(len(active)),
                "max_active_delta_difference_vs_ordinary_bps": max_active_difference,
                "correction_l2": float(np.linalg.norm((prediction - unscaled).to_numpy())),
            }
            output = args.output_dir / target / f"seed_{seed}"
            freeze_prediction_file(
                pd.DataFrame({"mol_id": prediction.index, "prediction": prediction.to_numpy()}),
                output / "predictions.parquet",
                metadata,
            )
            output.mkdir(parents=True, exist_ok=True)
            (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            records.append(metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(json.dumps({"targets": len(targets), "runs": len(records), "test_labels_materialized": False}, indent=2))


if __name__ == "__main__":
    main()
