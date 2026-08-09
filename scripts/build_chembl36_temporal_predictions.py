#!/usr/bin/env python3
"""Build and hash-freeze all temporal base/ensemble/BPS predictions label-free."""

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
    parser.add_argument("--svm-root", type=Path, required=True)
    parser.add_argument("--rf-root", type=Path, required=True)
    parser.add_argument("--gine-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    targets = json.loads(args.eligible_targets.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["seeds"]]
    reference = config["reference"]
    svm_weight = float(reference["svm_weight"])
    rf_weight = float(reference["rf_weight_inside_rf_gine"])
    gine_weight = float(reference["gine_weight_inside_rf_gine"])
    if abs(rf_weight + gine_weight - 1.0) > 1e-12:
        raise RuntimeError("RF/GINE reference weights must sum to one")
    operator = config["operator"]
    threshold, scale = float(operator["predicted_delta_threshold"]), float(operator["scale"])

    records: list[dict[str, object]] = []
    for target in targets:
        _, _, test_ids = temporal_split_ids(args.data_root, target)
        expected_ids = set(test_ids)
        all_edges = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        edges = all_edges[
            all_edges["mol_id_lo"].isin(expected_ids) & all_edges["mol_id_hi"].isin(expected_ids)
        ].copy()
        svm = read_frozen_prediction(args.svm_root / target / "seed_0" / "predictions.parquet")
        if set(svm.index) != expected_ids:
            raise RuntimeError(f"{target}: SVM test partition mismatch")
        for seed in seeds:
            rf = read_frozen_prediction(args.rf_root / target / f"seed_{seed}" / "predictions.parquet")
            gine = read_frozen_prediction(
                args.gine_root / target / "temporal" / f"seed_{seed}" / "predictions.parquet"
            )
            if set(rf.index) != expected_ids or set(gine.index) != expected_ids:
                raise RuntimeError(f"{target} seed {seed}: base test partition mismatch")
            rf_gine = rf_weight * rf + gine_weight * gine
            unscaled = svm_weight * svm + (1.0 - svm_weight) * rf_gine
            active = select_active_edges(unscaled, edges, threshold)
            bps = boundary_projection(unscaled, active, scale)
            metadata = {
                "target": target,
                "seed": seed,
                "protocol": str(config["name"]),
                "test_labels_materialized": False,
                "svm_weight": svm_weight,
                "rf_weight_inside_rf_gine": rf_weight,
                "gine_weight_inside_rf_gine": gine_weight,
                "predicted_delta_threshold": threshold,
                "scale": scale,
                "internal_test_mmp_edges": int(len(edges)),
                "active_test_mmp_edges": int(len(active)),
            }
            methods = {
                "svm": svm,
                "rf": rf,
                "gine": gine,
                "rf_gine": rf_gine,
                "unscaled_reference": unscaled,
                "bps": bps,
            }
            run_dir = args.output_dir / target / f"seed_{seed}"
            for method, prediction in methods.items():
                freeze_prediction_file(
                    pd.DataFrame({"mol_id": prediction.index, "prediction": prediction.to_numpy()}),
                    run_dir / method / "predictions.parquet",
                    {**metadata, "method": method},
                )
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            records.append(metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(json.dumps({"targets": len(targets), "runs": len(records), "test_labels_materialized": False}, indent=2))


if __name__ == "__main__":
    main()
