#!/usr/bin/env python3
"""Freeze the unchanged BPS rule on six label-private backbone matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from bps_gnn.io import freeze_prediction_file, sha256_file

try:
    from audit_gated_component_scale_validation import gated_edges, read_prediction
    from build_rf_gine_component_scale import scale_components
except ModuleNotFoundError:
    from scripts.audit_gated_component_scale_validation import gated_edges, read_prediction
    from scripts.build_rf_gine_component_scale import scale_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    input_root = Path(config["input_root"])
    fold_root = Path(config["fold_root"])
    mmp_root = Path(config["mmp_root"])
    threshold = float(config["operator"]["predicted_mmp_delta_threshold"])
    scale = float(config["operator"]["scale"])
    architectures = [str(value) for value in config["architectures"]]
    records = []
    for architecture in architectures:
        paths = sorted(
            (input_root / architecture).glob("*/fold_*/seed_*/predictions.parquet")
        )
        for path in paths:
            relative = path.relative_to(input_root / architecture)
            target, fold_name, seed_name = relative.parts[:3]
            fold = int(fold_name.removeprefix("fold_"))
            seed = int(seed_name.removeprefix("seed_"))
            raw = read_prediction(path)
            assignments = pd.read_parquet(fold_root / target / "outer_folds.parquet")
            test_ids = set(assignments.loc[assignments.outer_fold == fold, "mol_id"])
            if set(raw.index) != test_ids:
                raise RuntimeError(f"Partition mismatch: {architecture}/{target}/{fold}/{seed}")
            all_edges = pd.read_parquet(mmp_root / target / "eligible_mmp_edges.parquet")
            edges = all_edges[
                all_edges.mol_id_lo.isin(test_ids) & all_edges.mol_id_hi.isin(test_ids)
            ].copy()
            selected = gated_edges(raw, edges, threshold)
            candidate = scale_components(raw, selected, scale)
            output = args.output_dir / architecture / target / fold_name / seed_name
            metadata = {
                "architecture": architecture,
                "target": target,
                "fold": fold,
                "seed": seed,
                "method": "bps_fixed_transfer",
                "raw_prediction_sha256": sha256_file(path),
                "test_labels_materialized": False,
                "threshold": threshold,
                "scale": scale,
                "internal_test_mmp_edges": int(len(edges)),
                "selected_test_mmp_edges": int(len(selected)),
            }
            digest = freeze_prediction_file(
                pd.DataFrame(
                    {"mol_id": candidate.index, "prediction": candidate.to_numpy()}
                ),
                output / "predictions.parquet",
                metadata,
            )
            metadata["bps_prediction_sha256"] = digest
            output.mkdir(parents=True, exist_ok=True)
            (output / "run.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
            )
            records.append(metadata)
    expected = int(config["decision"]["expected_predictions"])
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} transferred predictions, found {len(records)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(json.dumps({"predictions": len(records), "test_labels_materialized": False}, indent=2))


if __name__ == "__main__":
    main()

