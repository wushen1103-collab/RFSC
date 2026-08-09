#!/usr/bin/env python3
"""Validation-only projection grid after the BindingDB-patent test reveal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_frozen_prediction, read_labels_for_ids
from bps_gnn.projection import (
    boundary_projection,
    confidence_weighted_boundary_projection,
    select_active_edges,
)
from bps_gnn.temporal import temporal_split_ids

try:
    from evaluate_chembl36_temporal import METRICS, metric_row
except ModuleNotFoundError:
    from scripts.evaluate_chembl36_temporal import METRICS, metric_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-config", type=Path, required=True)
    parser.add_argument("--eligible-targets", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mmp-root", type=Path, required=True)
    parser.add_argument("--svm-root", type=Path, required=True)
    parser.add_argument("--rf-root", type=Path, required=True)
    parser.add_argument("--gine-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-targets", type=int)
    return parser.parse_args()


def validation_prediction(root: Path, target: str, seed: int, temporal: bool = False) -> pd.Series:
    middle = Path(target) / "temporal" if temporal else Path(target)
    return read_frozen_prediction(root / middle / f"seed_{seed}" / "validation_predictions.parquet")


def summarize_grid(raw: pd.DataFrame, maximum_overall: float, maximum_cliff: float) -> pd.DataFrame:
    target = raw.groupby(
        ["target", "method", "threshold", "scale"], as_index=False, dropna=False
    )[METRICS].mean()
    reference = target[target["method"] == "unscaled"].set_index("target")
    records: list[dict[str, object]] = []
    candidates = target[target["method"] != "unscaled"]
    for (method, threshold, scale), frame in candidates.groupby(
        ["method", "threshold", "scale"], dropna=False
    ):
        frame = frame.set_index("target")
        record: dict[str, object] = {
            "method": method,
            "threshold": float(threshold),
            "scale": float(scale),
        }
        for metric in METRICS:
            paired = pd.concat(
                [frame[metric].rename("candidate"), reference[metric].rename("reference")],
                axis=1,
            ).dropna()
            prefix = metric.removesuffix("_mae")
            record[f"{prefix}_targets"] = int(len(paired))
            record[f"{prefix}_reference"] = float(paired["reference"].mean())
            record[f"{prefix}_candidate"] = float(paired["candidate"].mean())
            record[f"{prefix}_relative_change"] = float(
                paired["candidate"].mean() / paired["reference"].mean() - 1.0
            )
            record[f"{prefix}_favorable_fraction"] = float(
                (paired["candidate"] < paired["reference"]).mean()
            )
        record["passes_point_costs"] = bool(
            record["overall_relative_change"] <= maximum_overall
            and record["cliff_endpoint_relative_change"] <= maximum_cliff
        )
        records.append(record)
    return pd.DataFrame.from_records(records).sort_values(
        ["passes_point_costs", "pair_delta_relative_change"], ascending=[False, True]
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.diagnostic_config.read_text(encoding="utf-8"))
    targets = json.loads(args.eligible_targets.read_text(encoding="utf-8"))
    if args.maximum_targets is not None:
        targets = targets[: args.maximum_targets]
    thresholds = [float(value) for value in config["thresholds"]]
    scales = [float(value) for value in config["scales"]]
    centers = [str(value) for value in config["centers"]]
    weights = {name: float(value) for name, value in config["base_weights"].items()}
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise RuntimeError("Base weights must sum to one")
    floor_additive = float(config["uncertainty"]["median_floor_additive"])

    rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for target in targets:
        _, validation_ids, test_ids = temporal_split_ids(args.data_root, target)
        validation_set = set(validation_ids)
        if validation_set & set(test_ids):
            raise RuntimeError(f"{target}: validation/test overlap")
        truth = read_labels_for_ids(args.data_root, target, validation_ids).set_index("mol_id")[
            "y_pactivity"
        ]
        if set(truth.index) != validation_set:
            raise RuntimeError(f"{target}: validation response mismatch")
        all_edges = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        edges = all_edges[
            all_edges["mol_id_lo"].isin(validation_set)
            & all_edges["mol_id_hi"].isin(validation_set)
        ].copy()
        svm = validation_prediction(args.svm_root, target, 0)
        if set(svm.index) != validation_set:
            raise RuntimeError(f"{target}: SVM validation partition mismatch")
        for seed in [0, 1, 2]:
            rf = validation_prediction(args.rf_root, target, seed)
            gine = validation_prediction(args.gine_root, target, seed, temporal=True)
            if set(rf.index) != validation_set or set(gine.index) != validation_set:
                raise RuntimeError(f"{target} seed {seed}: validation partition mismatch")
            bases = {"svm": svm, "rf": rf, "gine": gine}
            unscaled = sum(weights[name] * prediction for name, prediction in bases.items())
            uncertainty = sum(
                weights[name] * np.square(prediction - unscaled)
                for name, prediction in bases.items()
            )
            precision = 1.0 / (uncertainty + float(np.median(uncertainty)) + floor_additive)
            rows.append(
                {
                    "target": target,
                    "seed": seed,
                    "method": "unscaled",
                    "threshold": -1.0,
                    "scale": 1.0,
                    "active_edges": 0,
                    **metric_row(unscaled, truth, edges),
                }
            )
            for threshold in thresholds:
                active = select_active_edges(unscaled, edges, threshold)
                for scale in scales:
                    predictions = {}
                    if "ordinary" in centers:
                        predictions["ordinary"] = boundary_projection(unscaled, active, scale)
                    if "confidence_weighted" in centers:
                        predictions["confidence_weighted"] = confidence_weighted_boundary_projection(
                            unscaled, active, scale, precision
                        )
                    for method, prediction in predictions.items():
                        rows.append(
                            {
                                "target": target,
                                "seed": seed,
                                "method": method,
                                "threshold": threshold,
                                "scale": scale,
                                "active_edges": int(len(active)),
                                **metric_row(prediction, truth, edges),
                            }
                        )
            audit.append(
                {
                    "target": target,
                    "n_validation": len(validation_ids),
                    "n_test_not_read": len(test_ids),
                    "internal_validation_mmp_edges": len(edges),
                }
            )

    raw = pd.DataFrame.from_records(rows)
    selection = config["selection"]
    grid = summarize_grid(
        raw,
        float(selection["maximum_overall_relative_cost"]),
        float(selection["maximum_cliff_relative_cost"]),
    )
    eligible = grid[grid["passes_point_costs"]]
    best = (eligible if len(eligible) else grid).iloc[0].to_dict()
    result = {
        "status": config["status"],
        "partition_read": "validation",
        "test_predictions_read": False,
        "test_labels_read": False,
        "targets": len(targets),
        "best_global_setting": best,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "raw_validation_metrics.csv", index=False)
    pd.DataFrame.from_records(audit).to_csv(args.output_dir / "partition_audit.csv", index=False)
    grid.to_csv(args.output_dir / "global_grid.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
