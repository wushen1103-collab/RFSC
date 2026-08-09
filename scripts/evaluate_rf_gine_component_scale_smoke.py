#!/usr/bin/env python3
"""Smoke a fixed component-mean-preserving pair-amplitude calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_labels_for_ids, sha256_file


def verify(path: Path) -> pd.Series:
    manifest = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Prediction hash mismatch: {path}")
    return pd.read_parquet(path).set_index("mol_id")["prediction"].sort_index()


def component_scaled(base: pd.Series, edges: pd.DataFrame, scale: float) -> pd.Series:
    parent = {value: value for value in base.index}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in edges.itertuples(index=False):
        union(row.mol_id_lo, row.mol_id_hi)
    groups = pd.Series({value: find(value) for value in base.index})
    result = base.copy()
    for _, ids in groups.groupby(groups).groups.items():
        values = base.loc[list(ids)]
        result.loc[list(ids)] = values.mean() + scale * (values - values.mean())
    return result


def metrics(prediction: pd.Series, truth: pd.Series, cliffs: pd.DataFrame) -> dict:
    endpoints = set(cliffs.mol_id_lo) | set(cliffs.mol_id_hi)
    endpoint_ids = [value for value in prediction.index if value in endpoints]
    delta = prediction.reindex(cliffs.mol_id_hi).to_numpy() - prediction.reindex(cliffs.mol_id_lo).to_numpy()
    return {
        "overall_mae": float(np.mean(abs(prediction - truth.loc[prediction.index]))),
        "cliff_mae": float(np.mean(abs(prediction.loc[endpoint_ids] - truth.loc[endpoint_ids]))),
        "pair_delta_mae": float(np.mean(abs(delta - cliffs.delta_y.to_numpy()))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    target, fold, seed = config["scope"]["target"], int(config["scope"]["fold"]), int(config["scope"]["seed"])
    rf = verify(Path("experiments/rf_scg/full") / target / f"fold_{fold}" / f"seed_{seed}" / "validation_predictions.parquet")
    gine_frame = pd.read_parquet(Path("experiments/sqrl_anchor/full") / target / f"fold_{fold}" / "validation_predictions.parquet")
    gine = gine_frame[(gine_frame.method == "base") & (gine_frame.seed == seed)].set_index("mol_id")["prediction"].sort_index()
    base = 0.5 * rf + 0.5 * gine
    truth = read_labels_for_ids(Path("data/processed/moleculeace"), target, base.index).set_index("mol_id")["y_pactivity"]
    edges = pd.read_parquet(Path("graphs/mmp") / target / "eligible_mmp_edges.parquet")
    edges = edges[edges.mol_id_lo.isin(base.index) & edges.mol_id_hi.isin(base.index)].copy()
    edges["delta_y"] = edges.mol_id_hi.map(truth) - edges.mol_id_lo.map(truth)
    cliffs = edges[edges.delta_y.abs() >= 1.0]
    candidate = component_scaled(base, edges, float(config["operator"]["scale"]))
    baseline_metrics, candidate_metrics = metrics(base, truth, cliffs), metrics(candidate, truth, cliffs)
    relative = {f"{key}_relative_change": candidate_metrics[key] / baseline_metrics[key] - 1.0 for key in baseline_metrics}
    decision = config["evaluation"]["safeguards"]
    result = {"baseline": baseline_metrics, "candidate": candidate_metrics, **relative, "pass": {
        "overall": relative["overall_mae_relative_change"] <= float(decision["maximum_overall_relative_change"]),
        "cliff": relative["cliff_mae_relative_change"] <= float(decision["maximum_cliff_relative_change"]),
        "pair": relative["pair_delta_mae_relative_change"] <= -float(decision["minimum_pair_relative_improvement"]),
    }}
    result["authorize_full"] = all(result["pass"].values())
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
