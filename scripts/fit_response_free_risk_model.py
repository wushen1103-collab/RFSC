#!/usr/bin/env python3
"""Fit the frozen MoleculeACE-only response-free harm model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

from bps_gnn.io import read_frozen_prediction, read_labels_for_ids
from bps_gnn.projection import active_components, select_active_edges


def features(base: pd.Series, candidate: pd.Series, svm: pd.Series, fixed: pd.Series, edges: pd.DataFrame) -> pd.DataFrame:
    scale = max(float(base.std(ddof=0)), 1e-8)
    active = select_active_edges(base, edges, 0.5)
    frame = pd.DataFrame(index=base.index)
    frame["correction_abs_scaled"] = (candidate - base).abs() / scale
    frame["base_disagreement_scaled"] = (svm - fixed).abs() / scale
    frame["active_degree"] = 0.0
    for left, right in active[["mol_id_lo", "mol_id_hi"]].itertuples(index=False, name=None):
        frame.loc[left, "active_degree"] += 1.0; frame.loc[right, "active_degree"] += 1.0
    frame["log_component_size"] = 0.0
    frame["component_density"] = 0.0
    frame["component_span_scaled"] = 0.0
    for members in active_components(base.index, active).values():
        ids = list(members)
        frame.loc[ids, "log_component_size"] = np.log1p(len(ids))
        if len(ids) <= 1:
            continue
        member_set = set(ids)
        count = int((active["mol_id_lo"].isin(member_set) & active["mol_id_hi"].isin(member_set)).sum())
        frame.loc[ids, "component_density"] = 2.0 * count / (len(ids) * (len(ids) - 1))
        frame.loc[ids, "component_span_scaled"] = float(base.loc[ids].max() - base.loc[ids].min()) / scale
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dev, rows = config["development"], []
    seen = set()
    for root_value in dev["prediction_roots"]:
        root = Path(root_value)
        for run_path in sorted(root.glob("*/fold_*/seed_*/run.json")):
            run = json.loads(run_path.read_text(encoding="utf-8"))
            key = (run["target"], int(run["fold"]), int(run["seed"]))
            if key in seen:
                raise RuntimeError(f"Duplicate development run: {key}")
            seen.add(key)
            base = read_frozen_prediction(run_path.parent / dev["base_method"] / "predictions.parquet")
            candidate = read_frozen_prediction(run_path.parent / dev["candidate_method"] / "predictions.parquet")
            svm = read_frozen_prediction(run_path.parent / "svm_ecfp" / "predictions.parquet")
            fixed = read_frozen_prediction(run_path.parent / "fixed_equal" / "predictions.parquet")
            all_edges = pd.read_parquet(Path(dev["mmp_root"]) / run["target"] / "eligible_mmp_edges.parquet")
            edges = all_edges[all_edges["mol_id_lo"].isin(base.index) & all_edges["mol_id_hi"].isin(base.index)]
            x = features(base, candidate, svm, fixed, edges)
            truth = read_labels_for_ids(Path(dev["data_root"]), run["target"], base.index).set_index("mol_id")["y_pactivity"].reindex(base.index)
            response_scale = max(float(base.std(ddof=0)), 1e-8)
            x["normalized_point_harm"] = ((candidate - truth).abs() - (base - truth).abs()) / response_scale
            x["target"] = run["target"]; x["fold"] = int(run["fold"]); x["seed"] = int(run["seed"]); x["mol_id"] = x.index
            rows.append(x.reset_index(drop=True))
    frame = pd.concat(rows, ignore_index=True)
    feature_names = list(config["features"])
    model_config = config["model"]
    model = HistGradientBoostingRegressor(max_depth=int(model_config["max_depth"]), learning_rate=float(model_config["learning_rate"]), max_iter=int(model_config["max_iter"]), l2_regularization=float(model_config["l2_regularization"]), random_state=int(model_config["random_state"]))
    model.fit(frame[feature_names], frame["normalized_point_harm"])
    frame["risk_score"] = model.predict(frame[feature_names])
    thresholds = {str(coverage): float(frame["risk_score"].quantile(float(coverage))) for coverage in config["coverage"]}
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_names, "thresholds": thresholds}, output / "risk_model.joblib")
    frame.groupby(["target", "fold", "seed"], as_index=False).agg(rows=("mol_id", "size"), mean_harm=("normalized_point_harm", "mean"), risk_harm_correlation=("risk_score", lambda value: value.corr(frame.loc[value.index, "normalized_point_harm"]))).to_csv(output / "development_run_audit.csv", index=False)
    manifest = {"development_runs": len(seen), "development_rows": len(frame), "features": feature_names, "thresholds": thresholds, "external_responses_read": False, "model": model_config}
    (output / "risk_model_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
