#!/usr/bin/env python3
"""Freeze AqSolDB reference/BPS predictions and response-free risk features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import freeze_prediction_file, read_frozen_prediction, sha256_file
from bps_gnn.projection import active_components, boundary_projection, select_active_edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--svm-root", type=Path, required=True)
    parser.add_argument("--rf-root", type=Path, required=True)
    parser.add_argument("--gine-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    target = str(config["target"])
    fold_root, mmp_root = Path(config["fold_root"]), Path(config["mmp_root"])
    assignments = pd.read_parquet(fold_root / target / "outer_folds.parquet")
    folds = sorted(int(value) for value in assignments["outer_fold"].unique())
    all_edges = pd.read_parquet(mmp_root / target / "eligible_mmp_edges.parquet")
    weights, operator = config["reference"], config["operator"]
    records = []
    for fold in folds:
        test_ids = set(assignments.loc[assignments["outer_fold"] == fold, "mol_id"])
        edges = all_edges[all_edges["mol_id_lo"].isin(test_ids) & all_edges["mol_id_hi"].isin(test_ids)].copy()
        svm = read_frozen_prediction(args.svm_root / target / f"fold_{fold}" / "seed_0" / "predictions.parquet")
        for seed in map(int, config["seeds"]):
            rf = read_frozen_prediction(args.rf_root / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet")
            gine = read_frozen_prediction(args.gine_root / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet")
            if set(svm.index) != test_ids or set(rf.index) != test_ids or set(gine.index) != test_ids:
                raise RuntimeError(f"Prediction partition mismatch fold={fold} seed={seed}")
            rf_gine = float(weights["rf_weight_inside_rf_gine"]) * rf + float(weights["gine_weight_inside_rf_gine"]) * gine
            base = float(weights["svm_weight"]) * svm + (1.0 - float(weights["svm_weight"])) * rf_gine
            active = select_active_edges(base, edges, float(operator["predicted_delta_threshold"]))
            bps = boundary_projection(base, active, float(operator["scale"]))
            run_dir = args.output_dir / target / f"fold_{fold}" / f"seed_{seed}"
            metadata = {
                "target": target, "fold": fold, "seed": seed,
                "test_labels_materialized": False, "protocol": config["name"],
                "internal_test_mmp_edges": int(len(edges)), "active_test_mmp_edges": int(len(active)),
                "threshold": float(operator["predicted_delta_threshold"]), "scale": float(operator["scale"]),
            }
            for method, prediction in {
                "svm": svm, "rf": rf, "gine": gine, "unscaled_reference": base, "bps": bps,
            }.items():
                freeze_prediction_file(pd.DataFrame({"mol_id": prediction.index, "prediction": prediction.to_numpy()}), run_dir / method / "predictions.parquet", {**metadata, "method": method})

            # These features never use a response and are frozen for later selective-risk analysis.
            diagnostics = pd.DataFrame(index=base.index)
            diagnostics.index.name = "mol_id"
            diagnostics["correction_abs"] = (bps - base).abs()
            diagnostics["base_disagreement_std"] = np.std(np.column_stack([svm.loc[base.index], rf.loc[base.index], gine.loc[base.index]]), axis=1)
            diagnostics["active_degree"] = 0
            for left, right in active[["mol_id_lo", "mol_id_hi"]].itertuples(index=False, name=None):
                diagnostics.loc[left, "active_degree"] += 1
                diagnostics.loc[right, "active_degree"] += 1
            diagnostics["component_size"] = 1
            diagnostics["component_density"] = 0.0
            diagnostics["component_span"] = 0.0
            for members in active_components(base.index, active).values():
                ids = list(members)
                if len(ids) <= 1:
                    continue
                member_set = set(ids)
                edge_count = int((active["mol_id_lo"].isin(member_set) & active["mol_id_hi"].isin(member_set)).sum())
                diagnostics.loc[ids, "component_size"] = len(ids)
                diagnostics.loc[ids, "component_density"] = 2.0 * edge_count / (len(ids) * (len(ids) - 1))
                diagnostics.loc[ids, "component_span"] = float(base.loc[ids].max() - base.loc[ids].min())
            diagnostics = diagnostics.reset_index()
            risk_path = run_dir / "response_free_risk_features.parquet"
            risk_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostics.sort_values("mol_id").to_parquet(risk_path, index=False)
            metadata["risk_feature_sha256"] = sha256_file(risk_path)
            (risk_path.with_suffix(risk_path.suffix + ".sha256.json")).write_text(json.dumps({"prediction_sha256": metadata["risk_feature_sha256"], "metadata": {**metadata, "response_columns": False}}, indent=2), encoding="utf-8")
            (run_dir / "run.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
            records.append(metadata)
    expected = len(folds) * len(config["seeds"])
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} runs, observed {len(records)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(json.dumps({"runs": len(records), "test_labels_materialized": False}, indent=2))


if __name__ == "__main__":
    main()
