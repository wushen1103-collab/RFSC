#!/usr/bin/env python3
"""Freeze AqSolDB risk scores without reading any AqSolDB response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from bps_gnn.io import read_frozen_prediction, sha256_file

try:
    from fit_response_free_risk_model import features
except ModuleNotFoundError:
    from scripts.fit_response_free_risk_model import features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = joblib.load(args.model)
    records = []
    for run_path in sorted(args.prediction_root.glob("*/fold_*/seed_*/run.json")):
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["test_labels_materialized"] is not False:
            raise RuntimeError("AqSolDB response firewall metadata failed")
        base = read_frozen_prediction(run_path.parent / "unscaled_reference" / "predictions.parquet")
        candidate = read_frozen_prediction(run_path.parent / "bps" / "predictions.parquet")
        svm = read_frozen_prediction(run_path.parent / "svm" / "predictions.parquet")
        rf = read_frozen_prediction(run_path.parent / "rf" / "predictions.parquet")
        gine = read_frozen_prediction(run_path.parent / "gine" / "predictions.parquet")
        fixed = 0.5 * rf + 0.5 * gine
        all_edges = pd.read_parquet(args.mmp_root / run["target"] / "eligible_mmp_edges.parquet")
        edges = all_edges[all_edges["mol_id_lo"].isin(base.index) & all_edges["mol_id_hi"].isin(base.index)]
        frame = features(base, candidate, svm, fixed, edges)
        frame["risk_score"] = payload["model"].predict(frame[payload["features"]])
        frame["mol_id"] = frame.index
        output = args.output_dir / run["target"] / f"fold_{run['fold']}" / f"seed_{run['seed']}" / "risk_scores.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.reset_index(drop=True).sort_values("mol_id").to_parquet(output, index=False)
        digest = sha256_file(output)
        metadata = {"target": run["target"], "fold": int(run["fold"]), "seed": int(run["seed"]), "response_columns_read": False, "risk_score_sha256": digest, "features": payload["features"], "thresholds": payload["thresholds"]}
        output.with_suffix(output.suffix + ".sha256.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        records.append(metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(json.dumps({"runs": len(records), "external_responses_read": False}, indent=2))


if __name__ == "__main__":
    main()
