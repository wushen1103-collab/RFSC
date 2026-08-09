#!/usr/bin/env python3
"""Evaluate the fixed query-degree support gate on validation labels only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from bps_gnn.io import read_labels_for_ids
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_tabpfn_chemeleon_smoke import gine_series, metrics, prediction_series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--rf-run", type=Path, required=True)
    parser.add_argument("--gine-run", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.new_run / "run.json").read_text(encoding="utf-8"))
    if not run["validation_only"] or run["outer_test_labels_materialized"] or run["outer_test_predictions_generated"]:
        raise RuntimeError("Support gate must remain validation-only")
    gate = prediction_series(args.new_run / "validation_predictions.parquet")
    rf = prediction_series(args.rf_run / "validation_predictions.parquet")
    gine = gine_series(args.gine_run / "validation_predictions.parquet", seed=0)
    if not gate.index.equals(rf.index) or not gate.index.equals(gine.index):
        raise RuntimeError("Validation molecule mismatch")
    truth = read_labels_for_ids(Path("data/processed/moleculeace"), args.target, gate.index).set_index("mol_id")["y_pactivity"]
    edges = pd.read_parquet(Path("graphs/mmp") / args.target / "eligible_mmp_edges.parquet")
    internal = edges[edges["mol_id_lo"].isin(truth.index) & edges["mol_id_hi"].isin(truth.index)].copy()
    delta = internal["mol_id_hi"].map(truth) - internal["mol_id_lo"].map(truth)
    cliffs = internal[delta.abs() >= 1.0]
    rows = []
    for name, prediction in {"gine": gine, "rf": rf, "fixed_rf_gine": 0.5 * rf + 0.5 * gine, "support_gate": gate}.items():
        rows.append({"method": name, **metrics(prediction, truth, cliffs)})
    table = pd.DataFrame(rows).set_index("method")
    base, result = table.loc["fixed_rf_gine"], table.loc["support_gate"]
    criteria = {
        "overall_not_worse_0_5pct": bool(result["overall_mae"] <= 1.005 * base["overall_mae"]),
        "cliff_improves_0_5pct": bool(result["cliff_endpoint_mae"] <= 0.995 * base["cliff_endpoint_mae"]),
        "pair_not_worse_2pct": bool(result["cliff_pair_delta_mae"] <= 1.02 * base["cliff_pair_delta_mae"]),
    }
    decision = {
        "status": "GO" if all(criteria.values()) else "NO-GO",
        "validation_only": True,
        "outer_test_labels_materialized": False,
        "criteria": criteria,
        "relative_to_fixed_rf_gine_pct": {
            metric: float(100.0 * (result[metric] / base[metric] - 1.0))
            for metric in ["overall_mae", "cliff_endpoint_mae", "cliff_pair_delta_mae"]
        },
        "action": "stop_without_test_evaluation_or_parameter_search" if not all(criteria.values()) else "hold_for_two_task_matrix_review",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    table.reset_index().to_csv(args.output_table, index=False)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
