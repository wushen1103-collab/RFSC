#!/usr/bin/env python3
"""Evaluate the validation-only strict-SCG LAC formula-adaptation smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_labels_for_ids, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_prediction(path: Path) -> pd.Series:
    manifest = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text())
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Prediction hash mismatch: {path}")
    frame = pd.read_parquet(path)
    if set(frame.columns) != {"mol_id", "prediction"} or frame["mol_id"].duplicated().any():
        raise RuntimeError(f"Invalid prediction schema: {path}")
    return frame.set_index("mol_id")["prediction"].sort_index()


def metrics(prediction: pd.Series, truth: pd.Series, cliffs: pd.DataFrame) -> dict:
    endpoints = set(cliffs["mol_id_lo"]) | set(cliffs["mol_id_hi"])
    endpoint_ids = [value for value in prediction.index if value in endpoints]
    predicted_delta = prediction.reindex(cliffs["mol_id_hi"]).to_numpy() - prediction.reindex(
        cliffs["mol_id_lo"]
    ).to_numpy()
    return {
        "overall_mae": float(np.mean(np.abs(prediction - truth.loc[prediction.index]))),
        "cliff_mae": float(
            np.mean(np.abs(prediction.loc[endpoint_ids] - truth.loc[endpoint_ids]))
        ),
        "pair_delta_mae": float(
            np.mean(np.abs(predicted_delta - cliffs["delta_y"].to_numpy()))
        ),
        "n_query": len(prediction),
        "n_cliff_endpoints": len(endpoint_ids),
        "n_cliff_pairs": len(cliffs),
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    smoke = config["smoke"]
    target, fold, seed = smoke["target"], int(smoke["fold"]), int(smoke["seed"])
    run = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    if run["validation_labels_materialized"] or run["outer_test_labels_materialized"]:
        raise RuntimeError("Query response entered LAC trainer")
    predictions = {
        variant: read_prediction(args.run_dir / variant / "validation_predictions.parquet")
        for variant in ["baseline", "lac"]
    }
    if not predictions["baseline"].index.equals(predictions["lac"].index):
        raise RuntimeError("Paired LAC validation molecule mismatch")
    truth = read_labels_for_ids(
        Path(config["data_root"]), target, predictions["baseline"].index
    ).set_index("mol_id")["y_pactivity"]
    edges = pd.read_parquet(Path(config["mmp_root"]) / target / "eligible_mmp_edges.parquet")
    edges = edges[
        edges["mol_id_lo"].isin(truth.index) & edges["mol_id_hi"].isin(truth.index)
    ].copy()
    edges["delta_y"] = edges["mol_id_hi"].map(truth) - edges["mol_id_lo"].map(truth)
    cliffs = edges[edges["delta_y"].abs() >= float(config["lac"]["cliff_threshold"])]
    table = pd.DataFrame(
        [
            {"method": variant, **metrics(prediction, truth, cliffs)}
            for variant, prediction in predictions.items()
        ]
    )
    values = table.set_index("method")
    relative = {
        metric: float(values.loc["lac", metric] / values.loc["baseline", metric] - 1.0)
        for metric in ["overall_mae", "cliff_mae", "pair_delta_mae"]
    }
    lac_training = next(row for row in run["training"] if row["variant"] == "lac")
    decision = config["decision"]
    criteria = {
        "overall_improvement": relative["overall_mae"]
        <= -float(decision["minimum_overall_improvement_vs_mlp"]),
        "cliff_improvement": relative["cliff_mae"]
        <= -float(decision["minimum_cliff_improvement_vs_mlp"]),
        "pair_not_worse": relative["pair_delta_mae"]
        <= float(decision["maximum_pair_delta_change_vs_mlp"]),
        "active_edge_batches": float(lac_training["active_edge_batch_fraction"])
        >= float(decision["minimum_active_edge_batch_fraction"]),
        "finite_metrics": bool(np.isfinite(table[["overall_mae", "cliff_mae", "pair_delta_mae"]]).all().all()),
    }
    result = {
        "method": config["name"],
        "adaptation_status": config["adaptation_status"],
        "target": target,
        "fold": fold,
        "seed": seed,
        "evaluated_partition": "validation",
        "outer_test_labels_materialized": False,
        "relative_changes_percent": {key: 100.0 * value for key, value in relative.items()},
        "active_edge_batch_fraction": lac_training["active_edge_batch_fraction"],
        "criteria": criteria,
        "advance_to_replication": bool(all(criteria.values())),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "lac_mlp_scg_smoke_table.csv", index=False)
    (args.output_dir / "lac_mlp_scg_smoke_decision.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    lines = [
        "# LAC ECFP-MLP strict-SCG formula-adaptation smoke",
        "",
        f"Decision: **{'GO' if result['advance_to_replication'] else 'NO-GO'}**.",
        "",
        "```csv",
        table.to_csv(index=False).strip(),
        "```",
        "",
        "This is a formula-level adaptation, not an official-code reproduction.",
        "Outer-test labels, predictions, and metrics were not materialized.",
        "",
    ]
    (args.output_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
