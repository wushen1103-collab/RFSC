#!/usr/bin/env python3
"""Freeze label-free gated component scaling for outer-test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from bps_gnn.io import freeze_prediction_file
try:
    from audit_gated_component_scale_validation import gated_edges, read_prediction
    from build_rf_gine_component_scale import scale_components
except ModuleNotFoundError:
    from scripts.audit_gated_component_scale_validation import gated_edges, read_prediction
    from scripts.build_rf_gine_component_scale import scale_components


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ensemble-dir", type=Path, default=Path("experiments/rf_gine_ensemble/full"))
    parser.add_argument("--fold-root", type=Path, default=Path("folds/scg"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    parameters = config["selection"]["per_target"]
    paths = sorted(args.ensemble_dir.glob("*/fold_*/seed_*/fixed_equal/predictions.parquet"))
    if len(paths) != 87:
        raise RuntimeError(f"Expected 87 fixed-equal predictions, observed {len(paths)}")
    records = []
    for path in paths:
        target = path.parts[-5]
        fold = int(path.parts[-4].removeprefix("fold_"))
        seed = int(path.parts[-3].removeprefix("seed_"))
        if target not in parameters:
            raise RuntimeError(f"Missing frozen parameters for {target}")
        scale = float(parameters[target]["scale"])
        threshold = float(parameters[target]["threshold"])
        base = read_prediction(path)
        assignments = pd.read_parquet(args.fold_root / target / "outer_folds.parquet")
        test_ids = set(assignments.loc[assignments.outer_fold == fold, "mol_id"])
        if set(base.index) != test_ids:
            raise RuntimeError(f"Test prediction partition mismatch: {target} fold {fold} seed {seed}")
        all_edges = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        edges = all_edges[all_edges.mol_id_lo.isin(test_ids) & all_edges.mol_id_hi.isin(test_ids)]
        selected_edges = gated_edges(base, edges, threshold)
        candidate = scale_components(base, selected_edges, scale)
        run_dir = args.output_dir / target / f"fold_{fold}" / f"seed_{seed}"
        metadata = {
            "target": target,
            "fold": fold,
            "seed": seed,
            "test_labels_materialized": False,
            "selection_protocol": "leave_one_target_out_validation",
            "scale": scale,
            "threshold": threshold,
            "internal_test_mmp_edges": len(edges),
            "selected_test_mmp_edges": len(selected_edges),
            "selected_test_nodes": len(set(selected_edges.mol_id_lo) | set(selected_edges.mol_id_hi)),
        }
        for method, prediction in [("fixed_equal", base), ("gated_component_scaled", candidate)]:
            freeze_prediction_file(
                pd.DataFrame({"mol_id": prediction.index, "prediction": prediction.to_numpy()}),
                run_dir / method / "predictions.parquet",
                {**metadata, "method": method},
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        records.append(metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(json.dumps({"runs": len(records), "test_labels_materialized": False}, indent=2))


if __name__ == "__main__":
    main()

