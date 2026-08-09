#!/usr/bin/env python3
"""Hash-freeze Phase 4 predictions without reading confirmatory labels."""

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
    parser.add_argument("--fold-root", type=Path, default=Path("folds/scg"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rf_root = Path(config["components"]["rf"])
    gine_root = Path(config["components"]["gine"])
    scale = float(config["operator"]["scale"])
    threshold = float(config["operator"]["predicted_mmp_delta_threshold"])
    records = []
    for target in config["confirmatory_targets"]:
        assignments = pd.read_parquet(args.fold_root / target / "outer_folds.parquet")
        n_folds = int(assignments.outer_fold.nunique())
        all_edges = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        for fold in range(n_folds):
            test_ids = set(assignments.loc[assignments.outer_fold == fold, "mol_id"])
            edges = all_edges[all_edges.mol_id_lo.isin(test_ids) & all_edges.mol_id_hi.isin(test_ids)]
            for seed in config["seeds"]:
                rf = read_prediction(rf_root / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet")
                gine = read_prediction(gine_root / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet")
                if not rf.index.equals(gine.index) or set(rf.index) != test_ids:
                    raise RuntimeError(f"Prediction partition mismatch: {target} fold {fold} seed {seed}")
                fixed = 0.5 * rf + 0.5 * gine
                selected_edges = gated_edges(fixed, edges, threshold)
                candidate = scale_components(fixed, selected_edges, scale)
                run_dir = args.output_dir / target / f"fold_{fold}" / f"seed_{seed}"
                metadata = {
                    "target": target,
                    "fold": fold,
                    "seed": int(seed),
                    "confirmatory": True,
                    "test_labels_materialized": False,
                    "scale": scale,
                    "threshold": threshold,
                    "internal_test_mmp_edges": len(edges),
                    "selected_test_mmp_edges": len(selected_edges),
                    "selected_test_nodes": len(set(selected_edges.mol_id_lo) | set(selected_edges.mol_id_hi)),
                }
                for method, prediction in [
                    ("rf", rf),
                    ("gine", gine),
                    ("fixed_equal", fixed),
                    ("gated_component_scaled", candidate),
                ]:
                    freeze_prediction_file(
                        pd.DataFrame({"mol_id": prediction.index, "prediction": prediction.to_numpy()}),
                        run_dir / method / "predictions.parquet",
                        {**metadata, "method": method},
                    )
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                records.append(metadata)
    expected = sum(
        int(pd.read_parquet(args.fold_root / target / "outer_folds.parquet").outer_fold.nunique())
        for target in config["confirmatory_targets"]
    ) * len(config["seeds"])
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} runs, observed {len(records)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(json.dumps({"runs": len(records), "targets": len(config["confirmatory_targets"]), "test_labels_materialized": False}, indent=2))


if __name__ == "__main__":
    main()

