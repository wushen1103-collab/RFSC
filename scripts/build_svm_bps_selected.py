#!/usr/bin/env python3
"""Build the frozen SVM-BPS candidate without materializing test labels."""

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


def mix_predictions(svm: pd.Series, fixed: pd.Series, svm_weight: float) -> pd.Series:
    if not svm.index.equals(fixed.index):
        raise RuntimeError("SVM and fixed-ensemble prediction partitions differ")
    return svm_weight * svm + (1.0 - svm_weight) * fixed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--svm-dir", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, default=Path("folds/scg"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    chosen = config["selection"]
    targets = set(config["scope"]["targets"])
    svm_weight = float(chosen["svm_weight"])
    scale = float(chosen["scale"])
    threshold = float(chosen["predicted_mmp_delta_threshold"])
    paths = sorted(args.ensemble_dir.glob("*/fold_*/seed_*/fixed_equal/predictions.parquet"))
    paths = [path for path in paths if path.parts[-5] in targets]
    expected = int(config["scope"]["expected_runs"])
    if len(paths) != expected:
        raise RuntimeError(f"Expected {expected} fixed predictions, observed {len(paths)}")
    records = []
    for path in paths:
        target = path.parts[-5]
        fold = int(path.parts[-4].removeprefix("fold_"))
        seed = int(path.parts[-3].removeprefix("seed_"))
        fixed = read_prediction(path)
        svm_path = args.svm_dir / target / f"fold_{fold}" / "seed_0" / "predictions.parquet"
        svm = read_prediction(svm_path)
        base = mix_predictions(svm, fixed, svm_weight)
        assignments = pd.read_parquet(args.fold_root / target / "outer_folds.parquet")
        test_ids = set(assignments.loc[assignments.outer_fold == fold, "mol_id"])
        if set(base.index) != test_ids:
            raise RuntimeError(f"Test partition mismatch: {target} fold {fold} seed {seed}")
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
            "selection_source": config["selection_source"],
            "svm_weight": svm_weight,
            "scale": scale,
            "threshold": threshold,
            "internal_test_mmp_edges": len(edges),
            "selected_test_mmp_edges": len(selected_edges),
        }
        methods = {
            "svm_ecfp": svm,
            "fixed_equal": fixed,
            "svm_fixed_mixture": base,
            "svm_bps_selected": candidate,
        }
        gated_path = path.parent.parent / "gated_component_scaled" / "predictions.parquet"
        if gated_path.exists():
            methods["gated_component_scaled"] = read_prediction(gated_path)
        for method, prediction in methods.items():
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
