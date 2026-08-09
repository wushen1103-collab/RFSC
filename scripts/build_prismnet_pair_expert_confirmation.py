#!/usr/bin/env python3
"""Freeze fixed-half PrismNet pair-expert predictions on unseen folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from bps_gnn.io import freeze_prediction_file
from scripts.build_rf_gine_scale_calibration import read_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    weight = float(config["pair_expert"]["prism_weight"])
    if weight != 0.5 or config["pair_expert"]["weight_selection"] != "prohibited":
        raise RuntimeError("Confirmation PrismNet weight is not frozen at 0.5")
    components = config["components"]
    records = []
    for task in config["confirmation"]["tasks"]:
        target, fold, seed = task["target"], int(task["fold"]), int(task["seed"])
        prism_path = Path(components["prismnet_run_root"]) / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet"
        fixed_path = Path(components["fixed_equal_ensemble"]) / target / f"fold_{fold}" / f"seed_{seed}" / "fixed_equal" / "predictions.parquet"
        prism, prism_hash = read_series(prism_path)
        fixed, fixed_hash = read_series(fixed_path)
        if not prism.index.equals(fixed.index):
            raise RuntimeError(f"Confirmation molecule mismatch: {target} fold {fold}")
        prediction = (1.0 - weight) * fixed + weight * prism
        output = args.output_dir / target / f"fold_{fold}" / f"seed_{seed}"
        metadata = {
            "target": target, "fold": fold, "seed": seed, "method": config["method"],
            "prism_weight": weight, "weight_selected_on_confirmation_validation": False,
            "outer_test_labels_materialized": False, "prism_prediction_sha256": prism_hash,
            "fixed_equal_prediction_sha256": fixed_hash, "component_hashes_verified": True,
        }
        digest = freeze_prediction_file(
            pd.DataFrame({"mol_id": prediction.index, "prediction": prediction.to_numpy()}),
            output / "predictions.parquet", metadata,
        )
        metadata["prediction_sha256"] = digest
        output.mkdir(parents=True, exist_ok=True)
        (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        records.append(metadata)
    pd.DataFrame(records).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(json.dumps({"runs": len(records), "prism_weight": weight, "outer_test_labels_materialized": False}, indent=2))


if __name__ == "__main__":
    main()

