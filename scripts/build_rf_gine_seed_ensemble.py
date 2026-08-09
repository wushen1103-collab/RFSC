#!/usr/bin/env python3
"""Average frozen RF/GINE predictions across seeds without reading test labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bps_gnn.io import freeze_prediction_file, sha256_file


def read_prediction(path: Path) -> pd.Series:
    manifest = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Prediction hash mismatch: {path}")
    frame = pd.read_parquet(path)
    if set(frame.columns) != {"mol_id", "prediction"}:
        raise RuntimeError(f"Unexpected prediction schema: {path}")
    return frame.set_index("mol_id")["prediction"].sort_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-dir", type=Path, default=Path("experiments/rf_gine_ensemble/full"))
    parser.add_argument("--gine-dir", type=Path, default=Path("experiments/h1_oracle/gine_scg_panel"))
    parser.add_argument("--rf-dir", type=Path, default=Path("experiments/rf_scg/full"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixed_paths = sorted(args.ensemble_dir.glob("*/fold_*/seed_*/fixed_equal/predictions.parquet"))
    if len(fixed_paths) != 87:
        raise RuntimeError(f"Expected 87 fixed-equal predictions, observed {len(fixed_paths)}")
    rows = []
    for target in sorted({path.parts[-5] for path in fixed_paths}):
        folds = sorted({int(path.parts[-4].split("_")[-1]) for path in fixed_paths if path.parts[-5] == target})
        for fold in folds:
            by_seed = [path for path in fixed_paths if path.parts[-5] == target and int(path.parts[-4].split("_")[-1]) == fold]
            if len(by_seed) != 3:
                raise RuntimeError(f"Expected 3 seeds for {target} fold {fold}")
            fixed = pd.concat([read_prediction(path).rename(str(i)) for i, path in enumerate(by_seed)], axis=1).mean(axis=1)
            selected_paths = [path.parent.parent / "validation_selected" / "predictions.parquet" for path in by_seed]
            selected = pd.concat([read_prediction(path).rename(str(i)) for i, path in enumerate(selected_paths)], axis=1).mean(axis=1)
            gine_paths = [args.gine_dir / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet" for seed in range(3)]
            rf_paths = [args.rf_dir / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet" for seed in range(3)]
            gine = pd.concat([read_prediction(path).rename(str(i)) for i, path in enumerate(gine_paths)], axis=1).mean(axis=1)
            rf = pd.concat([read_prediction(path).rename(str(i)) for i, path in enumerate(rf_paths)], axis=1).mean(axis=1)
            if not fixed.index.equals(selected.index) or not fixed.index.equals(gine.index) or not fixed.index.equals(rf.index):
                raise RuntimeError(f"Seed ensemble index mismatch: {target} fold {fold}")
            run_dir = args.output_dir / target / f"fold_{fold}"
            metadata = {"target": target, "fold": fold, "seed_count": 3, "test_labels_materialized": False}
            for method, prediction in [("fixed_equal_seed_mean", fixed), ("validation_selected_seed_mean", selected), ("gine_seed_mean", gine), ("rf_seed_mean", rf)]:
                freeze_prediction_file(
                    pd.DataFrame({"mol_id": prediction.index, "prediction": prediction.to_numpy()}),
                    run_dir / method / "predictions.parquet",
                    {**metadata, "method": method},
                )
            (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            rows.append(metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(json.dumps({"folds": len(rows), "targets": len(set(row["target"] for row in rows))}, indent=2))


if __name__ == "__main__":
    main()
