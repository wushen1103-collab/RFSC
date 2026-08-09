#!/usr/bin/env python3
"""Freeze structure-only component-mean-preserving scaling of fixed RF/GINE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from bps_gnn.io import freeze_prediction_file, sha256_file
from bps_gnn.projection import boundary_projection


def read_prediction(path: Path) -> pd.Series:
    manifest = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Prediction hash mismatch: {path}")
    frame = pd.read_parquet(path)
    return frame.set_index("mol_id")["prediction"].sort_index()


def scale_components(base: pd.Series, edges: pd.DataFrame, scale: float) -> pd.Series:
    """Backward-compatible entry point for the frozen BPS operator."""
    return boundary_projection(base, edges, scale)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ensemble-dir", type=Path, default=Path("experiments/rf_gine_ensemble/full"))
    parser.add_argument("--fold-root", type=Path, default=Path("folds/scg"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = sorted(args.ensemble_dir.glob("*/fold_*/seed_*/fixed_equal/predictions.parquet"))
    if len(paths) != 87:
        raise RuntimeError(f"Expected 87 fixed-equal predictions, observed {len(paths)}")
    records = []
    for path in paths:
        target = path.parts[-5]
        fold = int(path.parts[-4].split("_")[-1])
        seed = int(path.parts[-3].split("_")[-1])
        base = read_prediction(path)
        assignments = pd.read_parquet(args.fold_root / target / "outer_folds.parquet")
        test_ids = set(assignments.loc[assignments.outer_fold == fold, "mol_id"])
        if set(base.index) != test_ids:
            raise RuntimeError(f"Test prediction partition mismatch: {target} fold {fold} seed {seed}")
        edges = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        edges = edges[edges.mol_id_lo.isin(test_ids) & edges.mol_id_hi.isin(test_ids)]
        candidate = scale_components(base, edges, float(config["operator"]["scale"]))
        run_dir = args.output_dir / target / f"fold_{fold}" / f"seed_{seed}"
        metadata = {"target": target, "fold": fold, "seed": seed, "test_labels_materialized": False, "scale": float(config["operator"]["scale"]), "internal_test_mmp_edges": len(edges)}
        for method, prediction in [("fixed_equal", base), ("component_scaled", candidate)]:
            freeze_prediction_file(
                pd.DataFrame({"mol_id": prediction.index, "prediction": prediction.to_numpy()}),
                run_dir / method / "predictions.parquet",
                {**metadata, "method": method},
            )
        (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        records.append(metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(json.dumps({"runs": len(records), "test_labels_materialized": False}, indent=2))


if __name__ == "__main__":
    main()
