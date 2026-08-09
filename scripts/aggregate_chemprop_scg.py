#!/usr/bin/env python3
"""Audit and aggregate official Chemprop D-MPNN strict-SCG reruns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bps_gnn.io import sha256_file

try:
    from aggregate_modern_baselines import METRICS, aggregate_runs, completeness_table
except ModuleNotFoundError:
    from scripts.aggregate_modern_baselines import METRICS, aggregate_runs, completeness_table


def read_runs(root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for evaluation_path in sorted(root.glob("*/fold_*/seed_*/evaluation.json")):
        target, fold_name, seed_name = evaluation_path.relative_to(root).parts[:3]
        run_path = evaluation_path.parent / "run.json"
        prediction_path = evaluation_path.parent / "predictions.parquet"
        manifest_path = prediction_path.with_suffix(prediction_path.suffix + ".sha256.json")
        if not run_path.exists() or not prediction_path.exists() or not manifest_path.exists():
            raise RuntimeError(f"Incomplete Chemprop run beside {evaluation_path}")
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        run = json.loads(run_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_hash = sha256_file(prediction_path)
        hashes = {
            actual_hash,
            evaluation["prediction_sha256"],
            run["prediction_sha256"],
            manifest["prediction_sha256"],
        }
        if len(hashes) != 1:
            raise RuntimeError(f"Prediction provenance mismatch at {prediction_path}")
        record = {
            "method": "chemprop_dmpnn",
            "architecture": run["architecture"],
            "target": target,
            "fold": int(fold_name.removeprefix("fold_")),
            "seed": int(seed_name.removeprefix("seed_")),
            "prediction_sha256": actual_hash,
            "best_epoch": None,
            "wall_time_seconds": run["wall_time_seconds"],
            "peak_vram_bytes": int(run.get("peak_vram_bytes", 0)),
        }
        record.update({metric: evaluation[metric] for metric in METRICS})
        records.append(record)
    if not records:
        raise RuntimeError(f"No evaluated Chemprop runs found under {root}")
    frame = pd.DataFrame.from_records(records)
    keys = ["method", "target", "fold", "seed"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate Chemprop target/fold/seed runs")
    return frame.sort_values(keys).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--audit", type=Path, default=Path("results/tables/table1_data_audit.csv")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = read_runs(args.root)
    audit = pd.read_csv(args.audit)
    completeness = completeness_table(runs, audit)
    if not args.allow_partial and not completeness["complete"].all():
        raise RuntimeError("Official Chemprop strict-SCG matrix is incomplete")
    target, macro = aggregate_runs(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "run_metrics.csv", index=False)
    target.to_csv(args.output_dir / "target_metrics.csv", index=False)
    macro.to_csv(args.output_dir / "macro_metrics.csv", index=False)
    completeness.to_csv(args.output_dir / "completeness.csv", index=False)
    provenance = {
        "method": "Chemprop D-MPNN",
        "result_source": "Re-run",
        "implementation": "official Chemprop 2.2.1",
        "split": "identical frozen strict SCG",
        "selection": "validation-best checkpoint",
        "seeds": [0, 1, 2],
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
