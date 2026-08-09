#!/usr/bin/env python3
"""Audit and aggregate the common-protocol strict-SCG architecture matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bps_gnn.io import sha256_file


METRICS = [
    "overall_mae",
    "overall_rmse",
    "cliff_mae",
    "cliff_rmse",
    "pair_delta_mae",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=Path("results/tables/table1_data_audit.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def read_runs(root: Path) -> pd.DataFrame:
    records = []
    for evaluation_path in sorted(root.glob("*/*/fold_*/seed_*/evaluation.json")):
        relative = evaluation_path.relative_to(root)
        method, target, fold_name, seed_name = relative.parts[:4]
        run_path = evaluation_path.parent / "run.json"
        prediction_path = evaluation_path.parent / "predictions.parquet"
        manifest_path = prediction_path.with_suffix(prediction_path.suffix + ".sha256.json")
        if not run_path.exists() or not prediction_path.exists() or not manifest_path.exists():
            raise RuntimeError(f"Incomplete frozen run beside {evaluation_path}")
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
            "method": method,
            "architecture": run["architecture"],
            "target": target,
            "fold": int(fold_name.removeprefix("fold_")),
            "seed": int(seed_name.removeprefix("seed_")),
            "prediction_sha256": actual_hash,
            "best_epoch": run["best_epoch"],
            "wall_time_seconds": run["wall_time_seconds"],
            "peak_vram_bytes": run["peak_vram_bytes"],
        }
        record.update({metric: evaluation[metric] for metric in METRICS})
        records.append(record)
    if not records:
        raise RuntimeError(f"No evaluated runs found under {root}")
    frame = pd.DataFrame(records)
    keys = ["method", "target", "fold", "seed"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate method/target/fold/seed runs")
    return frame.sort_values(keys).reset_index(drop=True)


def aggregate_runs(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_target = runs.groupby(["method", "target", "seed"], as_index=False)[METRICS].mean()
    target_rows = []
    for (method, target), group in seed_target.groupby(["method", "target"], sort=True):
        row = {
            "method": method,
            "target": target,
            "n_seeds": int(group["seed"].nunique()),
            "n_runs": int(len(runs[(runs.method == method) & (runs.target == target)])),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        target_rows.append(row)
    targets = pd.DataFrame(target_rows)

    seed_macro = seed_target.groupby(["method", "seed"], as_index=False)[METRICS].mean()
    macro_rows = []
    for method, group in seed_macro.groupby("method", sort=True):
        method_runs = runs[runs.method == method]
        row = {
            "method": method,
            "n_targets": int(method_runs.target.nunique()),
            "n_runs": int(len(method_runs)),
            "n_seeds": int(group.seed.nunique()),
            "total_wall_time_hours": float(method_runs.wall_time_seconds.sum() / 3600.0),
            "peak_vram_gib": float(method_runs.peak_vram_bytes.max() / (1024**3)),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        macro_rows.append(row)
    return targets, pd.DataFrame(macro_rows)


def completeness_table(runs: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    eligible = audit[audit["scg_folds"].notna()].copy()
    expected_by_target = eligible.set_index("target")["scg_folds"].astype(int) * 3
    rows = []
    for method, group in runs.groupby("method", sort=True):
        actual = group.groupby("target").size()
        missing_targets = sorted(set(expected_by_target.index) - set(actual.index))
        count_mismatches = {
            target: {"expected": int(expected), "actual": int(actual.get(target, 0))}
            for target, expected in expected_by_target.items()
            if int(actual.get(target, 0)) != int(expected)
        }
        rows.append(
            {
                "method": method,
                "expected_runs": int(expected_by_target.sum()),
                "actual_runs": int(len(group)),
                "complete": not missing_targets and not count_mismatches,
                "missing_targets": json.dumps(missing_targets),
                "count_mismatches": json.dumps(count_mismatches, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    runs = read_runs(args.root)
    audit = pd.read_csv(args.audit)
    completeness = completeness_table(runs, audit)
    if not args.allow_partial and not completeness["complete"].all():
        failures = completeness.loc[~completeness.complete, "method"].tolist()
        raise RuntimeError(f"Incomplete methods: {failures}")
    targets, macro = aggregate_runs(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "run_metrics.csv", index=False)
    targets.to_csv(args.output_dir / "target_metrics.csv", index=False)
    macro.to_csv(args.output_dir / "macro_metrics.csv", index=False)
    completeness.to_csv(args.output_dir / "completeness.csv", index=False)
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()

