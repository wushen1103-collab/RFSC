#!/usr/bin/env python3
"""Run frozen Delta-LightGBM held-out folds with bounded CPU parallelism."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--threads-per-fold", type=int, default=32)
    parser.add_argument("--folds", type=int, nargs="*")
    return parser.parse_args()


def run_fold(fold: int, args: argparse.Namespace) -> dict[str, object]:
    fold_name = f"fold_{fold:02d}"
    prediction_root = args.output_root / "predictions" / fold_name
    evaluation_root = args.output_root / "evaluation" / fold_name
    decision_path = evaluation_root / "decision.json"
    if decision_path.exists():
        return {"fold": fold, "status": "existing"}
    prediction_root.mkdir(parents=True, exist_ok=True)
    evaluation_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    thread_count = str(args.threads_per_fold)
    environment.update(
        {
            "OMP_NUM_THREADS": thread_count,
            "MKL_NUM_THREADS": thread_count,
            "OPENBLAS_NUM_THREADS": thread_count,
            "NUMEXPR_NUM_THREADS": thread_count,
        }
    )
    build = subprocess.run(
        [
            sys.executable,
            "scripts/build_deepdelta_delta_lgbm_predictions.py",
            "--config",
            str(args.config),
            "--data-root",
            str(args.data_root),
            "--output-dir",
            str(prediction_root),
            "--held-out-fold",
            str(fold),
            "--n-jobs",
            thread_count,
        ],
        text=True,
        capture_output=True,
        env=environment,
    )
    (prediction_root / "launcher_build.stdout").write_text(build.stdout, encoding="utf-8")
    (prediction_root / "launcher_build.stderr").write_text(build.stderr, encoding="utf-8")
    if build.returncode:
        return {"fold": fold, "status": "build_failed", "returncode": build.returncode}
    evaluation = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_deepdelta_delta_lgbm_projection.py",
            "--config",
            str(args.config),
            "--data-root",
            str(args.data_root),
            "--prediction-root",
            str(prediction_root),
            "--output-dir",
            str(evaluation_root),
            "--held-out-fold",
            str(fold),
        ],
        text=True,
        capture_output=True,
        env=environment,
    )
    (evaluation_root / "launcher_eval.stdout").write_text(
        evaluation.stdout, encoding="utf-8"
    )
    (evaluation_root / "launcher_eval.stderr").write_text(
        evaluation.stderr, encoding="utf-8"
    )
    return {
        "fold": fold,
        "status": "completed" if evaluation.returncode == 0 else "evaluation_failed",
        "returncode": evaluation.returncode,
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    folds = args.folds or [int(value) for value in config["split"]["held_out_folds"]]
    if args.workers * args.threads_per_fold > (os.cpu_count() or 1) - 30:
        raise ValueError("Requested fold workers would leave fewer than 30 CPU cores free")
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_fold, fold, args): fold for fold in folds}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
    summary = pd.DataFrame(records).sort_values("fold")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_root / "launcher_summary.csv", index=False)
    failures = summary[~summary.status.isin(["completed", "existing"])]
    if len(failures):
        raise SystemExit(f"{len(failures)} folds failed")


if __name__ == "__main__":
    main()

