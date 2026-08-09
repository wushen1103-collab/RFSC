#!/usr/bin/env python3
"""Run deterministic ECFP-SVM across frozen SCG folds in parallel."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def run(task: dict) -> dict:
    output = task["output_dir"] / task["target"] / f"fold_{task['fold']}" / "seed_0"
    if (output / "predictions.parquet").exists():
        return {"target": task["target"], "fold": task["fold"], "status": "existing"}
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    process = subprocess.run(
        [sys.executable, "scripts/train_svm_scg.py", "--target", task["target"], "--fold", str(task["fold"]), "--config", str(task["config"]), "--output-dir", str(task["output_dir"])],
        text=True,
        capture_output=True,
        env=environment,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "launcher.stdout").write_text(process.stdout, encoding="utf-8")
    (output / "launcher.stderr").write_text(process.stderr, encoding="utf-8")
    return {"target": task["target"], "fold": task["fold"], "status": "completed" if process.returncode == 0 else "failed", "returncode": process.returncode}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase4-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()
    import yaml

    phase4 = yaml.safe_load(args.phase4_config.read_text(encoding="utf-8"))
    tasks = []
    for target in phase4["confirmatory_targets"]:
        assignments = pd.read_parquet(Path("folds/scg") / target / "outer_folds.parquet")
        for fold in range(int(assignments.outer_fold.nunique())):
            tasks.append({"target": target, "fold": fold, "config": args.config, "output_dir": args.output_dir})
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, task) for task in tasks]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
    summary = pd.DataFrame(records).sort_values(["target", "fold"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "launcher_summary.csv", index=False)
    if len(summary) != len(tasks) or (~summary.status.isin(["completed", "existing"])).any():
        raise SystemExit("SVM matrix incomplete or failed")


if __name__ == "__main__":
    main()
