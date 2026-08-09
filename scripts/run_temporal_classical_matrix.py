#!/usr/bin/env python3
"""Run frozen temporal SVM or RF predictions in parallel without test labels."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml


def run_task(task: dict) -> dict[str, object]:
    target, seed = task["target"], task["seed"]
    run_dir = task["output_dir"] / target / f"seed_{seed}"
    if (run_dir / "predictions.parquet").exists():
        return {"target": target, "seed": seed, "status": "existing"}
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    process = subprocess.run(
        [
            sys.executable,
            "scripts/train_temporal_classical.py",
            "--target", target,
            "--model", task["model"],
            "--seed", str(seed),
            "--config", str(task["config"]),
            "--output-dir", str(task["output_dir"]),
        ],
        text=True,
        capture_output=True,
        env=environment,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "launcher_train.stdout").write_text(process.stdout, encoding="utf-8")
    (run_dir / "launcher_train.stderr").write_text(process.stderr, encoding="utf-8")
    return {
        "target": target,
        "seed": seed,
        "status": "completed" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["svm", "rf"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eligible-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    targets = json.loads(args.eligible_targets.read_text(encoding="utf-8"))
    tasks = [
        {
            "target": target,
            "seed": int(seed),
            "model": args.model,
            "config": args.config,
            "output_dir": args.output_dir,
        }
        for target in targets
        for seed in config["seeds"]
    ]
    records: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
    summary = pd.DataFrame.from_records(records).sort_values(["target", "seed"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "launcher_summary.csv", index=False)
    if len(summary) != len(tasks) or (~summary["status"].isin(["completed", "existing"])).any():
        raise SystemExit(f"{args.model} temporal matrix incomplete or failed")


if __name__ == "__main__":
    main()
