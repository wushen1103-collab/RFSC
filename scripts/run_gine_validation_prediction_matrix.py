#!/usr/bin/env python3
"""Build missing response-free GINE validation predictions on audited GPUs."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eligible-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    return parser.parse_args()


def run_task(task: dict[str, object], available: queue.Queue) -> dict[str, object]:
    target, seed = str(task["target"]), int(task["seed"])
    run_dir = Path(task["output_dir"]) / target / "temporal" / f"seed_{seed}"
    if (run_dir / "validation_predictions.parquet.sha256.json").exists():
        return {"target": target, "seed": seed, "status": "existing", "gpu": None}
    gpu = available.get()
    try:
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "NUMEXPR_NUM_THREADS": "4",
            }
        )
        process = subprocess.run(
            [
                sys.executable,
                "scripts/build_gine_validation_predictions.py",
                "--target",
                target,
                "--seed",
                str(seed),
                "--config",
                str(task["config"]),
                "--output-dir",
                str(task["output_dir"]),
                "--device",
                "cuda",
            ],
            text=True,
            capture_output=True,
            env=environment,
        )
        (run_dir / "validation_launcher.stdout").write_text(process.stdout, encoding="utf-8")
        (run_dir / "validation_launcher.stderr").write_text(process.stderr, encoding="utf-8")
        return {
            "target": target,
            "seed": seed,
            "status": "completed" if process.returncode == 0 else "failed",
            "gpu": gpu,
            "returncode": process.returncode,
        }
    finally:
        available.put(gpu)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    targets = json.loads(args.eligible_targets.read_text(encoding="utf-8"))
    tasks = [
        {
            "target": target,
            "seed": int(seed),
            "config": args.config,
            "output_dir": args.output_dir,
        }
        for target in targets
        for seed in config["seeds"]
    ]
    available: queue.Queue = queue.Queue()
    for gpu in args.gpus:
        available.put(gpu)
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(run_task, task, available) for task in tasks]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
    summary = pd.DataFrame.from_records(records).sort_values(["target", "seed"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "validation_prediction_launcher_summary.csv", index=False)
    if len(summary) != len(tasks) or (~summary["status"].isin(["completed", "existing"])).any():
        raise SystemExit("GINE validation prediction matrix incomplete or failed")


if __name__ == "__main__":
    main()
