#!/usr/bin/env python3
"""Run the frozen temporal GINE matrix over an audited set of idle GPUs."""

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


def run_task(task: dict, available: queue.Queue) -> dict[str, object]:
    target, seed = task["target"], task["seed"]
    run_dir = task["output_dir"] / target / "temporal" / f"seed_{seed}"
    if (run_dir / "predictions.parquet").exists():
        return {"target": target, "seed": seed, "status": "existing", "gpu": None}
    gpu = available.get()
    try:
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "NUMEXPR_NUM_THREADS": "4",
            }
        )
        process = subprocess.run(
            [
                sys.executable,
                "scripts/train_gine.py",
                "--target", target,
                "--protocol", "temporal",
                "--config", str(task["config"]),
                "--seed", str(seed),
                "--output-dir", str(task["output_dir"]),
                "--device", "cuda",
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
            "gpu": gpu,
            "returncode": process.returncode,
        }
    finally:
        available.put(gpu)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eligible-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--targets", nargs="*", help="Optional audited smoke subset")
    return parser.parse_args()


def select_targets(requested: list[str] | None, eligible: list[str]) -> list[str]:
    if not requested:
        return eligible
    unknown = sorted(set(requested) - set(eligible))
    if unknown:
        raise ValueError(f"Targets absent from frozen eligibility list: {unknown}")
    return sorted(set(requested))


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    eligible = json.loads(args.eligible_targets.read_text(encoding="utf-8"))
    targets = select_targets(args.targets, eligible)
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
    summary.to_csv(args.output_dir / "launcher_summary.csv", index=False)
    if len(summary) != len(tasks) or (~summary["status"].isin(["completed", "existing"])).any():
        raise SystemExit("Temporal GINE matrix incomplete or failed")


if __name__ == "__main__":
    main()
