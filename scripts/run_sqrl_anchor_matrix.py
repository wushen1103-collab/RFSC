#!/usr/bin/env python3
"""Run labeled-anchor validation folds across the frozen panel."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--anchor-config", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    return parser.parse_args()


def run_task(task: dict, available: queue.Queue) -> dict:
    target, fold = task["target"], task["fold"]
    output = task["output_dir"] / target / f"fold_{fold}"
    if (output / "validation_predictions.parquet").exists():
        return {"target": target, "fold": fold, "status": "existing", "gpu": None}
    gpu = available.get()
    try:
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4"})
        command = [sys.executable, "scripts/train_sqrl_anchor_fold.py", "--target", target, "--fold", str(fold), "--base-run-dir", str(task["base_run_dir"]), "--base-config", str(task["base_config"]), "--anchor-config", str(task["anchor_config"]), "--output-dir", str(task["output_dir"]), "--device", "cuda"]
        result = subprocess.run(command, text=True, capture_output=True, env=environment)
        output.mkdir(parents=True, exist_ok=True)
        (output / "launcher.stdout").write_text(result.stdout, encoding="utf-8")
        (output / "launcher.stderr").write_text(result.stderr, encoding="utf-8")
        return {"target": target, "fold": fold, "status": "completed" if result.returncode == 0 else "failed", "gpu": gpu, "returncode": result.returncode}
    finally:
        available.put(gpu)


def main() -> None:
    args = parse_args()
    audit = pd.read_csv(args.audit).set_index("target")
    tasks = []
    for target in sorted(args.targets):
        folds = audit.loc[target, "scg_folds"]
        if pd.isna(folds):
            raise ValueError(f"SCG-infeasible target: {target}")
        tasks.extend({"target": target, "fold": fold, "base_run_dir": args.base_run_dir, "base_config": args.base_config, "anchor_config": args.anchor_config, "output_dir": args.output_dir} for fold in range(int(folds)))
    available: queue.Queue = queue.Queue()
    for gpu in args.gpus:
        available.put(gpu)
    records = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = {pool.submit(run_task, task, available): task for task in tasks}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
    summary = pd.DataFrame(records).sort_values(["target", "fold"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "launcher_summary.csv", index=False)
    if (~summary["status"].isin(["completed", "existing"])).any():
        raise SystemExit("Labeled-anchor matrix contains failed jobs")


if __name__ == "__main__":
    main()

