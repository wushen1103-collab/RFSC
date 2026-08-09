#!/usr/bin/env python3
"""Run train-only learned-gate development jobs across the frozen SCG panel."""

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
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    return parser.parse_args()


def run_task(task: dict, available: queue.Queue) -> dict:
    target, fold = task["target"], task["fold"]
    output = task["output_dir"] / target / f"fold_{fold}"
    if (output / "test_evaluation.csv").exists():
        return {"target": target, "fold": fold, "status": "existing", "gpu": None}
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
        train_command = [
            sys.executable,
            "scripts/train_learned_gate_fold.py",
            "--target",
            target,
            "--fold",
            str(fold),
            "--base-run-dir",
            str(task["base_run_dir"]),
            "--base-config",
            str(task["base_config"]),
            "--gate-config",
            str(task["gate_config"]),
            "--output-dir",
            str(task["output_dir"]),
            "--device",
            "cuda",
        ]
        trained = subprocess.run(train_command, text=True, capture_output=True, env=environment)
        output.mkdir(parents=True, exist_ok=True)
        (output / "launcher_train.stdout").write_text(trained.stdout, encoding="utf-8")
        (output / "launcher_train.stderr").write_text(trained.stderr, encoding="utf-8")
        if trained.returncode:
            return {"target": target, "fold": fold, "status": "train_failed", "gpu": gpu, "returncode": trained.returncode}
        for partition in ["validation", "test"]:
            evaluated = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_learned_gate.py",
                    "--target",
                    target,
                    "--fold",
                    str(fold),
                    "--partition",
                    partition,
                    "--predictions",
                    str(output / f"{partition}_predictions.parquet"),
                    "--gate",
                    str(output / f"{partition}_gate.parquet"),
                    "--output-dir",
                    str(output),
                ],
                text=True,
                capture_output=True,
                env=environment,
            )
            (output / f"launcher_{partition}_eval.stdout").write_text(
                evaluated.stdout, encoding="utf-8"
            )
            (output / f"launcher_{partition}_eval.stderr").write_text(
                evaluated.stderr, encoding="utf-8"
            )
            if evaluated.returncode:
                return {"target": target, "fold": fold, "status": f"{partition}_evaluation_failed", "gpu": gpu, "returncode": evaluated.returncode}
        return {"target": target, "fold": fold, "status": "completed", "gpu": gpu, "returncode": 0}
    finally:
        available.put(gpu)


def main() -> None:
    args = parse_args()
    audit = pd.read_csv(args.audit).set_index("target")
    tasks = []
    for target in sorted(args.targets):
        folds = audit.loc[target, "scg_folds"]
        if pd.isna(folds):
            raise ValueError(f"SCG-infeasible target requested: {target}")
        tasks.extend(
            {
                "target": target,
                "fold": fold,
                "base_run_dir": args.base_run_dir,
                "base_config": args.base_config,
                "gate_config": args.gate_config,
                "output_dir": args.output_dir,
            }
            for fold in range(int(folds))
        )
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
    failures = summary[~summary["status"].isin(["completed", "existing"])]
    if len(failures):
        raise SystemExit(f"{len(failures)} learned-gate jobs failed")


if __name__ == "__main__":
    main()
