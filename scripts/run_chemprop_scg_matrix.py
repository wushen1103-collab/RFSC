#!/usr/bin/env python3
"""Run official Chemprop D-MPNN over the frozen SCG matrix."""

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

try:
    from run_gine_matrix import select_targets
except ModuleNotFoundError:
    from scripts.run_gine_matrix import select_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--targets", nargs="*")
    parser.add_argument("--folds", nargs="*", type=int)
    parser.add_argument("--seeds", nargs="*", type=int)
    return parser.parse_args()


def task_environment(gpu: str) -> dict[str, str]:
    environment = os.environ.copy()
    project_root = str(Path.cwd().resolve())
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
            # Chemprop 2.2.1/Lightning reloads the checkpoint produced by this
            # same local process. PyTorch >=2.6 otherwise defaults torch.load
            # to weights_only=True, which rejects Chemprop metric classes.
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "PYTHONPATH": os.pathsep.join(
                value
                for value in [project_root, environment.get("PYTHONPATH", "")]
                if value
            ),
        }
    )
    return environment


def run_task(task: dict[str, object], available: queue.Queue) -> dict[str, object]:
    target, fold, seed = str(task["target"]), int(task["fold"]), int(task["seed"])
    run_dir = Path(task["output_dir"]) / target / f"fold_{fold}" / f"seed_{seed}"
    evaluation = run_dir / "evaluation.json"
    if evaluation.exists():
        return {"target": target, "fold": fold, "seed": seed, "status": "existing"}
    gpu = available.get()
    try:
        environment = task_environment(str(gpu))
        trained = subprocess.run(
            [
                sys.executable,
                "scripts/train_chemprop_scg.py",
                "--target",
                target,
                "--fold",
                str(fold),
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
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "launcher_train.stdout").write_text(trained.stdout, encoding="utf-8")
        (run_dir / "launcher_train.stderr").write_text(trained.stderr, encoding="utf-8")
        if trained.returncode:
            return {
                "target": target,
                "fold": fold,
                "seed": seed,
                "status": "train_failed",
                "gpu": gpu,
                "returncode": trained.returncode,
            }
        evaluated = subprocess.run(
            [
                sys.executable,
                "scripts/evaluate_predictions.py",
                "--target",
                target,
                "--protocol",
                "scg",
                "--predictions",
                str(run_dir / "predictions.parquet"),
                "--output",
                str(evaluation),
            ],
            text=True,
            capture_output=True,
            env=environment,
        )
        (run_dir / "launcher_eval.stdout").write_text(evaluated.stdout, encoding="utf-8")
        (run_dir / "launcher_eval.stderr").write_text(evaluated.stderr, encoding="utf-8")
        return {
            "target": target,
            "fold": fold,
            "seed": seed,
            "status": "completed" if evaluated.returncode == 0 else "evaluation_failed",
            "gpu": gpu,
            "returncode": evaluated.returncode,
        }
    finally:
        available.put(gpu)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    audit = pd.read_csv(args.audit)
    targets = select_targets(args.targets, audit, "scg")
    seeds = args.seeds or [int(value) for value in config["seeds"]]
    tasks = []
    for target in targets:
        n_folds = int(audit.loc[audit.target == target, "scg_folds"].iloc[0])
        folds = args.folds if args.folds is not None else list(range(n_folds))
        tasks.extend(
            {
                "target": target,
                "fold": fold,
                "seed": seed,
                "config": args.config,
                "output_dir": args.output_dir,
            }
            for fold in folds
            if fold < n_folds
            for seed in seeds
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
    summary = pd.DataFrame(records).sort_values(["target", "fold", "seed"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "launcher_summary.csv", index=False)
    failures = summary[~summary.status.isin(["completed", "existing"])]
    if len(failures):
        raise SystemExit(f"{len(failures)} Chemprop jobs failed")


if __name__ == "__main__":
    main()
