#!/usr/bin/env python3
"""Run a locked official- or SCG-split GINE matrix on the requested GPUs."""

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
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--protocol", choices=["official", "scg"], default="official")
    parser.add_argument(
        "--targets",
        nargs="*",
        help="Optional frozen target subset; defaults to every target in --audit.",
    )
    parser.add_argument("--folds", nargs="*", type=int, help="Optional smoke fold subset")
    parser.add_argument("--seeds", nargs="*", type=int, help="Optional frozen seed subset")
    return parser.parse_args()


def run_task(task: dict, available: queue.Queue) -> dict:
    target, seed, fold = task["target"], task["seed"], task["fold"]
    run_name = f"fold_{fold}" if task["protocol"] == "scg" else "official"
    run_dir = task["output_dir"] / target / run_name / f"seed_{seed}"
    evaluation = run_dir / "evaluation.json"
    if evaluation.exists():
        return {
            "target": target,
            "fold": fold,
            "seed": seed,
            "status": "existing",
            "gpu": None,
        }
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
            "scripts/train_gine.py",
            "--target",
            target,
            "--protocol",
            task["protocol"],
            "--config",
            str(task["config"]),
            "--seed",
            str(seed),
            "--output-dir",
            str(task["output_dir"]),
            "--device",
            "cuda",
        ]
        if fold is not None:
            train_command.extend(["--fold", str(fold)])
        trained = subprocess.run(
            train_command,
            text=True,
            capture_output=True,
            env=environment,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "launcher_train.stdout").write_text(trained.stdout, encoding="utf-8")
        (run_dir / "launcher_train.stderr").write_text(trained.stderr, encoding="utf-8")
        if trained.returncode:
            return {"target": target, "fold": fold, "seed": seed, "status": "train_failed", "gpu": gpu, "returncode": trained.returncode}
        evaluated = subprocess.run(
            [
                sys.executable,
                "scripts/evaluate_predictions.py",
                "--target",
                target,
                "--protocol",
                task["protocol"],
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


def select_targets(
    requested: list[str] | None,
    audit: pd.DataFrame,
    protocol: str,
) -> list[str]:
    if requested:
        targets = sorted(requested)
    elif protocol == "scg":
        targets = sorted(audit.loc[audit["scg_folds"].notna(), "target"].tolist())
    else:
        targets = sorted(audit["target"].tolist())
    unknown = sorted(set(targets) - set(audit["target"]))
    if unknown:
        raise ValueError(f"Targets missing from audit: {unknown}")
    return targets


def select_subset(requested: list[int] | None, allowed: list[int], name: str) -> list[int]:
    if requested is None:
        return allowed
    if not requested:
        raise ValueError(f"Explicit {name} subset cannot be empty")
    unknown = sorted(set(requested) - set(allowed))
    if unknown:
        raise ValueError(f"Invalid {name}: {unknown}; allowed={allowed}")
    return sorted(set(requested))


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    audit = pd.read_csv(args.audit)
    targets = select_targets(args.targets, audit, args.protocol)
    seeds = select_subset(args.seeds, [int(value) for value in config["seeds"]], "seeds")
    tasks = []
    for target in targets:
        if args.protocol == "scg":
            value = audit.loc[audit["target"] == target, "scg_folds"].iloc[0]
            if pd.isna(value):
                raise ValueError(f"SCG-infeasible target requested: {target}")
            folds = select_subset(args.folds, list(range(int(value))), f"folds for {target}")
        else:
            folds = [None]
        tasks.extend(
            {
                "target": target,
                "fold": fold,
                "seed": seed,
                "protocol": args.protocol,
                "config": args.config,
                "output_dir": args.output_dir,
            }
            for fold in folds
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
    summary = pd.DataFrame(records).sort_values(["target", "fold", "seed"], na_position="first")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "launcher_summary.csv", index=False)
    failures = summary[~summary["status"].isin(["completed", "existing"])]
    if len(failures):
        raise SystemExit(f"{len(failures)} GINE jobs failed; see launcher_summary.csv")


if __name__ == "__main__":
    main()
