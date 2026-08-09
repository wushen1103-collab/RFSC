#!/usr/bin/env python3
"""Run the locked 30-target x 3-seed ECFP-RF reproduction matrix."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def run_task(task: dict) -> dict:
    target, seed = task["target"], task["seed"]
    run_dir = task["output_dir"] / target / f"seed_{seed}"
    evaluation = run_dir / "evaluation.json"
    if evaluation.exists():
        return {"target": target, "seed": seed, "status": "existing"}
    command = [
        sys.executable,
        "scripts/train_rf_official.py",
        "--target",
        target,
        "--config",
        str(task["config"]),
        "--seed",
        str(seed),
        "--output-dir",
        str(task["output_dir"]),
    ]
    if task.get("selection_csv"):
        command.extend(["--selection-csv", str(task["selection_csv"])])
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    trained = subprocess.run(command, text=True, capture_output=True, env=environment)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "launcher_train.stdout").write_text(trained.stdout, encoding="utf-8")
    (run_dir / "launcher_train.stderr").write_text(trained.stderr, encoding="utf-8")
    if trained.returncode:
        return {"target": target, "seed": seed, "status": "train_failed", "returncode": trained.returncode}
    evaluate = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_predictions.py",
            "--target",
            target,
            "--protocol",
            "official",
            "--predictions",
            str(run_dir / "predictions.parquet"),
            "--output",
            str(evaluation),
        ],
        text=True,
        capture_output=True,
        env=environment,
    )
    (run_dir / "launcher_eval.stdout").write_text(evaluate.stdout, encoding="utf-8")
    (run_dir / "launcher_eval.stderr").write_text(evaluate.stderr, encoding="utf-8")
    status = "completed" if evaluate.returncode == 0 else "evaluation_failed"
    return {"target": target, "seed": seed, "status": status, "returncode": evaluate.returncode}


def execute(tasks: list[dict], workers: int) -> list[dict]:
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
    return records


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    targets = sorted(pd.read_csv(args.audit)["target"].tolist())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_one = [
        {"target": target, "seed": config["seeds"][0], "output_dir": args.output_dir, "config": args.config}
        for target in targets
    ]
    records = execute(stage_one, args.workers)
    failed_stage_one = {record["target"] for record in records if record["status"] not in {"completed", "existing"}}
    stage_two = []
    for target in targets:
        if target in failed_stage_one:
            continue
        selection = args.output_dir / target / f"seed_{config['seeds'][0]}" / "model_selection.csv"
        for seed in config["seeds"][1:]:
            stage_two.append(
                {
                    "target": target,
                    "seed": seed,
                    "output_dir": args.output_dir,
                    "config": args.config,
                    "selection_csv": selection,
                }
            )
    records.extend(execute(stage_two, args.workers))
    summary = pd.DataFrame(records).sort_values(["target", "seed"])
    summary.to_csv(args.output_dir / "launcher_summary.csv", index=False)
    failures = summary[~summary["status"].isin(["completed", "existing"])]
    if len(failures):
        raise SystemExit(f"{len(failures)} RF jobs failed; see launcher_summary.csv")


if __name__ == "__main__":
    main()

