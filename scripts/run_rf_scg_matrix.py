#!/usr/bin/env python3
"""Run the six-target ECFP-RF strict-SCG matrix."""

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
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=24)
    return parser.parse_args()


def run_task(task: dict) -> dict:
    target, fold, seed = task["target"], task["fold"], task["seed"]
    output = task["output_dir"] / target / f"fold_{fold}" / f"seed_{seed}"
    if (output / "evaluation.json").exists():
        return {"target": target, "fold": fold, "seed": seed, "status": "existing"}
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    train = subprocess.run(
        [sys.executable, "scripts/train_rf_scg.py", "--target", target, "--fold", str(fold), "--seed", str(seed), "--config", str(task["config"]), "--output-dir", str(task["output_dir"])],
        text=True, capture_output=True, env=environment,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "launcher_train.stdout").write_text(train.stdout, encoding="utf-8")
    (output / "launcher_train.stderr").write_text(train.stderr, encoding="utf-8")
    if train.returncode:
        return {"target": target, "fold": fold, "seed": seed, "status": "train_failed", "returncode": train.returncode}
    evaluate = subprocess.run(
        [sys.executable, "scripts/evaluate_predictions.py", "--target", target, "--protocol", "scg", "--predictions", str(output / "predictions.parquet"), "--output", str(output / "evaluation.json")],
        text=True, capture_output=True, env=environment,
    )
    (output / "launcher_eval.stdout").write_text(evaluate.stdout, encoding="utf-8")
    (output / "launcher_eval.stderr").write_text(evaluate.stderr, encoding="utf-8")
    return {"target": target, "fold": fold, "seed": seed, "status": "completed" if evaluate.returncode == 0 else "evaluation_failed", "returncode": evaluate.returncode}


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    audit = pd.read_csv(args.audit).set_index("target")
    tasks = []
    for target in sorted(args.targets):
        folds = audit.loc[target, "scg_folds"]
        if pd.isna(folds):
            raise ValueError(f"SCG-infeasible target: {target}")
        tasks.extend({"target": target, "fold": fold, "seed": int(seed), "config": args.config, "output_dir": args.output_dir} for fold in range(int(folds)) for seed in config["seeds"])
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
    summary = pd.DataFrame(records).sort_values(["target", "fold", "seed"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "launcher_summary.csv", index=False)
    if (~summary["status"].isin(["completed", "existing"])).any():
        raise SystemExit("RF-SCG matrix contains failed jobs")


if __name__ == "__main__":
    main()

