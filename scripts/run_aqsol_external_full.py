#!/usr/bin/env python3
"""Resume-safe parallel launcher for the frozen AqSolDB external matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


TARGET = "AQSOLDB_SOLUBILITY"
SEEDS = [0, 1, 2]


def complete(path: Path) -> bool:
    return path.exists() and path.with_suffix(path.suffix + ".sha256.json").exists()


def run(command: list[str], log: Path, env: dict[str, str] | None = None) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, check=False)
    return {"command": command, "log": str(log), "returncode": result.returncode}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("experiments/eswa_aqsol_external/full"))
    parser.add_argument("--gpus", default="1,2")
    parser.add_argument("--classical-workers", type=int, default=8)
    parser.add_argument("--gine-workers", type=int, default=4)
    args = parser.parse_args()
    folds = sorted(int(value) for value in pd.read_parquet(Path("folds/scg") / TARGET / "outer_folds.parquet")["outer_fold"].unique())
    python = sys.executable
    classical, neural = [], []
    svm_root, rf_root, gine_root = args.output_root / "svm", args.output_root / "rf", args.output_root / "gine"
    for fold in folds:
        if not complete(svm_root / TARGET / f"fold_{fold}" / "seed_0" / "predictions.parquet"):
            classical.append(([python, "scripts/train_svm_scg.py", "--target", TARGET, "--fold", str(fold), "--config", "configs/baselines/svm_aqsol_external.yaml", "--output-dir", str(svm_root)], args.output_root / "logs" / f"svm_fold{fold}.log", None))
        for seed in SEEDS:
            if not complete(rf_root / TARGET / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet"):
                classical.append(([python, "scripts/train_rf_scg.py", "--target", TARGET, "--fold", str(fold), "--seed", str(seed), "--config", "configs/baselines/rf_aqsol_external.yaml", "--output-dir", str(rf_root)], args.output_root / "logs" / f"rf_fold{fold}_seed{seed}.log", None))
            if not complete(gine_root / TARGET / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet"):
                neural.append((fold, seed))

    results = []
    with ThreadPoolExecutor(max_workers=args.classical_workers) as pool:
        futures = [pool.submit(run, *task) for task in classical]
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            print(json.dumps(result), flush=True)
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    with ThreadPoolExecutor(max_workers=args.gine_workers) as pool:
        futures = []
        for index, (fold, seed) in enumerate(neural):
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpus[index % len(gpus)]
            command = [python, "scripts/train_gine.py", "--target", TARGET, "--protocol", "scg", "--fold", str(fold), "--config", "configs/baselines/gine_aqsol_external.yaml", "--seed", str(seed), "--output-dir", str(gine_root), "--device", "cuda"]
            futures.append(pool.submit(run, command, args.output_root / "logs" / f"gine_fold{fold}_seed{seed}.log", env))
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            print(json.dumps(result), flush=True)
    failures = [result for result in results if result["returncode"] != 0]
    summary = {"folds": folds, "classical_jobs": len(classical), "gine_jobs": len(neural), "failures": failures}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "launcher_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(f"{len(failures)} AqSolDB jobs failed")


if __name__ == "__main__":
    main()
