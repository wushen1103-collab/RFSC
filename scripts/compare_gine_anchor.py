#!/usr/bin/env python3
"""Compare the modern GINE reproduction with the official MPNN magnitude anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.run_dir.glob("*/official/seed_*/evaluation.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["seed"] = int(path.parent.name.split("_")[-1])
        run = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
        payload.update(
            {
                "best_epoch": run["best_epoch"],
                "wall_time_seconds": run["wall_time_seconds"],
                "peak_vram_bytes": run["peak_vram_bytes"],
            }
        )
        rows.append(payload)
    raw = pd.DataFrame(rows)
    aggregate = raw.groupby("target").agg(
        gine_rmse=("overall_rmse", "mean"),
        gine_rmse_std=("overall_rmse", "std"),
        gine_cliff_rmse=("cliff_rmse", "mean"),
        gine_cliff_rmse_std=("cliff_rmse", "std"),
        seeds=("seed", "nunique"),
        median_best_epoch=("best_epoch", "median"),
        total_wall_time_seconds=("wall_time_seconds", "sum"),
        peak_vram_bytes=("peak_vram_bytes", "max"),
    ).reset_index()
    reference = pd.read_csv(args.reference)
    reference = reference[(reference["algorithm"] == "MPNN") & (reference["descriptor"] == "GRAPH")][
        ["dataset", "rmse", "cliff_rmse"]
    ].rename(columns={"dataset": "target", "rmse": "official_mpnn_rmse", "cliff_rmse": "official_mpnn_cliff_rmse"})
    table = aggregate.merge(reference, on="target", how="outer", validate="one_to_one")
    table["overall_relative_to_mpnn"] = (table["gine_rmse"] - table["official_mpnn_rmse"]) / table["official_mpnn_rmse"]
    table["cliff_relative_to_mpnn"] = (table["gine_cliff_rmse"] - table["official_mpnn_cliff_rmse"]) / table["official_mpnn_cliff_rmse"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "gine_seed_metrics.csv", index=False)
    table.to_csv(args.output_dir / "gine_anchor_table.csv", index=False)
    checks = {
        "targets_expected": 30,
        "targets_observed": int(len(table)),
        "all_targets_have_three_seeds": bool((table["seeds"] == 3).all()),
        "mean_overall_relative_to_official_mpnn": float(table["overall_relative_to_mpnn"].mean()),
        "mean_cliff_relative_to_official_mpnn": float(table["cliff_relative_to_mpnn"].mean()),
        "targets_gine_better_overall": int((table["overall_relative_to_mpnn"] < 0).sum()),
        "targets_gine_better_cliff": int((table["cliff_relative_to_mpnn"] < 0).sum()),
        "total_gpu_job_hours": float(raw["wall_time_seconds"].sum() / 3600.0),
        "maximum_peak_vram_gib": float(raw["peak_vram_bytes"].max() / 2**30),
    }
    checks["go"] = bool(
        checks["targets_observed"] == 30
        and checks["all_targets_have_three_seeds"]
        and checks["mean_overall_relative_to_official_mpnn"] <= 0.20
        and checks["mean_cliff_relative_to_official_mpnn"] <= 0.20
    )
    (args.output_dir / "gine_anchor_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
