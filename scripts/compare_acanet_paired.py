#!/usr/bin/env python3
"""Pair ACANet-v3 ACA-loss results with the same-seed GINE control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def bootstrap_mean_ci(values: pd.Series, *, seed: int = 20260718) -> list[float]:
    """Return a deterministic target-level percentile interval for the mean."""
    array = values.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(200_000, len(array)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aca-run-dir", type=Path, required=True)
    parser.add_argument("--gine-seed-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.aca_run_dir.glob("*/official/seed_*/evaluation.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["seed"] = int(path.parent.name.split("_")[-1])
        history = pd.read_json(path.parent / "history.jsonl", lines=True)
        run = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
        payload.update(
            {
                "aca_epochs": int(len(history)),
                "aca_epochs_with_active_triplets": int((history["aca_active_triplets"] > 0).sum()),
                "aca_active_triplets_total": int(history["aca_active_triplets"].sum()),
                "aca_wall_time_seconds": run["wall_time_seconds"],
                "aca_peak_vram_bytes": run["peak_vram_bytes"],
            }
        )
        rows.append(payload)
    aca = pd.DataFrame(rows).rename(
        columns={
            "overall_rmse": "aca_overall_rmse",
            "cliff_rmse": "aca_cliff_rmse",
            "overall_mae": "aca_overall_mae",
            "cliff_mae": "aca_cliff_mae",
        }
    )
    gine = pd.read_csv(args.gine_seed_table).rename(
        columns={
            "overall_rmse": "gine_overall_rmse",
            "cliff_rmse": "gine_cliff_rmse",
            "overall_mae": "gine_overall_mae",
            "cliff_mae": "gine_cliff_mae",
        }
    )
    keep = ["target", "seed", "gine_overall_rmse", "gine_cliff_rmse", "gine_overall_mae", "gine_cliff_mae"]
    paired = aca.merge(gine[keep], on=["target", "seed"], validate="one_to_one")
    paired["overall_relative_change"] = (paired["aca_overall_rmse"] - paired["gine_overall_rmse"]) / paired["gine_overall_rmse"]
    paired["cliff_relative_change"] = (paired["aca_cliff_rmse"] - paired["gine_cliff_rmse"]) / paired["gine_cliff_rmse"]
    table = paired.groupby("target").agg(
        aca_overall_rmse=("aca_overall_rmse", "mean"),
        gine_overall_rmse=("gine_overall_rmse", "mean"),
        overall_relative_change=("overall_relative_change", "mean"),
        aca_cliff_rmse=("aca_cliff_rmse", "mean"),
        gine_cliff_rmse=("gine_cliff_rmse", "mean"),
        cliff_relative_change=("cliff_relative_change", "mean"),
        seeds=("seed", "nunique"),
        active_triplets=("aca_active_triplets_total", "sum"),
    ).reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "acanet_seed_metrics.csv", index=False)
    table.to_csv(args.output_dir / "acanet_paired_table.csv", index=False)
    all_epochs_active = bool(
        (paired["aca_epochs"] == paired["aca_epochs_with_active_triplets"]).all()
    )
    overall_rel = table["overall_relative_change"]
    cliff_rel = table["cliff_relative_change"]
    checks = {
        "targets_expected": 30,
        "targets_observed": int(len(table)),
        "all_targets_have_three_seeds": bool((table["seeds"] == 3).all()),
        "all_training_epochs_have_active_triplets": all_epochs_active,
        "mean_overall_relative_change_vs_gine": float(paired["overall_relative_change"].mean()),
        "mean_cliff_relative_change_vs_gine": float(paired["cliff_relative_change"].mean()),
        "median_overall_relative_change_vs_gine": float(overall_rel.median()),
        "median_cliff_relative_change_vs_gine": float(cliff_rel.median()),
        "target_bootstrap95_mean_overall_relative_change": bootstrap_mean_ci(overall_rel),
        "target_bootstrap95_mean_cliff_relative_change": bootstrap_mean_ci(cliff_rel),
        "target_wilcoxon_two_sided_p_overall": float(wilcoxon(overall_rel).pvalue),
        "target_wilcoxon_two_sided_p_cliff": float(wilcoxon(cliff_rel).pvalue),
        "target_wilcoxon_aca_worse_p_overall": float(
            wilcoxon(overall_rel, alternative="greater").pvalue
        ),
        "target_wilcoxon_aca_worse_p_cliff": float(
            wilcoxon(cliff_rel, alternative="greater").pvalue
        ),
        "target_sign_test_two_sided_p_overall": float(
            binomtest(int((overall_rel < 0).sum()), len(overall_rel), 0.5).pvalue
        ),
        "target_sign_test_two_sided_p_cliff": float(
            binomtest(int((cliff_rel < 0).sum()), len(cliff_rel), 0.5).pvalue
        ),
        "targets_aca_better_overall": int((table["overall_relative_change"] < 0).sum()),
        "targets_aca_better_cliff": int((table["cliff_relative_change"] < 0).sum()),
        "total_gpu_job_hours": float(paired["aca_wall_time_seconds"].sum() / 3600.0),
        "maximum_peak_vram_gib": float(paired["aca_peak_vram_bytes"].max() / 2**30),
    }
    checks["go"] = bool(
        checks["targets_observed"] == 30
        and checks["all_targets_have_three_seeds"]
        and checks["all_training_epochs_have_active_triplets"]
        and checks["mean_overall_relative_change_vs_gine"] <= 0.10
        and checks["mean_cliff_relative_change_vs_gine"] <= 0.10
    )
    (args.output_dir / "acanet_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
