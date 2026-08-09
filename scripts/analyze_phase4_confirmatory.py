#!/usr/bin/env python3
"""Run locked target/fold/seed paired inference for Phase 4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRICS = ["overall_mae", "cliff_mae", "pair_delta_mae"]
CONTRASTS = [
    ("gated_component_scaled", "fixed_equal"),
    ("gated_component_scaled", "gine"),
    ("fixed_equal", "gine"),
    ("fixed_equal", "rf"),
]


def target_effects(frame: pd.DataFrame, candidate: str, baseline: str) -> pd.DataFrame:
    means = frame[frame.method.isin([candidate, baseline])].groupby(["target", "method"], as_index=False)[METRICS].mean()
    left = means[means.method == candidate].set_index("target")
    right = means[means.method == baseline].set_index("target")
    if not left.index.equals(right.index):
        raise RuntimeError(f"Unpaired targets for {candidate} vs {baseline}")
    return pd.DataFrame(
        {
            "target": left.index,
            "candidate": candidate,
            "baseline": baseline,
            **{metric: (left[metric] / right[metric] - 1.0).to_numpy() for metric in METRICS},
        }
    )


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    candidate: str,
    baseline: str,
    metric: str,
    rng: np.random.Generator,
    iterations: int,
) -> np.ndarray:
    paired = frame[frame.method.isin([candidate, baseline])].pivot(
        index=["target", "fold", "seed"], columns="method", values=metric
    )
    targets = paired.index.get_level_values("target").unique().to_numpy()
    samples = np.empty(iterations, dtype=float)
    by_target = {}
    for target in targets:
        target_frame = paired.xs(target, level="target")
        folds = target_frame.index.get_level_values("fold").unique().to_numpy()
        by_fold = {}
        for fold in folds:
            fold_frame = target_frame.xs(fold, level="fold")
            by_fold[fold] = (
                fold_frame.index.to_numpy(),
                fold_frame[candidate].to_numpy(),
                fold_frame[baseline].to_numpy(),
            )
        by_target[target] = (folds, by_fold)
    for iteration in range(iterations):
        sampled_targets = rng.choice(targets, size=len(targets), replace=True)
        effects = []
        for target in sampled_targets:
            folds, by_fold = by_target[target]
            sampled_folds = rng.choice(folds, size=len(folds), replace=True)
            candidate_values = []
            baseline_values = []
            for fold in sampled_folds:
                seeds, fold_candidate, fold_baseline = by_fold[fold]
                sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
                positions = np.searchsorted(seeds, sampled_seeds)
                candidate_values.extend(fold_candidate[positions])
                baseline_values.extend(fold_baseline[positions])
            effects.append(np.mean(candidate_values) / np.mean(baseline_values) - 1.0)
        samples[iteration] = float(np.mean(effects))
    return samples


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, (total - rank) * p_values[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2901)
    args = parser.parse_args()
    frame = pd.read_parquet(args.results)
    required = {"target", "fold", "seed", "method", *METRICS}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if frame.target.nunique() != 23:
        raise RuntimeError(f"Confirmatory inference requires 23 targets, observed {frame.target.nunique()}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_effects = []
    statistics = {}
    raw_p = {}
    rng = np.random.default_rng(args.seed)
    for candidate, baseline in CONTRASTS:
        contrast = f"{candidate}_vs_{baseline}"
        effects = target_effects(frame, candidate, baseline)
        all_effects.append(effects)
        statistics[contrast] = {}
        for metric in METRICS:
            values = effects[metric].to_numpy()
            bootstrap = hierarchical_bootstrap(frame, candidate, baseline, metric, rng, args.iterations)
            try:
                test = wilcoxon(values, alternative="less", zero_method="wilcox")
                p_value = float(test.pvalue)
                statistic = float(test.statistic)
            except ValueError:
                p_value = 1.0
                statistic = 0.0
            key = f"{contrast}:{metric}"
            raw_p[key] = p_value
            statistics[contrast][metric] = {
                "target_macro_relative_change": float(values.mean()),
                "bootstrap95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
                "improved_targets": int((values < 0).sum()),
                "targets": len(values),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_less": p_value,
            }
    adjusted = holm_adjust(raw_p)
    for contrast, measures in statistics.items():
        for metric, record in measures.items():
            record["holm_adjusted_p"] = adjusted[f"{contrast}:{metric}"]
    payload = {
        "confirmatory": True,
        "iterations": args.iterations,
        "seed": args.seed,
        "resampling": "target_then_fold_then_seed_with_replacement",
        "statistics": statistics,
    }
    pd.concat(all_effects, ignore_index=True).to_csv(args.output_dir / "target_relative_effects.csv", index=False)
    (args.output_dir / "paired_statistics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
