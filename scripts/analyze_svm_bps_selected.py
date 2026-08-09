#!/usr/bin/env python3
"""Hierarchical paired inference for the frozen SVM-BPS candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

from analyze_phase4_confirmatory import METRICS, hierarchical_bootstrap, holm_adjust, target_effects


CONTRASTS = [
    ("svm_bps_selected", "svm_ecfp"),
    ("svm_bps_selected", "svm_fixed_mixture"),
    ("svm_bps_selected", "fixed_equal"),
    ("svm_bps_selected", "gated_component_scaled"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2901)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frame = pd.read_parquet(args.results)
    expected_targets = len(config["scope"]["targets"])
    if frame.target.nunique() != expected_targets or "svm_bps_selected" not in set(frame.method):
        raise RuntimeError("SVM-BPS inference matrix is incomplete")
    rng = np.random.default_rng(args.seed)
    statistics = {}
    raw_p = {}
    effects_all = []
    for candidate, baseline in CONTRASTS:
        contrast = f"{candidate}_vs_{baseline}"
        effects = target_effects(frame, candidate, baseline)
        effects_all.append(effects)
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
        "scope": config["scope"]["labels"],
        "iterations": args.iterations,
        "seed": args.seed,
        "resampling": "target_then_fold_then_seed_with_deterministic_svm_repeated_for_pairing",
        "statistics": statistics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(effects_all, ignore_index=True).to_csv(args.output_dir / "target_relative_effects.csv", index=False)
    (args.output_dir / "paired_statistics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
