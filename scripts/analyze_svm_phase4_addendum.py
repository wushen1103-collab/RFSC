#!/usr/bin/env python3
"""Locked hierarchical inference for SVM addendum comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analyze_phase4_confirmatory import METRICS, hierarchical_bootstrap, holm_adjust, target_effects


CONTRASTS = [("gated_component_scaled", "svm_ecfp"), ("fixed_equal", "svm_ecfp")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2901)
    args = parser.parse_args()
    frame = pd.read_parquet(args.results)
    if frame.target.nunique() != 23 or "svm_ecfp" not in set(frame.method):
        raise RuntimeError("SVM addendum requires the complete 23-target joined matrix")
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
            test = wilcoxon(values, alternative="less", zero_method="wilcox")
            key = f"{contrast}:{metric}"
            raw_p[key] = float(test.pvalue)
            statistics[contrast][metric] = {
                "target_macro_relative_change": float(values.mean()),
                "bootstrap95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
                "improved_targets": int((values < 0).sum()),
                "targets": len(values),
                "wilcoxon_statistic": float(test.statistic),
                "wilcoxon_p_less": float(test.pvalue),
            }
    adjusted = holm_adjust(raw_p)
    for contrast, measures in statistics.items():
        for metric, record in measures.items():
            record["holm_adjusted_p"] = adjusted[f"{contrast}:{metric}"]
    payload = {
        "addendum": "public_global_mode_ecfp_svm",
        "iterations": args.iterations,
        "seed": args.seed,
        "resampling": "target_then_fold_then_seed_with_deterministic_svm_repeated_for_pairing",
        "statistics": statistics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(effects_all, ignore_index=True).to_csv(args.output_dir / "svm_target_relative_effects.csv", index=False)
    (args.output_dir / "svm_paired_statistics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
