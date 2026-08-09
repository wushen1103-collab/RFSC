#!/usr/bin/env python3
"""Lightweight integrity audit for the public RFSC release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ["overall_mae", "cliff_mae", "pair_delta_mae"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also require optional large regenerated artifacts when present.",
    )
    return parser.parse_args()


def require(path: str) -> Path:
    current = ROOT / path
    if not current.exists():
        raise RuntimeError(f"missing release artifact: {path}")
    return current


def assert_close(actual: np.ndarray, expected: np.ndarray, label: str, atol: float) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=atol):
        raise RuntimeError(f"table drift: {label}")


def verify_release_layout() -> int:
    required = [
        "README.md",
        "REPRODUCING.md",
        "pyproject.toml",
        "bps_gnn/projection.py",
        "bps_gnn/pair_flow.py",
        "scripts/build_expansion_tables.py",
        "scripts/draw_eswa_figures.py",
        "figure_data/figure_render_sources/fig2_domain_shift_selector_source.csv",
        "figures/final/Figure_1_final.pdf",
        "reproduction/main_tables.yaml",
    ]
    for path in required:
        require(path)
    return len(required)


def verify_decisions() -> int:
    checks = {
        "experiments/eswa_aqsol_external/evaluation/decision.json": [
            ("no_external_tuning", True),
        ],
        "experiments/eswa_response_free_risk/evaluation/decision.json": [
            ("external_scores_frozen_before_response_evaluation", True),
            ("useful_risk_control", True),
        ],
        "experiments/eswa_qm9_direct_pair/evaluation/decision.json": [
            ("held_out_response_opened_after_complete_hash_preflight", True),
            ("strong_mechanism_support", True),
        ],
        "results/delta_lgbm_10fold/decision.json": [
            ("robustness_go", True),
        ],
        "results/bps_backbone_transfer/decision.json": [
            ("broad_transfer_go", False),
        ],
    }
    verified = 0
    for relative, expectations in checks.items():
        record = json.loads(require(relative).read_text(encoding="utf-8"))
        for key, expected in expectations:
            if record.get(key) is not expected:
                raise RuntimeError(f"decision drift: {relative}/{key}")
            verified += 1
    return verified


def verify_benchmark_tables() -> int:
    verified = 0
    strict = pd.read_csv(
        require("paper/tables/benchmark_expansion/table_strict_scg_modern_baselines.csv")
    ).set_index("key")
    modern = pd.concat(
        [
            pd.read_csv(require("results/modern_baselines_scg/macro_metrics.csv")),
            pd.read_csv(require("results/chemprop_dmpnn_scg/macro_metrics.csv")),
        ],
        ignore_index=True,
    ).set_index("method")
    for method, row in modern.iterrows():
        if method not in strict.index:
            raise RuntimeError(f"modern baseline missing from release table: {method}")
        paper_row = strict.loc[method]
        for column in [
            "n_targets",
            "n_runs",
            "n_seeds",
            "overall_mae_mean",
            "overall_mae_std",
            "cliff_mae_mean",
            "cliff_mae_std",
            "pair_delta_mae_mean",
            "pair_delta_mae_std",
        ]:
            assert_close(
                np.atleast_1d(paper_row[column]),
                np.atleast_1d(row[column]),
                f"strict-SCG {method}/{column}",
                5e-12,
            )
    if len(strict) != 13:
        raise RuntimeError("strict-SCG table must contain 13 method rows")
    verified += 1

    ood_table = pd.read_csv(
        require("paper/tables/benchmark_expansion/table_temporal_ood_macro.csv")
    )
    ood_source = []
    for benchmark, relative in [
        ("ChEMBL36 earliest-year temporal", "results/modern_temporal_ood/chembl36/macro_mean_std.csv"),
        ("BindingDB patents latest-era", "results/modern_temporal_ood/bindingdb/macro_mean_std.csv"),
    ]:
        frame = pd.read_csv(require(relative))
        frame.insert(0, "benchmark", benchmark)
        ood_source.append(frame)
    ood_source = pd.concat(ood_source, ignore_index=True)
    key = ["benchmark", "method"]
    ood_table = ood_table.sort_values(key).reset_index(drop=True)
    ood_source = ood_source.sort_values(key).reset_index(drop=True)
    if not ood_table[key].equals(ood_source[key]) or len(ood_table) != 16:
        raise RuntimeError("OOD table key drift")
    for column in ood_source.columns.difference(key):
        assert_close(
            ood_table[column].to_numpy(),
            ood_source[column].to_numpy(),
            f"OOD {column}",
            5e-12,
        )
    verified += 1

    transfer_table = pd.read_csv(
        require("paper/tables/benchmark_expansion/table_bps_backbone_transfer.csv")
    )
    transfer_source = pd.read_csv(
        require("results/bps_backbone_transfer/architecture_summary.csv")
    ).sort_values("architecture")
    transfer_arch = transfer_table[transfer_table["scope"] != "family macro"].sort_values(
        "scope"
    )
    if transfer_arch["scope"].tolist() != transfer_source["architecture"].tolist():
        raise RuntimeError("BPS transfer architecture drift")
    for column in [
        "overall_mae_relative_change",
        "cliff_mae_relative_change",
        "pair_delta_mae_relative_change",
    ]:
        assert_close(
            transfer_arch[column].to_numpy(),
            transfer_source[column].to_numpy(),
            f"BPS transfer {column}",
            5e-12,
        )
    verified += 1

    coverage = pd.read_csv(
        require("paper/tables/benchmark_expansion/table_experiment_coverage.csv")
    )
    if coverage["status"].isna().any() or coverage["audit_axis"].nunique() != len(coverage):
        raise RuntimeError("experiment-coverage table drift")
    verified += 1
    return verified


def verify_figure_sources() -> int:
    required = [
        "figure_data/node_effects_moleculeace.csv",
        "figure_data/node_effects_chembl.csv",
        "figure_data/node_effects_aqsoldb.csv",
        "figure_data/selector_features.csv",
        "figure_data/deepdelta_pair_metrics.csv",
        "figure_data/delta_lightgbm_pair_metrics.csv",
        "figure_data/qm9_pair_metrics.csv",
        "figure_data/structured_missingness.csv",
        "figure_data/runtime_scaling.csv",
    ]
    for path in required:
        frame = pd.read_csv(require(path))
        if frame.empty:
            raise RuntimeError(f"empty figure source: {path}")
    return len(required)


def main() -> None:
    args = parse_args()
    report = {
        "layout_files_verified": verify_release_layout(),
        "decision_checks_verified": verify_decisions(),
        "benchmark_tables_verified": verify_benchmark_tables(),
        "figure_sources_verified": verify_figure_sources(),
        "strict_mode": args.strict,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
