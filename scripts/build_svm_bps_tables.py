#!/usr/bin/env python3
"""Build SVM-BPS paper tables for development, secondary, and all-29 scopes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = ["overall_mae", "cliff_mae", "pair_delta_mae"]
METHOD_LABELS = {
    "svm_bps_selected": "BPS-GNN+ (selected)",
    "svm_ecfp": "ECFP4-SVM",
    "gated_component_scaled": "BPS-GNN gated scale",
    "fixed_equal": "RF/GINE equal",
    "svm_fixed_mixture": "SVM/RF/GINE equal-shape mix",
}


def markdown(frame: pd.DataFrame) -> str:
    values = frame.copy()
    for column in values.columns:
        if pd.api.types.is_numeric_dtype(values[column]):
            values[column] = values[column].map(lambda value: f"{value:.4g}")
        else:
            values[column] = values[column].astype(str).str.replace("|", "\\|", regex=False)
    header = "| " + " | ".join(values.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in values.astype(str).to_numpy()]
    return "\n".join([header, separator, *rows]) + "\n"


def method_macro(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    target = frame.groupby(["target", "method"], as_index=False)[METRICS].mean()
    result = target.groupby("method", as_index=False)[METRICS].mean()
    result.insert(0, "scope", scope)
    result["method"] = result.method.map(METHOD_LABELS)
    return result.sort_values("overall_mae")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".md").write_text(markdown(frame), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-results", type=Path, required=True)
    parser.add_argument("--secondary-results", type=Path, required=True)
    parser.add_argument("--secondary-statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    development = pd.read_parquet(args.development_results)
    secondary = pd.read_parquet(args.secondary_results)
    statistics = json.loads(args.secondary_statistics.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    development_macro = method_macro(development, "development outer test (6 targets)")
    secondary_macro = method_macro(secondary, "Phase 4 secondary (23 targets)")
    all29_macro = method_macro(pd.concat([development, secondary], ignore_index=True), "all SCG-feasible evaluated targets (29)")
    write_table(development_macro, args.output_dir / "table_svm_bps_development_methods")
    write_table(secondary_macro, args.output_dir / "table_svm_bps_phase4_secondary_methods")
    write_table(all29_macro, args.output_dir / "table_svm_bps_all29_methods")

    target = pd.concat([development.assign(scope="development"), secondary.assign(scope="phase4_secondary")], ignore_index=True)
    target = target.groupby(["scope", "target", "method"], as_index=False)[METRICS].mean()
    target["method"] = target.method.map(METHOD_LABELS)
    write_table(target, args.output_dir / "table_svm_bps_all29_by_target")

    rows = []
    for contrast, measures in statistics["statistics"].items():
        candidate, baseline = contrast.split("_vs_", maxsplit=1)
        for metric, record in measures.items():
            rows.append({
                "candidate": METHOD_LABELS[candidate],
                "baseline": METHOD_LABELS[baseline],
                "metric": metric,
                "relative_change_pct": 100.0 * record["target_macro_relative_change"],
                "bootstrap95_low_pct": 100.0 * record["bootstrap95"][0],
                "bootstrap95_high_pct": 100.0 * record["bootstrap95"][1],
                "improved_targets": record["improved_targets"],
                "targets": record["targets"],
                "holm_adjusted_p": record["holm_adjusted_p"],
            })
    paired = pd.DataFrame(rows)
    write_table(paired, args.output_dir / "table_svm_bps_phase4_paired_statistics")
    print(json.dumps({"tables": 5, "plots": 0, "all29_target_method_rows": len(target)}, indent=2))


if __name__ == "__main__":
    main()
