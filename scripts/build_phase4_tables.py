#!/usr/bin/env python3
"""Build paper-ready CSV/Markdown tables for Phase 4; never draw figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = ["overall_mae", "cliff_mae", "pair_delta_mae"]
METHOD_LABELS = {
    "rf": "ECFP4-RF",
    "gine": "GINE",
    "fixed_equal": "RF/GINE equal",
    "gated_component_scaled": "BPS-GNN gated scale",
}


def markdown(frame: pd.DataFrame) -> str:
    values = frame.copy()
    for column in values.columns:
        if pd.api.types.is_numeric_dtype(values[column]):
            values[column] = values[column].map(lambda value: f"{value:.4f}")
        else:
            values[column] = values[column].astype(str).str.replace("|", "\\|", regex=False)
    header = "| " + " | ".join(values.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in values.astype(str).to_numpy()]
    return "\n".join([header, separator, *rows]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.results)
    statistics = json.loads(args.statistics.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target = frame.groupby(["target", "method"], as_index=False)[METRICS].mean()
    target["method"] = target.method.map(METHOD_LABELS)
    target.to_csv(args.output_dir / "table_phase4_by_target.csv", index=False)
    (args.output_dir / "table_phase4_by_target.md").write_text(markdown(target), encoding="utf-8")

    method_macro = target.groupby("method", as_index=False)[METRICS].mean()
    method_macro.to_csv(args.output_dir / "table_phase4_method_macro.csv", index=False)
    (args.output_dir / "table_phase4_method_macro.md").write_text(markdown(method_macro), encoding="utf-8")

    rows = []
    for contrast, measures in statistics["statistics"].items():
        candidate, baseline = contrast.split("_vs_", maxsplit=1)
        for metric, record in measures.items():
            rows.append(
                {
                    "candidate": METHOD_LABELS[candidate],
                    "baseline": METHOD_LABELS[baseline],
                    "metric": metric,
                    "relative_change_pct": 100.0 * record["target_macro_relative_change"],
                    "bootstrap95_low_pct": 100.0 * record["bootstrap95"][0],
                    "bootstrap95_high_pct": 100.0 * record["bootstrap95"][1],
                    "improved_targets": record["improved_targets"],
                    "targets": record["targets"],
                    "holm_adjusted_p": record["holm_adjusted_p"],
                }
            )
    comparisons = pd.DataFrame(rows)
    comparisons.to_csv(args.output_dir / "table_phase4_paired_statistics.csv", index=False)
    (args.output_dir / "table_phase4_paired_statistics.md").write_text(markdown(comparisons), encoding="utf-8")
    print(json.dumps({"tables": 3, "target_rows": len(target), "comparison_rows": len(comparisons), "plots": 0}, indent=2))


if __name__ == "__main__":
    main()
