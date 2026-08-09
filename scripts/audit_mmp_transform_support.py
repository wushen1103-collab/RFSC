#!/usr/bin/env python3
"""Label-free audit of MMP transformation reuse under frozen SCG folds."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


DEFAULT_TARGETS = [
    "CHEMBL287_Ki",
    "CHEMBL2971_Ki",
    "CHEMBL4203_Ki",
    "CHEMBL234_Ki",
    "CHEMBL4005_Ki",
    "CHEMBL4616_EC50",
]
MIN_COUNTS = (1, 2, 3, 5)


def undirected_transform_key(value: str) -> str:
    sides = str(value).split(">>", 1)
    if len(sides) != 2:
        raise ValueError(f"Malformed MMP transformation: {value!r}")
    return ">>".join(sorted(sides))


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def audit_fold(target: str, fold: int, fold_root: Path, graph_root: Path) -> dict:
    assignments = pd.read_parquet(fold_root / target / "outer_folds.parquet")
    required = {"mol_id", "component_id", "outer_fold"}
    if set(assignments.columns) != required:
        raise AssertionError(f"Unexpected SCG columns for {target}: {assignments.columns.tolist()}")
    validation_fold = (fold + 1) % 5
    train_ids = set(
        assignments.loc[~assignments["outer_fold"].isin([fold, validation_fold]), "mol_id"]
    )
    validation_ids = set(assignments.loc[assignments["outer_fold"] == validation_fold, "mol_id"])
    test_ids = set(assignments.loc[assignments["outer_fold"] == fold, "mol_id"])
    if train_ids & validation_ids or (train_ids | validation_ids) & test_ids:
        raise AssertionError("SCG partitions overlap")

    edges = pd.read_parquet(graph_root / target / "eligible_mmp_edges.parquet")
    if set(edges.columns) != {"mol_id_lo", "mol_id_hi", "transform", "constant"}:
        raise AssertionError(f"Unexpected edge columns for {target}: {edges.columns.tolist()}")
    edges = edges.assign(transform_key=edges["transform"].map(undirected_transform_key))
    train_edges = edges[
        edges["mol_id_lo"].isin(train_ids) & edges["mol_id_hi"].isin(train_ids)
    ].copy()
    validation_edges = edges[
        edges["mol_id_lo"].isin(validation_ids) & edges["mol_id_hi"].isin(validation_ids)
    ].copy()
    if (
        edges[edges["mol_id_lo"].isin(train_ids) & edges["mol_id_hi"].isin(validation_ids)].shape[0]
        or edges[edges["mol_id_lo"].isin(validation_ids) & edges["mol_id_hi"].isin(test_ids)].shape[0]
        or edges[edges["mol_id_lo"].isin(train_ids) & edges["mol_id_hi"].isin(test_ids)].shape[0]
    ):
        raise AssertionError("MMP edges cross SCG partitions")

    train_counts = Counter(train_edges["transform_key"])
    all_validation_endpoints = set(validation_edges["mol_id_lo"]) | set(validation_edges["mol_id_hi"])
    component_by_id = assignments.set_index("mol_id")["component_id"].to_dict()
    all_validation_components = {component_by_id[value] for value in all_validation_endpoints}
    row = {
        "target": target,
        "fold": fold,
        "validation_fold": validation_fold,
        "n_train_molecules": len(train_ids),
        "n_validation_molecules": len(validation_ids),
        "n_train_edges": len(train_edges),
        "n_validation_edges": len(validation_edges),
        "n_train_transform_keys": len(train_counts),
        "n_validation_edge_endpoints": len(all_validation_endpoints),
        "n_validation_edge_components": len(all_validation_components),
    }
    for minimum_count in MIN_COUNTS:
        supported = validation_edges[
            validation_edges["transform_key"].map(train_counts).fillna(0) >= minimum_count
        ]
        supported_endpoints = set(supported["mol_id_lo"]) | set(supported["mol_id_hi"])
        supported_components = {component_by_id[value] for value in supported_endpoints}
        prefix = f"train_count_ge_{minimum_count}"
        row[f"{prefix}_edges"] = len(supported)
        row[f"{prefix}_edge_coverage"] = ratio(len(supported), len(validation_edges))
        row[f"{prefix}_endpoints"] = len(supported_endpoints)
        row[f"{prefix}_endpoint_coverage"] = ratio(
            len(supported_endpoints), len(all_validation_endpoints)
        )
        row[f"{prefix}_components"] = len(supported_components)
        row[f"{prefix}_component_coverage"] = ratio(
            len(supported_components), len(all_validation_components)
        )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-root", type=Path, default=Path("folds/scg"))
    parser.add_argument("--graph-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        audit_fold(target, fold, args.fold_root, args.graph_root)
        for target in args.targets
        for fold in range(5)
    ]
    frame = pd.DataFrame(rows).sort_values(["target", "fold"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "support_by_fold.csv", index=False)
    coverage_columns = [column for column in frame if column.endswith("_coverage")]
    macro = frame[coverage_columns].mean().to_dict()
    totals = {
        "targets": len(args.targets),
        "folds": len(frame),
        "validation_edges": int(frame["n_validation_edges"].sum()),
    }
    for minimum_count in MIN_COUNTS:
        prefix = f"train_count_ge_{minimum_count}"
        supported_edges = int(frame[f"{prefix}_edges"].sum())
        totals[f"{prefix}_edges"] = supported_edges
        totals[f"{prefix}_edge_coverage_micro"] = ratio(
            supported_edges, totals["validation_edges"]
        )
    summary = {"scope": "label_free_outer_validation_support", "totals": totals, "macro": macro}
    (args.output_dir / "support_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
