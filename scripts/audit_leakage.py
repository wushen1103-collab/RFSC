#!/usr/bin/env python3
"""Audit exact, scaffold, and eligible-MMP leakage for every SCG fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def audit_target(structures_path: Path, edges_path: Path, folds_path: Path) -> dict:
    target = structures_path.parent.name
    structures = pd.read_parquet(structures_path)
    edges = pd.read_parquet(edges_path)
    folds = pd.read_parquet(folds_path)
    joined = structures.merge(folds, on="mol_id", validate="one_to_one")
    reports = []
    chosen = sorted(int(value) for value in joined["outer_fold"].unique() if value >= 0)
    for fold in chosen:
        train = joined[joined["outer_fold"] != fold]
        test = joined[joined["outer_fold"] == fold]
        exact = set(train["connectivity_key"]) & set(test["connectivity_key"])
        train_scaffolds = set(train.loc[train["scaffold_smiles"] != "", "scaffold_smiles"])
        test_scaffolds = set(test.loc[test["scaffold_smiles"] != "", "scaffold_smiles"])
        scaffolds = train_scaffolds & test_scaffolds
        train_ids, test_ids = set(train["mol_id"]), set(test["mol_id"])
        cross_mmp = edges[
            (edges["mol_id_lo"].isin(train_ids) & edges["mol_id_hi"].isin(test_ids))
            | (edges["mol_id_hi"].isin(train_ids) & edges["mol_id_lo"].isin(test_ids))
        ]
        report = {
            "fold": fold,
            "train_n": len(train),
            "test_n": len(test),
            "cross_split_exact_connectivity": len(exact),
            "cross_split_nonempty_scaffold": len(scaffolds),
            "cross_split_eligible_mmp_edges": len(cross_mmp),
            "test_labels_loaded_before_prediction": False,
            "prediction_hash_created_before_evaluation": None,
        }
        report["passed_structure_assertions"] = (
            report["cross_split_exact_connectivity"] == 0
            and report["cross_split_nonempty_scaffold"] == 0
            and report["cross_split_eligible_mmp_edges"] == 0
        )
        reports.append(report)
    if not chosen:
        return {
            "target": target,
            "protocol": "SCG",
            "status": "SCG_INFEASIBLE",
            "folds": [],
            "passed": None,
        }
    return {
        "target": target,
        "protocol": "SCG",
        "status": "SCG_FEASIBLE",
        "folds": reports,
        "passed": bool(reports) and all(r["passed_structure_assertions"] for r in reports),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--mmp-dir", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = []
    for structures in sorted(args.input_dir.glob("*/structures.parquet")):
        target = structures.parent.name
        folds = args.fold_dir / target / "outer_folds.parquet"
        edges = args.mmp_dir / target / "eligible_mmp_edges.parquet"
        reports.append(audit_target(structures, edges, folds))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2, sort_keys=True), encoding="utf-8")
    failed = [report["target"] for report in reports if report["passed"] is False]
    print(f"audited={len(reports)} failed={len(failed)}")
    if failed:
        print("failed_targets=" + ",".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
