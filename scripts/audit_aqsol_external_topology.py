#!/usr/bin/env python3
"""Apply the frozen response-free AqSolDB topology eligibility gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    target = str(config["target"])
    data_root, mmp_root, fold_root = map(Path, [config["data_root"], config["mmp_root"], config["fold_root"]])
    structures = pd.read_parquet(data_root / target / "structures.parquet")
    edges = pd.read_parquet(mmp_root / target / "eligible_mmp_edges.parquet")
    folds = pd.read_parquet(fold_root / target / "outer_folds.parquet")
    if any(str(column).lower().startswith("y_") for column in structures.columns):
        raise RuntimeError("Response column entered topology audit")
    chosen = sorted(int(value) for value in folds["outer_fold"].unique() if int(value) >= 0)
    internal = {}
    for fold in chosen:
        ids = set(folds.loc[folds["outer_fold"] == fold, "mol_id"])
        internal[str(fold)] = int((edges["mol_id_lo"].isin(ids) & edges["mol_id_hi"].isin(ids)).sum())
    gate = config["eligibility"]
    checks = {
        "minimum_unique_molecules": len(structures) >= int(gate["minimum_unique_molecules"]),
        "minimum_mmp_edges": len(edges) >= int(gate["minimum_mmp_edges"]),
        "minimum_folds": len(chosen) >= int(gate["minimum_folds"]),
        "minimum_internal_mmp_edges_per_fold": bool(internal) and min(internal.values()) >= int(gate["minimum_internal_mmp_edges_per_fold"]),
    }
    report = {
        "target": target, "n_unique_molecules": int(len(structures)),
        "n_mmp_edges": int(len(edges)), "folds": chosen,
        "internal_mmp_edges_by_fold": internal, "response_columns_read": False,
        "checks": checks, "eligible": bool(all(checks.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["eligible"]:
        raise SystemExit("AqSolDB is topology-ineligible under the frozen gate")


if __name__ == "__main__":
    main()
