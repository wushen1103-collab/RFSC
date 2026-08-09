#!/usr/bin/env python3
"""Diagnostic for component-centered TabPFN/GINE fusion (no fitted parameter)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bps_gnn.io import read_labels_for_ids, sha256_file
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_tabpfn_chemeleon_smoke import gine_series, metrics, prediction_series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--gine", type=Path, required=True)
    parser.add_argument("--rf", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = prediction_series(args.candidate)
    gine = gine_series(args.gine, seed=0)
    rf = prediction_series(args.rf)
    if not candidate.index.equals(gine.index) or not candidate.index.equals(rf.index):
        raise RuntimeError("Validation molecule mismatch")
    query = set(candidate.index)
    edges = pd.read_parquet(Path("graphs/mmp") / args.target / "eligible_mmp_edges.parquet")
    parent = {mol_id: mol_id for mol_id in query}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for left, right in zip(edges["mol_id_lo"], edges["mol_id_hi"]):
        if left in query and right in query:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left
    components: dict[str, list[str]] = {}
    for mol_id in query:
        components.setdefault(find(mol_id), []).append(mol_id)
    centered = candidate.astype(float).copy()
    for members in components.values():
        ids = pd.Index(members)
        centered.loc[ids] = candidate.loc[ids].mean() + (gine.loc[ids] - gine.loc[ids].mean())

    truth = read_labels_for_ids(Path("data/processed/moleculeace"), args.target, query).set_index("mol_id")["y_pactivity"]
    internal = edges[edges["mol_id_lo"].isin(query) & edges["mol_id_hi"].isin(query)].copy()
    delta = internal["mol_id_hi"].map(truth) - internal["mol_id_lo"].map(truth)
    cliffs = internal[delta.abs() >= 1.0]
    rows = []
    methods = {
        "gine": gine,
        "candidate_tabpfn": candidate,
        "fixed_rf_gine": 0.5 * rf + 0.5 * gine,
        "component_centered": centered,
    }
    for name, prediction in methods.items():
        rows.append({"method": name, **metrics(prediction, truth, cliffs)})
    table = pd.DataFrame(rows)
    summary = {
        "target": args.target,
        "fold": args.fold,
        "n_components": len(components),
        "component_query_fraction_non_singleton": float(np.mean([len(members) > 1 for members in components.values()])),
        "n_cliff_pairs": int(len(cliffs)),
        "metrics": table.to_dict("records"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps({k: summary[k] for k in ["target", "n_components", "component_query_fraction_non_singleton", "n_cliff_pairs"]}))


if __name__ == "__main__":
    main()
