#!/usr/bin/env python3
"""Generate data/graph/fold manifests and Table 1 without hand-entered counts."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def connected_component_sizes(nodes: list[str], edges: pd.DataFrame) -> list[int]:
    parent = {node: node for node in nodes}
    size = {node: 1 for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left, right = find(left), find(right)
        if left == right:
            return
        if size[left] < size[right]:
            left, right = right, left
        parent[right] = left
        size[left] += size[right]

    for row in edges.itertuples(index=False):
        union(str(row.mol_id_lo), str(row.mol_id_hi))
    counts: dict[str, int] = {}
    for node in nodes:
        counts[find(node)] = counts.get(find(node), 0) + 1
    return sorted(counts.values(), reverse=True)


def count_knn_edges(smiles: list[str], k: int = 10, floor: float = 0.6) -> int:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(smile)) for smile in smiles]
    undirected: set[tuple[int, int]] = set()
    for idx, fp in enumerate(fps):
        similarities = DataStructs.BulkTanimotoSimilarity(fp, fps)
        candidates = [
            (sim, other)
            for other, sim in enumerate(similarities)
            if other != idx and sim >= floor
        ]
        candidates.sort(key=lambda item: (-item[0], item[1]))
        for _, other in candidates[:k]:
            undirected.add((min(idx, other), max(idx, other)))
    return len(undirected)


def audit_target(
    structures_path: Path, mmp_root: Path, fold_root: Path
) -> tuple[dict, list[dict]]:
    target = structures_path.parent.name
    target_dir = structures_path.parent
    structures = pd.read_parquet(structures_path)
    labels = pd.read_parquet(target_dir / "labels_private.parquet")
    edges = pd.read_parquet(mmp_root / target / "eligible_mmp_edges.parquet")
    folds = pd.read_parquet(fold_root / target / "outer_folds.parquet")
    standard = json.loads((target_dir / "standardization_manifest.json").read_text())
    split = json.loads((fold_root / target / "split_manifest.json").read_text())
    component_sizes = connected_component_sizes(structures["mol_id"].tolist(), edges)
    ecfp_edges = count_knn_edges(structures["canonical_isomeric_smiles"].tolist())
    fold_records: list[dict] = []
    joined_labels = labels[["mol_id", "y_pactivity"]].merge(
        folds, on="mol_id", validate="one_to_one"
    )
    y_map = dict(zip(joined_labels["mol_id"], joined_labels["y_pactivity"]))
    for fold in sorted(int(value) for value in folds["outer_fold"].unique() if value >= 0):
        test_ids = set(folds.loc[folds["outer_fold"] == fold, "mol_id"])
        test_edges = edges[
            edges["mol_id_lo"].isin(test_ids) & edges["mol_id_hi"].isin(test_ids)
        ]
        deltas = test_edges.apply(
            lambda row: abs(y_map[row["mol_id_lo"]] - y_map[row["mol_id_hi"]]), axis=1
        )
        cliff_edges = test_edges[deltas.to_numpy() >= 1.0]
        cliff_endpoints = set(cliff_edges["mol_id_lo"]) | set(cliff_edges["mol_id_hi"])
        fold_records.append(
            {
                "target": target,
                "outer_fold": fold,
                "test_n": len(test_ids),
                "test_mmp_pairs": len(test_edges),
                "test_mmp_cliff_pairs_dy_ge_1": len(cliff_edges),
                "test_mmp_cliff_endpoints_dy_ge_1": len(cliff_endpoints),
            }
        )
    row = {
        **standard,
        "unique_scaffolds_nonempty": int(
            structures.loc[structures["scaffold_smiles"] != "", "scaffold_smiles"].nunique()
        ),
        "acyclic_structures": int((structures["scaffold_smiles"] == "").sum()),
        "eligible_mmp_edges": int(len(edges)),
        "mmp_components_including_isolates": len(component_sizes),
        "largest_mmp_component_fraction": max(component_sizes) / len(structures),
        "ecfp_edges_k10_floor_0_6": ecfp_edges,
        "scg_status": split["status"],
        "scg_folds": split["chosen_folds"],
        "scg_components": split["n_components"],
        "largest_scg_component_fraction": split["largest_component_fraction"],
        "scg_fold_sizes": json.dumps(split["fold_sizes"]),
        "license": "ChEMBL CC BY-SA 3.0",
    }
    return row, fold_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--mmp-dir", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--table-out", type=Path, required=True)
    parser.add_argument("--fold-out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    structures = sorted(args.input_dir.glob("*/structures.parquet"))
    rows: list[dict] = []
    fold_rows: list[dict] = []
    failures: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(structures))) as pool:
        futures = {
            pool.submit(audit_target, path, args.mmp_dir, args.fold_dir): path
            for path in structures
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                row, target_folds = future.result()
                rows.append(row)
                fold_rows.extend(target_folds)
                print(f"OK {path.parent.name}", flush=True)
            except Exception as exc:
                failures.append(
                    {"target": path.parent.name, "error": f"{type(exc).__name__}: {exc}"}
                )
                print(f"FAIL {path.parent.name}: {exc}", flush=True)
    if failures:
        failure_path = args.out.with_suffix(".failures.json")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise SystemExit(f"{len(failures)} target audit(s) failed")
    frame = pd.DataFrame(rows).sort_values("target")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    table_columns = [
        "target",
        "unique_isomeric_smiles",
        "unique_scaffolds_nonempty",
        "eligible_mmp_edges",
        "mmp_components_including_isolates",
        "ecfp_edges_k10_floor_0_6",
        "scg_folds",
        "largest_scg_component_fraction",
        "official_cliff_rows",
        "license",
    ]
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    frame[table_columns].to_csv(args.table_out, index=False)
    pd.DataFrame(fold_rows).sort_values(["target", "outer_fold"]).to_csv(
        args.fold_out, index=False
    )


if __name__ == "__main__":
    main()

