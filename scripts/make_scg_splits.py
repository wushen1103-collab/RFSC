#!/usr/bin/env python3
"""Construct deterministic, label-free Structure-Component Grouped splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def assert_label_free_columns(columns) -> None:
    prohibited_exact = {
        "y",
        "label",
        "cliff",
        "split",
        "y_pactivity",
        "y_legacy_log_nm",
        "official_cliff_mol",
        "official_split",
    }
    present = {str(column).lower() for column in columns} & prohibited_exact
    if present:
        raise AssertionError(f"SCG received prohibited columns: {sorted(present)}")


class DisjointSet:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def deterministic_tie(seed: int, component: str) -> str:
    return hashlib.sha256(f"{seed}\0{component}".encode()).hexdigest()


def assign_components(
    components: dict[str, list[str]], n_folds: int, seed: int
) -> tuple[dict[str, int], list[int]]:
    ordered = sorted(
        components.items(),
        key=lambda item: (-len(item[1]), deterministic_tie(seed, item[0])),
    )
    fold_sizes = [0] * n_folds
    assignment: dict[str, int] = {}
    for component, members in ordered:
        fold = min(range(n_folds), key=lambda idx: (fold_sizes[idx], idx))
        assignment[component] = fold
        fold_sizes[fold] += len(members)
    return assignment, fold_sizes


def balanced_enough(fold_sizes: list[int], n_nodes: int) -> bool:
    if not fold_sizes or min(fold_sizes) == 0:
        return False
    ideal = n_nodes / len(fold_sizes)
    return max(fold_sizes) <= 1.5 * ideal and min(fold_sizes) >= 0.5 * ideal


def build_target(
    structures_path: Path,
    mmp_path: Path,
    output_root: Path,
    requested_folds: int,
    minimum_folds: int,
    seed: int,
) -> dict:
    target = structures_path.parent.name
    structures = pd.read_parquet(structures_path)
    assert_label_free_columns(structures.columns)
    edges = pd.read_parquet(mmp_path)
    ids = structures["mol_id"].astype(str).tolist()
    dsu = DisjointSet(ids)

    for _, group in structures.groupby("connectivity_key", sort=False):
        members = group["mol_id"].astype(str).tolist()
        for member in members[1:]:
            dsu.union(members[0], member)
    nonempty_scaffolds = structures[structures["scaffold_smiles"].astype(str) != ""]
    for _, group in nonempty_scaffolds.groupby("scaffold_smiles", sort=False):
        members = group["mol_id"].astype(str).tolist()
        for member in members[1:]:
            dsu.union(members[0], member)
    for row in edges.itertuples(index=False):
        dsu.union(str(row.mol_id_lo), str(row.mol_id_hi))

    components: dict[str, list[str]] = {}
    for mol_id in ids:
        root = dsu.find(mol_id)
        components.setdefault(root, []).append(mol_id)

    chosen_folds = None
    component_assignment: dict[str, int] = {}
    fold_sizes: list[int] = []
    for n_folds in range(requested_folds, minimum_folds - 1, -1):
        trial_assignment, trial_sizes = assign_components(components, n_folds, seed)
        if balanced_enough(trial_sizes, len(ids)):
            chosen_folds = n_folds
            component_assignment = trial_assignment
            fold_sizes = trial_sizes
            break

    out = output_root / target
    out.mkdir(parents=True, exist_ok=True)
    root_for_id = {mol_id: dsu.find(mol_id) for mol_id in ids}
    if chosen_folds is None:
        assignments = pd.DataFrame(
            {
                "mol_id": ids,
                "component_id": [root_for_id[mol_id] for mol_id in ids],
                "outer_fold": -1,
            }
        )
        status = "SCG_INFEASIBLE"
    else:
        assignments = pd.DataFrame(
            {
                "mol_id": ids,
                "component_id": [root_for_id[mol_id] for mol_id in ids],
                "outer_fold": [
                    component_assignment[root_for_id[mol_id]] for mol_id in ids
                ],
            }
        )
        status = "SCG_FEASIBLE"
    assignments.to_parquet(out / "outer_folds.parquet", index=False)

    largest_component = max(map(len, components.values()))
    manifest = {
        "target": target,
        "status": status,
        "n_nodes": len(ids),
        "n_components": len(components),
        "largest_component": largest_component,
        "largest_component_fraction": largest_component / len(ids),
        "requested_folds": requested_folds,
        "chosen_folds": chosen_folds,
        "fold_sizes": fold_sizes,
        "seed": seed,
        "label_columns_read": False,
        "balance_rule": "max<=1.5*ideal and min>=0.5*ideal",
    }
    (out / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--mmp-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requested-folds", type=int, default=5)
    parser.add_argument("--minimum-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2901)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = []
    for structures in sorted(args.input_dir.glob("*/structures.parquet")):
        mmp = args.mmp_dir / structures.parent.name / "eligible_mmp_edges.parquet"
        manifest = build_target(
            structures,
            mmp,
            args.output_dir,
            args.requested_folds,
            args.minimum_folds,
            args.seed,
        )
        manifests.append(manifest)
        print(
            f"{manifest['target']}: {manifest['status']} folds={manifest['chosen_folds']} "
            f"components={manifest['n_components']} max_fraction="
            f"{manifest['largest_component_fraction']:.3f}",
            flush=True,
        )
    pd.DataFrame(manifests).sort_values("target").to_csv(
        args.output_dir / "scg_manifest.csv", index=False
    )


if __name__ == "__main__":
    main()
