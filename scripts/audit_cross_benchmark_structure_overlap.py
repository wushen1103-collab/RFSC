#!/usr/bin/env python3
"""Response-free exact structure-overlap audit across benchmark evidence sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from standardize_molecules import standardize_one
except ModuleNotFoundError:
    from scripts.standardize_molecules import standardize_one


IDENTITY_COLUMN = "canonical_no_stereo_smiles"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moleculeace-root", type=Path, required=True)
    parser.add_argument("--chembl-root", type=Path, required=True)
    parser.add_argument("--patent-root", type=Path, required=True)
    parser.add_argument("--deepdelta-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-deepdelta-datasets", type=int)
    return parser.parse_args()


def task_structure_sets(root: Path) -> tuple[dict[str, set[str]], set[str]]:
    tasks: dict[str, set[str]] = {}
    evaluation: set[str] = set()
    for structure_path in sorted(root.glob("*/structures.parquet")):
        frame = pd.read_parquet(structure_path, columns=["mol_id", IDENTITY_COLUMN])
        if frame[IDENTITY_COLUMN].isna().any():
            raise RuntimeError(f"missing structure identity: {structure_path}")
        tasks[structure_path.parent.name] = set(frame[IDENTITY_COLUMN].astype(str))
        partition_path = structure_path.parent / "partitions.parquet"
        if partition_path.exists():
            partition = pd.read_parquet(partition_path, columns=["mol_id", "partition"])
            test_ids = set(partition.loc[partition["partition"] == "test", "mol_id"].astype(str))
            mapping = frame.set_index("mol_id")[IDENTITY_COLUMN]
            missing = test_ids - set(mapping.index.astype(str))
            if missing:
                raise RuntimeError(f"partition IDs missing from structures: {structure_path.parent}")
            evaluation.update(mapping.loc[sorted(test_ids)].astype(str))
    if not tasks:
        raise RuntimeError(f"no structure tasks found under {root}")
    all_structures = set().union(*tasks.values())
    if not evaluation:
        evaluation = set(all_structures)
    return tasks, evaluation


def deepdelta_structure_sets(
    root: Path, maximum_datasets: int | None
) -> tuple[dict[str, set[str]], int, int]:
    paths = sorted(root.glob("*.csv"))
    if maximum_datasets is not None:
        paths = paths[:maximum_datasets]
    tasks: dict[str, set[str]] = {}
    invalid = 0
    raw_rows = 0
    cache: dict[str, str | None] = {}
    for path in paths:
        smiles = pd.read_csv(path, usecols=["SMILES"])["SMILES"].astype(str)
        raw_rows += len(smiles)
        identities: set[str] = set()
        for raw in smiles:
            if raw not in cache:
                try:
                    cache[raw] = str(standardize_one(raw)[IDENTITY_COLUMN])
                except Exception:
                    cache[raw] = None
                    invalid += 1
            identity = cache[raw]
            if identity is not None:
                identities.add(identity)
        tasks[path.stem] = identities
    if not tasks:
        raise RuntimeError(f"no DeepDelta CSV files found under {root}")
    return tasks, raw_rows, invalid


def overlap_record(name_a: str, values_a: set[str], name_b: str, values_b: set[str]) -> dict:
    intersection = len(values_a & values_b)
    union = len(values_a | values_b)
    return {
        "source_a": name_a,
        "source_b": name_b,
        "structures_a": len(values_a),
        "structures_b": len(values_b),
        "intersection": intersection,
        "containment_a": intersection / len(values_a) if values_a else 0.0,
        "containment_b": intersection / len(values_b) if values_b else 0.0,
        "jaccard": intersection / union if union else 0.0,
    }


def normalized_chembl_target(task: str) -> str:
    return task.split("_")[0]


def main() -> None:
    args = parse_args()
    moleculeace_tasks, moleculeace_evaluation = task_structure_sets(args.moleculeace_root)
    chembl_tasks, chembl_evaluation = task_structure_sets(args.chembl_root)
    patent_tasks, patent_evaluation = task_structure_sets(args.patent_root)
    deepdelta_tasks, deepdelta_rows, deepdelta_invalid = deepdelta_structure_sets(
        args.deepdelta_root, args.maximum_deepdelta_datasets
    )

    task_sets = {
        "MoleculeACE": moleculeace_tasks,
        "ChEMBL36_earliest": chembl_tasks,
        "BindingDB_patents": patent_tasks,
        "DeepDelta": deepdelta_tasks,
    }
    all_sets = {name: set().union(*tasks.values()) for name, tasks in task_sets.items()}
    evaluation_sets = {
        "MoleculeACE": moleculeace_evaluation,
        "ChEMBL36_earliest": chembl_evaluation,
        "BindingDB_patents": patent_evaluation,
        "DeepDelta": set(all_sets["DeepDelta"]),
    }
    evaluation_kind = {
        "MoleculeACE": "cross_validation_all",
        "ChEMBL36_earliest": "temporal_test",
        "BindingDB_patents": "latest_era_test",
        "DeepDelta": "ten_fold_cv_all",
    }

    summary_rows = []
    for name in all_sets:
        summary_rows.append(
            {
                "source": name,
                "tasks": len(task_sets[name]),
                "unique_structures_all": len(all_sets[name]),
                "evaluation_scope": evaluation_kind[name],
                "unique_structures_evaluation": len(evaluation_sets[name]),
            }
        )

    names = list(all_sets)
    pairwise_rows = []
    evaluation_pairwise_rows = []
    evaluation_vs_all_rows = []
    for index, name_a in enumerate(names):
        for name_b in names[index + 1 :]:
            pairwise_rows.append(overlap_record(name_a, all_sets[name_a], name_b, all_sets[name_b]))
            evaluation_pairwise_rows.append(
                overlap_record(
                    name_a, evaluation_sets[name_a], name_b, evaluation_sets[name_b]
                )
            )
        for reference in names:
            record = overlap_record(
                name_a, evaluation_sets[name_a], reference, all_sets[reference]
            )
            record["evaluation_scope_a"] = evaluation_kind[name_a]
            evaluation_vs_all_rows.append(record)

    chembl_by_target = {
        normalized_chembl_target(task): values for task, values in chembl_tasks.items()
    }
    target_rows = []
    for moleculeace_task, values in sorted(moleculeace_tasks.items()):
        target = normalized_chembl_target(moleculeace_task)
        if target in chembl_by_target:
            record = overlap_record(
                moleculeace_task, values, target, chembl_by_target[target]
            )
            record["chembl_target"] = target
            target_rows.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(summary_rows).to_csv(args.output_dir / "source_summary.csv", index=False)
    pd.DataFrame.from_records(pairwise_rows).to_csv(
        args.output_dir / "all_structure_pairwise_overlap.csv", index=False
    )
    pd.DataFrame.from_records(evaluation_pairwise_rows).to_csv(
        args.output_dir / "evaluation_pairwise_overlap.csv", index=False
    )
    pd.DataFrame.from_records(evaluation_vs_all_rows).to_csv(
        args.output_dir / "evaluation_vs_all_overlap.csv", index=False
    )
    pd.DataFrame.from_records(target_rows).to_csv(
        args.output_dir / "moleculeace_chembl_shared_target_overlap.csv", index=False
    )
    manifest = {
        "response_columns_read": False,
        "identity": IDENTITY_COLUMN,
        "deepdelta_raw_rows": deepdelta_rows,
        "deepdelta_unique_raw_smiles_standardization_failures": deepdelta_invalid,
        "sources": len(all_sets),
        "pairwise_comparisons": len(pairwise_rows),
        "shared_moleculeace_chembl_targets": len(target_rows),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "summary": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
