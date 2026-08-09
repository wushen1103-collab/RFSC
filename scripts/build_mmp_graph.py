#!/usr/bin/env python3
"""Build structure-only eligible MMP graphs with mmpdb."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.warning")


CSV_COLUMNS = ["smiles_1", "smiles_2", "mol_id_1", "mol_id_2", "transform", "constant"]


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
        raise AssertionError(f"Structure table contains prohibited columns: {sorted(present)}")


def run_checked(command: list[str], log_path: Path) -> None:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT\n" + result.stdout
        + "\nSTDERR\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log_path}")


def read_mmp_csv(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        first = handle.readline().rstrip("\n")
    has_header = "id1" in first.lower() or "smiles1" in first.lower()
    frame = pd.read_csv(
        path,
        sep="\t",
        header=0 if has_header else None,
        names=None if has_header else CSV_COLUMNS,
        dtype=str,
    )
    if has_header:
        if len(frame.columns) != 6:
            raise ValueError(f"Unexpected mmpdb columns: {frame.columns.tolist()}")
        frame.columns = CSV_COLUMNS
    return frame


def variable_heavy_atom_score(transform: str) -> tuple[int, int, int]:
    try:
        left, right = transform.split(">>", 1)
        left_mol, right_mol = Chem.MolFromSmiles(left), Chem.MolFromSmiles(right)
        if left_mol is None or right_mol is None:
            raise ValueError("unparseable variable fragment")
        left_heavy = sum(atom.GetAtomicNum() > 1 for atom in left_mol.GetAtoms())
        right_heavy = sum(atom.GetAtomicNum() > 1 for atom in right_mol.GetAtoms())
        return max(left_heavy, right_heavy), left_heavy + right_heavy, abs(left_heavy - right_heavy)
    except Exception:
        return 10**9, 10**9, 10**9


def process_target(
    target_dir: Path,
    output_root: Path,
    mmpdb: Path,
    jobs_per_target: int,
) -> dict:
    target = target_dir.name
    out = output_root / target
    out.mkdir(parents=True, exist_ok=True)
    structures = pd.read_parquet(target_dir / "structures.parquet")
    assert_label_free_columns(structures.columns)

    smi = out / f"{target}.smi"
    structures[["canonical_isomeric_smiles", "mol_id"]].to_csv(
        smi, sep="\t", header=False, index=False
    )
    fragdb = out / f"{target}.fragdb"
    raw_pairs = out / f"{target}.pairs.csv.gz"
    run_checked(
        [
            str(mmpdb),
            "fragment",
            str(smi),
            "--delimiter",
            "tab",
            "--num-cuts",
            "1",
            "--salt-remover",
            "<none>",
            "--min-heavies-total-const-frag",
            "1",
            "--num-jobs",
            str(jobs_per_target),
            "--output",
            str(fragdb),
        ],
        out / "fragment.log",
    )
    run_checked(
        [
            str(mmpdb),
            "index",
            str(fragdb),
            "--out",
            "csv.gz",
            "--max-variable-ratio",
            "0.33",
            "--max-variable-heavies",
            "12",
            "--max-heavies-transf",
            "12",
            "--max-radius",
            "3",
            "--output",
            str(raw_pairs),
        ],
        out / "index.log",
    )
    pairs = read_mmp_csv(raw_pairs)
    raw_candidate_pairs = len(pairs)
    valid_ids = set(structures["mol_id"])
    pairs = pairs[
        pairs["mol_id_1"].isin(valid_ids) & pairs["mol_id_2"].isin(valid_ids)
    ].copy()
    pairs = pairs[pairs["mol_id_1"] != pairs["mol_id_2"]]
    ordered = pairs[["mol_id_1", "mol_id_2"]].apply(
        lambda row: sorted((row.iloc[0], row.iloc[1])), axis=1, result_type="expand"
    )
    if len(ordered):
        pairs["mol_id_lo"] = ordered[0]
        pairs["mol_id_hi"] = ordered[1]
        scores = pairs["transform"].map(variable_heavy_atom_score)
        pairs[["score_max_variable_heavies", "score_total_variable_heavies", "score_heavy_difference"]] = pd.DataFrame(
            scores.tolist(), index=pairs.index
        )
        pairs = pairs.sort_values(
            [
                "mol_id_lo",
                "mol_id_hi",
                "score_max_variable_heavies",
                "score_total_variable_heavies",
                "score_heavy_difference",
                "transform",
                "constant",
            ]
        ).drop_duplicates(["mol_id_lo", "mol_id_hi"], keep="first")
    else:
        pairs["mol_id_lo"] = pd.Series(dtype=str)
        pairs["mol_id_hi"] = pd.Series(dtype=str)
    edge_columns = ["mol_id_lo", "mol_id_hi", "transform", "constant"]
    pairs[edge_columns].to_parquet(out / "eligible_mmp_edges.parquet", index=False)
    manifest = {
        "target": target,
        "n_structures": int(len(structures)),
        "eligible_mmp_edges": int(len(pairs)),
        "raw_mmp_candidates": int(raw_candidate_pairs),
        "num_cuts": 1,
        "min_core_heavy_atom_fraction": 0.67,
        "max_variable_heavy_atoms": 12,
        "max_heavy_atom_difference": 12,
        "smallest_transformation_only": False,
        "smallest_transform_postselection": "per_pair_minimum_max_variable_heavy_atoms",
        "upstream_smallest_transform_flag_disabled_reason": "mmpdb 3.1.4 attachment-mask compatibility failure with RDKit 2025.9.6",
        "max_radius": 3,
        "label_columns_read": False,
    }
    (out / "mmp_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mmpdb", type=Path, required=True)
    parser.add_argument("--target-workers", type=int, default=24)
    parser.add_argument("--jobs-per-target", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = sorted(p.parent for p in args.input_dir.glob("*/structures.parquet"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[dict] = []
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(args.target_workers, len(targets))) as pool:
        futures = {
            pool.submit(
                process_target,
                target,
                args.output_dir,
                args.mmpdb,
                args.jobs_per_target,
            ): target
            for target in targets
        }
        for future in as_completed(futures):
            target = futures[future]
            try:
                manifest = future.result()
                manifests.append(manifest)
                print(
                    f"OK {target.name}: {manifest['eligible_mmp_edges']} MMP edges",
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    {"target": target.name, "error": f"{type(exc).__name__}: {exc}"}
                )
                print(f"FAIL {target.name}: {exc}", flush=True)
    manifest_frame = pd.DataFrame(manifests)
    if not manifest_frame.empty:
        manifest_frame = manifest_frame.sort_values("target")
    manifest_frame.to_csv(args.output_dir / "mmp_graph_manifest.csv", index=False)
    (args.output_dir / "mmp_failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True), encoding="utf-8"
    )
    if failures:
        raise SystemExit(f"{len(failures)} target(s) failed MMP construction")


if __name__ == "__main__":
    main()
