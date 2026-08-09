#!/usr/bin/env python3
"""Standardize MoleculeACE while physically separating structures and labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_mol_id(target: str, canonical_isomeric_smiles: str) -> str:
    value = f"{target}\0{canonical_isomeric_smiles}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:24]


def standardize_one(raw_smiles: str) -> dict[str, object]:
    mol = Chem.MolFromSmiles(str(raw_smiles), sanitize=True)
    if mol is None:
        raise ValueError("rdkit_parse_or_sanitize_failed")

    raw_fragments = Chem.GetMolFrags(mol)
    raw_charge = int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    cleaned = rdMolStandardize.Cleanup(mol)
    parent = rdMolStandardize.FragmentParent(cleaned)
    salt_removed = len(raw_fragments) > 1 or parent.GetNumAtoms() < cleaned.GetNumAtoms()
    uncharged = rdMolStandardize.Uncharger().uncharge(parent)
    Chem.SanitizeMol(uncharged)

    canonical = Chem.MolToSmiles(uncharged, canonical=True, isomericSmiles=True)
    canonical_no_stereo = Chem.MolToSmiles(
        uncharged, canonical=True, isomericSmiles=False
    )
    full_inchi_key = Chem.MolToInchiKey(uncharged)
    if not full_inchi_key:
        raise ValueError("inchikey_generation_failed")
    connectivity_key = full_inchi_key.split("-")[0]

    scaffold_mol = MurckoScaffold.GetScaffoldForMol(uncharged)
    scaffold = Chem.MolToSmiles(scaffold_mol, canonical=True, isomericSmiles=True)
    scaffold_group = scaffold if scaffold else f"ACYCLIC::{connectivity_key}"
    standardized_charge = int(
        sum(atom.GetFormalCharge() for atom in uncharged.GetAtoms())
    )
    return {
        "canonical_isomeric_smiles": canonical,
        "canonical_no_stereo_smiles": canonical_no_stereo,
        "full_inchikey": full_inchi_key,
        "connectivity_key": connectivity_key,
        "scaffold_smiles": scaffold,
        "scaffold_group": scaffold_group,
        "heavy_atoms": int(Descriptors.HeavyAtomCount(uncharged)),
        "raw_formal_charge": raw_charge,
        "standardized_formal_charge": standardized_charge,
        "salt_removed": bool(salt_removed),
    }


def _consistent_or_conflict(values: pd.Series) -> tuple[object, bool]:
    clean = values.dropna().astype(str).unique().tolist()
    if len(clean) == 1:
        return clean[0], False
    if not clean:
        return None, False
    return None, True


def process_target(source: Path, output_root: Path, discordance: float) -> dict:
    target = source.stem
    target_dir = output_root / target
    target_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source)
    required = {
        "smiles",
        "y",
        "cliff_mol",
        "split",
        "y [pEC50/pKi]",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{target}: missing columns {sorted(missing)}")

    y_legacy = pd.to_numeric(raw["y"], errors="coerce")
    y_p = pd.to_numeric(raw["y [pEC50/pKi]"], errors="coerce")
    finite = np.isfinite(y_legacy) & np.isfinite(y_p)
    if not finite.all():
        raise ValueError(f"{target}: non-finite labels")
    relation_error = np.abs((y_legacy + 9.0) - y_p)
    if float(relation_error.max()) > 1e-8:
        raise ValueError(
            f"{target}: y[pEC50/pKi] != y[log nM]+9; max error "
            f"{relation_error.max()}"
        )

    records: list[dict] = []
    invalid: list[dict] = []
    for raw_id, row in raw.reset_index(drop=True).iterrows():
        try:
            structure = standardize_one(row["smiles"])
            records.append(
                {
                    "raw_id": int(raw_id),
                    "raw_smiles": row["smiles"],
                    "y_legacy_log_nm": float(row["y"]),
                    "y_pactivity": float(row["y [pEC50/pKi]"]),
                    "official_cliff_mol": int(row["cliff_mol"]),
                    "official_split": str(row["split"]),
                    **structure,
                }
            )
        except Exception as exc:  # audited, never silently dropped
            invalid.append(
                {
                    "raw_id": int(raw_id),
                    "raw_smiles": str(row.get("smiles", "")),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    rows = pd.DataFrame.from_records(records)
    if rows.empty:
        raise ValueError(f"{target}: no valid structures")
    rows["mol_id"] = rows["canonical_isomeric_smiles"].map(
        lambda value: stable_mol_id(target, value)
    )
    rows.to_parquet(target_dir / "raw_valid_rows.parquet", index=False)
    pd.DataFrame.from_records(invalid).to_json(
        target_dir / "invalid_rows.jsonl", orient="records", lines=True
    )

    structure_columns = [
        "mol_id",
        "canonical_isomeric_smiles",
        "canonical_no_stereo_smiles",
        "full_inchikey",
        "connectivity_key",
        "scaffold_smiles",
        "scaffold_group",
        "heavy_atoms",
        "raw_formal_charge",
        "standardized_formal_charge",
        "salt_removed",
    ]
    structures = (
        rows.sort_values("raw_id")
        .drop_duplicates("mol_id", keep="first")[structure_columns]
        .reset_index(drop=True)
    )
    if structures["mol_id"].duplicated().any():
        raise AssertionError(f"{target}: duplicate mol_id after aggregation")
    structures.to_parquet(target_dir / "structures.parquet", index=False)

    label_records: list[dict] = []
    exact_duplicate_groups = 0
    discordant_duplicate_groups = 0
    for mol_id, group in rows.groupby("mol_id", sort=True):
        split, split_conflict = _consistent_or_conflict(group["official_split"])
        cliff, cliff_conflict = _consistent_or_conflict(group["official_cliff_mol"])
        replicate_count = int(len(group))
        exact_duplicate_groups += int(replicate_count > 1)
        spread = float(group["y_pactivity"].max() - group["y_pactivity"].min())
        is_discordant = bool(spread > discordance)
        discordant_duplicate_groups += int(replicate_count > 1 and is_discordant)
        label_records.append(
            {
                "mol_id": mol_id,
                "y_pactivity": float(group["y_pactivity"].median()),
                "y_legacy_log_nm": float(group["y_legacy_log_nm"].median()),
                "replicate_count": replicate_count,
                "replicate_mad": float(
                    np.median(
                        np.abs(
                            group["y_pactivity"]
                            - float(group["y_pactivity"].median())
                        )
                    )
                ),
                "replicate_range": spread,
                "discordant_duplicate": is_discordant,
                "official_split": split,
                "official_split_conflict": split_conflict,
                "official_cliff_mol": int(cliff) if cliff is not None else None,
                "official_cliff_conflict": cliff_conflict,
            }
        )
    labels = pd.DataFrame.from_records(label_records)
    labels.to_parquet(target_dir / "labels_private.parquet", index=False)

    manifest = {
        "dataset": "MoleculeACE",
        "target": target,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "download_time_utc": datetime.now(timezone.utc).isoformat(),
        "raw_rows": int(len(raw)),
        "valid_rows": int(len(rows)),
        "invalid_smiles": int(len(invalid)),
        "unique_isomeric_smiles": int(len(structures)),
        "unique_inchikey": int(structures["full_inchikey"].nunique()),
        "unique_connectivity_key": int(structures["connectivity_key"].nunique()),
        "exact_duplicate_groups": exact_duplicate_groups,
        "discordant_duplicate_groups": discordant_duplicate_groups,
        "official_train_rows": int((raw["split"] == "train").sum()),
        "official_test_rows": int((raw["split"] == "test").sum()),
        "official_cliff_rows": int(pd.to_numeric(raw["cliff_mol"]).sum()),
        "label_relation_max_abs_error": float(relation_error.max()),
        "rdkit_version": rdBase.rdkitVersion,
        "code_license": "MIT",
        "data_license": "ChEMBL CC BY-SA 3.0",
        "redistribution_allowed": "share-alike with attribution and provenance",
    }
    (target_dir / "standardization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--discordance", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = sorted(
        p for p in args.input_root.glob("*.csv") if p.name != "datasets.csv"
    )
    if not sources:
        raise SystemExit(f"No CSV files found in {args.input_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[dict] = []
    failures: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(sources))) as pool:
        jobs = {
            pool.submit(process_target, src, args.output_dir, args.discordance): src
            for src in sources
        }
        for future in as_completed(jobs):
            source = jobs[future]
            try:
                manifest = future.result()
                manifests.append(manifest)
                print(
                    f"OK {source.stem}: {manifest['valid_rows']} rows -> "
                    f"{manifest['unique_isomeric_smiles']} molecules",
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    {
                        "target": source.stem,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"FAIL {source.stem}: {exc}", flush=True)
    pd.DataFrame(manifests).sort_values("target").to_csv(
        args.output_dir / "standardization_summary.csv", index=False
    )
    (args.output_dir / "standardization_failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True), encoding="utf-8"
    )
    if failures:
        raise SystemExit(f"{len(failures)} target(s) failed standardization")


if __name__ == "__main__":
    main()

