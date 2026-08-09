#!/usr/bin/env python3
"""Select external targets from counts and MMP topology without reading responses."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


PROHIBITED_RESPONSE_COLUMNS = {
    "y", "label", "y_pactivity", "ki_nm", "standard_value", "cliff",
}


def audit_target(target_dir: Path, graph_dir: Path, config: dict) -> dict[str, object]:
    structures = pd.read_parquet(target_dir / "structures.parquet")
    partitions = pd.read_parquet(target_dir / "partitions.parquet")
    edges = pd.read_parquet(graph_dir / "eligible_mmp_edges.parquet")
    exposed = PROHIBITED_RESPONSE_COLUMNS & {
        str(column).lower() for column in [*structures.columns, *partitions.columns]
    }
    if exposed:
        raise AssertionError(f"{target_dir.name}: response columns exposed: {sorted(exposed)}")
    if structures["mol_id"].duplicated().any() or partitions["mol_id"].duplicated().any():
        raise AssertionError(f"{target_dir.name}: duplicate molecule IDs")
    if set(structures["mol_id"]) != set(partitions["mol_id"]):
        raise AssertionError(f"{target_dir.name}: structure/partition IDs differ")
    if not set(partitions["partition"]).issubset({"train", "validation", "test"}):
        raise AssertionError(f"{target_dir.name}: invalid temporal partition")
    valid_ids = set(partitions["mol_id"])
    edge_ids = set(edges["mol_id_lo"]) | set(edges["mol_id_hi"])
    if not edge_ids.issubset(valid_ids):
        raise AssertionError(f"{target_dir.name}: MMP graph contains unknown molecule IDs")

    counts = partitions["partition"].value_counts()
    test_ids = set(partitions.loc[partitions["partition"] == "test", "mol_id"])
    internal_test = edges[
        edges["mol_id_lo"].isin(test_ids) & edges["mol_id_hi"].isin(test_ids)
    ]
    thresholds = config["eligibility_structure_only"]
    count_eligible = (
        int(counts.get("train", 0)) >= int(thresholds["minimum_train_molecules"])
        and int(counts.get("validation", 0)) >= int(thresholds["minimum_validation_molecules"])
        and int(counts.get("test", 0)) >= int(thresholds["minimum_test_molecules"])
    )
    graph_eligible = len(internal_test) >= int(thresholds["minimum_internal_test_mmp_edges"])
    manifest_path = target_dir / "target_manifest.json"
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "target_chembl_id": target_dir.name,
        "target_name": metadata["target_name"],
        "train_molecules": int(counts.get("train", 0)),
        "validation_molecules": int(counts.get("validation", 0)),
        "test_molecules": int(counts.get("test", 0)),
        "all_mmp_edges": int(len(edges)),
        "internal_test_mmp_edges": int(len(internal_test)),
        "count_eligible": bool(count_eligible),
        "graph_eligible": bool(graph_eligible),
        "eligible": bool(count_eligible and graph_eligible),
        "response_columns_read": False,
    }


def run_audit(input_dir: Path, graph_root: Path, config: dict) -> pd.DataFrame:
    targets = sorted(path.parent for path in input_dir.glob("*/structures.parquet"))
    if not targets:
        raise RuntimeError(f"No curated target directories in {input_dir}")
    records = [audit_target(target, graph_root / target.name, config) for target in targets]
    return pd.DataFrame.from_records(records).sort_values("target_chembl_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    audit = run_audit(args.input_dir, args.graph_root, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "target_eligibility.csv", index=False)
    eligible = audit.loc[audit["eligible"], "target_chembl_id"].tolist()
    (args.output_dir / "eligible_targets.json").write_text(
        json.dumps(eligible, indent=2), encoding="utf-8"
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "targets_audited": int(len(audit)),
        "eligible_targets": int(len(eligible)),
        "minimum_required_targets": int(
            config["eligibility_structure_only"]["minimum_eligible_targets"]
        ),
        "response_columns_read": False,
        "eligible_target_ids": eligible,
    }
    (args.output_dir / "eligibility_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    if len(eligible) < manifest["minimum_required_targets"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
