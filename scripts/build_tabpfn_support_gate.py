#!/usr/bin/env python3
"""Build the fixed query-degree TabPFN/GINE support-gated prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from bps_gnn.io import freeze_prediction_file, sha256_file


def verified(path: Path, *, method: str | None = None, seed: int | None = None) -> pd.Series:
    manifest = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Prediction hash mismatch: {path}")
    frame = pd.read_parquet(path)
    if method is not None:
        frame = frame[(frame["method"] == method) & (frame["seed"] == seed)]
    return frame.set_index("mol_id")["prediction"].sort_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--gine", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate = verified(args.candidate)
    gine = verified(args.gine, method="base", seed=0)
    if not candidate.index.equals(gine.index):
        raise RuntimeError("Validation molecule mismatch")
    edges = pd.read_parquet(Path("graphs/mmp") / args.target / "eligible_mmp_edges.parquet")
    query = set(candidate.index)
    degree = {mol_id: 0 for mol_id in query}
    for left, right in zip(edges["mol_id_lo"], edges["mol_id_hi"]):
        if left in query and right in query:
            degree[left] += 1
            degree[right] += 1
    use_gine = [mol_id for mol_id, count in degree.items() if count > 0]
    gated = candidate.copy()
    gated.loc[use_gine] = gine.loc[use_gine]
    output = args.output_dir / args.target / f"fold_{args.fold}" / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "target": args.target,
        "fold": args.fold,
        "seed": args.seed,
        "method": "tabpfn_query_degree_support_gate",
        "protocol": "strict_scg_validation_only",
        "support_rule": "validation_validation_mmp_degree_gt_zero_uses_gine_else_tabpfn",
        "supported_query_fraction": float(len(use_gine) / len(query)),
        "candidate_sha256": sha256_file(args.candidate),
        "gine_sha256": sha256_file(args.gine),
        "validation_labels_materialized": False,
        "outer_test_labels_materialized": False,
        "outer_test_predictions_generated": False,
        "validation_only": True,
    }
    metadata["prediction_sha256"] = freeze_prediction_file(
        pd.DataFrame({"mol_id": gated.index, "prediction": gated.to_numpy()}),
        output / "validation_predictions.parquet",
        metadata,
    )
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
