#!/usr/bin/env python3
"""Freeze label-private validation predictions with context-local MMP constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse
from scipy.sparse.linalg import spsolve

try:
    from audit_mmp_transform_effect_train_only import read_oriented_edges
except ModuleNotFoundError:  # package import from the repository root (e.g. pytest)
    from scripts.audit_mmp_transform_effect_train_only import read_oriented_edges
from bps_gnn.io import freeze_prediction_file, read_labels_for_ids, sha256_file


def verify(path: Path) -> None:
    manifest = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Frozen prediction hash mismatch: {path}")


def read_rf(path: Path) -> pd.Series:
    verify(path)
    frame = pd.read_parquet(path)
    if set(frame.columns) != {"mol_id", "prediction"}:
        raise RuntimeError(f"Unexpected RF prediction schema: {path}")
    return frame.set_index("mol_id")["prediction"].sort_index()


def read_gine(path: Path, seed: int) -> pd.Series:
    verify(path)
    frame = pd.read_parquet(path)
    frame = frame[(frame["method"] == "base") & (frame["seed"] == seed)]
    if frame["mol_id"].duplicated().any():
        raise RuntimeError("Duplicate GINE validation prediction")
    return frame.set_index("mol_id")["prediction"].sort_index()


def fragment_fingerprints(values: pd.Series, bits: int) -> dict[str, object]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits)
    result = {}
    for value in values.drop_duplicates():
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Cannot parse MMP constant fragment: {value!r}")
        result[value] = generator.GetFingerprint(molecule)
    return result


def build_constraints(
    fit: pd.DataFrame,
    query: pd.DataFrame,
    fingerprints: dict[str, object],
    prior: dict,
) -> pd.DataFrame:
    minimum_fit = int(prior["minimum_fit_occurrences"])
    reference = {
        key: group.reset_index(drop=True)
        for key, group in fit.groupby("transform_key")
        if len(group) >= minimum_fit
    }
    rows = []
    for row in query.itertuples(index=False):
        candidates = reference.get(row.transform_key)
        if candidates is None:
            continue
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                fingerprints[row.constant],
                [fingerprints[value] for value in candidates["constant"]],
            ),
            dtype=float,
        )
        order = np.argsort(-similarities, kind="stable")
        order = order[
            similarities[order] >= float(prior["context_similarity_floor"])
        ][: int(prior["maximum_context_neighbors"])]
        if len(order) < int(prior["minimum_context_neighbors"]):
            continue
        weights = similarities[order]
        canonical_delta = float(
            np.average(candidates.iloc[order]["actual_delta"], weights=weights)
        )
        minimum_abs = float(prior.get("minimum_predicted_pair_abs", 0.0))
        if abs(canonical_delta) < minimum_abs:
            continue
        raw_to_lohi = 1 if row.mol_id_1 == row.mol_id_lo else -1
        desired_hi_minus_lo = row.direction_sign * raw_to_lohi * canonical_delta
        rows.append(
            {
                "mol_id_lo": row.mol_id_lo,
                "mol_id_hi": row.mol_id_hi,
                "transform_key": row.transform_key,
                "predicted_pair_delta": desired_hi_minus_lo,
                "edge_weight": float(weights.mean()),
                "fit_count": len(candidates),
                "context_neighbor_count": len(order),
                "maximum_context_similarity": float(weights.max()),
                "mean_context_similarity": float(weights.mean()),
            }
        )
    return pd.DataFrame(rows)


def project_predictions(
    base: pd.Series, constraints: pd.DataFrame, pair_strength: float
) -> tuple[pd.Series, float]:
    if constraints.empty:
        return base.copy(), 0.0
    positions = {value: index for index, value in enumerate(base.index)}
    rows = np.repeat(np.arange(len(constraints)), 2)
    columns = np.column_stack(
        [
            constraints["mol_id_lo"].map(positions).to_numpy(),
            constraints["mol_id_hi"].map(positions).to_numpy(),
        ]
    ).reshape(-1)
    values = np.tile(np.asarray([-1.0, 1.0]), len(constraints))
    incidence = sparse.csr_matrix((values, (rows, columns)), shape=(len(constraints), len(base)))
    weights = constraints["edge_weight"].to_numpy(dtype=float)
    weighted = sparse.diags(weights)
    system = sparse.eye(len(base), format="csr") + pair_strength * incidence.T @ weighted @ incidence
    right = base.to_numpy(dtype=float) + pair_strength * incidence.T @ (
        weights * constraints["predicted_pair_delta"].to_numpy(dtype=float)
    )
    projected = pd.Series(spsolve(system, right), index=base.index, name="prediction")
    mean_shift = float(abs((projected - base).mean()))
    if mean_shift > 1e-10:
        raise RuntimeError(f"Incidence projection changed the component-weighted global mean: {mean_shift}")
    return projected, mean_shift


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--fold-root", type=Path, default=Path("folds/scg"))
    parser.add_argument("--graph-root", type=Path, default=Path("graphs/mmp"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    scope = config["scope"]
    target, fold = scope["target"], int(scope["fold"])
    assignments = pd.read_parquet(args.fold_root / target / "outer_folds.parquet")
    validation_fold = (fold + 1) % 5
    train_ids = set(
        assignments.loc[~assignments["outer_fold"].isin([fold, validation_fold]), "mol_id"]
    )
    validation_ids = set(assignments.loc[assignments["outer_fold"] == validation_fold, "mol_id"])
    test_ids = set(assignments.loc[assignments["outer_fold"] == fold, "mol_id"])
    labels = read_labels_for_ids(args.data_root, target, train_ids).set_index("mol_id")["y_pactivity"]
    if set(labels.index) & (validation_ids | test_ids):
        raise RuntimeError("Query label entered context-expert builder")
    edges = read_oriented_edges(args.graph_root, target)
    fit = edges[edges["mol_id_1"].isin(train_ids) & edges["mol_id_2"].isin(train_ids)].copy()
    fit["actual_delta"] = fit["direction_sign"] * (
        fit["mol_id_2"].map(labels) - fit["mol_id_1"].map(labels)
    )
    query = edges[
        edges["mol_id_1"].isin(validation_ids) & edges["mol_id_2"].isin(validation_ids)
    ].copy()
    fingerprints = fragment_fingerprints(
        pd.concat([fit["constant"], query["constant"]], ignore_index=True),
        int(config["transformation_prior"]["fingerprint_bits"]),
    )
    constraints = build_constraints(
        fit, query, fingerprints, config["transformation_prior"]
    )
    if (set(constraints.get("mol_id_lo", [])) | set(constraints.get("mol_id_hi", []))) - validation_ids:
        raise RuntimeError("Constraint escaped validation partition")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    constraints.to_parquet(args.output_dir / "label_private_pair_constraints.parquet", index=False)
    records = []
    components = config["components"]
    for seed in scope["seeds"]:
        rf_path = Path(components["rf"]) / target / f"fold_{fold}" / f"seed_{seed}" / "validation_predictions.parquet"
        gine_path = Path(components["gine_validation_source"]) / target / f"fold_{fold}" / "validation_predictions.parquet"
        rf, gine = read_rf(rf_path), read_gine(gine_path, int(seed))
        if not rf.index.equals(gine.index) or set(rf.index) != validation_ids:
            raise RuntimeError("Baseline/query partition mismatch")
        base = 0.5 * rf + 0.5 * gine
        projected, mean_shift = project_predictions(
            base, constraints, float(config["projection"]["pair_strength"])
        )
        run_dir = args.output_dir / target / f"fold_{fold}" / f"seed_{seed}"
        metadata = {
            "target": target,
            "fold": fold,
            "seed": int(seed),
            "partition": "outer_validation_pseudo_query",
            "outer_train_labels_materialized": True,
            "validation_labels_materialized": False,
            "outer_test_labels_or_predictions_materialized": False,
            "supported_query_edges": len(constraints),
            "pair_strength": float(config["projection"]["pair_strength"]),
            "absolute_global_mean_shift": mean_shift,
        }
        for method, prediction in [("fixed_equal", base), ("context_expert", projected)]:
            freeze_prediction_file(
                pd.DataFrame({"mol_id": prediction.index, "prediction": prediction.to_numpy()}),
                run_dir / method / "predictions.parquet",
                {**metadata, "method": method},
            )
        (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        records.append(metadata)
    pd.DataFrame(records).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(json.dumps({"runs": len(records), "supported_query_edges": len(constraints)}, indent=2))


if __name__ == "__main__":
    main()
