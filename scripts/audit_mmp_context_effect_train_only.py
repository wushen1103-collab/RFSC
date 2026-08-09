#!/usr/bin/env python3
"""Context-local MMP effect transfer audit using outer-training labels only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from audit_mmp_transform_effect_train_only import inner_fold, metrics, read_oriented_edges
from bps_gnn.io import read_labels_for_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--fold-root", type=Path, default=Path("folds/scg"))
    parser.add_argument("--graph-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def build_fingerprints(values: pd.Series, bits: int) -> dict[str, object]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits)
    result = {}
    for value in values.drop_duplicates():
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Cannot parse MMP constant fragment: {value!r}")
        result[value] = generator.GetFingerprint(molecule)
    return result


def predict_local(
    fit: pd.DataFrame,
    query: pd.DataFrame,
    fingerprints: dict[str, object],
    minimum_fit: int,
    minimum_neighbors: int,
    maximum_neighbors: int,
    similarity_floor: float,
) -> pd.DataFrame:
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
        order = order[similarities[order] >= similarity_floor][:maximum_neighbors]
        if len(order) < minimum_neighbors:
            continue
        weights = similarities[order]
        predicted = float(np.average(candidates.iloc[order]["actual_delta"], weights=weights))
        rows.append(
            {
                "mol_id_lo": row.mol_id_lo,
                "mol_id_hi": row.mol_id_hi,
                "transform_key": row.transform_key,
                "actual_delta": row.actual_delta,
                "predicted_delta": predicted,
                "fit_count": len(candidates),
                "context_neighbor_count": len(order),
                "maximum_context_similarity": float(weights.max()),
                "mean_context_similarity": float(weights.mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    targets = config["scope"]["targets"]
    inner_config = config["inner_evaluation"]
    prior = config["transformation_prior"]
    n_inner_folds = int(inner_config["folds"])
    seed = int(inner_config["deterministic_hash_seed"])
    cliff_threshold = float(config["evaluation"]["cliff_delta_threshold"])
    prediction_rows, coverage_rows = [], []
    for target in targets:
        assignments = pd.read_parquet(args.fold_root / target / "outer_folds.parquet")
        component_by_id = assignments.set_index("mol_id")["component_id"].to_dict()
        edges = read_oriented_edges(args.graph_root, target)
        fingerprints = build_fingerprints(edges["constant"], int(prior["fingerprint_bits"]))
        for outer_fold in range(5):
            validation_fold = (outer_fold + 1) % 5
            outer_train_ids = set(
                assignments.loc[
                    ~assignments["outer_fold"].isin([outer_fold, validation_fold]), "mol_id"
                ]
            )
            prohibited_ids = set(assignments["mol_id"]) - outer_train_ids
            labels = read_labels_for_ids(args.data_root, target, outer_train_ids).set_index("mol_id")[
                "y_pactivity"
            ]
            if set(labels.index) & prohibited_ids:
                raise RuntimeError("Non-training label entered train-only feasibility audit")
            train_edges = edges[
                edges["mol_id_1"].isin(outer_train_ids)
                & edges["mol_id_2"].isin(outer_train_ids)
            ].copy()
            train_edges["component_id"] = train_edges["mol_id_1"].map(component_by_id)
            if not (
                train_edges["component_id"] == train_edges["mol_id_2"].map(component_by_id)
            ).all():
                raise RuntimeError("An MMP edge crosses components")
            train_edges["inner_fold"] = train_edges["component_id"].map(
                lambda value: inner_fold(target, outer_fold, value, n_inner_folds, seed)
            )
            train_edges["actual_delta"] = train_edges["direction_sign"] * (
                train_edges["mol_id_2"].map(labels) - train_edges["mol_id_1"].map(labels)
            )
            for heldout_fold in range(n_inner_folds):
                fit = train_edges[train_edges["inner_fold"] != heldout_fold]
                query = train_edges[train_edges["inner_fold"] == heldout_fold]
                supported = predict_local(
                    fit,
                    query,
                    fingerprints,
                    int(prior["minimum_fit_occurrences"]),
                    int(prior["minimum_context_neighbors"]),
                    int(prior["maximum_context_neighbors"]),
                    float(prior["context_similarity_floor"]),
                )
                coverage_rows.append(
                    {
                        "target": target,
                        "outer_fold": outer_fold,
                        "inner_fold": heldout_fold,
                        "fit_edges": len(fit),
                        "query_edges": len(query),
                        "supported_edges": len(supported),
                        "support_coverage": len(supported) / len(query) if len(query) else float("nan"),
                    }
                )
                if len(supported):
                    prediction_rows.append(
                        supported.assign(
                            target=target, outer_fold=outer_fold, inner_fold=heldout_fold
                        )
                    )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    coverage = pd.DataFrame(coverage_rows)
    target_metrics = pd.DataFrame(
        [
            {"target": target, **metrics(group, cliff_threshold)}
            for target, group in predictions.groupby("target")
        ]
    )
    micro = metrics(predictions, cliff_threshold)
    decision = config["decision"]
    criteria = {
        "supported_edges": micro["supported_edges"] >= int(decision["minimum_supported_edges"]),
        "pair_delta": micro["relative_change_vs_zero"] <= -float(
            decision["minimum_pair_delta_improvement_vs_zero"]
        ),
        "cliff_pair_delta": micro["cliff_relative_change_vs_zero"] <= -float(
            decision["minimum_cliff_pair_delta_improvement_vs_zero"]
        ),
        "pearson": micro["pearson"] >= float(decision["minimum_pearson_correlation"]),
        "sign": micro["sign_accuracy"] >= float(decision["minimum_sign_accuracy"]),
        "cliff_sign": micro["cliff_sign_accuracy"] >= float(decision["minimum_cliff_sign_accuracy"]),
        "improved_targets": int((target_metrics["relative_change_vs_zero"] < 0).sum())
        >= int(decision["minimum_improved_targets"]),
    }
    summary = {
        "scope": "outer_train_labels_only_component_heldout_context_local",
        "micro": micro,
        "coverage_macro": float(coverage["support_coverage"].mean()),
        "improved_targets": int((target_metrics["relative_change_vs_zero"] < 0).sum()),
        "criteria": criteria,
        "advance_to_outer_validation": all(criteria.values()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.output_dir / "train_only_supported_predictions.parquet", index=False)
    coverage.to_csv(args.output_dir / "coverage_by_fold.csv", index=False)
    target_metrics.to_csv(args.output_dir / "metrics_by_target.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
