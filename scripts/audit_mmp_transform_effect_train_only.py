#!/usr/bin/env python3
"""Test MMP transformation-effect transfer using outer-training labels only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_labels_for_ids


RAW_COLUMNS = ["smiles_1", "smiles_2", "mol_id_1", "mol_id_2", "transform", "constant"]


def transform_parts(value: str) -> tuple[str, str]:
    parts = str(value).split(">>", 1)
    if len(parts) != 2:
        raise ValueError(f"Malformed transformation: {value!r}")
    return parts[0], parts[1]


def canonical_key_and_sign(value: str) -> tuple[str, int]:
    left, right = transform_parts(value)
    if left <= right:
        return f"{left}>>{right}", 1
    return f"{right}>>{left}", -1


def inner_fold(target: str, outer_fold: int, component_id: str, folds: int, seed: int) -> int:
    token = f"{seed}|{target}|{outer_fold}|{component_id}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % folds


def read_oriented_edges(graph_root: Path, target: str) -> pd.DataFrame:
    target_root = graph_root / target
    eligible = pd.read_parquet(target_root / "eligible_mmp_edges.parquet")
    raw = pd.read_csv(
        target_root / f"{target}.pairs.csv.gz",
        sep="\t",
        header=None,
        names=RAW_COLUMNS,
        dtype=str,
    )
    ordered = np.sort(raw[["mol_id_1", "mol_id_2"]].to_numpy(dtype=str), axis=1)
    raw["mol_id_lo"] = ordered[:, 0]
    raw["mol_id_hi"] = ordered[:, 1]
    keys = ["mol_id_lo", "mol_id_hi", "transform", "constant"]
    oriented = eligible.merge(
        raw[keys + ["mol_id_1", "mol_id_2"]], on=keys, how="left", validate="one_to_many"
    ).drop_duplicates(keys)
    if len(oriented) != len(eligible) or oriented[["mol_id_1", "mol_id_2"]].isna().any().any():
        raise RuntimeError(f"Could not recover unique raw MMP orientation for {target}")
    key_sign = oriented["transform"].map(canonical_key_and_sign)
    oriented["transform_key"] = [value[0] for value in key_sign]
    oriented["direction_sign"] = [value[1] for value in key_sign]
    return oriented


def safe_corr(actual: pd.Series, predicted: pd.Series, method: str) -> float:
    if len(actual) < 2 or actual.nunique() < 2 or predicted.nunique() < 2:
        return float("nan")
    return float(actual.corr(predicted, method=method))


def metrics(frame: pd.DataFrame, cliff_threshold: float) -> dict:
    actual = frame["actual_delta"]
    predicted = frame["predicted_delta"]
    zero_mae = float(actual.abs().mean())
    prediction_mae = float((actual - predicted).abs().mean())
    cliff = frame[actual.abs() >= cliff_threshold]
    cliff_zero_mae = float(cliff["actual_delta"].abs().mean()) if len(cliff) else float("nan")
    cliff_prediction_mae = (
        float((cliff["actual_delta"] - cliff["predicted_delta"]).abs().mean())
        if len(cliff)
        else float("nan")
    )
    return {
        "supported_edges": int(len(frame)),
        "supported_cliff_edges": int(len(cliff)),
        "zero_mae": zero_mae,
        "prediction_mae": prediction_mae,
        "relative_change_vs_zero": prediction_mae / zero_mae - 1.0,
        "pearson": safe_corr(actual, predicted, "pearson"),
        "spearman": safe_corr(actual, predicted, "spearman"),
        "sign_accuracy": float((np.sign(actual) == np.sign(predicted)).mean()),
        "cliff_zero_mae": cliff_zero_mae,
        "cliff_prediction_mae": cliff_prediction_mae,
        "cliff_relative_change_vs_zero": cliff_prediction_mae / cliff_zero_mae - 1.0,
        "cliff_sign_accuracy": (
            float((np.sign(cliff["actual_delta"]) == np.sign(cliff["predicted_delta"])).mean())
            if len(cliff)
            else float("nan")
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--fold-root", type=Path, default=Path("folds/scg"))
    parser.add_argument("--graph-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    targets = config["scope"]["targets"]
    n_inner_folds = int(config["inner_evaluation"]["folds"])
    seed = int(config["inner_evaluation"]["deterministic_hash_seed"])
    minimum_count = int(config["transformation_prior"]["minimum_fit_occurrences"])
    cliff_threshold = float(config["evaluation"]["cliff_delta_threshold"])
    prediction_rows = []
    coverage_rows = []
    for target in targets:
        assignments = pd.read_parquet(args.fold_root / target / "outer_folds.parquet")
        component_by_id = assignments.set_index("mol_id")["component_id"].to_dict()
        edges = read_oriented_edges(args.graph_root, target)
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
                query = train_edges[train_edges["inner_fold"] == heldout_fold].copy()
                statistics = fit.groupby("transform_key")["actual_delta"].agg(
                    fit_count="size", predicted_delta="median"
                )
                query = query.join(statistics, on="transform_key")
                supported = query[query["fit_count"] >= minimum_count].copy()
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
                prediction_rows.append(
                    supported[
                        [
                            "mol_id_lo",
                            "mol_id_hi",
                            "transform_key",
                            "actual_delta",
                            "predicted_delta",
                            "fit_count",
                        ]
                    ].assign(target=target, outer_fold=outer_fold, inner_fold=heldout_fold)
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
    decision_config = config["decision"]
    criteria = {
        "supported_edges": micro["supported_edges"] >= int(decision_config["minimum_supported_edges"]),
        "pair_delta": micro["relative_change_vs_zero"] <= -float(
            decision_config["minimum_pair_delta_improvement_vs_zero"]
        ),
        "cliff_pair_delta": micro["cliff_relative_change_vs_zero"] <= -float(
            decision_config["minimum_cliff_pair_delta_improvement_vs_zero"]
        ),
        "pearson": micro["pearson"] >= float(decision_config["minimum_pearson_correlation"]),
        "sign": micro["sign_accuracy"] >= float(decision_config["minimum_sign_accuracy"]),
        "cliff_sign": micro["cliff_sign_accuracy"] >= float(
            decision_config["minimum_cliff_sign_accuracy"]
        ),
        "improved_targets": int((target_metrics["relative_change_vs_zero"] < 0).sum())
        >= int(decision_config["minimum_improved_targets"]),
    }
    summary = {
        "scope": "outer_train_labels_only_component_heldout",
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

