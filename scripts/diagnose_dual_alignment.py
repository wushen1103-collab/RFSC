#!/usr/bin/env python3
"""Post-freeze diagnosis of boundary transfer and signed high-pass alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from bps_gnn.io import read_labels_for_ids, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--gamma", type=float, default=0.25)
    return parser.parse_args()


def verify(path: Path) -> None:
    manifest = json.loads(
        path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8")
    )
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Frozen artifact hash mismatch: {path}")


def safe_auprc(labels: pd.Series, score: pd.Series) -> float:
    if labels.nunique() < 2:
        return float("nan")
    return float(average_precision_score(labels.to_numpy(), score.to_numpy()))


def safe_corr(left: pd.Series, right: pd.Series) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or left[valid].std() <= 1e-12 or right[valid].std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def target_row(target: str, query: pd.DataFrame, edge: pd.DataFrame) -> dict:
    endpoint = query[query["is_internal_cliff_endpoint"]]
    nonendpoint = query[~query["is_internal_cliff_endpoint"]]
    aligned = endpoint[endpoint["message_nonzero"]]
    top = query[query["weighted_q"] >= query["weighted_q"].quantile(0.75)]
    bottom = query[query["weighted_q"] <= query["weighted_q"].quantile(0.25)]
    return {
        "target": target,
        "queries": int(len(query)),
        "internal_cliff_endpoint_prevalence": float(query["is_internal_cliff_endpoint"].mean()),
        "query_weighted_q_endpoint_auprc": safe_auprc(
            query["is_internal_cliff_endpoint"], query["weighted_q"]
        ),
        "endpoint_rate_top_q_quartile": float(top["is_internal_cliff_endpoint"].mean()),
        "endpoint_rate_bottom_q_quartile": float(bottom["is_internal_cliff_endpoint"].mean()),
        "query_train_boundary_auprc": safe_auprc(edge["is_boundary"], edge["q_learned"]),
        "query_train_boundary_prevalence": float(edge["is_boundary"].mean()),
        "endpoint_message_nonzero_fraction": float(endpoint["message_nonzero"].mean()),
        "endpoint_message_required_residual_correlation": safe_corr(
            endpoint["boundary_message"], endpoint["required_from_cut"]
        ),
        "endpoint_message_sign_accuracy": float(aligned["message_sign_matches_required"].mean()),
        "endpoint_fraction_helped_at_gamma": float(endpoint["dual_helped_vs_cut"].mean()),
        "nonendpoint_fraction_helped_at_gamma": float(nonendpoint["dual_helped_vs_cut"].mean()),
        "endpoint_base_mae": float(endpoint["base_abs_error"].mean()),
        "endpoint_cut_mae": float(endpoint["cut_abs_error"].mean()),
        "endpoint_dual_mae": float(endpoint["dual_abs_error"].mean()),
        "nonendpoint_base_mae": float(nonendpoint["base_abs_error"].mean()),
        "nonendpoint_cut_mae": float(nonendpoint["cut_abs_error"].mean()),
        "nonendpoint_dual_mae": float(nonendpoint["dual_abs_error"].mean()),
        "endpoint_median_required_gamma": float(aligned["required_gamma"].median()),
        "endpoint_positive_required_gamma_fraction": float((aligned["required_gamma"] > 0).mean()),
    }


def main() -> None:
    args = parse_args()
    suffix = str(float(args.gamma)).replace(".", "p")
    dual_method = f"learned_dual_g{suffix}"
    query_rows, edge_rows = [], []
    completed = 0
    for run_path in sorted(args.run_dir.glob("*/fold_*/run.json")):
        metadata = json.loads(run_path.read_text(encoding="utf-8"))
        if metadata["validation_labels_materialized"] or metadata["outer_test_labels_materialized"]:
            raise RuntimeError(f"Query label entered trainer: {run_path}")
        target = metadata["target"]
        fold = int(metadata["fold"])
        prediction_path = run_path.parent / "test_predictions.parquet"
        gate_path = run_path.parent / "test_gate.parquet"
        verify(prediction_path)
        verify(gate_path)
        predictions = pd.read_parquet(prediction_path)
        gate = pd.read_parquet(gate_path)
        if dual_method not in set(predictions["method"]):
            raise RuntimeError(f"Missing {dual_method}: {prediction_path}")

        query_ids = predictions.loc[predictions["method"] == "base", "mol_id"].unique().tolist()
        reference_ids = gate["reference_id"].unique().tolist()
        labels = read_labels_for_ids(
            args.data_root, target, [*query_ids, *reference_ids]
        ).set_index("mol_id")["y_pactivity"]
        mmp = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        internal = mmp[
            mmp["mol_id_lo"].isin(query_ids) & mmp["mol_id_hi"].isin(query_ids)
        ].copy()
        internal["delta_y"] = (
            internal["mol_id_hi"].map(labels) - internal["mol_id_lo"].map(labels)
        )
        cliffs = internal[internal["delta_y"].abs() >= 1.0]
        endpoints = set(cliffs["mol_id_lo"]) | set(cliffs["mol_id_hi"])

        for seed in sorted(predictions["seed"].unique()):
            seed_predictions = predictions[predictions["seed"] == seed]
            series = {
                method: seed_predictions[
                    (seed_predictions["method"] == method)
                    & (seed_predictions["replicate"] == -1)
                ].set_index("mol_id")["prediction"]
                for method in ["base", "learned", dual_method]
            }
            if any(set(value.index) != set(query_ids) for value in series.values()):
                raise RuntimeError(f"Incomplete deterministic prediction in {run_path}")
            seed_gate = gate[gate["seed"] == seed].copy()
            seed_gate["query_y"] = seed_gate["query_id"].map(labels)
            seed_gate["reference_y"] = seed_gate["reference_id"].map(labels)
            seed_gate["is_boundary"] = (
                (seed_gate["query_y"] - seed_gate["reference_y"]).abs() >= 1.0
            )
            edge_rows.append(
                seed_gate.assign(target=target, fold=fold)[
                    [
                        "target",
                        "fold",
                        "seed",
                        "query_id",
                        "reference_id",
                        "similarity",
                        "q_learned",
                        "is_boundary",
                    ]
                ]
            )
            gate_summary = seed_gate.groupby("query_id").apply(
                lambda group: pd.Series(
                    {
                        "neighbors": len(group),
                        "mean_q": group["q_learned"].mean(),
                        "max_q": group["q_learned"].max(),
                        "weighted_q": np.average(
                            group["q_learned"], weights=group["similarity"]
                        ),
                        "actual_boundary_rate": group["is_boundary"].mean(),
                    }
                ),
                include_groups=False,
            )
            for mol_id in query_ids:
                base = float(series["base"].loc[mol_id])
                cut = float(series["learned"].loc[mol_id])
                dual = float(series[dual_method].loc[mol_id])
                truth = float(labels.loc[mol_id])
                message = (dual - cut) / args.gamma
                required = truth - cut
                summary = gate_summary.loc[mol_id] if mol_id in gate_summary.index else None
                nonzero = abs(message) > 1e-12
                query_rows.append(
                    {
                        "target": target,
                        "fold": fold,
                        "seed": int(seed),
                        "mol_id": mol_id,
                        "is_internal_cliff_endpoint": mol_id in endpoints,
                        "neighbors": int(summary["neighbors"]) if summary is not None else 0,
                        "mean_q": float(summary["mean_q"]) if summary is not None else 0.0,
                        "max_q": float(summary["max_q"]) if summary is not None else 0.0,
                        "weighted_q": float(summary["weighted_q"]) if summary is not None else 0.0,
                        "actual_boundary_rate": float(summary["actual_boundary_rate"]) if summary is not None else 0.0,
                        "boundary_message": message,
                        "required_from_cut": required,
                        "message_nonzero": nonzero,
                        "message_sign_matches_required": bool(nonzero and np.sign(message) == np.sign(required)),
                        "required_gamma": required / message if nonzero else float("nan"),
                        "dual_helped_vs_cut": abs(dual - truth) < abs(cut - truth),
                        "base_abs_error": abs(base - truth),
                        "cut_abs_error": abs(cut - truth),
                        "dual_abs_error": abs(dual - truth),
                    }
                )
        completed += 1
    if completed != 29:
        raise RuntimeError(f"Expected 29 folds, observed {completed}")
    query = pd.DataFrame(query_rows)
    edge = pd.concat(edge_rows, ignore_index=True)
    target = pd.DataFrame(
        [target_row(name, group, edge[edge["target"] == name]) for name, group in query.groupby("target")]
    )
    numeric = [column for column in target.columns if column != "target"]
    macro = {column: float(target[column].mean()) for column in numeric}
    macro.update({"targets": int(len(target)), "folds": completed, "gamma": args.gamma})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    query.to_csv(args.output_dir / "dual_alignment_query.csv", index=False)
    edge.to_csv(args.output_dir / "dual_alignment_edges.csv", index=False)
    target.to_csv(args.output_dir / "dual_alignment_target_table.csv", index=False)
    (args.output_dir / "dual_alignment_macro.json").write_text(
        json.dumps(macro, indent=2), encoding="utf-8"
    )
    print(json.dumps(macro, indent=2))


if __name__ == "__main__":
    main()
