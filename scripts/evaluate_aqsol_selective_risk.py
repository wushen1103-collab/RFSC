#!/usr/bin/env python3
"""Evaluate the frozen response-free AqSolDB coverage-risk curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_frozen_prediction, read_labels_for_ids, sha256_file


def paired_bootstrap(candidate: np.ndarray, reference: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed); values = np.empty(iterations)
    for index in range(iterations):
        chosen = rng.integers(0, len(candidate), len(candidate))
        values[index] = candidate[chosen].mean() / reference[chosen].mean() - 1.0
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def pair_mae(prediction: pd.Series, truth: pd.Series, cliffs: pd.DataFrame) -> float:
    if cliffs.empty:
        return float("nan")
    predicted = prediction.reindex(cliffs["mol_id_hi"]).to_numpy() - prediction.reindex(cliffs["mol_id_lo"]).to_numpy()
    actual = cliffs["mol_id_hi"].map(truth).to_numpy() - cliffs["mol_id_lo"].map(truth).to_numpy()
    return float(np.mean(np.abs(predicted - actual)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    payload = joblib.load(args.model)
    external, frozen = config["external"], []
    target = external["target"]
    run_paths = sorted(args.prediction_root.glob(f"{target}/fold_*/seed_*/run.json"))
    for run_path in run_paths:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        base = read_frozen_prediction(run_path.parent / "unscaled_reference" / "predictions.parquet")
        candidate = read_frozen_prediction(run_path.parent / "bps" / "predictions.parquet")
        score_path = args.score_root / target / f"fold_{run['fold']}" / f"seed_{run['seed']}" / "risk_scores.parquet"
        metadata = json.loads(score_path.with_suffix(score_path.suffix + ".sha256.json").read_text(encoding="utf-8"))
        if metadata["response_columns_read"] is not False or sha256_file(score_path) != metadata["risk_score_sha256"]:
            raise RuntimeError("Risk score preflight failed")
        score = pd.read_parquet(score_path).set_index("mol_id")["risk_score"]
        if set(score.index) != set(base.index) or set(candidate.index) != set(base.index):
            raise RuntimeError("Risk/prediction partition mismatch")
        frozen.append((run, base, candidate, score.reindex(base.index)))
    if not frozen:
        raise RuntimeError("No frozen AqSolDB risk runs")

    all_edges = pd.read_parquet(Path(external["mmp_root"]) / target / "eligible_mmp_edges.parquet")
    rows = []
    for run, base, candidate, score in frozen:
        truth = read_labels_for_ids(Path(external["data_root"]), target, base.index).set_index("mol_id")["y_pactivity"].reindex(base.index)
        edges = all_edges[all_edges["mol_id_lo"].isin(base.index) & all_edges["mol_id_hi"].isin(base.index)].copy()
        edges["truth_delta"] = edges["mol_id_hi"].map(truth) - edges["mol_id_lo"].map(truth)
        cliffs = edges[edges["truth_delta"].abs() >= float(external["cliff_threshold"])]
        cliff_endpoints = set(cliffs["mol_id_lo"]) | set(cliffs["mol_id_hi"])
        for coverage in map(float, config["coverage"]):
            accepted = pd.Series(True, index=base.index) if coverage == 1.0 else score <= float(payload["thresholds"][str(coverage)])
            accepted_ids = set(accepted[accepted].index)
            accepted_cliff_endpoints = sorted(cliff_endpoints & accepted_ids)
            accepted_cliffs = cliffs[cliffs["mol_id_lo"].isin(accepted_ids) & cliffs["mol_id_hi"].isin(accepted_ids)]
            hybrid = base.copy(); hybrid.loc[list(accepted_ids)] = candidate.loc[list(accepted_ids)]
            base_error = (base - truth).abs(); candidate_error = (candidate - truth).abs()
            rows.append({
                "fold": int(run["fold"]), "seed": int(run["seed"]), "nominal_coverage": coverage,
                "realized_node_coverage": float(accepted.mean()), "accepted_nodes": int(accepted.sum()),
                "accepted_true_cliff_pairs": int(len(accepted_cliffs)), "true_cliff_pair_coverage": float(len(accepted_cliffs) / len(cliffs)) if len(cliffs) else float("nan"),
                "accepted_cliff_endpoints": int(len(accepted_cliff_endpoints)), "cliff_endpoint_coverage": float(len(accepted_cliff_endpoints) / len(cliff_endpoints)) if cliff_endpoints else float("nan"),
                "accepted_reference_mae": float(base_error[accepted].mean()), "accepted_bps_mae": float(candidate_error[accepted].mean()),
                "accepted_point_relative_change": float(candidate_error[accepted].mean() / base_error[accepted].mean() - 1.0),
                "accepted_harmed_fraction": float((candidate_error[accepted] > base_error[accepted]).mean()),
                "accepted_reference_cliff_mae": float(base_error.reindex(accepted_cliff_endpoints).mean()) if accepted_cliff_endpoints else float("nan"),
                "accepted_bps_cliff_mae": float(candidate_error.reindex(accepted_cliff_endpoints).mean()) if accepted_cliff_endpoints else float("nan"),
                "accepted_reference_pair_mae": pair_mae(base, truth, accepted_cliffs), "accepted_bps_pair_mae": pair_mae(candidate, truth, accepted_cliffs),
                "accepted_pair_relative_change": pair_mae(candidate, truth, accepted_cliffs) / pair_mae(base, truth, accepted_cliffs) - 1.0 if len(accepted_cliffs) else float("nan"),
                "hybrid_overall_mae": float((hybrid - truth).abs().mean()), "reference_overall_mae": float(base_error.mean()), "bps_overall_mae": float(candidate_error.mean()),
                "hybrid_cliff_mae": float((hybrid.reindex(sorted(cliff_endpoints)) - truth.reindex(sorted(cliff_endpoints))).abs().mean()) if cliff_endpoints else float("nan"),
                "reference_cliff_mae": float(base_error.reindex(sorted(cliff_endpoints)).mean()) if cliff_endpoints else float("nan"),
                "bps_cliff_mae": float(candidate_error.reindex(sorted(cliff_endpoints)).mean()) if cliff_endpoints else float("nan"),
                "hybrid_pair_mae": pair_mae(hybrid, truth, cliffs), "reference_pair_mae": pair_mae(base, truth, cliffs), "bps_pair_mae": pair_mae(candidate, truth, cliffs),
            })
    run_metrics = pd.DataFrame(rows)
    fold_metrics = run_metrics.groupby(["fold", "nominal_coverage"], as_index=False).mean(numeric_only=True)
    summary = fold_metrics.groupby("nominal_coverage", as_index=False).agg({column: ["mean", "std"] for column in fold_metrics.columns if column not in {"fold", "nominal_coverage"}})
    statistics = []
    contrasts = [
        ("accepted_point", "accepted_bps_mae", "accepted_reference_mae"),
        ("accepted_cliff", "accepted_bps_cliff_mae", "accepted_reference_cliff_mae"),
        ("accepted_pair", "accepted_bps_pair_mae", "accepted_reference_pair_mae"),
        ("hybrid_overall", "hybrid_overall_mae", "reference_overall_mae"),
        ("hybrid_cliff", "hybrid_cliff_mae", "reference_cliff_mae"),
        ("hybrid_pair", "hybrid_pair_mae", "reference_pair_mae"),
    ]
    for coverage_index, (coverage, group) in enumerate(fold_metrics.groupby("nominal_coverage")):
        for metric_index, (name, candidate_column, reference_column) in enumerate(contrasts):
            paired = group[[candidate_column, reference_column]].dropna()
            lower, upper = paired_bootstrap(paired[candidate_column].to_numpy(), paired[reference_column].to_numpy(), int(config["evaluation"]["bootstrap_iterations"]), int(config["evaluation"]["bootstrap_seed"]) + 10 * coverage_index + metric_index)
            statistics.append({"nominal_coverage": coverage, "contrast": name, "folds": len(paired), "relative_change": paired[candidate_column].mean() / paired[reference_column].mean() - 1.0, "bootstrap95_lower": lower, "bootstrap95_upper": upper, "favorable_folds": int((paired[candidate_column] < paired[reference_column]).sum())})
    statistics_frame = pd.DataFrame(statistics)
    point = fold_metrics.drop(columns=["fold", "seed"], errors="ignore").groupby("nominal_coverage", as_index=False).mean(numeric_only=True).set_index("nominal_coverage")
    primary_coverage = float(config["primary_coverage"])
    primary = point.loc[primary_coverage]
    full = point.loc[1.0]
    checks = {
        "realized_coverage": float(primary["realized_node_coverage"]) >= float(config["decision"]["minimum_realized_node_coverage"]),
        "point_cost_lower_than_full": float(primary["accepted_point_relative_change"]) < float(full["accepted_point_relative_change"]),
        "pair_effect_favorable": float(primary["accepted_pair_relative_change"]) < 0.0,
    }
    decision = {"primary_nominal_coverage": primary_coverage, "checks": checks, "useful_risk_control": bool(all(checks.values())), "external_scores_frozen_before_response_evaluation": True}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_metrics.to_csv(args.output_dir / "seed_fold_coverage_metrics.csv", index=False)
    fold_metrics.to_csv(args.output_dir / "fold_coverage_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "coverage_mean_std.csv", index=False)
    statistics_frame.to_csv(args.output_dir / "paired_statistics.csv", index=False)
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"decision": decision, "primary": primary.to_dict(), "full": full.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
