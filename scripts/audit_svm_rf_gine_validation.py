#!/usr/bin/env python3
"""Development-validation-only screen of SVM/RF/GINE complementarity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from bps_gnn.io import read_labels_for_ids
try:
    from audit_gated_component_scale_validation import gated_edges, metrics, read_prediction
    from build_rf_gine_component_scale import scale_components
except ModuleNotFoundError:
    from scripts.audit_gated_component_scale_validation import gated_edges, metrics, read_prediction
    from scripts.build_rf_gine_component_scale import scale_components


METRICS = ["overall_mae", "cliff_mae", "pair_delta_mae"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, default=Path("folds/scg"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-target")
    parser.add_argument("--limit-fold", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    components = config["components"]
    svm_paths = sorted(Path(components["svm"]).glob("*/fold_*/seed_0/validation_predictions.parquet"))
    records = []
    for svm_path in svm_paths:
        target = svm_path.parts[-4]
        fold = int(svm_path.parts[-3].removeprefix("fold_"))
        if args.limit_target is not None and target != args.limit_target:
            continue
        if args.limit_fold is not None and fold != args.limit_fold:
            continue
        assignments = pd.read_parquet(args.fold_root / target / "outer_folds.parquet")
        n_folds = int(assignments.outer_fold.nunique())
        validation_fold = (fold + 1) % n_folds
        validation_ids = set(assignments.loc[assignments.outer_fold == validation_fold, "mol_id"])
        svm = read_prediction(svm_path)
        rf_paths = sorted((Path(components["rf"]) / target / f"fold_{fold}").glob("seed_*/validation_predictions.parquet"))
        rf = sum((read_prediction(path) for path in rf_paths)) / len(rf_paths)
        gine_frame = pd.read_parquet(Path(components["gine_validation_source"]) / target / f"fold_{fold}" / "validation_predictions.parquet")
        gine = gine_frame[gine_frame.method == "base"].groupby("mol_id").prediction.mean().sort_index()
        fixed = 0.5 * rf + 0.5 * gine
        if not svm.index.equals(fixed.index) or set(svm.index) != validation_ids:
            raise RuntimeError(f"Validation partition mismatch: {target} fold {fold}")
        truth = read_labels_for_ids(args.data_root, target, validation_ids).set_index("mol_id")["y_pactivity"]
        all_edges = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        edges = all_edges[all_edges.mol_id_lo.isin(validation_ids) & all_edges.mol_id_hi.isin(validation_ids)].copy()
        edges["delta_y"] = edges.mol_id_hi.map(truth) - edges.mol_id_lo.map(truth)
        cliffs = edges[edges.delta_y.abs() >= 1.0]
        svm_metrics = metrics(svm, truth, cliffs)
        for svm_weight in config["candidates"]["svm_weights"]:
            base = float(svm_weight) * svm + (1.0 - float(svm_weight)) * fixed
            base_metrics = metrics(base, truth, cliffs)
            for scale in config["candidates"]["scales"]:
                thresholds = [0.0] if float(scale) == 1.0 else config["candidates"]["predicted_mmp_delta_thresholds"]
                for threshold in thresholds:
                    selected = gated_edges(base, edges, float(threshold)) if float(scale) != 1.0 else edges.iloc[0:0]
                    prediction = base if float(scale) == 1.0 else scale_components(base, selected, float(scale))
                    measured = metrics(prediction, truth, cliffs)
                    records.append(
                        {
                            "target": target,
                            "fold": fold,
                            "svm_weight": float(svm_weight),
                            "scale": float(scale),
                            "threshold": float(threshold),
                            **measured,
                            **{f"relative_{metric}_pct_vs_svm": 100.0 * (measured[metric] / svm_metrics[metric] - 1.0) for metric in METRICS},
                            **{f"relative_{metric}_pct_vs_unscaled_mixture": 100.0 * (measured[metric] / base_metrics[metric] - 1.0) for metric in METRICS},
                        }
                    )
    if not records:
        raise RuntimeError("No development validation records")
    frame = pd.DataFrame(records)
    relative = [f"relative_{metric}_pct_vs_svm" for metric in METRICS]
    relative_mixture = [f"relative_{metric}_pct_vs_unscaled_mixture" for metric in METRICS]
    target = frame.groupby(["target", "svm_weight", "scale", "threshold"], as_index=False)[METRICS + relative + relative_mixture].mean()
    summary = target.groupby(["svm_weight", "scale", "threshold"], as_index=False)[METRICS + relative + relative_mixture].mean()
    direction = []
    for keys, group in target.groupby(["svm_weight", "scale", "threshold"]):
        direction.append({
            "svm_weight": keys[0],
            "scale": keys[1],
            "threshold": keys[2],
            **{f"improved_targets_{metric}": int((group[f'relative_{metric}_pct_vs_svm'] < 0).sum()) for metric in METRICS},
            **{f"improved_targets_{metric}_vs_unscaled_mixture": int((group[f'relative_{metric}_pct_vs_unscaled_mixture'] < 0).sum()) for metric in METRICS},
        })
    summary = summary.merge(pd.DataFrame(direction), on=["svm_weight", "scale", "threshold"], validate="one_to_one")
    decision = config["decision"]
    if decision.get("baseline", "svm") == "unscaled_mixture":
        summary["passes"] = (
            (summary.relative_overall_mae_pct_vs_unscaled_mixture <= 100 * float(decision["maximum_overall_relative_change"]))
            & (summary.relative_cliff_mae_pct_vs_unscaled_mixture <= 100 * float(decision["maximum_cliff_relative_change"]))
            & (summary.relative_pair_delta_mae_pct_vs_unscaled_mixture <= -100 * float(decision["minimum_pair_relative_improvement"]))
            & (summary.improved_targets_pair_delta_mae_vs_unscaled_mixture >= int(decision["minimum_pair_improved_targets"]))
        )
        ranking_columns = relative_mixture
    else:
        summary["passes"] = (
            (summary.relative_overall_mae_pct_vs_svm <= 100 * float(decision["maximum_overall_relative_change_vs_svm"]))
            & (summary.relative_cliff_mae_pct_vs_svm <= 100 * float(decision["maximum_cliff_relative_change_vs_svm"]))
            & (summary.relative_pair_delta_mae_pct_vs_svm <= 100 * float(decision["maximum_pair_relative_change_vs_svm"]))
            & (summary[[f"improved_targets_{metric}" for metric in METRICS]].min(axis=1) >= int(decision["minimum_improved_targets_per_metric"]))
        )
        ranking_columns = relative
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "validation_run_grid.csv", index=False)
    target.to_csv(args.output_dir / "validation_target_grid.csv", index=False)
    summary.to_csv(args.output_dir / "validation_summary.csv", index=False)
    best = summary.assign(score=summary[ranking_columns].sum(axis=1)).sort_values("score").head(12)
    payload = {"labels": "development_outer_validation_only", "folds": int(frame[["target", "fold"]].drop_duplicates().shape[0]), "passing_candidates": int(summary.passes.sum()), "best": best.to_dict(orient="records")}
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
