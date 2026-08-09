#!/usr/bin/env python3
"""Screen high-predicted-delta component scaling on outer validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_labels_for_ids, sha256_file
try:
    from build_rf_gine_component_scale import scale_components
except ModuleNotFoundError:
    from scripts.build_rf_gine_component_scale import scale_components


METRICS = ["overall_mae", "cliff_mae", "pair_delta_mae"]


def read_prediction(path: Path) -> pd.Series:
    manifest = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Prediction hash mismatch: {path}")
    return pd.read_parquet(path).set_index("mol_id")["prediction"].sort_index()


def metrics(prediction: pd.Series, truth: pd.Series, cliffs: pd.DataFrame) -> dict[str, float]:
    endpoints = set(cliffs.mol_id_lo) | set(cliffs.mol_id_hi)
    endpoint_ids = [value for value in prediction.index if value in endpoints]
    delta = prediction.reindex(cliffs.mol_id_hi).to_numpy() - prediction.reindex(cliffs.mol_id_lo).to_numpy()
    return {
        "overall_mae": float(np.mean(abs(prediction - truth.loc[prediction.index]))),
        "cliff_mae": float(np.mean(abs(prediction.loc[endpoint_ids] - truth.loc[endpoint_ids]))),
        "pair_delta_mae": float(np.mean(abs(delta - cliffs.delta_y.to_numpy()))),
    }


def gated_edges(base: pd.Series, edges: pd.DataFrame, threshold: float) -> pd.DataFrame:
    delta = abs(edges.mol_id_hi.map(base) - edges.mol_id_lo.map(base))
    return edges.loc[delta >= threshold].copy()


def select_loto(target_macro: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    selection = config["selection"]
    weights = selection["objective_weights"]
    targets = sorted(target_macro.target.unique())
    for held_out in targets:
        source = target_macro[target_macro.target != held_out]
        aggregate = source.groupby(["scale", "threshold"], as_index=False)[
            [f"relative_{metric}_pct" for metric in METRICS]
        ].mean()
        improved = (
            source.assign(pair_improved=source.relative_pair_delta_mae_pct < 0)
            .groupby(["scale", "threshold"], as_index=False)
            .agg(pair_improved_source_targets=("pair_improved", "sum"))
        )
        aggregate = aggregate.merge(improved, on=["scale", "threshold"], validate="one_to_one")
        aggregate["eligible"] = (
            (aggregate.relative_overall_mae_pct <= 100.0 * float(selection["maximum_overall_relative_change"]))
            & (aggregate.relative_cliff_mae_pct <= 100.0 * float(selection["maximum_cliff_relative_change"]))
            & (aggregate.relative_pair_delta_mae_pct < 0)
            & (aggregate.pair_improved_source_targets >= int(selection["minimum_pair_improved_source_targets"]))
        )
        aggregate["objective"] = (
            float(weights["pair_delta_mae"]) * aggregate.relative_pair_delta_mae_pct
            + float(weights["cliff_mae"]) * aggregate.relative_cliff_mae_pct
            + float(weights["overall_mae"]) * aggregate.relative_overall_mae_pct
        )
        pool = aggregate[aggregate.eligible]
        if pool.empty:
            chosen = aggregate[(aggregate.scale == 1.0) & (aggregate.threshold == 0.0)].iloc[0]
        else:
            chosen = pool.sort_values(["objective", "scale", "threshold"], kind="stable").iloc[0]
        rows.append({"target": held_out, **chosen.to_dict(), "source_targets": len(targets) - 1})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rf-dir", type=Path, default=Path("experiments/rf_scg/full"))
    parser.add_argument("--gine-validation-source", type=Path, default=Path("experiments/sqrl_anchor/full"))
    parser.add_argument("--fold-root", type=Path, default=Path("folds/scg"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-target")
    parser.add_argument("--limit-fold", type=int)
    parser.add_argument("--limit-seed", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    candidates = [(1.0, 0.0)] + [
        (float(scale), float(threshold))
        for scale in config["operator"]["scales"]
        for threshold in config["operator"]["predicted_mmp_delta_thresholds"]
    ]

    paths = sorted(args.rf_dir.glob("*/fold_*/seed_*/validation_predictions.parquet"))
    records: list[dict] = []
    for rf_path in paths:
        target = rf_path.parts[-4]
        fold = int(rf_path.parts[-3].removeprefix("fold_"))
        seed = int(rf_path.parts[-2].removeprefix("seed_"))
        if args.limit_target is not None and target != args.limit_target:
            continue
        if args.limit_fold is not None and fold != args.limit_fold:
            continue
        if args.limit_seed is not None and seed != args.limit_seed:
            continue
        assignments = pd.read_parquet(args.fold_root / target / "outer_folds.parquet")
        n_folds = int(assignments.outer_fold.nunique())
        validation_fold = (fold + 1) % n_folds
        validation_ids = set(assignments.loc[assignments.outer_fold == validation_fold, "mol_id"])
        rf_valid = read_prediction(rf_path)
        gine_frame = pd.read_parquet(args.gine_validation_source / target / f"fold_{fold}" / "validation_predictions.parquet")
        gine_valid = gine_frame[(gine_frame.method == "base") & (gine_frame.seed == seed)].set_index("mol_id")["prediction"].sort_index()
        base = 0.5 * rf_valid + 0.5 * gine_valid
        if set(base.index) != validation_ids:
            raise RuntimeError(f"Validation partition mismatch: {target} fold {fold} seed {seed}")
        truth = read_labels_for_ids(args.data_root, target, validation_ids).set_index("mol_id")["y_pactivity"]
        all_edges = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        edges = all_edges[all_edges.mol_id_lo.isin(validation_ids) & all_edges.mol_id_hi.isin(validation_ids)].copy()
        edges["delta_y"] = edges.mol_id_hi.map(truth) - edges.mol_id_lo.map(truth)
        cliffs = edges[edges.delta_y.abs() >= 1.0]
        identity = metrics(base, truth, cliffs)
        for scale, threshold in candidates:
            selected_edges = edges.iloc[0:0] if scale == 1.0 else gated_edges(base, edges, threshold)
            prediction = base if scale == 1.0 else scale_components(base, selected_edges, scale)
            measures = metrics(prediction, truth, cliffs)
            records.append(
                {
                    "target": target,
                    "fold": fold,
                    "seed": seed,
                    "scale": scale,
                    "threshold": threshold,
                    "selected_edges": len(selected_edges),
                    "selected_nodes": len(set(selected_edges.mol_id_lo) | set(selected_edges.mol_id_hi)),
                    **measures,
                    **{f"relative_{metric}_pct": 100.0 * (measures[metric] / identity[metric] - 1.0) for metric in METRICS},
                }
            )
    if not records:
        raise RuntimeError("No validation runs matched the requested scope")
    frame = pd.DataFrame(records)
    relative = [f"relative_{metric}_pct" for metric in METRICS]
    target_macro = frame.groupby(["target", "scale", "threshold"], as_index=False)[METRICS + relative + ["selected_edges", "selected_nodes"]].mean()
    summary = target_macro.groupby(["scale", "threshold"], as_index=False)[METRICS + relative + ["selected_edges", "selected_nodes"]].mean()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "validation_run_grid.csv", index=False)
    target_macro.to_csv(args.output_dir / "validation_target_macro_grid.csv", index=False)
    summary.to_csv(args.output_dir / "validation_grid_summary.csv", index=False)
    selected = None
    if frame.target.nunique() >= 2:
        selected = select_loto(target_macro, config)
        selected.to_csv(args.output_dir / "loto_selected_parameters.csv", index=False)
    payload = {
        "label_scope": "outer_validation_only",
        "runs": int(frame[["target", "fold", "seed"]].drop_duplicates().shape[0]),
        "candidates": len(candidates),
        "targets": int(frame.target.nunique()),
        "best_pair_candidates": summary.sort_values("relative_pair_delta_mae_pct").head(8).to_dict(orient="records"),
        "loto_selected": None if selected is None else selected.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

