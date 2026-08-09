#!/usr/bin/env python3
"""Leave-one-target-out RF/GINE utility predictability audit on frozen OOF predictions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import yaml

from bps_gnn.io import read_labels_for_ids, sha256_file


FEATURES = [
    "ensemble_mean_z",
    "signed_disagreement_z",
    "absolute_disagreement_z",
    "rf_seed_std_z",
    "gine_seed_std_z",
    "seed_std_difference_z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_hashed_prediction(path: Path) -> pd.Series:
    manifest_path = path.with_suffix(path.suffix + ".sha256.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Prediction hash mismatch: {path}")
    frame = pd.read_parquet(path)
    if set(frame.columns) != {"mol_id", "prediction"}:
        raise RuntimeError(f"Unexpected prediction schema: {path}")
    if frame["mol_id"].duplicated().any():
        raise RuntimeError(f"Duplicate molecule prediction: {path}")
    return frame.set_index("mol_id")["prediction"].sort_index()


def bounded_router_weight(probability_rf_better: np.ndarray, amplitude: float) -> np.ndarray:
    probability = np.asarray(probability_rf_better, dtype=float)
    return 0.5 + float(amplitude) * (2.0 * probability - 1.0)


def load_target_rows(config: dict, target: str) -> pd.DataFrame:
    assignments = pd.read_parquet(Path(config["fold_root"]) / target / "outer_folds.parquet")
    folds = sorted(int(value) for value in assignments["outer_fold"].unique())
    seeds = [int(value) for value in config["seeds"]]
    frames = []
    for fold in folds:
        rf_predictions = []
        gine_predictions = []
        for seed in seeds:
            rf_path = (
                Path(config["rf_root"])
                / target
                / f"fold_{fold}"
                / f"seed_{seed}"
                / "predictions.parquet"
            )
            gine_path = (
                Path(config["gine_root"])
                / target
                / f"fold_{fold}"
                / f"seed_{seed}"
                / "predictions.parquet"
            )
            rf_predictions.append(read_hashed_prediction(rf_path))
            gine_predictions.append(read_hashed_prediction(gine_path))
        reference_index = rf_predictions[0].index
        for prediction in [*rf_predictions, *gine_predictions]:
            if not reference_index.equals(prediction.index):
                raise RuntimeError(f"RF/GINE molecule mismatch: {target} fold {fold}")
        gine_run = json.loads(
            (
                Path(config["gine_root"])
                / target
                / f"fold_{fold}"
                / f"seed_{seeds[0]}"
                / "run.json"
            ).read_text(encoding="utf-8")
        )
        center = float(gine_run["response_center"])
        scale = max(float(gine_run["response_mad"]), 1e-8)
        rf_matrix = np.column_stack([value.to_numpy() for value in rf_predictions])
        gine_matrix = np.column_stack([value.to_numpy() for value in gine_predictions])
        rf_mean = rf_matrix.mean(axis=1)
        gine_mean = gine_matrix.mean(axis=1)
        rf_std = rf_matrix.std(axis=1, ddof=0)
        gine_std = gine_matrix.std(axis=1, ddof=0)
        frames.append(
            pd.DataFrame(
                {
                    "target": target,
                    "fold": fold,
                    "mol_id": reference_index,
                    "rf_prediction": rf_mean,
                    "gine_prediction": gine_mean,
                    "fixed_equal_prediction": 0.5 * (rf_mean + gine_mean),
                    "ensemble_mean_z": (0.5 * (rf_mean + gine_mean) - center) / scale,
                    "signed_disagreement_z": (rf_mean - gine_mean) / scale,
                    "absolute_disagreement_z": np.abs(rf_mean - gine_mean) / scale,
                    "rf_seed_std_z": rf_std / scale,
                    "gine_seed_std_z": gine_std / scale,
                    "seed_std_difference_z": (rf_std - gine_std) / scale,
                }
            )
        )
    result = pd.concat(frames, ignore_index=True)
    if result["mol_id"].duplicated().any():
        raise RuntimeError(f"Outer-fold predictions do not partition {target}")
    truth = read_labels_for_ids(
        Path(config["data_root"]), target, result["mol_id"]
    ).set_index("mol_id")["y_pactivity"]
    result["truth"] = result["mol_id"].map(truth)
    rf_error = np.abs(result["rf_prediction"] - result["truth"])
    gine_error = np.abs(result["gine_prediction"] - result["truth"])
    result["rf_better"] = (rf_error < gine_error).astype(int)
    result["component_utility_gap"] = np.abs(rf_error - gine_error)
    return result


def fit_router(source: pd.DataFrame, seed: int) -> Pipeline:
    counts = source.groupby("target")["mol_id"].transform("count").to_numpy(dtype=float)
    sample_weight = 1.0 / counts
    sample_weight *= len(sample_weight) / sample_weight.sum()
    router = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    router.fit(
        source[FEATURES].to_numpy(),
        source["rf_better"].to_numpy(),
        logistic__sample_weight=sample_weight,
    )
    return router


def point_and_pair_metrics(
    frame: pd.DataFrame,
    prediction_column: str,
    edges: pd.DataFrame,
    cliff_threshold: float,
) -> dict:
    prediction = frame.set_index("mol_id")[prediction_column]
    truth = frame.set_index("mol_id")["truth"]
    cliffs = edges[
        edges["mol_id_lo"].isin(prediction.index)
        & edges["mol_id_hi"].isin(prediction.index)
    ].copy()
    cliffs["delta_y"] = cliffs["mol_id_hi"].map(truth) - cliffs["mol_id_lo"].map(truth)
    cliffs = cliffs[cliffs["delta_y"].abs() >= cliff_threshold]
    endpoints = set(cliffs["mol_id_lo"]) | set(cliffs["mol_id_hi"])
    endpoint_ids = [value for value in prediction.index if value in endpoints]
    predicted_delta = prediction.reindex(cliffs["mol_id_hi"]).to_numpy() - prediction.reindex(
        cliffs["mol_id_lo"]
    ).to_numpy()
    return {
        "overall_mae": float(np.mean(np.abs(prediction - truth.loc[prediction.index]))),
        "cliff_mae": float(np.mean(np.abs(prediction.loc[endpoint_ids] - truth.loc[endpoint_ids]))),
        "pair_delta_mae": float(np.mean(np.abs(predicted_delta - cliffs["delta_y"].to_numpy()))),
        "n_molecules": len(frame),
        "n_cliff_endpoints": len(endpoint_ids),
        "n_cliff_pairs": len(cliffs),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    targets = list(config["targets"])
    complete = pd.concat(
        [load_target_rows(config, target) for target in targets], ignore_index=True
    )
    routed_frames = []
    classification_rows = []
    for held_target in targets:
        source = complete[complete["target"] != held_target].copy()
        held = complete[complete["target"] == held_target].copy()
        if held_target in set(source["target"]):
            raise RuntimeError("Held target entered router training")
        router = fit_router(source, int(config["router"]["seed"]))
        classes = router.named_steps["logistic"].classes_.tolist()
        probability = router.predict_proba(held[FEATURES].to_numpy())[:, classes.index(1)]
        weight = bounded_router_weight(probability, float(config["router"]["amplitude"]))
        held["probability_rf_better"] = probability
        held["rf_weight"] = weight
        held["router_prediction"] = (
            weight * held["rf_prediction"] + (1.0 - weight) * held["gine_prediction"]
        )
        classification_rows.append(
            {
                "target": held_target,
                "n": len(held),
                "rf_better_prevalence": float(held["rf_better"].mean()),
                "auroc": float(roc_auc_score(held["rf_better"], probability)),
                "auprc": float(average_precision_score(held["rf_better"], probability)),
                "auprc_lift_over_prevalence": float(
                    average_precision_score(held["rf_better"], probability)
                    - held["rf_better"].mean()
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(held["rf_better"], probability >= 0.5)
                ),
                "brier": float(brier_score_loss(held["rf_better"], probability)),
                "mean_rf_weight": float(weight.mean()),
            }
        )
        routed_frames.append(held)
    routed = pd.concat(routed_frames, ignore_index=True)
    classification = pd.DataFrame(classification_rows)
    metric_rows = []
    for target in targets:
        target_frame = routed[routed["target"] == target].copy()
        edges = pd.read_parquet(
            Path(config["mmp_root"]) / target / "eligible_mmp_edges.parquet"
        )
        for method, column in [
            ("fixed_equal", "fixed_equal_prediction"),
            ("cross_target_router", "router_prediction"),
        ]:
            metric_rows.append(
                {
                    "target": target,
                    "method": method,
                    **point_and_pair_metrics(
                        target_frame,
                        column,
                        edges,
                        float(config["evaluation"]["cliff_threshold"]),
                    ),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    wide = metrics.pivot(index="target", columns="method")
    relative_rows = []
    for target in targets:
        row = {"target": target}
        for metric in ["overall_mae", "cliff_mae", "pair_delta_mae"]:
            row[f"relative_{metric}"] = float(
                wide.loc[target, (metric, "cross_target_router")]
                / wide.loc[target, (metric, "fixed_equal")]
                - 1.0
            )
        relative_rows.append(row)
    relative = pd.DataFrame(relative_rows)
    macro = {
        metric: float(
            metrics[metrics["method"] == "cross_target_router"][metric].mean()
            / metrics[metrics["method"] == "fixed_equal"][metric].mean()
            - 1.0
        )
        for metric in ["overall_mae", "cliff_mae", "pair_delta_mae"]
    }
    decision = config["decision"]
    criteria = {
        "macro_auroc": float(classification["auroc"].mean())
        >= float(decision["minimum_macro_auroc"]),
        "macro_auprc_lift": float(classification["auprc_lift_over_prevalence"].mean())
        >= float(decision["minimum_macro_auprc_lift"]),
        "auroc_target_directions": int((classification["auroc"] > 0.5).sum())
        >= int(decision["minimum_auroc_targets_above_chance"]),
        "overall_improvement": macro["overall_mae"]
        <= -float(decision["minimum_overall_improvement_vs_fixed_equal"]),
        "cliff_improvement": macro["cliff_mae"]
        <= -float(decision["minimum_cliff_improvement_vs_fixed_equal"]),
        "pair_not_worse": macro["pair_delta_mae"]
        <= float(decision["maximum_pair_delta_change_vs_fixed_equal"]),
        "cliff_target_directions": int((relative["relative_cliff_mae"] < 0).sum())
        >= int(decision["minimum_cliff_targets_improved"]),
        "finite": bool(
            np.isfinite(classification.select_dtypes(include=[np.number])).all().all()
            and np.isfinite(metrics.select_dtypes(include=[np.number])).all().all()
        ),
    }
    safe_predictions = routed.drop(
        columns=["truth", "rf_better", "component_utility_gap"]
    ).sort_values(["target", "fold", "mol_id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "cross_target_router_predictions.parquet"
    safe_predictions.to_parquet(prediction_path, index=False)
    prediction_hash = sha256_file(prediction_path)
    classification.to_csv(args.output_dir / "classification_by_target.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics_by_target.csv", index=False)
    relative.to_csv(args.output_dir / "relative_changes_by_target.csv", index=False)
    result = {
        "audit_status": "exploratory_existing_frozen_oof_predictions",
        "held_target_excluded_from_every_router_fit": True,
        "historical_outer_test_labels_materialized_for_evaluation": True,
        "prediction_sha256": prediction_hash,
        "features": FEATURES,
        "router": config["router"],
        "macro_classification": {
            "auroc": float(classification["auroc"].mean()),
            "auprc_lift_over_prevalence": float(
                classification["auprc_lift_over_prevalence"].mean()
            ),
            "balanced_accuracy": float(classification["balanced_accuracy"].mean()),
        },
        "macro_relative_changes_percent": {
            key: 100.0 * value for key, value in macro.items()
        },
        "directions": {
            "auroc_targets_above_chance": int((classification["auroc"] > 0.5).sum()),
            "overall_targets_improved": int((relative["relative_overall_mae"] < 0).sum()),
            "cliff_targets_improved": int((relative["relative_cliff_mae"] < 0).sum()),
            "pair_targets_improved": int((relative["relative_pair_delta_mae"] < 0).sum()),
        },
        "criteria": criteria,
        "advance_to_fresh_target_smoke": bool(all(criteria.values())),
        "fresh_target_if_passed": config["expansion"]["fresh_target_if_passed"],
        "wall_time_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "decision.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
