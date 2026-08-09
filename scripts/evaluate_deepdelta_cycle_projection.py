#!/usr/bin/env python3
"""Evaluate the once-frozen DeepDelta cycle-consistency projection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold

from bps_gnn.io import read_frozen_prediction
from bps_gnn.pair_flow import antisymmetric_pair_flow

try:
    from evaluate_chembl36_temporal import bootstrap_relative
except ModuleNotFoundError:
    from scripts.evaluate_chembl36_temporal import bootstrap_relative


LOWER_METRICS = ["pair_rmse", "pair_mae", "high_delta_pair_mae"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_flow(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise RuntimeError(f"{path}: expected one prediction row")
    flat = frame.iloc[0].to_numpy(dtype=float)
    n_nodes = int(round(np.sqrt(len(flat))))
    if n_nodes * n_nodes != len(flat):
        raise RuntimeError(f"{path}: prediction count is not square")
    return flat.reshape(n_nodes, n_nodes)


def ordered_position_values(prediction: pd.Series, n_nodes: int) -> np.ndarray:
    """Restore numeric matrix position after the generic freezer sorts IDs as strings."""
    expected = pd.Index([f"position_{index}" for index in range(n_nodes)])
    if set(prediction.index) != set(expected):
        missing = sorted(set(expected) - set(prediction.index))
        unexpected = sorted(set(prediction.index) - set(expected))
        raise RuntimeError(
            f"frozen position IDs mismatch: missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    return prediction.reindex(expected).to_numpy(dtype=float)


def flow_metrics(prediction: np.ndarray, truth: np.ndarray, high_mask: np.ndarray) -> dict[str, float]:
    off_diagonal = ~np.eye(len(truth), dtype=bool)
    predicted = prediction[off_diagonal]
    actual = truth[off_diagonal]
    error = predicted - actual
    correlation = float(np.corrcoef(predicted, actual)[0, 1])
    return {
        "pair_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "pair_mae": float(np.mean(np.abs(error))),
        "high_delta_pair_mae": float(np.mean(np.abs(prediction[high_mask] - truth[high_mask]))),
        "pearson_r": correlation,
    }


def comparison_statistics(
    task_metrics: pd.DataFrame, candidate_method: str, reference_method: str, iterations: int, seed: int
) -> list[dict[str, object]]:
    candidate = task_metrics[task_metrics["method"] == candidate_method].set_index("dataset")
    reference = task_metrics[task_metrics["method"] == reference_method].set_index("dataset")
    records: list[dict[str, object]] = []
    for metric_index, metric in enumerate(LOWER_METRICS):
        paired = pd.concat(
            [candidate[metric].rename("candidate"), reference[metric].rename("reference")], axis=1
        ).dropna()
        lower, upper = bootstrap_relative(
            paired["candidate"].to_numpy(),
            paired["reference"].to_numpy(),
            iterations,
            seed + metric_index,
        )
        records.append(
            {
                "candidate": candidate_method,
                "reference": reference_method,
                "metric": metric,
                "datasets": int(len(paired)),
                "reference_macro": float(paired["reference"].mean()),
                "candidate_macro": float(paired["candidate"].mean()),
                "relative_change": float(
                    paired["candidate"].mean() / paired["reference"].mean() - 1.0
                ),
                "bootstrap95_lower": lower,
                "bootstrap95_upper": upper,
                "favorable_datasets": int((paired["candidate"] < paired["reference"]).sum()),
                "favorable_fraction": float((paired["candidate"] < paired["reference"]).mean()),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    datasets = [str(value) for value in config["datasets"]]
    n_folds = int(config["cross_validation"]["folds"])

    # Hash and partition preflight before any benchmark response is loaded.
    for dataset in datasets:
        for fold in range(1, n_folds + 1):
            source = args.source_root / f"{dataset}_DeepDelta_Individual_Fold_{fold}.csv"
            run_dir = args.prediction_root / dataset / f"fold_{fold}"
            metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            if sha256_file(source) != metadata["source_prediction_sha256"]:
                raise RuntimeError(f"{source}: source hash changed")
            raw = load_raw_flow(source)
            potential = read_frozen_prediction(run_dir / "potential.parquet")
            if len(potential) != len(raw):
                raise RuntimeError(f"{dataset} fold {fold}: candidate partition mismatch")
            ordered_position_values(potential, len(raw))

    rows: list[dict[str, object]] = []
    theorem_rows: list[dict[str, object]] = []
    quantile = float(config["evaluation"]["high_delta_quantile_within_fold"])
    tolerance = float(config["decision"]["pythagorean_relative_tolerance"])
    for dataset in datasets:
        data = pd.read_csv(args.data_root / f"{dataset}.csv")
        if not {"SMILES", "Y"}.issubset(data.columns):
            raise RuntimeError(f"{dataset}: benchmark columns missing")
        response = data["Y"].to_numpy(dtype=float)
        splitter = KFold(
            n_splits=n_folds,
            shuffle=bool(config["cross_validation"]["shuffle"]),
            random_state=int(config["cross_validation"]["random_state"]),
        )
        for fold, (_, test_index) in enumerate(splitter.split(data), start=1):
            source = args.source_root / f"{dataset}_DeepDelta_Individual_Fold_{fold}.csv"
            raw = load_raw_flow(source)
            fold_response = response[test_index]
            if raw.shape != (len(fold_response), len(fold_response)):
                raise RuntimeError(f"{dataset} fold {fold}: official split shape mismatch")
            truth = fold_response[None, :] - fold_response[:, None]
            anti = antisymmetric_pair_flow(raw)
            potential = ordered_position_values(
                read_frozen_prediction(
                    args.prediction_root / dataset / f"fold_{fold}" / "potential.parquet"
                ),
                len(raw),
            )
            hodge = potential[None, :] - potential[:, None]
            off_diagonal = ~np.eye(len(truth), dtype=bool)
            cutoff = float(np.quantile(np.abs(truth[off_diagonal]), quantile))
            high_mask = off_diagonal & (np.abs(truth) >= cutoff)
            for method, prediction in {
                "published_deepdelta": raw,
                "antisymmetric": anti,
                "hodge_cycle_consistent": hodge,
            }.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "method": method,
                        "n_nodes": len(truth),
                        "n_off_diagonal_pairs": int(off_diagonal.sum()),
                        "high_delta_cutoff": cutoff,
                        "n_high_delta_pairs": int(high_mask.sum()),
                        **flow_metrics(prediction, truth, high_mask),
                    }
                )
            raw_sse = float(np.square(raw - truth).sum())
            hodge_sse = float(np.square(hodge - truth).sum())
            removed_sse = float(np.square(raw - hodge).sum())
            discrepancy = abs(raw_sse - hodge_sse - removed_sse) / max(raw_sse, 1.0)
            theorem_rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "raw_sse": raw_sse,
                    "hodge_sse": hodge_sse,
                    "removed_cycle_energy": removed_sse,
                    "pythagorean_relative_discrepancy": discrepancy,
                    "passed": bool(discrepancy <= tolerance),
                }
            )

    raw_metrics = pd.DataFrame.from_records(rows)
    task_metrics = raw_metrics.groupby(["dataset", "method"], as_index=False)[
        [*LOWER_METRICS, "pearson_r"]
    ].mean()
    iterations = int(config["evaluation"]["bootstrap_iterations"])
    statistics = comparison_statistics(
        task_metrics, "hodge_cycle_consistent", "published_deepdelta", iterations, 290001
    ) + comparison_statistics(
        task_metrics, "hodge_cycle_consistent", "antisymmetric", iterations, 290101
    )
    statistics_frame = pd.DataFrame.from_records(statistics)
    primary = statistics_frame[
        (statistics_frame["candidate"] == "hodge_cycle_consistent")
        & (statistics_frame["reference"] == "published_deepdelta")
    ].set_index("metric")
    anti_comparison = statistics_frame[
        (statistics_frame["candidate"] == "hodge_cycle_consistent")
        & (statistics_frame["reference"] == "antisymmetric")
    ].set_index("metric")
    decision_config = config["decision"]
    checks = {
        "rmse_all_datasets_improve": bool(
            primary.loc["pair_rmse", "favorable_fraction"]
            >= float(decision_config["minimum_pair_rmse_improved_dataset_fraction"])
        ),
        "rmse_bootstrap_upper_below_zero": bool(
            primary.loc["pair_rmse", "bootstrap95_upper"]
            < float(decision_config["maximum_pair_rmse_bootstrap_upper"])
        ),
        "mae_at_least_8_of_10_improve": bool(
            primary.loc["pair_mae", "favorable_fraction"]
            >= float(decision_config["minimum_pair_mae_improved_dataset_fraction"])
        ),
        "mae_bootstrap_upper_below_zero": bool(
            primary.loc["pair_mae", "bootstrap95_upper"]
            < float(decision_config["maximum_pair_mae_bootstrap_upper"])
        ),
        "high_delta_mae_not_worse": bool(
            primary.loc["high_delta_pair_mae", "relative_change"]
            <= float(decision_config["maximum_high_delta_pair_mae_relative_change"])
        ),
        "cycle_projection_beats_antisymmetrization": bool(
            anti_comparison.loc["pair_rmse", "bootstrap95_upper"]
            < float(decision_config["maximum_hodge_vs_antisymmetric_rmse_bootstrap_upper"])
        ),
        "pythagorean_identity_all_folds": bool(
            pd.DataFrame.from_records(theorem_rows)["passed"].all()
        ),
    }
    decision = {
        "dataset": config["name"],
        "candidate": "hodge_cycle_consistent",
        "baseline": "published_deepdelta",
        "checks": checks,
        "strong_go": bool(all(checks.values())),
        "prediction_hash_preflight_before_benchmark_responses": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    task_metrics.to_csv(args.output_dir / "dataset_metrics.csv", index=False)
    statistics_frame.to_csv(args.output_dir / "statistics.csv", index=False)
    pd.DataFrame.from_records(theorem_rows).to_csv(
        args.output_dir / "pythagorean_audit.csv", index=False
    )
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "statistics": statistics}, indent=2))


if __name__ == "__main__":
    main()
