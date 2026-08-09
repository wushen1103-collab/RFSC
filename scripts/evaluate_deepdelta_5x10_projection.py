#!/usr/bin/env python3
"""Evaluate hash-frozen Hodge repair on official DeepDelta 5x10 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_frozen_prediction
from bps_gnn.pair_flow import antisymmetric_pair_flow
try:
    from build_deepdelta_5x10_predictions import (
        expected_ids,
        infer_kfold_sizes,
        load_prediction_row_only,
        split_square_blocks,
    )
    from evaluate_deepdelta_cycle_projection import flow_metrics, load_raw_flow
except ModuleNotFoundError:
    from scripts.build_deepdelta_5x10_predictions import (
        expected_ids,
        infer_kfold_sizes,
        load_prediction_row_only,
        split_square_blocks,
    )
    from scripts.evaluate_deepdelta_cycle_projection import flow_metrics, load_raw_flow

try:
    from evaluate_chembl36_temporal import bootstrap_relative
except ModuleNotFoundError:
    from scripts.evaluate_chembl36_temporal import bootstrap_relative


LOWER_METRICS = ["pair_rmse", "pair_mae", "high_delta_pair_mae"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--individual-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def ordered_potentials(series: pd.Series, sizes: list[int]) -> list[np.ndarray]:
    ids = expected_ids(sizes)
    if set(series.index) != set(ids):
        raise RuntimeError("frozen 5x10 position IDs do not match the official fold partition")
    values = series.reindex(ids).to_numpy(dtype=float)
    blocks: list[np.ndarray] = []
    offset = 0
    for size in sizes:
        blocks.append(values[offset : offset + size])
        offset += size
    return blocks


def load_official_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if len(frame) != 2:
        raise RuntimeError(f"{path}: expected one response row and one prediction row")
    truth = frame.iloc[0].to_numpy(dtype=float)
    prediction = frame.iloc[1].to_numpy(dtype=float)
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise RuntimeError(f"{path}: non-finite official value")
    return truth, prediction


def comparison_statistics(
    dataset_metrics: pd.DataFrame,
    scope: str,
    candidate_method: str,
    reference_method: str,
    iterations: int,
    seed: int,
) -> list[dict[str, object]]:
    candidate = dataset_metrics[dataset_metrics["method"] == candidate_method].set_index(
        "dataset"
    )
    reference = dataset_metrics[dataset_metrics["method"] == reference_method].set_index(
        "dataset"
    )
    records: list[dict[str, object]] = []
    for metric_index, metric in enumerate(LOWER_METRICS):
        paired = pd.concat(
            [candidate[metric].rename("candidate"), reference[metric].rename("reference")],
            axis=1,
        ).dropna()
        lower, upper = bootstrap_relative(
            paired["candidate"].to_numpy(),
            paired["reference"].to_numpy(),
            iterations,
            seed + metric_index,
        )
        records.append(
            {
                "scope": scope,
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
    repeats = [int(value) for value in config["cross_validation"]["repeats"]]
    n_folds = int(config["cross_validation"]["folds"])
    prior_repeat = int(config["cross_validation"]["previously_evaluated_repeat"])
    decision_config = config["decision"]

    # Candidate/hash/partition preflight. No aggregate response row is parsed here.
    preflight: dict[tuple[str, int], tuple[list[int], list[np.ndarray]]] = {}
    for dataset in datasets:
        for repeat in repeats:
            source = args.source_root / f"{dataset}_DeepDelta5_{repeat}.csv"
            flat, row_hash = load_prediction_row_only(source)
            run_dir = args.prediction_root / dataset / f"repeat_{repeat}"
            metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            if row_hash != metadata["source_prediction_row_sha256"]:
                raise RuntimeError(f"{source}: prediction-row hash changed")
            sizes = infer_kfold_sizes(len(flat), n_folds)
            if sizes != [int(value) for value in metadata["fold_sizes"]]:
                raise RuntimeError(f"{source}: fold partition changed")
            potentials = ordered_potentials(
                read_frozen_prediction(run_dir / "potential.parquet"), sizes
            )
            preflight[(dataset, repeat)] = (sizes, potentials)

    # Response-free audit that official repeat 1 is the previously used release.
    alignment_rows: list[dict[str, object]] = []
    alignment_tolerance = float(decision_config["repeat_one_alignment_absolute_tolerance"])
    for dataset in datasets:
        aggregate_prediction, _ = load_prediction_row_only(
            args.source_root / f"{dataset}_DeepDelta5_{prior_repeat}.csv"
        )
        sizes, _ = preflight[(dataset, prior_repeat)]
        aggregate_blocks = split_square_blocks(aggregate_prediction, sizes)
        for fold, aggregate in enumerate(aggregate_blocks, start=1):
            individual = load_raw_flow(
                args.individual_root / f"{dataset}_DeepDelta_Individual_Fold_{fold}.csv"
            )
            same_shape = aggregate.shape == individual.shape
            discrepancy = (
                float(np.max(np.abs(aggregate - individual))) if same_shape else float("inf")
            )
            alignment_rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "same_shape": same_shape,
                    "maximum_absolute_discrepancy": discrepancy,
                    "passed": bool(same_shape and discrepancy <= alignment_tolerance),
                }
            )

    rows: list[dict[str, object]] = []
    theorem_rows: list[dict[str, object]] = []
    quantile = float(config["evaluation"]["high_delta_quantile_within_fold"])
    theorem_tolerance = float(decision_config["pythagorean_relative_tolerance"])

    # Only now may official aggregate response rows be parsed.
    for dataset in datasets:
        for repeat in repeats:
            source = args.source_root / f"{dataset}_DeepDelta5_{repeat}.csv"
            truth_flat, raw_flat = load_official_rows(source)
            sizes, potentials = preflight[(dataset, repeat)]
            truth_blocks = split_square_blocks(truth_flat, sizes)
            raw_blocks = split_square_blocks(raw_flat, sizes)
            for fold, (truth, raw, potential) in enumerate(
                zip(truth_blocks, raw_blocks, potentials, strict=True), start=1
            ):
                hodge = potential[None, :] - potential[:, None]
                anti = antisymmetric_pair_flow(raw)
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
                            "repeat": repeat,
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
                        "repeat": repeat,
                        "fold": fold,
                        "raw_sse": raw_sse,
                        "hodge_sse": hodge_sse,
                        "removed_energy": removed_sse,
                        "pythagorean_relative_discrepancy": discrepancy,
                        "passed": bool(discrepancy <= theorem_tolerance),
                    }
                )

    fold_metrics = pd.DataFrame.from_records(rows)
    repeat_metrics = fold_metrics.groupby(["dataset", "repeat", "method"], as_index=False)[
        [*LOWER_METRICS, "pearson_r"]
    ].mean()
    all_dataset_metrics = repeat_metrics.groupby(["dataset", "method"], as_index=False)[
        [*LOWER_METRICS, "pearson_r"]
    ].mean()
    new_repeat_metrics = repeat_metrics[repeat_metrics["repeat"] != prior_repeat]
    new_dataset_metrics = new_repeat_metrics.groupby(["dataset", "method"], as_index=False)[
        [*LOWER_METRICS, "pearson_r"]
    ].mean()

    iterations = int(config["evaluation"]["bootstrap_iterations"])
    statistics = (
        comparison_statistics(
            all_dataset_metrics,
            "all_five_repeats",
            "hodge_cycle_consistent",
            "published_deepdelta",
            iterations,
            295001,
        )
        + comparison_statistics(
            new_dataset_metrics,
            "four_new_repeats",
            "hodge_cycle_consistent",
            "published_deepdelta",
            iterations,
            295101,
        )
        + comparison_statistics(
            all_dataset_metrics,
            "all_five_repeats",
            "hodge_cycle_consistent",
            "antisymmetric",
            iterations,
            295201,
        )
    )
    statistics_frame = pd.DataFrame.from_records(statistics)

    repeat_pivot = repeat_metrics.pivot(
        index=["dataset", "repeat"], columns="method", values="pair_rmse"
    )
    favorable_repeat_fraction = float(
        (
            repeat_pivot["hodge_cycle_consistent"]
            < repeat_pivot["published_deepdelta"]
        ).mean()
    )

    def stat(scope: str, reference: str, metric: str) -> pd.Series:
        selected = statistics_frame[
            (statistics_frame["scope"] == scope)
            & (statistics_frame["candidate"] == "hodge_cycle_consistent")
            & (statistics_frame["reference"] == reference)
            & (statistics_frame["metric"] == metric)
        ]
        if len(selected) != 1:
            raise RuntimeError(f"missing statistic {scope}/{reference}/{metric}")
        return selected.iloc[0]

    all_rmse = stat("all_five_repeats", "published_deepdelta", "pair_rmse")
    new_rmse = stat("four_new_repeats", "published_deepdelta", "pair_rmse")
    all_mae = stat("all_five_repeats", "published_deepdelta", "pair_mae")
    all_high = stat("all_five_repeats", "published_deepdelta", "high_delta_pair_mae")
    anti_rmse = stat("all_five_repeats", "antisymmetric", "pair_rmse")
    theorem_frame = pd.DataFrame.from_records(theorem_rows)
    alignment_frame = pd.DataFrame.from_records(alignment_rows)
    checks = {
        "all_repeat_rmse_all_datasets_improve": bool(
            all_rmse["favorable_fraction"]
            >= float(
                decision_config["minimum_pair_rmse_improved_dataset_fraction_all_repeats"]
            )
        ),
        "all_repeat_rmse_bootstrap_upper_below_zero": bool(
            all_rmse["bootstrap95_upper"]
            < float(decision_config["maximum_pair_rmse_bootstrap_upper_all_repeats"])
        ),
        "new_repeat_rmse_all_datasets_improve": bool(
            new_rmse["favorable_fraction"]
            >= float(
                decision_config["minimum_pair_rmse_improved_dataset_fraction_new_repeats"]
            )
        ),
        "new_repeat_rmse_bootstrap_upper_below_zero": bool(
            new_rmse["bootstrap95_upper"]
            < float(decision_config["maximum_pair_rmse_bootstrap_upper_new_repeats"])
        ),
        "rmse_at_least_48_of_50_repeats_improve": bool(
            favorable_repeat_fraction
            >= float(decision_config["minimum_pair_rmse_improved_repeat_fraction"])
        ),
        "all_repeat_mae_at_least_8_of_10_improve": bool(
            all_mae["favorable_fraction"]
            >= float(
                decision_config["minimum_pair_mae_improved_dataset_fraction_all_repeats"]
            )
        ),
        "all_repeat_mae_bootstrap_upper_below_zero": bool(
            all_mae["bootstrap95_upper"]
            < float(decision_config["maximum_pair_mae_bootstrap_upper_all_repeats"])
        ),
        "all_repeat_high_delta_mae_not_worse": bool(
            all_high["relative_change"]
            <= float(
                decision_config["maximum_high_delta_pair_mae_relative_change_all_repeats"]
            )
        ),
        "hodge_beats_antisymmetrization": bool(
            anti_rmse["bootstrap95_upper"]
            < float(decision_config["maximum_hodge_vs_antisymmetric_rmse_bootstrap_upper"])
        ),
        "pythagorean_identity_all_500_folds": bool(theorem_frame["passed"].all()),
        "repeat_one_matches_individual_release": bool(alignment_frame["passed"].all()),
    }
    decision = {
        "dataset": config["name"],
        "candidate": "hodge_cycle_consistent",
        "baseline": "published_deepdelta",
        "checks": checks,
        "strong_robustness_go": bool(all(checks.values())),
        "favorable_dataset_repeats": int(
            (
                repeat_pivot["hodge_cycle_consistent"]
                < repeat_pivot["published_deepdelta"]
            ).sum()
        ),
        "total_dataset_repeats": int(len(repeat_pivot)),
        "prediction_hash_preflight_before_response_rows": True,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    repeat_metrics.to_csv(args.output_dir / "repeat_metrics.csv", index=False)
    all_dataset_metrics.assign(scope="all_five_repeats").to_csv(
        args.output_dir / "dataset_metrics.csv", index=False
    )
    new_dataset_metrics.assign(scope="four_new_repeats").to_csv(
        args.output_dir / "new_repeat_dataset_metrics.csv", index=False
    )
    statistics_frame.to_csv(args.output_dir / "statistics.csv", index=False)
    theorem_frame.to_csv(args.output_dir / "pythagorean_audit.csv", index=False)
    alignment_frame.to_csv(args.output_dir / "repeat_one_alignment_audit.csv", index=False)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "statistics": statistics}, indent=2))


if __name__ == "__main__":
    main()
