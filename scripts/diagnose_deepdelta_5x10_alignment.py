#!/usr/bin/env python3
"""Post-hoc diagnose repeat-1 split alignment versus stochastic prediction identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold

try:
    from build_deepdelta_5x10_predictions import infer_kfold_sizes, split_square_blocks
    from evaluate_deepdelta_5x10_projection import load_official_rows
    from evaluate_deepdelta_cycle_projection import load_raw_flow
except ModuleNotFoundError:
    from scripts.build_deepdelta_5x10_predictions import infer_kfold_sizes, split_square_blocks
    from scripts.evaluate_deepdelta_5x10_projection import load_official_rows
    from scripts.evaluate_deepdelta_cycle_projection import load_raw_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--individual-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    datasets = [str(value) for value in config["datasets"]]
    repeat = int(config["cross_validation"]["previously_evaluated_repeat"])
    n_folds = int(config["cross_validation"]["folds"])
    rows: list[dict[str, object]] = []

    for dataset in datasets:
        truth_flat, prediction_flat = load_official_rows(
            args.source_root / f"{dataset}_DeepDelta5_{repeat}.csv"
        )
        sizes = infer_kfold_sizes(len(prediction_flat), n_folds)
        official_truth = split_square_blocks(truth_flat, sizes)
        official_prediction = split_square_blocks(prediction_flat, sizes)
        data = pd.read_csv(args.data_root / f"{dataset}.csv")
        response = data["Y"].to_numpy(dtype=float)
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=repeat)
        for fold, ((_, test_index), released_truth, released_prediction) in enumerate(
            zip(splitter.split(data), official_truth, official_prediction, strict=True), start=1
        ):
            expected_truth = response[test_index][None, :] - response[test_index][:, None]
            individual_prediction = load_raw_flow(
                args.individual_root / f"{dataset}_DeepDelta_Individual_Fold_{fold}.csv"
            )
            truth_discrepancy = float(np.max(np.abs(released_truth - expected_truth)))
            prediction_discrepancy = float(
                np.max(np.abs(released_prediction - individual_prediction))
            )
            prediction_rmse = float(
                np.sqrt(np.mean(np.square(released_prediction - individual_prediction)))
            )
            prediction_correlation = float(
                np.corrcoef(released_prediction.ravel(), individual_prediction.ravel())[0, 1]
            )
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "n_nodes": len(test_index),
                    "truth_maximum_absolute_discrepancy": truth_discrepancy,
                    "truth_split_aligned": bool(truth_discrepancy <= 1e-12),
                    "prediction_maximum_absolute_discrepancy": prediction_discrepancy,
                    "prediction_between_release_rmse": prediction_rmse,
                    "prediction_between_release_pearson_r": prediction_correlation,
                    "prediction_exactly_aligned": bool(prediction_discrepancy <= 1e-12),
                }
            )

    frame = pd.DataFrame.from_records(rows)
    summary = {
        "status": "post_hoc_provenance_diagnostic",
        "repeat": repeat,
        "folds": int(len(frame)),
        "truth_split_aligned_folds": int(frame["truth_split_aligned"].sum()),
        "prediction_exactly_aligned_folds": int(frame["prediction_exactly_aligned"].sum()),
        "maximum_truth_discrepancy": float(
            frame["truth_maximum_absolute_discrepancy"].max()
        ),
        "mean_prediction_between_release_rmse": float(
            frame["prediction_between_release_rmse"].mean()
        ),
        "mean_prediction_between_release_pearson_r": float(
            frame["prediction_between_release_pearson_r"].mean()
        ),
        "interpretation": (
            "Same random-state-1 folds if all truth rows align; non-identical predictions "
            "then indicate a distinct stochastic model realization rather than a split mismatch."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "fold_alignment.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

