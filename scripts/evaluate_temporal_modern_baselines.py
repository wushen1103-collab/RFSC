#!/usr/bin/env python3
"""Hash-audit and summarize modern baselines on a frozen temporal benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bps_gnn.io import read_frozen_prediction, read_labels_for_ids
from bps_gnn.temporal import temporal_split_ids

try:
    from evaluate_chembl36_temporal import METRICS, metric_row
except ModuleNotFoundError:
    from scripts.evaluate_chembl36_temporal import METRICS, metric_row


def parse_source(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be METHOD=PATH_TEMPLATE")
    method, template = value.split("=", 1)
    if not method or "{target}" not in template or "{seed}" not in template:
        raise argparse.ArgumentTypeError(
            "source requires a method and {target}/{seed} placeholders"
        )
    return method, template


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = (
        raw.groupby(["target", "method"], as_index=False)[METRICS]
        .agg(["mean", "std", "count"])
    )
    target.columns = [
        "target",
        "method",
        *[f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std", "count")],
    ]
    seed_macro = raw.groupby(["method", "seed"], as_index=False)[METRICS].mean()
    macro = seed_macro.groupby("method", as_index=False)[METRICS].agg(["mean", "std", "count"])
    macro.columns = [
        "method",
        *[f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std", "count")],
    ]
    return target, seed_macro, macro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eligible-targets", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mmp-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--prediction-source", action="append", type=parse_source, required=True,
        help="METHOD=PATH_TEMPLATE with {target} and {seed}",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = json.loads(args.eligible_targets.read_text(encoding="utf-8"))
    sources = dict(args.prediction_source)
    if len(sources) != len(args.prediction_source):
        raise RuntimeError("Duplicate prediction method")

    rows: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    for target in targets:
        _, _, test_ids = temporal_split_ids(args.data_root, target)
        expected = set(test_ids)
        truth = read_labels_for_ids(args.data_root, target, test_ids).set_index("mol_id")[
            "y_pactivity"
        ]
        all_edges = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        edges = all_edges[
            all_edges["mol_id_lo"].isin(expected) & all_edges["mol_id_hi"].isin(expected)
        ].copy()
        for seed in args.seeds:
            for method, template in sources.items():
                path = Path(template.format(target=target, seed=seed))
                prediction = read_frozen_prediction(path)
                matches = set(prediction.index) == expected
                coverage.append(
                    {
                        "target": target,
                        "seed": seed,
                        "method": method,
                        "path": str(path),
                        "hash_verified": True,
                        "test_partition_matches": matches,
                        "n_predictions": len(prediction),
                    }
                )
                if not matches:
                    raise RuntimeError(f"{target} seed {seed} {method}: test partition mismatch")
                rows.append(
                    {
                        "target": target,
                        "seed": seed,
                        "method": method,
                        **metric_row(prediction, truth, edges),
                    }
                )

    raw = pd.DataFrame.from_records(rows).sort_values(["target", "seed", "method"])
    coverage_frame = pd.DataFrame.from_records(coverage).sort_values(
        ["target", "seed", "method"]
    )
    expected_rows = len(targets) * len(args.seeds) * len(sources)
    if len(raw) != expected_rows or len(coverage_frame) != expected_rows:
        raise RuntimeError("Temporal modern-baseline matrix is incomplete")
    target, seed_macro, macro = summarize(raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "raw_seed_metrics.csv", index=False)
    target.to_csv(args.output_dir / "target_seed_summary.csv", index=False)
    seed_macro.to_csv(args.output_dir / "macro_by_seed.csv", index=False)
    macro.to_csv(args.output_dir / "macro_mean_std.csv", index=False)
    coverage_frame.to_csv(args.output_dir / "coverage_hash_audit.csv", index=False)
    provenance = {
        method: {
            "result_source": "Ours"
            if method in {"bps", "unscaled_reference", "confidence_weighted_bps"}
            else "Re-run",
            "prediction_template": template,
            "split": "identical frozen temporal split",
            "seeds": args.seeds,
        }
        for method, template in sources.items()
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
