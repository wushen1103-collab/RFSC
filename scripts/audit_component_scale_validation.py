#!/usr/bin/env python3
"""Aggregate component-scale evidence using validation records only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["overall_mae", "cliff_mae", "pair_delta_mae", "selection_objective"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records: list[pd.DataFrame] = []
    for path in sorted(args.input_dir.glob("*/fold_*/seed_*/scale_selection.csv")):
        target, fold, seed = path.parts[-4], path.parts[-3], path.parts[-2]
        frame = pd.read_csv(path)
        frame.insert(0, "target", target)
        frame.insert(1, "fold", int(fold.removeprefix("fold_")))
        frame.insert(2, "seed", int(seed.removeprefix("seed_")))
        identity = frame.loc[np.isclose(frame.scale, 1.0)].iloc[0]
        for metric in METRICS:
            frame[f"relative_{metric}_pct"] = 100.0 * (frame[metric] / float(identity[metric]) - 1.0)
        records.append(frame)
    if len(records) != 87:
        raise RuntimeError(f"Expected 87 validation selections, observed {len(records)}")

    all_rows = pd.concat(records, ignore_index=True)
    relative = [f"relative_{metric}_pct" for metric in METRICS]
    target_macro = all_rows.groupby(["target", "scale"], as_index=False)[METRICS + relative].mean()
    summary = target_macro.groupby("scale", as_index=False)[METRICS + relative].mean()
    rates = (
        all_rows.assign(
            pair_improved=all_rows.relative_pair_delta_mae_pct < 0,
            absolute_safe=(all_rows.relative_overall_mae_pct <= 1.0) & (all_rows.relative_cliff_mae_pct <= 1.0),
        )
        .groupby("scale", as_index=False)
        .agg(pair_improved_runs=("pair_improved", "sum"), absolute_safe_runs=("absolute_safe", "sum"), runs=("pair_improved", "size"))
    )
    summary = summary.merge(rates, on="scale", validate="one_to_one")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(args.output_dir / "validation_run_metrics.csv", index=False)
    target_macro.to_csv(args.output_dir / "validation_target_macro.csv", index=False)
    summary.to_csv(args.output_dir / "validation_scale_summary.csv", index=False)
    payload = {
        "label_scope": "outer_validation_only",
        "runs": len(records),
        "targets": int(all_rows.target.nunique()),
        "summary": summary.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
