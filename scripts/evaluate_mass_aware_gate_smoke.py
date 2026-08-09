#!/usr/bin/env python3
"""Evaluate the pre-registered validation-only mass-aware gate smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


METRICS = ["overall_mae", "cliff_mae", "pair_delta_mae"]
REQUIRED_METHODS = {
    "base",
    "learned",
    "learned_mass_aware",
    "random_mass_aware",
    "similarity_mass_aware",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_METHODS - set(frame["method"])
    if missing:
        raise RuntimeError(f"Missing smoke methods: {sorted(missing)}")
    if set(frame["partition"]) != {"validation"}:
        raise RuntimeError("Mass-aware smoke must contain validation metrics only")
    rows = []
    for method, group in frame.groupby("method", sort=True):
        # Random controls are averaged within each seed before paired seed aggregation.
        seed = group.groupby("seed", as_index=False)[METRICS].mean()
        row = {"method": method, "n_seeds": int(seed["seed"].nunique())}
        row.update({metric: float(seed[metric].mean()) for metric in METRICS})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("method", kind="stable").reset_index(drop=True)


def relative_change(candidate: float, reference: float) -> float:
    return candidate / reference - 1.0


def decide(table: pd.DataFrame, smoke: dict) -> dict:
    values = table.set_index("method")
    base = values.loc["base"]
    learned = values.loc["learned"]
    candidate = values.loc["learned_mass_aware"]
    random = values.loc["random_mass_aware"]
    similarity = values.loc["similarity_mass_aware"]

    cliff_change = relative_change(candidate.cliff_mae, learned.cliff_mae)
    pair_change = relative_change(candidate.pair_delta_mae, learned.pair_delta_mae)
    overall_change = relative_change(candidate.overall_mae, learned.overall_mae)
    raw_pair_gap = float(learned.pair_delta_mae - base.pair_delta_mae)
    if raw_pair_gap > 0:
        pair_gap_reduction = float(
            (learned.pair_delta_mae - candidate.pair_delta_mae) / raw_pair_gap
        )
    else:
        pair_gap_reduction = float(candidate.pair_delta_mae <= learned.pair_delta_mae)

    criteria = {
        "cliff_improvement_vs_normalized_soft": bool(
            cliff_change
            <= -float(smoke["minimum_cliff_improvement_vs_normalized_soft"])
        ),
        "pair_improvement_vs_normalized_soft": bool(
            pair_change
            <= -float(smoke["minimum_pair_improvement_vs_normalized_soft"])
        ),
        "overall_not_worse_vs_normalized_soft": bool(
            overall_change
            <= float(smoke["maximum_overall_change_vs_normalized_soft"])
        ),
        "pair_gap_reduction_toward_base": bool(
            pair_gap_reduction
            >= float(smoke["minimum_pair_gap_reduction_toward_base"])
        ),
        "beats_mass_matched_random_on_cliff": bool(
            candidate.cliff_mae < random.cliff_mae
        ),
        "beats_mass_matched_similarity_on_cliff": bool(
            candidate.cliff_mae < similarity.cliff_mae
        ),
    }
    return {
        "relative_changes_percent": {
            "mass_aware_vs_normalized_overall": 100.0 * overall_change,
            "mass_aware_vs_normalized_cliff": 100.0 * cliff_change,
            "mass_aware_vs_normalized_pair_delta": 100.0 * pair_change,
            "mass_aware_vs_base_overall": 100.0
            * relative_change(candidate.overall_mae, base.overall_mae),
            "mass_aware_vs_base_cliff": 100.0
            * relative_change(candidate.cliff_mae, base.cliff_mae),
            "mass_aware_vs_base_pair_delta": 100.0
            * relative_change(candidate.pair_delta_mae, base.pair_delta_mae),
            "mass_aware_vs_random_cliff": 100.0
            * relative_change(candidate.cliff_mae, random.cliff_mae),
            "mass_aware_vs_similarity_cliff": 100.0
            * relative_change(candidate.cliff_mae, similarity.cliff_mae),
        },
        "pair_gap_reduction_toward_base": pair_gap_reduction,
        "criteria": criteria,
        "advance_to_unseen_replication": bool(all(criteria.values())),
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.evaluation)
    smoke = config["smoke"]
    if set(frame["target"]) != {smoke["target"]} or set(frame["fold"]) != {
        int(smoke["fold"])
    }:
        raise RuntimeError("Evaluation task does not match the pre-registered smoke")
    table = summarize(frame)
    decision = decide(table, smoke)
    decision.update(
        {
            "method": config["name"],
            "target": smoke["target"],
            "fold": int(smoke["fold"]),
            "evaluated_partition": "validation",
            "outer_test_labels_materialized": False,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "mass_aware_gate_smoke_table.csv", index=False)
    (args.output_dir / "mass_aware_gate_smoke_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    lines = [
        "# Mass-aware learned-gate validation smoke",
        "",
        f"Decision: **{'GO' if decision['advance_to_unseen_replication'] else 'NO-GO'}**.",
        "",
        "```csv",
        table.to_csv(index=False).strip(),
        "```",
        "",
        "## Pre-registered checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in decision["criteria"].items()
    )
    lines.extend(
        [
            "",
            "Outer-test labels and metrics were not materialized for this smoke.",
            "",
        ]
    )
    (args.output_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
