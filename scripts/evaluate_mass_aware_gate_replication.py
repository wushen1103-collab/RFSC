#!/usr/bin/env python3
"""Aggregate the pre-registered two-point mass-aware validation replication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_mass_aware_gate_smoke import decide, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    replication = config["replication"]
    task_rows, method_tables, task_decisions = [], [], []
    criteria_config = {
        key: replication[key]
        for key in [
            "minimum_cliff_improvement_vs_normalized_soft",
            "minimum_pair_improvement_vs_normalized_soft",
            "maximum_overall_change_vs_normalized_soft",
            "minimum_pair_gap_reduction_toward_base",
        ]
    }
    for task in replication["tasks"]:
        target, fold = task["target"], int(task["fold"])
        path = args.root / target / f"fold_{fold}" / "validation_evaluation.csv"
        frame = pd.read_csv(path)
        if set(frame["target"]) != {target} or set(frame["fold"]) != {fold}:
            raise RuntimeError(f"Task mismatch in {path}")
        table = summarize(frame)
        table.insert(0, "fold", fold)
        table.insert(0, "target", target)
        method_tables.append(table)
        decision = decide(table, criteria_config)
        passed = bool(decision["advance_to_unseen_replication"])
        task_decisions.append(
            {
                "target": target,
                "fold": fold,
                "role": task["role"],
                "criteria": decision["criteria"],
                "relative_changes_percent": decision["relative_changes_percent"],
                "pair_gap_reduction_toward_base": decision[
                    "pair_gap_reduction_toward_base"
                ],
                "passed": passed,
            }
        )
        task_rows.append(
            {
                "target": target,
                "fold": fold,
                "role": task["role"],
                **decision["relative_changes_percent"],
                "pair_gap_reduction_toward_base": decision[
                    "pair_gap_reduction_toward_base"
                ],
                "passed": passed,
            }
        )
    required = int(replication["required_task_passes"])
    passed_count = sum(row["passed"] for row in task_decisions)
    result = {
        "method": config["name"],
        "evaluated_partition": "validation",
        "outer_test_labels_materialized": False,
        "task_decisions": task_decisions,
        "passed_tasks": int(passed_count),
        "required_task_passes": required,
        "advance_to_validation_panel": bool(passed_count == required),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(method_tables, ignore_index=True).to_csv(
        args.output_dir / "mass_aware_replication_method_table.csv", index=False
    )
    pd.DataFrame(task_rows).to_csv(
        args.output_dir / "mass_aware_replication_task_table.csv", index=False
    )
    (args.output_dir / "mass_aware_replication_decision.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    lines = [
        "# Mass-aware learned-gate two-point replication",
        "",
        f"Decision: **{'GO' if result['advance_to_validation_panel'] else 'NO-GO'}** "
        f"({passed_count}/{required} tasks passed).",
        "",
        "```csv",
        pd.DataFrame(task_rows).to_csv(index=False).strip(),
        "```",
        "",
        "Outer-test labels and metrics were not materialized.",
        "",
    ]
    (args.output_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
