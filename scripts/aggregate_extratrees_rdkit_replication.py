#!/usr/bin/env python3
"""Aggregate fixed per-task descriptor-ExtraTrees replication gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", required=True, help="TARGET:FOLD:SEED")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    return parser.parse_args()


def aggregate_decisions(decisions: list[dict]) -> dict:
    pass_count = sum(decision["status"] == "GO" for decision in decisions)
    all_criteria = [
        value
        for decision in decisions
        for value in decision["criteria"].values()
    ]
    passed = pass_count == len(decisions) and all(all_criteria)
    return {
        "status": "GO" if passed else "NO-GO",
        "replication_tasks_passed": pass_count,
        "replication_tasks_total": len(decisions),
        "criteria_passed": int(sum(all_criteria)),
        "criteria_total": len(all_criteria),
        "action": (
            "authorize_panel_preregistration"
            if passed
            else "stop_without_test_evaluation_or_parameter_search"
        ),
        "test_labels_materialized": False,
    }


def main() -> None:
    args = parse_args()
    decisions, tables = [], []
    for task_text in args.task:
        target, fold_text, seed_text = task_text.split(":")
        fold, seed = int(fold_text), int(seed_text)
        stem = f"{target}_fold{fold}_seed{seed}"
        decision = json.loads((args.result_root / f"{stem}_decision.json").read_text(encoding="utf-8"))
        if decision["test_labels_materialized"]:
            raise RuntimeError(f"Test-label firewall failed for {stem}")
        decision["target"] = target
        decision["fold"] = fold
        decision["seed"] = seed
        decisions.append(decision)
        table = pd.read_csv(args.result_root / f"{stem}_metrics.csv")
        table.insert(0, "seed", seed)
        table.insert(0, "fold", fold)
        table.insert(0, "target", target)
        tables.append(table)
    aggregate = aggregate_decisions(decisions)
    aggregate["tasks"] = decisions
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    pd.concat(tables, ignore_index=True).to_csv(args.output_table, index=False)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
