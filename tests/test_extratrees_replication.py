from __future__ import annotations

from scripts.aggregate_extratrees_rdkit_replication import aggregate_decisions
from scripts.train_extratrees_rdkit_scg import task_is_authorized


def test_replication_task_whitelist() -> None:
    evaluation = {
        "tasks": [
            {"target": "a", "fold": 0, "seed": 0},
            {"target": "b", "fold": 1, "seed": 2},
        ]
    }
    assert task_is_authorized(evaluation, "a", 0, 0)
    assert task_is_authorized(evaluation, "b", 1, 2)
    assert not task_is_authorized(evaluation, "b", 0, 2)


def test_replication_requires_every_task_and_criterion() -> None:
    passed = {"status": "GO", "criteria": {"a": True, "b": True}}
    failed = {"status": "NO-GO", "criteria": {"a": True, "b": False}}
    assert aggregate_decisions([passed, passed])["status"] == "GO"
    result = aggregate_decisions([passed, failed])
    assert result["status"] == "NO-GO"
    assert result["replication_tasks_passed"] == 1
