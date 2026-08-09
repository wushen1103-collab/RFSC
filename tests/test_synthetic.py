from __future__ import annotations

import numpy as np

from scripts.run_synthetic import build_problem, run_case


def test_synthetic_problem_and_smoothers_are_finite() -> None:
    problem = build_problem(seed=0, n=120, k=6, jump=1.0, noise=0.2)
    assert len(problem["edges"]) > 0
    assert 0 < int(problem["true_boundary"].sum()) < len(problem["edges"])

    rows = run_case(
        seed=0,
        cfg={"n_nodes": 120, "k": 6},
        jump=1.0,
        noise=0.2,
        methods=["global", "oracle"],
        lambdas=[0.1],
    )
    assert len(rows) == 2
    assert all(np.isfinite(row["overall_mse"]) for row in rows)
    assert all(np.isfinite(row["boundary_mse"]) for row in rows)
    assert rows[1]["gate_auprc"] == 1.0
