from scripts.evaluate_prismnet_pilot import pilot_gate


DECISION = {
    "maximum_overall_relative_change": 0.10,
    "maximum_cliff_relative_change": 0.10,
    "maximum_pair_delta_relative_change": 0.10,
    "overall_must_pass": True,
    "minimum_passed_endpoints": 2,
}


def test_pilot_gate_requires_overall_and_two_endpoints() -> None:
    criteria, advance = pilot_gate(
        {"overall": 0.05, "cliff": 0.09, "pair_delta": 0.20}, DECISION
    )
    assert criteria == {"overall": True, "cliff": True, "pair_delta": False}
    assert advance


def test_pilot_gate_rejects_overall_failure() -> None:
    _, advance = pilot_gate(
        {"overall": 0.11, "cliff": -0.10, "pair_delta": -0.10}, DECISION
    )
    assert not advance

