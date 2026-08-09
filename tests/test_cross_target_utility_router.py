import numpy as np

from scripts.audit_cross_target_utility_router import bounded_router_weight


def test_bounded_router_recovers_identity_and_respects_range():
    probability = np.asarray([0.0, 0.5, 1.0])
    weight = bounded_router_weight(probability, 0.25)
    assert np.allclose(weight, [0.25, 0.5, 0.75])


def test_bounded_router_cannot_leave_component_convex_hull():
    probability = np.linspace(0.0, 1.0, 101)
    weight = bounded_router_weight(probability, 0.25)
    assert np.all(weight >= 0.25)
    assert np.all(weight <= 0.75)
