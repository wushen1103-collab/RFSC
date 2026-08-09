import numpy as np

from bps_gnn.dataset_graph import (
    apply_dual_propagation,
    apply_mass_aware_soft_propagation,
    apply_soft_propagation,
    similarity_matched_gate,
    symmetric_edge_features,
)


def test_symmetric_edge_features_are_swap_invariant():
    left = np.array([[1.0, 2.0]])
    right = np.array([[3.0, 4.0]])
    first = symmetric_edge_features(left, right, np.array([1.0]), np.array([2.0]), np.array([0.7]))
    second = symmetric_edge_features(right, left, np.array([2.0]), np.array([1.0]), np.array([0.7]))
    assert np.array_equal(first, second)


def test_soft_gate_all_boundary_falls_back_to_base():
    result = apply_soft_propagation(
        np.array([1.0]),
        np.array([3.0, 5.0]),
        [np.array([0, 1])],
        [np.array([0.7, 0.6])],
        [np.ones(2)],
        0.5,
    )
    assert np.array_equal(result, [1.0])


def test_mass_aware_zero_gate_equals_global_soft_propagation():
    arguments = (
        np.array([1.0]),
        np.array([3.0, 5.0]),
        [np.array([0, 1])],
        [np.array([0.75, 0.25])],
        [np.zeros(2)],
    )
    expected = apply_soft_propagation(*arguments, 0.5)
    observed = apply_mass_aware_soft_propagation(*arguments, 0.5)
    assert np.allclose(observed, expected)


def test_mass_aware_uniform_gate_scales_global_residual():
    arguments = (
        np.array([1.0]),
        np.array([3.0, 5.0]),
        [np.array([0, 1])],
        [np.array([0.75, 0.25])],
    )
    global_prediction = apply_mass_aware_soft_propagation(
        *arguments, [np.zeros(2)], 0.5
    )
    observed = apply_mass_aware_soft_propagation(
        *arguments, [np.full(2, 0.6)], 0.5
    )
    expected = arguments[0] + 0.4 * (global_prediction - arguments[0])
    assert np.allclose(observed, expected)


def test_mass_aware_all_boundary_returns_base():
    result = apply_mass_aware_soft_propagation(
        np.array([1.0]),
        np.array([3.0, 5.0]),
        [np.array([0, 1])],
        [np.array([0.7, 0.6])],
        [np.ones(2)],
        0.5,
    )
    assert np.array_equal(result, [1.0])


def test_dual_gate_restores_signed_boundary_jump():
    result = apply_dual_propagation(
        np.array([3.0]),
        np.array([1.0]),
        [np.array([0])],
        [np.array([0.8])],
        [np.ones(1)],
        0.5,
        0.25,
    )
    assert np.allclose(result, [3.5])


def test_dual_gate_without_boundary_mass_equals_soft_propagation():
    arguments = (
        np.array([1.0]),
        np.array([3.0, 5.0]),
        [np.array([0, 1])],
        [np.array([0.75, 0.25])],
        [np.zeros(2)],
    )
    expected = apply_soft_propagation(*arguments, 0.5)
    observed = apply_dual_propagation(*arguments, 0.5, 0.25)
    assert np.allclose(observed, expected)


def test_similarity_control_preserves_q_multiset_and_orders_it():
    learned = [np.array([0.2, 0.9, 0.5])]
    similarity = [np.array([0.8, 0.4, 0.6])]
    matched = similarity_matched_gate(learned, similarity)[0]
    assert np.array_equal(np.sort(matched), np.sort(learned[0]))
    assert matched[1] == 0.9  # Lowest-similarity edge receives the largest q.
