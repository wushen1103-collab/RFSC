import numpy as np
import pytest

from bps_gnn.pair_flow import (
    antisymmetric_pair_flow,
    hodge_project_complete_pair_flow,
    hodge_project_observed_pair_flow,
)


def gradient(values: np.ndarray) -> np.ndarray:
    return values[None, :] - values[:, None]


def test_exact_gradient_is_fixed() -> None:
    potential = np.asarray([-1.0, 0.25, 0.75, 2.0])
    centered, projected = hodge_project_complete_pair_flow(gradient(potential))
    np.testing.assert_allclose(centered, potential - potential.mean(), atol=1e-12)
    np.testing.assert_allclose(projected, gradient(potential), atol=1e-12)


def test_pythagorean_error_identity_against_true_gradient() -> None:
    truth = gradient(np.asarray([-0.7, 0.1, 0.4, 1.3]))
    perturbation = np.asarray(
        [
            [0.4, -0.1, 0.3, 0.2],
            [0.5, -0.2, 0.7, -0.4],
            [-0.6, 0.9, 0.1, 0.8],
            [0.2, -0.3, 0.4, 0.5],
        ]
    )
    observed = truth + perturbation
    _, projected = hodge_project_complete_pair_flow(observed)
    raw_error = np.square(observed - truth).sum()
    projected_error = np.square(projected - truth).sum()
    removed_energy = np.square(observed - projected).sum()
    assert projected_error < raw_error
    assert raw_error == pytest.approx(projected_error + removed_energy, abs=1e-12)


def test_antisymmetric_ablation_and_permutation_equivariance() -> None:
    rng = np.random.default_rng(29)
    flow = rng.normal(size=(7, 7))
    anti = antisymmetric_pair_flow(flow)
    np.testing.assert_allclose(anti, -anti.T, atol=1e-12)
    permutation = np.asarray([3, 0, 6, 1, 5, 2, 4])
    potential, projected = hodge_project_complete_pair_flow(flow)
    permuted_potential, permuted_projected = hodge_project_complete_pair_flow(
        flow[np.ix_(permutation, permutation)]
    )
    np.testing.assert_allclose(permuted_potential, potential[permutation], atol=1e-12)
    np.testing.assert_allclose(
        permuted_projected, projected[np.ix_(permutation, permutation)], atol=1e-12
    )


@pytest.mark.parametrize("bad", [np.ones(3), np.ones((2, 3)), np.asarray([[np.nan]])])
def test_invalid_pair_flow_rejected(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        hodge_project_complete_pair_flow(bad)


def test_observed_graph_recovers_exact_gradient_and_centers_components() -> None:
    truth = np.asarray([-2.0, 1.0, 3.0, 8.0, 10.0])
    source = np.asarray([0, 1, 1, 3, 4])
    target = np.asarray([1, 0, 2, 4, 3])
    observed = truth[target] - truth[source]
    potential, fitted = hodge_project_observed_pair_flow(
        len(truth), source, target, observed
    )
    np.testing.assert_allclose(fitted, observed, atol=1e-12)
    np.testing.assert_allclose(potential[:3].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(potential[3:].mean(), 0.0, atol=1e-12)


def test_observed_graph_pythagorean_identity_and_weights() -> None:
    source = np.asarray([0, 1, 2, 3, 0, 2])
    target = np.asarray([1, 2, 3, 0, 2, 0])
    truth_potential = np.asarray([-0.7, 0.1, 0.4, 1.3])
    truth = truth_potential[target] - truth_potential[source]
    raw = truth + np.asarray([0.4, -0.1, 0.3, 0.2, 0.5, -0.6])
    weight = np.asarray([1.0, 2.0, 0.5, 1.5, 3.0, 0.8])
    _, projected = hodge_project_observed_pair_flow(
        4, source, target, raw, weight
    )
    raw_error = np.sum(weight * np.square(raw - truth))
    projected_error = np.sum(weight * np.square(projected - truth))
    removed_energy = np.sum(weight * np.square(raw - projected))
    assert raw_error == pytest.approx(projected_error + removed_energy, abs=1e-12)


@pytest.mark.parametrize(
    "args",
    [
        (0, np.asarray([0]), np.asarray([0]), np.asarray([0.0]), None),
        (2, np.asarray([0]), np.asarray([2]), np.asarray([0.0]), None),
        (2, np.asarray([0]), np.asarray([1]), np.asarray([np.nan]), None),
        (2, np.asarray([0]), np.asarray([1]), np.asarray([0.0]), np.asarray([0.0])),
    ],
)
def test_invalid_observed_pair_flow_rejected(args: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        hodge_project_observed_pair_flow(*args)
