import numpy as np
import pytest

from scripts.build_deepdelta_sparse_projection import (
    is_connected,
    sample_connected_undirected_mask,
)
from scripts.evaluate_deepdelta_sparse_projection import (
    bootstrap_relative,
    masked_metrics,
)


def test_sparse_sampler_is_connected_deterministic_and_density_bounded() -> None:
    n_nodes = 11
    mask = sample_connected_undirected_mask(n_nodes, 0.10, 2901, 3)
    repeated = sample_connected_undirected_mask(n_nodes, 0.10, 2901, 3)
    other_dataset = sample_connected_undirected_mask(n_nodes, 0.10, 2901, 4)
    upper_i, upper_j = np.triu_indices(n_nodes, k=1)
    assert np.array_equal(mask, repeated)
    assert not np.array_equal(mask, other_dataset)
    assert int(mask.sum()) == n_nodes - 1
    assert is_connected(n_nodes, upper_i[mask], upper_j[mask])


def test_sparse_sampler_fills_above_spanning_tree() -> None:
    n_nodes = 10
    mask = sample_connected_undirected_mask(n_nodes, 0.50, 2905, 0)
    assert int(mask.sum()) == 23


def test_masked_metrics_and_bootstrap_relative() -> None:
    truth = np.asarray([[0.0, 2.0], [-2.0, 0.0]])
    prediction = np.asarray([[9.0, 1.0], [-3.0, 9.0]])
    mask = np.asarray([[False, True], [True, False]])
    metrics = masked_metrics(prediction, truth, mask)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(1.0)
    lower, upper = bootstrap_relative(
        np.asarray([0.8, 1.6]), np.asarray([1.0, 2.0]), 1000, 29
    )
    assert lower == pytest.approx(-0.2)
    assert upper == pytest.approx(-0.2)


@pytest.mark.parametrize("density", [0.0, -0.1, 1.1])
def test_sparse_sampler_rejects_invalid_density(density: float) -> None:
    with pytest.raises(ValueError):
        sample_connected_undirected_mask(5, density, 1, 0)

