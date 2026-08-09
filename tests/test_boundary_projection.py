import numpy as np
import pandas as pd

from bps_gnn.projection import (
    active_centered_signal,
    boundary_projection,
    confidence_weighted_boundary_projection,
    select_active_edges,
)


def _edges(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(pairs, columns=["mol_id_lo", "mol_id_hi"])


def test_projection_exactly_scales_active_differences_and_preserves_means() -> None:
    base = pd.Series([0.5, 2.0, -1.0, 4.0, 8.0], index=list("abcde"))
    edges = _edges([("a", "b"), ("b", "c"), ("d", "e")])
    scale = 1.7
    result = boundary_projection(base, edges, scale)

    for left, right in edges.itertuples(index=False, name=None):
        assert np.isclose(result[left] - result[right], scale * (base[left] - base[right]))
    assert np.isclose(result.loc[list("abc")].mean(), base.loc[list("abc")].mean())
    assert np.isclose(result.loc[list("de")].mean(), base.loc[list("de")].mean())


def test_projection_is_identity_on_inactive_singletons_and_at_scale_one() -> None:
    base = pd.Series([1.0, 4.0, 9.0], index=list("abc"))
    edges = _edges([("a", "b")])
    assert boundary_projection(base, edges, 1.8)["c"] == base["c"]
    pd.testing.assert_series_equal(boundary_projection(base, edges, 1.0), base)


def test_centering_is_an_orthogonal_projector() -> None:
    base = pd.Series([0.0, 2.0, 5.0, -3.0], index=list("abcd"))
    edges = _edges([("a", "b"), ("b", "c")])
    once = active_centered_signal(base, edges)
    twice = active_centered_signal(once, edges)
    pd.testing.assert_series_equal(twice, once)
    assert np.isclose(once.loc[list("abc")].sum(), 0.0)
    assert once["d"] == 0.0


def test_correction_is_minimum_norm_among_feasible_corrections() -> None:
    base = pd.Series([0.0, 2.0, 5.0, -3.0], index=list("abcd"))
    edges = _edges([("a", "b"), ("b", "c")])
    correction = boundary_projection(base, edges, 1.5) - base
    alternative = correction.copy()
    alternative.loc[list("abc")] += 2.0
    alternative.loc["d"] += 4.0
    for left, right in edges.itertuples(index=False, name=None):
        assert np.isclose(alternative[left] - alternative[right], correction[left] - correction[right])
    assert np.linalg.norm(correction) < np.linalg.norm(alternative)


def test_projection_is_permutation_equivariant() -> None:
    base = pd.Series([0.0, 2.0, 5.0, -3.0], index=list("abcd"))
    edges = _edges([("a", "b"), ("b", "c")])
    expected = boundary_projection(base, edges, 1.5)
    order = ["d", "b", "a", "c"]
    actual = boundary_projection(base.loc[order], edges.iloc[::-1], 1.5)
    pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index())


def test_operator_spectral_norm_equals_scale_for_nonempty_active_graph() -> None:
    node_ids = list("abcd")
    edges = _edges([("a", "b"), ("b", "c")])
    columns = []
    for node in node_ids:
        basis = pd.Series(0.0, index=node_ids)
        basis[node] = 1.0
        columns.append(boundary_projection(basis, edges, 1.5).to_numpy())
    matrix = np.column_stack(columns)
    assert np.isclose(np.linalg.svd(matrix, compute_uv=False)[0], 1.5)


def test_projection_rejects_unknown_edge_nodes_and_nonpositive_scale() -> None:
    base = pd.Series([0.0, 1.0], index=["a", "b"])
    with np.testing.assert_raises(ValueError):
        boundary_projection(base, _edges([("a", "z")]), 1.5)
    with np.testing.assert_raises(ValueError):
        boundary_projection(base, _edges([("a", "b")]), 0.0)


def test_active_edge_selection_uses_only_absolute_predicted_difference() -> None:
    base = pd.Series([0.0, 0.4, 1.2], index=list("abc"))
    edges = _edges([("a", "b"), ("b", "c")])
    selected = select_active_edges(base, edges, 0.5)
    assert list(selected.itertuples(index=False, name=None)) == [("b", "c")]


def test_weighted_projection_preserves_active_differences_and_weighted_mean() -> None:
    base = pd.Series([0.0, 2.0, 5.0], index=list("abc"))
    precision = pd.Series([10.0, 2.0, 1.0], index=list("abc"))
    edges = _edges([("a", "b"), ("b", "c")])
    result = confidence_weighted_boundary_projection(base, edges, 1.5, precision)
    for left, right in edges.itertuples(index=False, name=None):
        assert np.isclose(result[left] - result[right], 1.5 * (base[left] - base[right]))
    assert np.isclose(np.dot(result, precision), np.dot(base, precision))


def test_weighted_projection_moves_high_precision_node_less() -> None:
    base = pd.Series([0.0, 2.0], index=["confident", "uncertain"])
    precision = pd.Series([100.0, 1.0], index=base.index)
    result = confidence_weighted_boundary_projection(
        base, _edges([("confident", "uncertain")]), 1.5, precision
    )
    correction = (result - base).abs()
    assert correction["confident"] < correction["uncertain"]


def test_weighted_projection_reduces_to_ordinary_for_equal_precision() -> None:
    base = pd.Series([0.0, 2.0, 5.0, -1.0], index=list("abcd"))
    edges = _edges([("a", "b"), ("b", "c")])
    weighted = confidence_weighted_boundary_projection(
        base, edges, 1.5, pd.Series(1.0, index=base.index)
    )
    pd.testing.assert_series_equal(weighted, boundary_projection(base, edges, 1.5))
