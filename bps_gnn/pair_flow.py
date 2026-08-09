"""Orthogonal projections for directed molecular pair predictions."""

from __future__ import annotations

import numpy as np


def antisymmetric_pair_flow(flow: np.ndarray) -> np.ndarray:
    """Return the Frobenius projection onto antisymmetric matrices."""
    values = np.asarray(flow, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("pair flow must be a square matrix")
    if not np.isfinite(values).all():
        raise ValueError("pair flow must be finite")
    return 0.5 * (values - values.T)


def hodge_project_complete_pair_flow(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project a complete ordered pair-flow onto exact node-potential differences.

    ``flow[i, j]`` is interpreted as a prediction of ``x[j] - x[i]``. The
    returned potential is centered at zero and the returned flow is its exact
    ordered difference matrix.
    """
    values = np.asarray(flow, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("pair flow must be a square matrix")
    if not np.isfinite(values).all():
        raise ValueError("pair flow must be finite")
    n_nodes = values.shape[0]
    if n_nodes == 0:
        raise ValueError("pair flow must contain at least one node")
    potential = (values.sum(axis=0) - values.sum(axis=1)) / (2.0 * n_nodes)
    potential -= float(potential.mean())
    gradient = potential[None, :] - potential[:, None]
    return potential, gradient


def hodge_project_observed_pair_flow(
    n_nodes: int,
    source: np.ndarray,
    target: np.ndarray,
    flow: np.ndarray,
    weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares integrate a directed flow observed on an arbitrary graph.

    Each row represents ``flow[e] ~= potential[target[e]] -
    potential[source[e]]``. The minimum-norm solution centers the potential
    independently within every connected component. The returned fitted vector
    follows the input edge order.
    """
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    source_values = np.asarray(source, dtype=int)
    target_values = np.asarray(target, dtype=int)
    flow_values = np.asarray(flow, dtype=float)
    if source_values.ndim != 1 or target_values.ndim != 1 or flow_values.ndim != 1:
        raise ValueError("source, target, and flow must be one-dimensional")
    if not (len(source_values) == len(target_values) == len(flow_values)):
        raise ValueError("source, target, and flow lengths must match")
    if len(flow_values) == 0:
        raise ValueError("at least one observed edge is required")
    if (
        (source_values < 0).any()
        or (target_values < 0).any()
        or (source_values >= n_nodes).any()
        or (target_values >= n_nodes).any()
    ):
        raise ValueError("edge endpoint is outside the node range")
    if not np.isfinite(flow_values).all():
        raise ValueError("observed flow must be finite")
    weights = (
        np.ones(len(flow_values), dtype=float)
        if weight is None
        else np.asarray(weight, dtype=float)
    )
    if weights.ndim != 1 or len(weights) != len(flow_values):
        raise ValueError("weight must be one-dimensional and edge-aligned")
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise ValueError("weights must be finite and strictly positive")

    incidence = np.zeros((len(flow_values), n_nodes), dtype=float)
    row = np.arange(len(flow_values))
    incidence[row, source_values] -= 1.0
    incidence[row, target_values] += 1.0
    square_root_weight = np.sqrt(weights)
    potential, _, _, _ = np.linalg.lstsq(
        incidence * square_root_weight[:, None],
        flow_values * square_root_weight,
        rcond=None,
    )
    fitted = potential[target_values] - potential[source_values]
    return potential, fitted
