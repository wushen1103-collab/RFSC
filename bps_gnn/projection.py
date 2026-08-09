"""Closed-form boundary projection on a label-free active-edge graph."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable

import numpy as np
import pandas as pd


def active_components(
    node_ids: Iterable[Hashable],
    edges: pd.DataFrame,
    left_column: str = "mol_id_lo",
    right_column: str = "mol_id_hi",
) -> dict[Hashable, tuple[Hashable, ...]]:
    """Return deterministic connected components, including singleton nodes."""
    nodes = tuple(node_ids)
    if len(set(nodes)) != len(nodes):
        raise ValueError("node_ids must be unique")
    required = {left_column, right_column}
    if not required.issubset(edges.columns):
        raise ValueError(f"edges lack columns {sorted(required - set(edges.columns))}")
    parent = {node: node for node in nodes}

    def find(node: Hashable) -> Hashable:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges[[left_column, right_column]].itertuples(index=False, name=None):
        if left not in parent or right not in parent:
            raise ValueError("active edge contains a node absent from predictions")
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    groups: dict[Hashable, list[Hashable]] = defaultdict(list)
    for node in nodes:
        groups[find(node)].append(node)
    return {root: tuple(members) for root, members in groups.items()}


def active_centered_signal(
    signal: pd.Series,
    edges: pd.DataFrame,
    left_column: str = "mol_id_lo",
    right_column: str = "mol_id_hi",
) -> pd.Series:
    """Apply the active-edge row-space projector without forming a matrix."""
    if signal.index.has_duplicates:
        raise ValueError("signal index must be unique")
    if not np.isfinite(signal.to_numpy(dtype=float)).all():
        raise ValueError("signal must be finite")
    centered = pd.Series(0.0, index=signal.index, dtype=float)
    components = active_components(signal.index, edges, left_column, right_column)
    for members in components.values():
        if len(members) <= 1:
            continue
        values = signal.loc[list(members)].astype(float)
        centered.loc[list(members)] = values - float(values.mean())
    return centered


def boundary_projection(
    signal: pd.Series,
    edges: pd.DataFrame,
    scale: float,
    left_column: str = "mol_id_lo",
    right_column: str = "mol_id_hi",
) -> pd.Series:
    """Return ``signal + (scale - 1) C_A signal`` in matrix-free form.

    ``C_A`` is the orthogonal projector onto the row space of the active-edge
    incidence matrix. Consequently, this is the unique minimum-L2 correction
    whose differences on every active edge are multiplied by ``scale``.
    """
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    centered = active_centered_signal(signal, edges, left_column, right_column)
    return signal.astype(float) + (scale - 1.0) * centered


def select_active_edges(
    signal: pd.Series,
    edges: pd.DataFrame,
    threshold: float,
    left_column: str = "mol_id_lo",
    right_column: str = "mol_id_hi",
) -> pd.DataFrame:
    """Select graph edges by absolute label-free predicted difference."""
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be non-negative and finite")
    required = {left_column, right_column}
    if not required.issubset(edges.columns):
        raise ValueError(f"edges lack columns {sorted(required - set(edges.columns))}")
    unknown = (set(edges[left_column]) | set(edges[right_column])) - set(signal.index)
    if unknown:
        raise ValueError("edge table contains nodes absent from predictions")
    difference = edges[right_column].map(signal) - edges[left_column].map(signal)
    return edges.loc[difference.abs() >= threshold].copy()


def confidence_weighted_boundary_projection(
    signal: pd.Series,
    edges: pd.DataFrame,
    scale: float,
    precision: pd.Series,
    left_column: str = "mol_id_lo",
    right_column: str = "mol_id_hi",
) -> pd.Series:
    """Minimum weighted-L2 boundary projection using positive node precision."""
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    if set(precision.index) != set(signal.index) or precision.index.has_duplicates:
        raise ValueError("precision and signal must have identical unique node IDs")
    precision = precision.loc[signal.index].astype(float)
    if not np.isfinite(precision.to_numpy()).all() or (precision <= 0.0).any():
        raise ValueError("precision must be positive and finite")
    result = signal.astype(float).copy()
    components = active_components(signal.index, edges, left_column, right_column)
    for members in components.values():
        if len(members) <= 1:
            continue
        ids = list(members)
        values = signal.loc[ids].astype(float)
        weights = precision.loc[ids]
        center = float(np.dot(weights.to_numpy(), values.to_numpy()) / weights.sum())
        result.loc[ids] = center + scale * (values - center)
    return result
