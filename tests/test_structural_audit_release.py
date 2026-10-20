"""Small implementation checks; these are not additional scientific experiments."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from bps_gnn.projection import boundary_projection, select_active_edges
from scripts.run_structural_audits import fair_operators
from scripts.evaluate_audit_matched_controls import laplacian_unsharp, naive_active_edge_push, stable_rng

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_structural_archive():
    result = subprocess.run([sys.executable, str(ROOT/'scripts/audit_structural_release.py')],
                            cwd=ROOT, check=True, capture_output=True, text=True)
    assert '"privacy_scan": "PASS"' in result.stdout


def test_vectorized_postprocessors_match_reference_implementations():
    base = pd.Series([-.2, .4, 1.4, -1., .6], index=list('abcde'))
    edges = pd.DataFrame([('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('a', 'e')],
                         columns=['mol_id_lo', 'mol_id_hi'])
    left = base.index.get_indexer(edges.mol_id_lo)
    right = base.index.get_indexer(edges.mol_id_hi)
    for family, _, scale, threshold, actual, _ in fair_operators(
            base.to_numpy(), left, right, 'toy', 0, [1.1, 1.5], [.25, .5, .75]):
        active = select_active_edges(base, edges, threshold)
        if family == 'identity':
            expected = base
        elif family == 'BPS':
            expected = boundary_projection(base, active, scale)
        elif family == 'Laplacian_unsharp':
            expected = laplacian_unsharp(base, active, scale-1.)
        elif family == 'naive_active_edge_push':
            expected = naive_active_edge_push(base, active, scale)
        else:
            rng = stable_rng(2917, 'earliest', 'toy', 0, threshold)
            picks = np.sort(rng.choice(len(edges), size=len(active), replace=False))
            expected = boundary_projection(base, edges.iloc[picks], scale)
        np.testing.assert_allclose(actual, expected.to_numpy(), rtol=0., atol=1e-12)
