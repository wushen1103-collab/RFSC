import numpy as np

from bps_gnn.pair_flow import hodge_project_complete_pair_flow
from scripts.analyze_deepdelta_projection_mechanism import (
    cycle_residual_rms,
    gradient,
    one_sided_potentials,
)


def test_one_sided_average_equals_hodge_potential() -> None:
    rng = np.random.default_rng(2902)
    flow = rng.normal(size=(13, 13))
    incoming, outgoing = one_sided_potentials(flow)
    hodge, _ = hodge_project_complete_pair_flow(flow)
    np.testing.assert_allclose(0.5 * (incoming + outgoing), hodge, atol=1e-12)


def test_cycle_residual_is_zero_for_gradient_flow() -> None:
    flow = gradient(np.asarray([-0.8, 0.2, 0.5, 1.1]))
    assert cycle_residual_rms(flow) < 1e-15
