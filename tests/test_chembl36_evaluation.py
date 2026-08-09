import numpy as np
import pandas as pd

from scripts.evaluate_chembl36_temporal import bootstrap_relative, holm_adjust, metric_row


def test_temporal_metrics_use_only_true_cliff_edges() -> None:
    truth = pd.Series([0.0, 2.0, 2.2], index=list("abc"))
    prediction = pd.Series([0.0, 1.5, 2.1], index=list("abc"))
    edges = pd.DataFrame(
        {"mol_id_lo": ["a", "b"], "mol_id_hi": ["b", "c"]}
    )
    result = metric_row(prediction, truth, edges)
    assert result["n_cliff_pairs"] == 1
    assert result["n_cliff_endpoints"] == 2
    assert np.isclose(result["pair_delta_mae"], 0.5)


def test_bootstrap_and_holm_are_deterministic() -> None:
    candidate = np.array([0.8, 0.9, 1.0])
    reference = np.array([1.0, 1.0, 1.0])
    first = bootstrap_relative(candidate, reference, 100, 7)
    second = bootstrap_relative(candidate, reference, 100, 7)
    assert first == second
    adjusted = holm_adjust([0.01, 0.03, 0.2])
    assert adjusted == [0.03, 0.06, 0.2]
