import pandas as pd

from scripts.build_mmp_context_expert_validation import project_predictions


def test_pair_projection_preserves_mean_and_moves_delta() -> None:
    base = pd.Series([1.0, 1.0], index=["a", "b"])
    constraints = pd.DataFrame(
        {"mol_id_lo": ["a"], "mol_id_hi": ["b"], "predicted_pair_delta": [2.0], "edge_weight": [1.0]}
    )
    result, shift = project_predictions(base, constraints, pair_strength=0.25)
    assert shift < 1e-12
    assert abs(result.mean() - base.mean()) < 1e-12
    assert result["b"] - result["a"] > 0
