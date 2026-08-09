import pandas as pd

from scripts.build_rf_gine_component_scale import scale_components


def test_component_scaling_preserves_each_component_mean() -> None:
    base = pd.Series([0.0, 2.0, 10.0], index=["a", "b", "c"])
    edges = pd.DataFrame({"mol_id_lo": ["a"], "mol_id_hi": ["b"]})
    result = scale_components(base, edges, 1.25)
    assert abs(result.loc[["a", "b"]].mean() - base.loc[["a", "b"]].mean()) < 1e-12
    assert result["b"] - result["a"] > base["b"] - base["a"]
    assert result["c"] == base["c"]
