import pandas as pd

from scripts.audit_gated_component_scale_validation import gated_edges


def test_gated_edges_uses_absolute_predicted_delta() -> None:
    base = pd.Series({"a": 0.0, "b": 0.2, "c": 1.0})
    edges = pd.DataFrame({"mol_id_lo": ["a", "b"], "mol_id_hi": ["b", "c"]})
    selected = gated_edges(base, edges, threshold=0.5)
    assert selected[["mol_id_lo", "mol_id_hi"]].to_dict(orient="records") == [
        {"mol_id_lo": "b", "mol_id_hi": "c"}
    ]
