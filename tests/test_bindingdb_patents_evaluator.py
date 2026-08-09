from scripts.evaluate_bindingdb_patents_external import METHODS


def test_patent_primary_method_set_is_fixed() -> None:
    assert METHODS == ["unscaled_reference", "bps", "confidence_weighted_bps"]
