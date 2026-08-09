import pandas as pd

from scripts.analyze_bindingdb_patent_structure_novelty import classify_cliff_edges


def test_cliff_edges_are_partitioned_by_endpoint_novelty() -> None:
    truth = pd.Series({"a": 0.0, "b": 1.2, "c": 2.5, "d": 2.0})
    novel = pd.Series({"a": True, "b": True, "c": False, "d": False})
    edges = pd.DataFrame(
        {
            "mol_id_lo": ["a", "a", "c", "b"],
            "mol_id_hi": ["b", "c", "d", "c"],
        }
    )
    strata = classify_cliff_edges(edges, truth, novel)
    assert len(strata["all"]) == 3
    assert len(strata["both_novel"]) == 1
    assert len(strata["mixed"]) == 2
    assert len(strata["both_prior_overlap"]) == 0
