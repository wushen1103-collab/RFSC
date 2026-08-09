from scripts.audit_cross_benchmark_structure_overlap import overlap_record


def test_overlap_record_reports_both_containments_and_jaccard() -> None:
    record = overlap_record("a", {"x", "y", "z"}, "b", {"y", "z", "w", "q"})
    assert record["intersection"] == 2
    assert record["containment_a"] == 2 / 3
    assert record["containment_b"] == 1 / 2
    assert record["jaccard"] == 2 / 5


def test_overlap_record_handles_empty_set() -> None:
    record = overlap_record("empty", set(), "one", {"x"})
    assert record["intersection"] == 0
    assert record["containment_a"] == 0.0
    assert record["jaccard"] == 0.0
