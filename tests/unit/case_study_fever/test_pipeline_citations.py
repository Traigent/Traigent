from paper_experiments.case_study_fever import pipeline


def test_format_numbered_evidence_handles_empty():
    result = pipeline._format_numbered_evidence([])
    assert "no evidence" in result.lower()


def test_extract_structured_response_parses_all_fields():
    response = """CITED_EVIDENCE: [1, 3]
REASONING: Sentence 1 supports the claim while 3 provides context.
VERDICT: SUPPORTS
"""
    verdict, reasoning, cited = pipeline._extract_structured_response(response)
    assert verdict == "SUPPORTS"
    assert "supports the claim" in reasoning
    assert cited == [1, 3]


def test_map_citations_to_evidence_filters_invalid_entries():
    evidence_items = [
        {"page": "Page_A", "line": 2, "text": "A"},
        {"page": "Page_B", "line": "3", "text": "B"},
    ]
    mapped = pipeline._map_citations_to_evidence([2, 5, 2], evidence_items)
    assert mapped == [{"page": "Page_B", "line": 3}]


def test_normalize_pipeline_output_filters_bad_evidence():
    result = pipeline._normalize_pipeline_output(
        verdict="SUPPORTS",
        justification="Because evidence 1 exists",
        evidence=[
            {"page": "Valid", "line": 1},
            {"page": "", "line": 2},
            {"page": "AlsoValid", "line": "5"},
            {"page": "Bad", "line": "not_int"},
        ],
    )
    assert result["evidence"] == [
        {"page": "Valid", "line": 1},
        {"page": "AlsoValid", "line": 5},
    ]
