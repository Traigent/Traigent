"""Tests for config_generator.subsystems.tvar_ranges."""

from __future__ import annotations

from traigent.config_generator.subsystems.tvar_ranges import (
    _heuristic_from_value,
    _parse_llm_range_response,
    generate_tvar_specs,
)
from traigent.tuned_variables.detection_types import (
    CandidateType,
    DetectionConfidence,
    DetectionResult,
    SourceLocation,
    SuggestedRange,
    TunedVariableCandidate,
)


def _make_candidate(
    name: str = "temperature",
    canonical_name: str | None = "temperature",
    confidence: DetectionConfidence = DetectionConfidence.HIGH,
    value: object = 0.7,
    candidate_type: CandidateType = CandidateType.NUMERIC_CONTINUOUS,
    suggested_range: SuggestedRange | None = None,
) -> TunedVariableCandidate:
    return TunedVariableCandidate(
        name=name,
        canonical_name=canonical_name,
        candidate_type=candidate_type,
        confidence=confidence,
        current_value=value,
        location=SourceLocation(line=1, col_offset=0),
        reasoning="test",
        detection_source="ast",
        suggested_range=suggested_range,
    )


def _make_result(candidates: list[TunedVariableCandidate]) -> DetectionResult:
    return DetectionResult(
        function_name="test_func",
        candidates=tuple(candidates),
        warnings=(),
        source_hash="abc123",
        detection_strategies_used=("ast",),
    )


class TestGenerateTvarSpecs:
    def test_canonical_preset_temperature(self) -> None:
        result = _make_result([_make_candidate()])
        specs = generate_tvar_specs([result])
        assert len(specs) == 1
        assert specs[0].name == "temperature"
        assert specs[0].range_type == "Range"
        assert specs[0].source == "preset"
        assert specs[0].range_kwargs["low"] == 0.0
        assert specs[0].range_kwargs["high"] == 1.0

    def test_canonical_preset_max_tokens(self) -> None:
        candidate = _make_candidate(
            name="max_tokens",
            canonical_name="max_tokens",
            value=1024,
            candidate_type=CandidateType.NUMERIC_INTEGER,
        )
        specs = generate_tvar_specs([_make_result([candidate])])
        assert len(specs) == 1
        assert specs[0].range_type == "IntRange"
        assert specs[0].source == "preset"

    def test_canonical_preset_model(self) -> None:
        candidate = _make_candidate(
            name="model_name",
            canonical_name="model",
            value="gpt-4o",
            candidate_type=CandidateType.CATEGORICAL,
        )
        specs = generate_tvar_specs([_make_result([candidate])])
        assert len(specs) == 1
        assert specs[0].range_type == "Choices"

    def test_fallback_to_suggested_range(self) -> None:
        candidate = _make_candidate(
            name="custom_param",
            canonical_name=None,
            value=0.5,
            suggested_range=SuggestedRange(
                range_type="Range", kwargs={"low": 0.1, "high": 0.9}
            ),
        )
        specs = generate_tvar_specs([_make_result([candidate])])
        assert len(specs) == 1
        assert specs[0].source == "detection"
        assert specs[0].range_kwargs == {"low": 0.1, "high": 0.9}

    def test_fallback_to_heuristic(self) -> None:
        candidate = _make_candidate(
            name="custom_float",
            canonical_name=None,
            value=0.5,
            suggested_range=None,
        )
        specs = generate_tvar_specs([_make_result([candidate])])
        assert len(specs) == 1
        assert specs[0].source == "heuristic"
        assert specs[0].range_type == "Range"

    def test_skips_low_confidence(self) -> None:
        candidate = _make_candidate(confidence=DetectionConfidence.LOW)
        specs = generate_tvar_specs([_make_result([candidate])])
        assert len(specs) == 0

    def test_deduplicates_by_name(self) -> None:
        c1 = _make_candidate(name="temperature")
        c2 = _make_candidate(name="temperature")
        result = _make_result([c1, c2])
        specs = generate_tvar_specs([result])
        assert len(specs) == 1

    def test_multiple_results(self) -> None:
        r1 = _make_result([_make_candidate(name="temperature")])
        r2 = _make_result(
            [
                _make_candidate(
                    name="max_tokens",
                    canonical_name="max_tokens",
                    value=1024,
                    candidate_type=CandidateType.NUMERIC_INTEGER,
                )
            ]
        )
        specs = generate_tvar_specs([r1, r2])
        assert len(specs) == 2
        names = {s.name for s in specs}
        assert names == {"temperature", "max_tokens"}

    def test_medium_confidence_included(self) -> None:
        candidate = _make_candidate(confidence=DetectionConfidence.MEDIUM)
        specs = generate_tvar_specs([_make_result([candidate])])
        assert len(specs) == 1
        assert specs[0].confidence == 0.7


class TestHeuristicFromValue:
    def test_float_positive(self) -> None:
        spec = _heuristic_from_value("x", 0.5)
        assert spec is not None
        assert spec.range_type == "Range"
        assert spec.range_kwargs["low"] == 0.25
        assert spec.range_kwargs["high"] == 1.0

    def test_float_zero(self) -> None:
        spec = _heuristic_from_value("x", 0.0)
        assert spec is not None
        assert spec.range_kwargs["low"] == 0.0
        assert spec.range_kwargs["high"] == 1.0

    def test_int_positive(self) -> None:
        spec = _heuristic_from_value("x", 100)
        assert spec is not None
        assert spec.range_type == "IntRange"
        assert spec.range_kwargs["low"] == 50
        assert spec.range_kwargs["high"] == 200

    def test_int_small(self) -> None:
        spec = _heuristic_from_value("x", 1)
        assert spec is not None
        assert spec.range_kwargs["low"] == 1
        assert spec.range_kwargs["high"] >= 2

    def test_string(self) -> None:
        spec = _heuristic_from_value("x", "gpt-4o")
        assert spec is not None
        assert spec.range_type == "Choices"
        assert spec.range_kwargs["values"] == ["gpt-4o"]

    def test_bool(self) -> None:
        spec = _heuristic_from_value("x", True)
        assert spec is not None
        assert spec.range_type == "Choices"
        assert set(spec.range_kwargs["values"]) == {True, False}

    def test_none_returns_none(self) -> None:
        assert _heuristic_from_value("x", None) is None

    def test_list_returns_none(self) -> None:
        assert _heuristic_from_value("x", [1, 2, 3]) is None


class TestParseLlmRangeResponse:
    def test_valid_range_json(self) -> None:
        response = '{"range_type": "Range", "kwargs": {"low": 0.0, "high": 2.0}}'
        spec = _parse_llm_range_response("x", response)
        assert spec is not None
        assert spec.range_type == "Range"
        assert spec.range_kwargs == {"low": 0.0, "high": 2.0}
        assert spec.source == "llm"

    def test_valid_choices_json(self) -> None:
        response = '{"range_type": "Choices", "kwargs": {"values": ["a", "b"]}}'
        spec = _parse_llm_range_response("x", response)
        assert spec is not None
        assert spec.range_type == "Choices"

    def test_markdown_fenced_json(self) -> None:
        response = (
            '```json\n{"range_type": "IntRange", "kwargs": {"low": 1, "high": 10}}\n```'
        )
        spec = _parse_llm_range_response("x", response)
        assert spec is not None
        assert spec.range_type == "IntRange"

    def test_invalid_json(self) -> None:
        assert _parse_llm_range_response("x", "not json") is None

    def test_missing_range_type(self) -> None:
        assert _parse_llm_range_response("x", '{"kwargs": {"low": 0}}') is None

    def test_invalid_range_type(self) -> None:
        response = '{"range_type": "BadType", "kwargs": {"low": 0}}'
        assert _parse_llm_range_response("x", response) is None

    def test_missing_kwargs(self) -> None:
        response = '{"range_type": "Range"}'
        assert _parse_llm_range_response("x", response) is None
