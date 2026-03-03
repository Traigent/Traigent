"""Unit tests for detection_types.py data model.

Tests cover:
- Enum values and StrEnum behaviour
- Frozen dataclass construction and immutability
- SuggestedRange.to_parameter_range_code()
- DetectionResult properties and to_configuration_space()
"""

from __future__ import annotations

import pytest

from traigent.api.parameter_ranges import Choices, IntRange, Range
from traigent.tuned_variables.detection_types import (
    CandidateType,
    DetectionConfidence,
    DetectionResult,
    SourceLocation,
    SuggestedRange,
    TunedVariableCandidate,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestDetectionConfidence:
    def test_values_are_strings(self) -> None:
        assert DetectionConfidence.HIGH == "high"
        assert DetectionConfidence.MEDIUM == "medium"
        assert DetectionConfidence.LOW == "low"

    def test_construction_from_string(self) -> None:
        assert DetectionConfidence("high") is DetectionConfidence.HIGH
        assert DetectionConfidence("low") is DetectionConfidence.LOW

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            DetectionConfidence("ultra")


class TestCandidateType:
    def test_all_values_are_strings(self) -> None:
        assert CandidateType.NUMERIC_CONTINUOUS == "numeric_continuous"
        assert CandidateType.NUMERIC_INTEGER == "numeric_integer"
        assert CandidateType.CATEGORICAL == "categorical"
        assert CandidateType.BOOLEAN == "boolean"

    def test_construction_from_string(self) -> None:
        assert CandidateType("boolean") is CandidateType.BOOLEAN


# ---------------------------------------------------------------------------
# SourceLocation
# ---------------------------------------------------------------------------


class TestSourceLocation:
    def test_required_fields(self) -> None:
        loc = SourceLocation(line=10, col_offset=4)
        assert loc.line == 10
        assert loc.col_offset == 4
        assert loc.end_line is None
        assert loc.end_col_offset is None

    def test_full_construction(self) -> None:
        loc = SourceLocation(line=5, col_offset=2, end_line=5, end_col_offset=20)
        assert loc.end_line == 5
        assert loc.end_col_offset == 20

    def test_is_frozen(self) -> None:
        loc = SourceLocation(line=1, col_offset=0)
        with pytest.raises(AttributeError):
            loc.line = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SuggestedRange
# ---------------------------------------------------------------------------


class TestSuggestedRange:
    def test_to_code_range(self) -> None:
        sr = SuggestedRange(range_type="Range", kwargs={"low": 0.0, "high": 2.0})
        code = sr.to_parameter_range_code()
        assert code == "Range(low=0.0, high=2.0)", f"Unexpected: {code}"

    def test_to_code_int_range(self) -> None:
        sr = SuggestedRange(range_type="IntRange", kwargs={"low": 256, "high": 4096})
        code = sr.to_parameter_range_code()
        assert code == "IntRange(low=256, high=4096)", f"Unexpected: {code}"

    def test_to_code_choices(self) -> None:
        sr = SuggestedRange(
            range_type="Choices", kwargs={"values": ["gpt-4", "gpt-3.5-turbo"]}
        )
        code = sr.to_parameter_range_code()
        assert "Choices(" in code
        assert "gpt-4" in code
        assert "gpt-3.5-turbo" in code

    def test_to_code_with_default(self) -> None:
        sr = SuggestedRange(
            range_type="Range", kwargs={"low": 0.0, "high": 2.0, "default": 0.7}
        )
        code = sr.to_parameter_range_code()
        assert "default=0.7" in code

    def test_empty_kwargs(self) -> None:
        sr = SuggestedRange(range_type="Range")
        code = sr.to_parameter_range_code()
        assert code == "Range()"

    def test_to_parameter_range_object_range(self) -> None:
        sr = SuggestedRange(range_type="Range", kwargs={"low": 0.0, "high": 1.0})
        obj = sr.to_parameter_range()
        assert isinstance(obj, Range)
        assert obj.low == 0.0
        assert obj.high == 1.0

    def test_to_parameter_range_object_int_range(self) -> None:
        sr = SuggestedRange(range_type="IntRange", kwargs={"low": 128, "high": 1024})
        obj = sr.to_parameter_range()
        assert isinstance(obj, IntRange)
        assert obj.low == 128
        assert obj.high == 1024

    def test_to_parameter_range_object_choices(self) -> None:
        sr = SuggestedRange(range_type="Choices", kwargs={"values": ["gpt-4", "gpt-4o"]})
        obj = sr.to_parameter_range()
        assert isinstance(obj, Choices)
        assert "gpt-4" in obj
        assert "gpt-4o" in obj

    def test_to_parameter_range_unknown_type_raises(self) -> None:
        sr = SuggestedRange(range_type="UnknownRange", kwargs={"x": 1})
        with pytest.raises(ValueError, match="Unsupported range_type"):
            sr.to_parameter_range()

    def test_is_frozen(self) -> None:
        sr = SuggestedRange(range_type="Range", kwargs={"low": 0.0, "high": 1.0})
        with pytest.raises(AttributeError):
            sr.range_type = "IntRange"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TunedVariableCandidate
# ---------------------------------------------------------------------------


class TestTunedVariableCandidate:
    def _make(self, **kwargs) -> TunedVariableCandidate:
        defaults = {
            "name": "temperature",
            "candidate_type": CandidateType.NUMERIC_CONTINUOUS,
            "confidence": DetectionConfidence.HIGH,
            "location": SourceLocation(line=5, col_offset=4),
        }
        defaults.update(kwargs)
        return TunedVariableCandidate(**defaults)

    def test_basic_construction(self) -> None:
        c = self._make()
        assert c.name == "temperature"
        assert c.candidate_type == CandidateType.NUMERIC_CONTINUOUS
        assert c.confidence == DetectionConfidence.HIGH
        assert c.current_value is None
        assert c.suggested_range is None
        assert c.detection_source == "ast"
        assert c.reasoning == ""
        assert c.canonical_name is None

    def test_with_suggested_range(self) -> None:
        sr = SuggestedRange(range_type="Range", kwargs={"low": 0.0, "high": 2.0})
        c = self._make(suggested_range=sr, current_value=0.7)
        assert c.suggested_range is sr
        assert c.current_value == 0.7
        code = c.suggested_range.to_parameter_range_code()
        assert "Range(" in code

    def test_is_frozen(self) -> None:
        c = self._make()
        with pytest.raises(AttributeError):
            c.name = "top_p"  # type: ignore[misc]

    def test_detection_source_variants(self) -> None:
        for source in ("ast", "llm", "combined"):
            c = self._make(detection_source=source)
            assert c.detection_source == source


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------


def _make_candidate(
    name: str,
    confidence: DetectionConfidence,
    *,
    with_range: bool = True,
) -> TunedVariableCandidate:
    sr = (
        SuggestedRange(range_type="Range", kwargs={"low": 0.0, "high": 1.0})
        if with_range
        else None
    )
    return TunedVariableCandidate(
        name=name,
        candidate_type=CandidateType.NUMERIC_CONTINUOUS,
        confidence=confidence,
        location=SourceLocation(line=1, col_offset=0),
        suggested_range=sr,
    )


class TestDetectionResult:
    def test_empty_result(self) -> None:
        r = DetectionResult(function_name="fn")
        assert r.count == 0
        assert len(r.high_confidence) == 0
        assert r.to_configuration_space() == {}

    def test_count_matches_candidates(self) -> None:
        c1 = _make_candidate("temperature", DetectionConfidence.HIGH)
        c2 = _make_candidate("model", DetectionConfidence.LOW)
        r = DetectionResult(function_name="fn", candidates=(c1, c2))
        assert r.count == 2, f"Expected 2, got {r.count}"

    def test_high_confidence_filters_correctly(self) -> None:
        high = _make_candidate("temperature", DetectionConfidence.HIGH)
        medium = _make_candidate("top_p", DetectionConfidence.MEDIUM)
        low = _make_candidate("model", DetectionConfidence.LOW)
        r = DetectionResult(function_name="fn", candidates=(high, medium, low))

        result = r.high_confidence
        assert len(result) == 1, f"Expected 1 HIGH candidate, got {len(result)}"
        assert result[0].name == "temperature"

    def test_to_configuration_space_excludes_low(self) -> None:
        high = _make_candidate("temperature", DetectionConfidence.HIGH)
        medium = _make_candidate("top_p", DetectionConfidence.MEDIUM)
        low = _make_candidate("seed", DetectionConfidence.LOW)
        r = DetectionResult(function_name="fn", candidates=(high, medium, low))

        cs = r.to_configuration_space()
        assert "temperature" in cs, "HIGH candidate should be in config space"
        assert "top_p" in cs, "MEDIUM candidate should be in config space"
        assert "seed" not in cs, "LOW candidate must be excluded from config space"

    def test_to_configuration_space_empty_when_no_range(self) -> None:
        c = _make_candidate("temperature", DetectionConfidence.HIGH, with_range=False)
        r = DetectionResult(function_name="fn", candidates=(c,))
        cs = r.to_configuration_space()
        assert cs == {}, "No range means nothing to put in config space"

    def test_to_configuration_space_ranges_format(self) -> None:
        high = _make_candidate("temperature", DetectionConfidence.HIGH)
        r = DetectionResult(function_name="fn", candidates=(high,))
        cs = r.to_configuration_space(format="ranges")
        assert isinstance(cs["temperature"], Range)

    def test_to_configuration_space_min_confidence(self) -> None:
        high = _make_candidate("temperature", DetectionConfidence.HIGH)
        medium = _make_candidate("top_p", DetectionConfidence.MEDIUM)
        r = DetectionResult(function_name="fn", candidates=(high, medium))

        cs = r.to_configuration_space(min_confidence="high")
        assert "temperature" in cs
        assert "top_p" not in cs

    def test_to_configuration_space_include_exclude(self) -> None:
        high = _make_candidate("temperature", DetectionConfidence.HIGH)
        medium = _make_candidate("top_p", DetectionConfidence.MEDIUM)
        r = DetectionResult(function_name="fn", candidates=(high, medium))

        included = r.to_configuration_space(include={"top_p"})
        assert set(included.keys()) == {"top_p"}

        excluded = r.to_configuration_space(exclude={"temperature"})
        assert "temperature" not in excluded
        assert "top_p" in excluded

    def test_to_configuration_space_invalid_format_raises(self) -> None:
        r = DetectionResult(
            function_name="fn",
            candidates=(_make_candidate("temperature", DetectionConfidence.HIGH),),
        )
        with pytest.raises(ValueError, match="format must be"):
            r.to_configuration_space(format="bad")  # type: ignore[arg-type]

    def test_warnings_tuple(self) -> None:
        r = DetectionResult(function_name="fn", warnings=("something went wrong",))
        assert len(r.warnings) == 1
        assert "went wrong" in r.warnings[0]

    def test_strategies_used_tuple(self) -> None:
        r = DetectionResult(
            function_name="fn",
            detection_strategies_used=("ASTDetectionStrategy", "LLMDetectionStrategy"),
        )
        assert "ASTDetectionStrategy" in r.detection_strategies_used
        assert len(r.detection_strategies_used) == 2

    def test_is_frozen(self) -> None:
        r = DetectionResult(function_name="fn")
        with pytest.raises(AttributeError):
            r.function_name = "other"  # type: ignore[misc]
