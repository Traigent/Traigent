"""Unit tests for detection_strategies.py.

Tests cover:
- ASTDetectionStrategy: all detection patterns with explicit confidence assertions
- LLMDetectionStrategy: graceful degradation, JSON parsing, error handling
- _suggest_range: range generation for known and unknown parameter names
- _infer_candidate_type: type inference from literal values
"""

from __future__ import annotations

import textwrap

import pytest

from traigent.tuned_variables.detection_strategies import (
    ASTDetectionStrategy,
    LLMDetectionStrategy,
    _infer_candidate_type,
    _suggest_range,
)
from traigent.tuned_variables.detection_types import CandidateType, DetectionConfidence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dedent(src: str) -> str:
    return textwrap.dedent(src).strip()


def _names(candidates) -> list[str]:
    return [c.name for c in candidates]


def _by_name(candidates, name: str):
    for c in candidates:
        if c.name == name:
            return c
    return None


# ---------------------------------------------------------------------------
# _infer_candidate_type
# ---------------------------------------------------------------------------


class TestInferCandidateType:
    def test_float_is_continuous(self) -> None:
        assert _infer_candidate_type(0.7) == CandidateType.NUMERIC_CONTINUOUS

    def test_int_is_integer(self) -> None:
        assert _infer_candidate_type(1024) == CandidateType.NUMERIC_INTEGER

    def test_bool_is_boolean(self) -> None:
        # bool must be checked before int (bool is subclass of int)
        assert _infer_candidate_type(True) == CandidateType.BOOLEAN
        assert _infer_candidate_type(False) == CandidateType.BOOLEAN

    def test_str_is_categorical(self) -> None:
        assert _infer_candidate_type("gpt-4") == CandidateType.CATEGORICAL

    def test_list_is_categorical(self) -> None:
        assert _infer_candidate_type(["a", "b"]) == CandidateType.CATEGORICAL


# ---------------------------------------------------------------------------
# _suggest_range
# ---------------------------------------------------------------------------


class TestSuggestRange:
    def test_temperature_uses_canonical_range(self) -> None:
        sr = _suggest_range("temperature", CandidateType.NUMERIC_CONTINUOUS, 0.7)
        assert sr is not None
        assert sr.range_type == "Range"
        assert sr.kwargs["low"] == pytest.approx(0.0)
        assert sr.kwargs["high"] == pytest.approx(2.0)

    def test_max_tokens_uses_int_range(self) -> None:
        sr = _suggest_range("max_tokens", CandidateType.NUMERIC_INTEGER, 512)
        assert sr is not None
        assert sr.range_type == "IntRange"
        assert sr.kwargs["low"] < sr.kwargs["high"]

    def test_unknown_float_uses_heuristic(self) -> None:
        sr = _suggest_range(None, CandidateType.NUMERIC_CONTINUOUS, 2.0)
        assert sr is not None
        assert sr.range_type == "Range"
        assert sr.kwargs["low"] == pytest.approx(1.0)
        assert sr.kwargs["high"] == pytest.approx(4.0)

    def test_unknown_int_uses_heuristic(self) -> None:
        sr = _suggest_range(None, CandidateType.NUMERIC_INTEGER, 100)
        assert sr is not None
        assert sr.range_type == "IntRange"
        assert sr.kwargs["low"] == 50
        assert sr.kwargs["high"] == 200

    def test_string_value_becomes_choices(self) -> None:
        sr = _suggest_range(None, CandidateType.CATEGORICAL, "gpt-4")
        assert sr is not None
        assert sr.range_type == "Choices"
        assert "gpt-4" in sr.kwargs["values"]

    def test_list_value_becomes_choices(self) -> None:
        sr = _suggest_range(None, CandidateType.CATEGORICAL, ["a", "b"])
        assert sr is not None
        assert sr.range_type == "Choices"
        assert "a" in sr.kwargs["values"]

    def test_boolean_becomes_choices_of_true_false(self) -> None:
        sr = _suggest_range(None, CandidateType.BOOLEAN, True)
        assert sr is not None
        assert sr.range_type == "Choices"
        assert True in sr.kwargs["values"]
        assert False in sr.kwargs["values"]


# ---------------------------------------------------------------------------
# ASTDetectionStrategy
# ---------------------------------------------------------------------------


class TestASTDetectionStrategy:
    @pytest.fixture
    def strategy(self) -> ASTDetectionStrategy:
        return ASTDetectionStrategy()

    # -- HIGH confidence: direct name match -----------------------------------

    def test_detects_temperature_assignment(self, strategy) -> None:
        src = _dedent("""
            def my_func():
                temperature = 0.7
                return temperature
        """)
        candidates = strategy.detect(src, "my_func")
        c = _by_name(candidates, "temperature")
        assert c is not None, "temperature should be detected"
        assert c.confidence == DetectionConfidence.HIGH
        assert c.current_value == pytest.approx(0.7)
        assert c.candidate_type == CandidateType.NUMERIC_CONTINUOUS

    def test_detects_model_assignment(self, strategy) -> None:
        src = _dedent("""
            def call_llm():
                model = "gpt-4"
                return model
        """)
        candidates = strategy.detect(src, "call_llm")
        c = _by_name(candidates, "model")
        assert c is not None, "model should be detected"
        assert c.confidence == DetectionConfidence.HIGH
        assert c.current_value == "gpt-4"
        assert c.candidate_type == CandidateType.CATEGORICAL

    def test_detects_max_tokens_assignment(self, strategy) -> None:
        src = _dedent("""
            def generate():
                max_tokens = 1024
                return max_tokens
        """)
        candidates = strategy.detect(src, "generate")
        c = _by_name(candidates, "max_tokens")
        assert c is not None, "max_tokens should be detected"
        assert c.confidence == DetectionConfidence.HIGH
        assert c.current_value == 1024
        assert c.candidate_type == CandidateType.NUMERIC_INTEGER

    def test_detects_annotated_assignment(self, strategy) -> None:
        src = _dedent("""
            def my_func():
                temperature: float = 0.9
                return temperature
        """)
        candidates = strategy.detect(src, "my_func")
        c = _by_name(candidates, "temperature")
        assert c is not None, "annotated assignment should be detected"
        assert c.confidence == DetectionConfidence.HIGH
        assert c.current_value == pytest.approx(0.9)

    # -- HIGH confidence: kwargs in API calls ---------------------------------

    def test_detects_kwarg_in_call(self, strategy) -> None:
        src = _dedent("""
            def my_func():
                result = client.create(temperature=0.7, model="gpt-4")
                return result
        """)
        candidates = strategy.detect(src, "my_func")
        names = _names(candidates)
        assert "temperature" in names, f"temperature kwarg not detected; got {names}"
        c = _by_name(candidates, "temperature")
        assert c.confidence == DetectionConfidence.HIGH
        assert c.current_value == pytest.approx(0.7)

    def test_detects_model_kwarg_in_call(self, strategy) -> None:
        src = _dedent("""
            def chat():
                response = openai.ChatCompletion.create(model="gpt-4o", max_tokens=512)
                return response
        """)
        candidates = strategy.detect(src, "chat")
        names = _names(candidates)
        assert "model" in names, f"model kwarg not detected; got {names}"
        assert "max_tokens" in names, f"max_tokens kwarg not detected; got {names}"

    # -- MEDIUM confidence: dict keys -----------------------------------------

    def test_detects_dict_key(self, strategy) -> None:
        src = _dedent("""
            def build_config():
                config = {"model": "gpt-4", "temperature": 0.5}
                return config
        """)
        candidates = strategy.detect(src, "build_config")
        names = _names(candidates)
        assert "model" in names, f"model dict key not detected; got {names}"
        c = _by_name(candidates, "model")
        assert c.confidence == DetectionConfidence.MEDIUM

    # -- MEDIUM confidence: fuzzy name match ----------------------------------

    def test_detects_fuzzy_match_temp(self, strategy) -> None:
        src = _dedent("""
            def my_func():
                temp = 0.8
                return temp
        """)
        candidates = strategy.detect(src, "my_func")
        # "temp" is a known variant of "temperature" in the universal mapping,
        # so it gets HIGH confidence (direct known-param match), not MEDIUM.
        assert (
            len(candidates) == 1
        ), f"'temp' should yield exactly 1 candidate; got {[c.name for c in candidates]}"
        c = candidates[0]
        assert c.name == "temp"
        assert c.confidence == DetectionConfidence.HIGH
        assert c.current_value == pytest.approx(0.8)

    # -- MEDIUM confidence: model string heuristic ----------------------------

    def test_detects_model_string_heuristic(self, strategy) -> None:
        src = _dedent("""
            def infer():
                llm_model = "claude-3-opus"
                return llm_model
        """)
        candidates = strategy.detect(src, "infer")
        # "llm_model" is not an exact match but "claude-3-opus" looks like a model
        assert (
            len(candidates) == 1
        ), f"model string heuristic should fire exactly once; got {[c.name for c in candidates]}"
        c = candidates[0]
        assert c.name == "llm_model"
        assert c.current_value == "claude-3-opus"
        assert (
            c.confidence == DetectionConfidence.MEDIUM
        )  # model string heuristic is MEDIUM
        assert c.canonical_name == "model"

    # -- Skips ParameterRange assignments ------------------------------------

    def test_skips_parameter_range_assignment(self, strategy) -> None:
        src = _dedent("""
            def my_func():
                temperature = Range(0.0, 2.0)
                return temperature
        """)
        candidates = strategy.detect(src, "my_func")
        c = _by_name(candidates, "temperature")
        assert (
            c is None
        ), "Variable already using ParameterRange must not be re-detected"

    def test_skips_choices_assignment(self, strategy) -> None:
        src = _dedent("""
            def my_func():
                model = Choices(["gpt-4", "gpt-3.5-turbo"])
                return model
        """)
        candidates = strategy.detect(src, "my_func")
        c = _by_name(candidates, "model")
        assert c is None, "Variable already using Choices must not be re-detected"

    # -- Nested scope isolation -----------------------------------------------

    def test_skips_nested_function_assignment(self, strategy) -> None:
        src = _dedent("""
            def outer():
                def inner():
                    temperature = 0.9
                return inner
        """)
        candidates = strategy.detect(src, "outer")
        c = _by_name(candidates, "temperature")
        assert c is None, "Assignment inside nested function must be ignored"

    def test_skips_existing_tvar_context(self, strategy) -> None:
        src = _dedent("""
            def my_func():
                temperature = 0.7
                return temperature
        """)
        candidates = strategy.detect(
            src, "my_func", context={"existing_tvars": {"temperature"}}
        )
        c = _by_name(candidates, "temperature")
        assert c is None, "Already-configured variable must be skipped"

    # -- Wrong function name returns nothing ----------------------------------

    def test_returns_empty_for_unknown_function(self, strategy) -> None:
        src = _dedent("""
            def actual_func():
                temperature = 0.7
        """)
        candidates = strategy.detect(src, "nonexistent_func")
        assert candidates == [], "Should return empty list for unknown function"

    # -- Syntax error handling ------------------------------------------------

    def test_handles_syntax_error_gracefully(self, strategy) -> None:
        src = "def broken(: pass"
        candidates = strategy.detect(src, "broken")
        assert candidates == [], "Syntax errors should return empty list, not raise"

    # -- Suggested ranges are populated ---------------------------------------

    def test_suggested_range_populated_for_temperature(self, strategy) -> None:
        src = _dedent("""
            def fn():
                temperature = 0.7
        """)
        candidates = strategy.detect(src, "fn")
        c = _by_name(candidates, "temperature")
        assert c is not None
        assert (
            c.suggested_range is not None
        ), "temperature should have a suggested range"
        assert c.suggested_range.range_type == "Range"
        code = c.suggested_range.to_parameter_range_code()
        assert "Range(" in code

    def test_suggested_range_populated_for_model(self, strategy) -> None:
        src = _dedent("""
            def fn():
                model = "gpt-4"
        """)
        candidates = strategy.detect(src, "fn")
        c = _by_name(candidates, "model")
        assert c is not None
        assert c.suggested_range is not None, "model should have a suggested range"
        assert c.suggested_range.range_type == "Choices"

    def test_non_literal_assignment_not_detected(self, strategy) -> None:
        """Variables assigned complex expressions (not literals) must be skipped."""
        src = _dedent("""
            def fn():
                temperature = compute_temperature()
                model = os.getenv("MODEL", "gpt-4")
        """)
        candidates = strategy.detect(src, "fn")
        # No literal value → cannot extract; should be empty
        assert (
            candidates == []
        ), f"Non-literal assignments must not be detected; got {[c.name for c in candidates]}"

    def test_detects_stop_parameter_with_list_value(self, strategy) -> None:
        """'stop' parameter with a list of strings should be detected as CATEGORICAL."""
        src = _dedent("""
            def fn():
                stop = ["\\n", "END"]
                return stop
        """)
        candidates = strategy.detect(src, "fn")
        c = _by_name(candidates, "stop")
        assert c is not None, "stop parameter should be detected"
        assert c.confidence == DetectionConfidence.HIGH
        assert c.candidate_type == CandidateType.CATEGORICAL
        assert c.current_value == ["\n", "END"]

    def test_detects_stream_parameter_with_bool_value(self, strategy) -> None:
        """'stream' parameter with a bool value should be detected as BOOLEAN."""
        src = _dedent("""
            def fn():
                stream = True
                return stream
        """)
        candidates = strategy.detect(src, "fn")
        c = _by_name(candidates, "stream")
        assert c is not None, "stream parameter should be detected"
        assert c.confidence == DetectionConfidence.HIGH
        assert c.candidate_type == CandidateType.BOOLEAN

    def test_canonical_name_populated_for_variant(self, strategy) -> None:
        """Variant names (e.g. model_name) should map to canonical (model)."""
        src = _dedent("""
            def fn():
                model_name = "gpt-4"
        """)
        candidates = strategy.detect(src, "fn")
        c = _by_name(candidates, "model_name")
        assert c is not None, "model_name should be detected as a known variant"
        assert (
            c.canonical_name == "model"
        ), f"Expected canonical 'model', got {c.canonical_name!r}"

    # -- Source location is set -----------------------------------------------

    def test_source_location_line_is_positive(self, strategy) -> None:
        src = _dedent("""
            def fn():
                temperature = 0.5
        """)
        candidates = strategy.detect(src, "fn")
        c = _by_name(candidates, "temperature")
        assert c is not None
        assert c.location.line > 0, f"Line number should be > 0, got {c.location.line}"

    # -- Multiple candidates in same function ---------------------------------

    def test_detects_multiple_candidates(self, strategy) -> None:
        src = _dedent("""
            def pipeline():
                model = "gpt-4"
                temperature = 0.7
                max_tokens = 512
                return model, temperature, max_tokens
        """)
        candidates = strategy.detect(src, "pipeline")
        names = _names(candidates)
        assert "model" in names, f"model not detected; got {names}"
        assert "temperature" in names, f"temperature not detected; got {names}"
        assert "max_tokens" in names, f"max_tokens not detected; got {names}"
        assert (
            len(candidates) == 3
        ), f"Expected exactly 3 candidates, got {len(candidates)}"


# ---------------------------------------------------------------------------
# LLMDetectionStrategy
# ---------------------------------------------------------------------------


class TestLLMDetectionStrategy:
    def test_returns_empty_when_no_llm(self) -> None:
        strategy = LLMDetectionStrategy(llm_callable=None)
        src = "def fn():\n    temperature = 0.7\n"
        result = strategy.detect(src, "fn")
        assert result == [], "No LLM callable should return empty list"

    def test_parses_valid_json_response(self) -> None:
        response = """[
            {"name": "temperature", "type": "numeric_continuous",
             "current_value": 0.7, "reasoning": "Controls creativity"},
            {"name": "model", "type": "categorical",
             "current_value": "gpt-4", "reasoning": "Model choice"}
        ]"""

        def mock_llm(prompt: str) -> str:
            return response

        strategy = LLMDetectionStrategy(llm_callable=mock_llm)
        src = _dedent("""
            def fn():
                temperature = 0.7
                model = "gpt-4"
        """)
        candidates = strategy.detect(src, "fn")

        assert len(candidates) == 2, f"Expected 2 candidates, got {len(candidates)}"
        names = _names(candidates)
        assert "temperature" in names
        assert "model" in names

    def test_all_llm_candidates_start_at_low_confidence(self) -> None:
        response = '[{"name": "temperature", "type": "numeric_continuous", "current_value": 0.7, "reasoning": "r"}]'

        def mock_llm(prompt: str) -> str:
            return response

        strategy = LLMDetectionStrategy(llm_callable=mock_llm)
        src = "def fn():\n    temperature = 0.7\n"
        candidates = strategy.detect(src, "fn")

        assert len(candidates) == 1
        assert (
            candidates[0].confidence == DetectionConfidence.LOW
        ), "LLM-only candidates must start at LOW confidence"
        assert candidates[0].detection_source == "llm"

    def test_handles_malformed_json_gracefully(self) -> None:
        def mock_llm(prompt: str) -> str:
            return "this is not json at all"

        strategy = LLMDetectionStrategy(llm_callable=mock_llm)
        src = "def fn():\n    pass\n"
        candidates = strategy.detect(src, "fn")
        assert candidates == [], "Malformed JSON should return empty list, not raise"

    def test_handles_llm_exception_gracefully(self) -> None:
        def failing_llm(prompt: str) -> str:
            raise RuntimeError("LLM service unavailable")

        strategy = LLMDetectionStrategy(llm_callable=failing_llm)
        src = "def fn():\n    temperature = 0.5\n"
        candidates = strategy.detect(src, "fn")
        assert candidates == [], "LLM exception should return empty list, not raise"

    def test_parses_markdown_wrapped_json(self) -> None:
        response = '```json\n[{"name": "top_p", "type": "numeric_continuous", "current_value": 0.9, "reasoning": "r"}]\n```'

        def mock_llm(prompt: str) -> str:
            return response

        strategy = LLMDetectionStrategy(llm_callable=mock_llm)
        src = "def fn():\n    top_p = 0.9\n"
        candidates = strategy.detect(src, "fn")
        assert len(candidates) == 1
        assert candidates[0].name == "top_p"

    def test_returns_empty_for_nonexistent_function(self) -> None:
        def mock_llm(prompt: str) -> str:
            return "[]"

        strategy = LLMDetectionStrategy(llm_callable=mock_llm)
        src = "def actual_fn():\n    temperature = 0.7\n"
        # function_name doesn't match → source extraction returns None
        candidates = strategy.detect(src, "nonexistent")
        assert candidates == []

    def test_canonical_name_mapped_for_known_params(self) -> None:
        response = '[{"name": "temperature", "type": "numeric_continuous", "current_value": 0.5, "reasoning": "r"}]'

        def mock_llm(prompt: str) -> str:
            return response

        strategy = LLMDetectionStrategy(llm_callable=mock_llm)
        src = "def fn():\n    temperature = 0.5\n"
        candidates = strategy.detect(src, "fn")
        assert len(candidates) == 1
        assert candidates[0].canonical_name == "temperature"

    def test_unknown_type_falls_back_to_categorical(self) -> None:
        response = '[{"name": "strategy", "type": "completely_unknown_type", "current_value": "a", "reasoning": "r"}]'

        def mock_llm(prompt: str) -> str:
            return response

        strategy = LLMDetectionStrategy(llm_callable=mock_llm)
        src = "def fn():\n    strategy = 'a'\n"
        candidates = strategy.detect(src, "fn")
        assert len(candidates) == 1
        assert candidates[0].candidate_type == CandidateType.CATEGORICAL

    def test_response_not_a_list_returns_empty(self) -> None:
        """If LLM returns a JSON object instead of array, return empty list."""
        response = '{"name": "temperature", "type": "numeric_continuous"}'

        def mock_llm(prompt: str) -> str:
            return response

        strategy = LLMDetectionStrategy(llm_callable=mock_llm)
        src = "def fn():\n    temperature = 0.7\n"
        candidates = strategy.detect(src, "fn")
        assert candidates == [], "JSON object (not array) should return empty list"

    def test_items_without_name_key_are_skipped(self) -> None:
        """Malformed items missing 'name' key should be silently skipped."""
        response = (
            '[{"type": "numeric_continuous", "current_value": 0.7, "reasoning": "r"}]'
        )

        def mock_llm(prompt: str) -> str:
            return response

        strategy = LLMDetectionStrategy(llm_callable=mock_llm)
        src = "def fn():\n    temperature = 0.7\n"
        candidates = strategy.detect(src, "fn")
        assert candidates == [], "Items without 'name' key must be skipped"


class TestSuggestRangeEdgeCases:
    """Edge cases for _suggest_range not covered by TestSuggestRange."""

    def test_zero_float_value_uses_fallback_range(self) -> None:
        """Float value of 0.0 should use 0..1 heuristic fallback."""
        sr = _suggest_range(None, CandidateType.NUMERIC_CONTINUOUS, 0.0)
        assert sr is not None
        assert sr.range_type == "Range"
        assert sr.kwargs["low"] == pytest.approx(0.0)
        assert sr.kwargs["high"] == pytest.approx(1.0)  # fallback for non-positive

    def test_none_canonical_with_none_value_returns_none(self) -> None:
        """Unknown type + None value has no heuristic → returns None."""
        sr = _suggest_range(None, CandidateType.NUMERIC_CONTINUOUS, None)
        assert sr is None, "No value and no canonical → no suggested range"
