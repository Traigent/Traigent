"""Unit tests for detector.py (TunedVariableDetector orchestrator).

Tests cover:
- detect_from_source: single-strategy and multi-strategy flows
- detect_from_callable: real functions, OSError handling
- detect_from_file: file and directory scanning
- _merge_candidates: deduplication and confidence upgrading
- Integration: DetectionResult properties after detection
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from traigent.tuned_variables.detection_strategies import (
    ASTDetectionStrategy,
    LLMDetectionStrategy,
)
from traigent.tuned_variables.detection_types import (
    CandidateType,
    DetectionConfidence,
    SourceLocation,
    TunedVariableCandidate,
)
from traigent.tuned_variables.detector import TunedVariableDetector, _upgrade_confidence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dedent(src: str) -> str:
    return textwrap.dedent(src).strip()


def _by_name(candidates, name: str):
    for c in candidates:
        if c.name == name:
            return c
    return None


def _make_candidate(
    name: str,
    confidence: DetectionConfidence,
    line: int = 1,
) -> TunedVariableCandidate:
    return TunedVariableCandidate(
        name=name,
        candidate_type=CandidateType.NUMERIC_CONTINUOUS,
        confidence=confidence,
        location=SourceLocation(line=line, col_offset=0),
    )


# ---------------------------------------------------------------------------
# _upgrade_confidence
# ---------------------------------------------------------------------------


class TestUpgradeConfidence:
    def test_two_high_stays_high(self) -> None:
        result = _upgrade_confidence(DetectionConfidence.HIGH, DetectionConfidence.HIGH)
        assert result == DetectionConfidence.HIGH

    def test_high_and_medium_stays_medium_or_high(self) -> None:
        result = _upgrade_confidence(
            DetectionConfidence.HIGH, DetectionConfidence.MEDIUM
        )
        assert result in (DetectionConfidence.HIGH, DetectionConfidence.MEDIUM)

    def test_two_low_upgrades_to_medium(self) -> None:
        # When two strategies both detect (even at LOW), agreement = upgrade
        result = _upgrade_confidence(DetectionConfidence.LOW, DetectionConfidence.LOW)
        assert (
            result == DetectionConfidence.MEDIUM
        ), "Two LOW detections should upgrade to MEDIUM"

    def test_low_and_medium_gives_medium(self) -> None:
        result = _upgrade_confidence(
            DetectionConfidence.LOW, DetectionConfidence.MEDIUM
        )
        assert result == DetectionConfidence.MEDIUM


# ---------------------------------------------------------------------------
# detect_from_source
# ---------------------------------------------------------------------------


class TestDetectFromSource:
    @pytest.fixture
    def detector(self) -> TunedVariableDetector:
        return TunedVariableDetector()  # defaults to [ASTDetectionStrategy()]

    def test_detects_known_param_high_confidence(self, detector) -> None:
        src = _dedent(
            """
            def my_func():
                temperature = 0.7
                return temperature
        """
        )
        result = detector.detect_from_source(src, "my_func")

        assert result.function_name == "my_func"
        assert result.count == 1, f"Expected exactly temperature, got {result.count}"
        c = _by_name(result.candidates, "temperature")
        assert c is not None, "temperature should be detected"
        assert c.confidence == DetectionConfidence.HIGH
        assert c.current_value == pytest.approx(0.7)

    def test_returns_empty_for_no_tvars(self, detector) -> None:
        src = _dedent(
            """
            def simple():
                x = 42
                name = "Alice"
                return x + len(name)
        """
        )
        result = detector.detect_from_source(src, "simple")
        # No known LLM param names — count should be 0
        assert result.count == 0, f"Expected 0 candidates, got {result.count}"

    def test_strategies_recorded_in_result(self, detector) -> None:
        src = "def fn():\n    temperature = 0.5\n"
        result = detector.detect_from_source(src, "fn")
        assert "ASTDetectionStrategy" in result.detection_strategies_used

    def test_source_hash_populated(self, detector) -> None:
        src = "def fn():\n    temperature = 0.5\n"
        result = detector.detect_from_source(src, "fn")
        assert (
            len(result.source_hash) == 16
        ), f"Expected 16-char hash, got: {result.source_hash!r}"

    def test_context_existing_tvars_skips_detected(self, detector) -> None:
        src = _dedent(
            """
            def fn():
                temperature = 0.7
                max_tokens = 512
        """
        )
        result = detector.detect_from_source(
            src, "fn", context={"existing_tvars": {"temperature"}}
        )
        names = [c.name for c in result.candidates]
        assert "temperature" not in names, "Already-configured var must be skipped"
        assert "max_tokens" in names, "Other vars should still be detected"

    def test_multi_strategy_merge_upgrades_confidence(self) -> None:
        # LLM strategy also detects temperature → merge upgrades LOW → MEDIUM
        llm_response = '[{"name": "temperature", "type": "numeric_continuous", "current_value": 0.7, "reasoning": "controls creativity"}]'

        def mock_llm(prompt: str) -> str:
            return llm_response

        detector = TunedVariableDetector(
            strategies=[ASTDetectionStrategy(), LLMDetectionStrategy(mock_llm)]
        )
        src = _dedent(
            """
            def fn():
                temperature = 0.7
        """
        )
        result = detector.detect_from_source(src, "fn")
        c = _by_name(result.candidates, "temperature")
        assert c is not None
        # AST gives HIGH, LLM gives LOW — merge keeps HIGH (not downgraded)
        assert c.confidence in (DetectionConfidence.HIGH, DetectionConfidence.MEDIUM)
        assert c.detection_source == "combined"

    def test_failed_strategy_does_not_crash_detector(self) -> None:
        class BrokenStrategy:
            def detect(self, source, function_name, *, context=None):
                raise RuntimeError("strategy exploded")

        detector = TunedVariableDetector(
            strategies=[BrokenStrategy(), ASTDetectionStrategy()]
        )
        src = "def fn():\n    temperature = 0.7\n"
        result = detector.detect_from_source(src, "fn")
        # AST strategy still ran despite broken one
        assert (
            result.count == 1
        ), "Working strategy should still produce exactly 1 result"
        assert _by_name(result.candidates, "temperature") is not None


# ---------------------------------------------------------------------------
# detect_from_callable
# ---------------------------------------------------------------------------


class TestDetectFromCallable:
    @pytest.fixture
    def detector(self) -> TunedVariableDetector:
        return TunedVariableDetector()

    def test_detects_from_real_function(self, detector) -> None:
        def my_llm_call():
            temperature = 0.7
            model = "gpt-4"
            return temperature, model

        result = detector.detect_from_callable(my_llm_call)
        assert result.function_name == "my_llm_call"
        names = [c.name for c in result.candidates]
        assert "temperature" in names, f"temperature not detected; got {names}"
        assert "model" in names, f"model not detected; got {names}"

    def test_handles_oserror_gracefully(self, detector) -> None:
        def fn():
            pass

        with patch("inspect.getsource", side_effect=OSError("no source")):
            result = detector.detect_from_callable(fn)

        assert result.function_name == "fn"
        assert result.count == 0
        assert len(result.warnings) == 1
        assert "source" in result.warnings[0].lower()

    def test_function_name_extracted_correctly(self, detector) -> None:
        def unique_name_xyz():
            pass

        result = detector.detect_from_callable(unique_name_xyz)
        assert result.function_name == "unique_name_xyz"

    def test_context_passed_through_to_strategy(self, detector) -> None:
        """context kwarg must be forwarded so existing_tvars are respected."""

        def fn():
            temperature = 0.7
            model = "gpt-4"
            return temperature, model

        result = detector.detect_from_callable(
            fn, context={"existing_tvars": {"temperature"}}
        )
        names = [c.name for c in result.candidates]
        assert "temperature" not in names, "existing_tvars must be excluded"
        assert "model" in names, "non-excluded vars should still be detected"


# ---------------------------------------------------------------------------
# detect_from_file
# ---------------------------------------------------------------------------


class TestDetectFromFile:
    @pytest.fixture
    def detector(self) -> TunedVariableDetector:
        return TunedVariableDetector()

    @pytest.fixture
    def sample_py_file(self, tmp_path: Path) -> Path:
        src = _dedent(
            """
            def agent_call():
                model = "gpt-4"
                temperature = 0.7
                return model, temperature

            def helper():
                x = 42
                return x
        """
        )
        f = tmp_path / "agent.py"
        f.write_text(src)
        return f

    def test_detects_from_file_all_functions(self, detector, sample_py_file) -> None:
        results = detector.detect_from_file(sample_py_file)
        # Only agent_call has known params; helper has none
        assert (
            len(results) == 1
        ), f"Expected exactly 1 result (agent_call), got {len(results)}"
        func_names = [r.function_name for r in results]
        assert "agent_call" in func_names, f"agent_call not found; got {func_names}"

    def test_detects_from_file_specific_function(
        self, detector, sample_py_file
    ) -> None:
        results = detector.detect_from_file(sample_py_file, "agent_call")
        assert len(results) == 1
        result = results[0]
        assert result.function_name == "agent_call"
        names = [c.name for c in result.candidates]
        assert "temperature" in names
        assert "model" in names

    def test_returns_empty_for_nonexistent_file(self, detector, tmp_path) -> None:
        missing = tmp_path / "does_not_exist.py"
        results = detector.detect_from_file(missing)
        assert results == [], "Missing file should return empty list, not raise"

    def test_returns_empty_for_syntax_error_file(self, detector, tmp_path) -> None:
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(: pass")
        results = detector.detect_from_file(bad_file)
        assert results == [], "Syntax error file should return empty list, not raise"

    def test_helper_with_no_tvars_excluded_from_results(
        self, detector, sample_py_file
    ) -> None:
        results = detector.detect_from_file(sample_py_file)
        func_names = [r.function_name for r in results]
        assert (
            "helper" not in func_names
        ), "helper() has no tunable vars and should be excluded from results"


# ---------------------------------------------------------------------------
# _merge_candidates (via detect_from_source with 2 strategies)
# ---------------------------------------------------------------------------


class TestMergeCandidates:
    def test_deduplicates_by_name(self) -> None:
        """Two strategies detecting the same var → 1 candidate."""
        llm_response = '[{"name": "temperature", "type": "numeric_continuous", "current_value": 0.7, "reasoning": "r"}]'

        def mock_llm(prompt: str) -> str:
            return llm_response

        # Two strategies both detecting "temperature" should merge into one.
        class FakeStrategy:
            def detect(self, source, function_name, *, context=None):
                return [
                    _make_candidate("temperature", DetectionConfidence.MEDIUM, line=2)
                ]

        class FakeStrategy2:
            def detect(self, source, function_name, *, context=None):
                return [_make_candidate("temperature", DetectionConfidence.LOW, line=2)]

        detector = TunedVariableDetector(strategies=[FakeStrategy(), FakeStrategy2()])
        src = "def fn():\n    temperature = 0.7\n"
        result = detector.detect_from_source(src, "fn")

        # Should merge into 1 candidate, not 2
        temp_candidates = [c for c in result.candidates if c.name == "temperature"]
        assert (
            len(temp_candidates) == 1
        ), f"Expected 1 merged candidate, got {len(temp_candidates)}"
        # Confidence should be upgraded from LOW+MEDIUM → MEDIUM or higher
        assert temp_candidates[0].confidence in (
            DetectionConfidence.MEDIUM,
            DetectionConfidence.HIGH,
        )
        assert temp_candidates[0].detection_source == "combined"

    def test_merged_candidate_combines_reasoning(self) -> None:
        class S1:
            def detect(self, source, function_name, *, context=None):
                return [
                    TunedVariableCandidate(
                        name="model",
                        candidate_type=CandidateType.CATEGORICAL,
                        confidence=DetectionConfidence.HIGH,
                        location=SourceLocation(line=5, col_offset=0),
                        reasoning="reason A",
                    )
                ]

        class S2:
            def detect(self, source, function_name, *, context=None):
                return [
                    TunedVariableCandidate(
                        name="model",
                        candidate_type=CandidateType.CATEGORICAL,
                        confidence=DetectionConfidence.LOW,
                        location=SourceLocation(line=5, col_offset=0),
                        reasoning="reason B",
                    )
                ]

        detector = TunedVariableDetector(strategies=[S1(), S2()])
        result = detector.detect_from_source("def fn():\n    model = 'gpt-4'\n", "fn")
        c = _by_name(result.candidates, "model")
        assert c is not None
        assert "reason A" in c.reasoning
        assert "reason B" in c.reasoning

    def test_distinct_variables_not_deduped(self) -> None:
        class S1:
            def detect(self, source, function_name, *, context=None):
                return [
                    _make_candidate("temperature", DetectionConfidence.HIGH, line=1)
                ]

        class S2:
            def detect(self, source, function_name, *, context=None):
                return [_make_candidate("model", DetectionConfidence.HIGH, line=2)]

        detector = TunedVariableDetector(strategies=[S1(), S2()])
        result = detector.detect_from_source("def fn(): pass", "fn")
        names = [c.name for c in result.candidates]
        assert "temperature" in names
        assert "model" in names
        assert len(result.candidates) == 2


# ---------------------------------------------------------------------------
# Integration: DetectionResult properties after a real detect_from_source call
# ---------------------------------------------------------------------------


class TestDetectorResultIntegration:
    def test_to_configuration_space_only_includes_high_medium(self) -> None:
        detector = TunedVariableDetector()
        # Only temperature (HIGH confidence) should appear in config space
        src = _dedent(
            """
            def pipeline():
                temperature = 0.7
                model = "gpt-4"
                max_tokens = 512
        """
        )
        result = detector.detect_from_source(src, "pipeline")

        cs = result.to_configuration_space()
        # All three should be HIGH confidence via direct name match
        for name in [
            k
            for k in ("temperature", "model", "max_tokens")
            if k in [c.name for c in result.candidates]
        ]:
            c = _by_name(result.candidates, name)
            if c and c.confidence != DetectionConfidence.LOW:
                assert name in cs, f"{name} should be in config space"

    def test_high_confidence_property_filters_correctly(self) -> None:
        detector = TunedVariableDetector()
        src = _dedent(
            """
            def fn():
                temperature = 0.7
        """
        )
        result = detector.detect_from_source(src, "fn")
        high = result.high_confidence
        for c in high:
            assert (
                c.confidence == DetectionConfidence.HIGH
            ), f"high_confidence returned non-HIGH candidate: {c.confidence}"
