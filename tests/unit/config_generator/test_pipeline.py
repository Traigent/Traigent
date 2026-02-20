"""Tests for config_generator.pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

from traigent.config_generator.pipeline import ALL_SUBSYSTEMS, ConfigGeneratorPipeline
from traigent.config_generator.types import AutoConfigResult
from traigent.tuned_variables.detection_types import (
    CandidateType,
    DetectionConfidence,
    DetectionResult,
    SourceLocation,
    TunedVariableCandidate,
)


def _make_detection_result(
    names: list[str] | None = None,
) -> DetectionResult:
    """Build a minimal DetectionResult."""
    if names is None:
        names = ["temperature"]
    candidates = tuple(
        TunedVariableCandidate(
            name=n,
            canonical_name=n,
            candidate_type=CandidateType.NUMERIC_CONTINUOUS,
            confidence=DetectionConfidence.HIGH,
            location=SourceLocation(line=1, col_offset=0),
        )
        for n in names
    )
    return DetectionResult(
        function_name="my_func",
        candidates=candidates,
    )


class TestConfigGeneratorPipeline:
    def test_returns_auto_config_result(self) -> None:
        pipeline = ConfigGeneratorPipeline()
        result = pipeline.generate([_make_detection_result()])
        assert isinstance(result, AutoConfigResult)

    def test_tvars_populated(self) -> None:
        pipeline = ConfigGeneratorPipeline()
        result = pipeline.generate([_make_detection_result(["temperature"])])
        assert len(result.tvars) >= 1
        names = {t.name for t in result.tvars}
        assert "temperature" in names

    def test_objectives_always_have_accuracy(self) -> None:
        pipeline = ConfigGeneratorPipeline()
        result = pipeline.generate(
            [_make_detection_result()],
            source_code="def f(): pass",
        )
        obj_names = {o.name for o in result.objectives}
        assert "accuracy" in obj_names

    def test_agent_type_classified(self) -> None:
        rag_code = (
            "from langchain.vectorstores import Chroma\n"
            "def my_func():\n"
            "    retriever = Chroma()\n"
            "    docs = retriever.similarity_search(query)\n"
        )
        pipeline = ConfigGeneratorPipeline()
        result = pipeline.generate(
            [_make_detection_result()],
            source_code=rag_code,
        )
        assert result.agent_type == "rag"

    def test_safety_constraints_populated_for_rag(self) -> None:
        rag_code = (
            "from langchain.vectorstores import Chroma\n"
            "def my_func():\n"
            "    retriever = Chroma()\n"
            "    docs = retriever.similarity_search(query)\n"
        )
        pipeline = ConfigGeneratorPipeline()
        result = pipeline.generate(
            [_make_detection_result()],
            source_code=rag_code,
        )
        assert len(result.safety_constraints) >= 1

    def test_structural_constraints_for_matching_tvars(self) -> None:
        pipeline = ConfigGeneratorPipeline()
        result = pipeline.generate(
            [_make_detection_result(["temperature", "top_p"])],
        )
        assert len(result.structural_constraints) >= 1

    def test_benchmarks_populated(self) -> None:
        pipeline = ConfigGeneratorPipeline()
        result = pipeline.generate(
            [_make_detection_result()],
            source_code="def f(): pass",
        )
        assert len(result.benchmarks) >= 1

    def test_recommendations_populated(self) -> None:
        pipeline = ConfigGeneratorPipeline()
        result = pipeline.generate(
            [_make_detection_result()],
            source_code="def f(): pass",
        )
        # general_llm should recommend at least prompting_strategy
        names = {r.name for r in result.recommendations}
        assert "prompting_strategy" in names

    def test_subsystem_filter(self) -> None:
        pipeline = ConfigGeneratorPipeline(subsystems=frozenset({"tvars"}))
        result = pipeline.generate([_make_detection_result()])
        assert len(result.tvars) >= 1
        assert len(result.objectives) == 0
        assert len(result.safety_constraints) == 0
        assert len(result.structural_constraints) == 0
        assert len(result.benchmarks) == 0
        assert len(result.recommendations) == 0

    def test_empty_detection_results(self) -> None:
        empty_result = [DetectionResult(function_name="empty", candidates=())]
        pipeline = ConfigGeneratorPipeline()
        result = pipeline.generate(empty_result, source_code="def f(): pass")
        assert isinstance(result, AutoConfigResult)
        assert len(result.tvars) == 0

    def test_no_source_code_skips_classification(self) -> None:
        pipeline = ConfigGeneratorPipeline()
        result = pipeline.generate([_make_detection_result()])
        assert result.agent_type is None

    def test_multiple_detection_results(self) -> None:
        r1 = _make_detection_result(["temperature"])
        r2 = _make_detection_result(["max_tokens"])
        pipeline = ConfigGeneratorPipeline()
        result = pipeline.generate([r1, r2])
        names = {t.name for t in result.tvars}
        assert "temperature" in names
        assert "max_tokens" in names

    def test_all_subsystems_constant(self) -> None:
        assert "tvars" in ALL_SUBSYSTEMS
        assert "objectives" in ALL_SUBSYSTEMS
        assert "safety" in ALL_SUBSYSTEMS
        assert "structural" in ALL_SUBSYSTEMS
        assert "benchmarks" in ALL_SUBSYSTEMS
        assert "recommendations" in ALL_SUBSYSTEMS

    def test_classification_skipped_when_no_consumers(self) -> None:
        """Classification should not run when only tvars/structural are enabled."""
        llm = MagicMock()
        llm.complete.return_value = "[]"
        llm.calls_made = 0
        llm._spent_usd = 0.0
        pipeline = ConfigGeneratorPipeline(
            llm=llm,
            subsystems=frozenset({"tvars", "structural"}),
        )
        result = pipeline.generate(
            [_make_detection_result()],
            source_code="def f(): pass",
        )
        # Classification never ran, so agent_type should be None
        assert result.agent_type is None

    def test_llm_stats_collected(self) -> None:
        llm = MagicMock()
        # Return valid JSON dict (for classify_agent) or array (for other subsystems)
        llm.complete.return_value = '{"agent_type": "general_llm", "confidence": 0.5}'
        llm.calls_made = 5
        llm._spent_usd = 0.03
        pipeline = ConfigGeneratorPipeline(llm=llm)
        result = pipeline.generate(
            [_make_detection_result()],
            source_code="def f(): pass",
        )
        assert result.llm_calls_made == 5
        assert abs(result.llm_cost_usd - 0.03) < 1e-9
