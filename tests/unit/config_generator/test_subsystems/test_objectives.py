"""Tests for config_generator.subsystems.objectives."""

from __future__ import annotations

import pytest

from traigent.config_generator.agent_classifier import ClassificationResult
from traigent.config_generator.subsystems.objectives import (
    _has_llm_imports,
    _normalize_weights,
    generate_objectives,
)
from traigent.config_generator.types import ObjectiveSpec, TVarSpec

LLM_SOURCE = """\
from langchain_openai import ChatOpenAI

def my_agent(query: str) -> str:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    return llm.invoke(query)
"""

GENERIC_SOURCE = """\
def compute(x: int) -> int:
    return x * 2
"""


class TestGenerateObjectives:
    def test_always_includes_accuracy(self) -> None:
        objectives = generate_objectives(GENERIC_SOURCE, [])
        names = {o.name for o in objectives}
        assert "accuracy" in names

    def test_model_tvar_adds_cost(self) -> None:
        tvars = [
            TVarSpec(
                name="model",
                range_type="Choices",
                range_kwargs={"values": ["gpt-4o"]},
            ),
        ]
        objectives = generate_objectives(GENERIC_SOURCE, tvars)
        names = {o.name for o in objectives}
        assert "cost" in names

    def test_llm_imports_add_cost_and_latency(self) -> None:
        objectives = generate_objectives(LLM_SOURCE, [])
        names = {o.name for o in objectives}
        assert "cost" in names
        assert "latency" in names

    def test_rag_classification_adds_faithfulness(self) -> None:
        classification = ClassificationResult(
            agent_type="rag", confidence=0.8, source="heuristic", reasoning="test"
        )
        objectives = generate_objectives(
            GENERIC_SOURCE, [], classification=classification
        )
        names = {o.name for o in objectives}
        assert "faithfulness" in names

    def test_classification_agent_adds_f1(self) -> None:
        classification = ClassificationResult(
            agent_type="classification",
            confidence=0.8,
            source="heuristic",
            reasoning="test",
        )
        objectives = generate_objectives(
            GENERIC_SOURCE, [], classification=classification
        )
        names = {o.name for o in objectives}
        assert "f1" in names

    def test_weights_sum_to_one(self) -> None:
        objectives = generate_objectives(LLM_SOURCE, [])
        total = sum(o.weight for o in objectives)
        assert abs(total - 1.0) < 0.01

    def test_all_objectives_have_valid_orientation(self) -> None:
        objectives = generate_objectives(LLM_SOURCE, [])
        for obj in objectives:
            assert obj.orientation in ("maximize", "minimize")


class TestHasLlmImports:
    def test_langchain(self) -> None:
        assert _has_llm_imports("from langchain import LLM") is True

    def test_openai(self) -> None:
        assert _has_llm_imports("import openai") is True

    def test_no_imports(self) -> None:
        assert _has_llm_imports("def f(): return 1") is False


class TestNormalizeWeights:
    def test_normalizes_to_one(self) -> None:
        objectives = [
            ObjectiveSpec(name="a", weight=3.0),
            ObjectiveSpec(name="b", weight=7.0),
        ]
        normalized = _normalize_weights(objectives)
        assert abs(sum(o.weight for o in normalized) - 1.0) < 0.001
        assert normalized[0].weight == pytest.approx(0.3, abs=0.001)
        assert normalized[1].weight == pytest.approx(0.7, abs=0.001)

    def test_zero_total_unchanged(self) -> None:
        objectives = [
            ObjectiveSpec(name="a", weight=0.0),
        ]
        normalized = _normalize_weights(objectives)
        assert normalized[0].weight == 0.0

    def test_preserves_names(self) -> None:
        objectives = [
            ObjectiveSpec(name="accuracy", weight=0.5),
            ObjectiveSpec(name="cost", weight=0.5, orientation="minimize"),
        ]
        normalized = _normalize_weights(objectives)
        assert normalized[0].name == "accuracy"
        assert normalized[1].name == "cost"
        assert normalized[1].orientation == "minimize"


class TestLLMEnrichment:
    def test_non_numeric_weight_handled(self) -> None:
        """LLM returning non-numeric weight should use default, not crash."""
        import json
        from unittest.mock import MagicMock

        llm = MagicMock()
        llm.complete.return_value = json.dumps(
            [
                {
                    "name": "custom_metric",
                    "orientation": "maximize",
                    "weight": "medium",
                    "reasoning": "test",
                }
            ]
        )
        objectives = generate_objectives("def f(): pass", [], llm=llm)
        custom = [o for o in objectives if o.name == "custom_metric"]
        assert len(custom) == 1
        # Weight is the fallback 0.15 after normalization (may differ slightly)
        assert custom[0].weight > 0
