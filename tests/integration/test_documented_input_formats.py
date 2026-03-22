"""Smoke tests for documented optimize() input formats."""

from __future__ import annotations

import pytest

import traigent
from traigent import Choices, Range
from traigent.api.decorators import EvaluationOptions


def _answer_question(query: str) -> str:
    answers = {
        "What is Python?": "A programming language",
        "What is 2+2?": "4",
        "Capital of France?": "Paris",
    }
    return answers[query]


class TestDocumentedInputFormats:
    """Verify documented decorator input forms run end-to-end in mock mode."""

    @pytest.fixture
    def inline_examples(self) -> list[dict[str, object]]:
        return [
            {
                "input": {"query": "What is Python?"},
                "expected": "A programming language",
            },
            {"input": {"query": "What is 2+2?"}, "expected": "4"},
            {"input": {"query": "Capital of France?"}, "expected": "Paris"},
        ]

    def test_offline_quickstart_inline_dataset_smoke(
        self, monkeypatch, inline_examples
    ) -> None:
        """The offline guide's inline dataset example should optimize successfully."""
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

        @traigent.optimize(
            eval_dataset=inline_examples,
            model=Choices(["gpt-4o-mini", "gpt-4o"]),
            temperature=Range(0.0, 1.0),
            objectives=["accuracy"],
        )
        def answer(query: str) -> str:
            return _answer_question(query)

        result = answer.optimize_sync(max_trials=3)

        assert len(result.trials) == 3
        assert result.best_config is not None

    @pytest.mark.parametrize(
        "evaluation",
        [
            pytest.param(
                {
                    "eval_dataset": [
                        {"input": {"query": "What is 2+2?"}, "expected": "4"},
                        {"input": {"query": "Capital of France?"}, "expected": "Paris"},
                    ]
                },
                id="dict-bundle",
            ),
            pytest.param(
                EvaluationOptions(
                    eval_dataset=[
                        {"input": {"query": "What is 2+2?"}, "expected": "4"},
                        {"input": {"query": "Capital of France?"}, "expected": "Paris"},
                    ]
                ),
                id="model-bundle",
            ),
        ],
    )
    def test_evaluation_bundle_inline_dataset_smoke(
        self, monkeypatch, evaluation
    ) -> None:
        """Grouped evaluation options should preserve the documented inline dataset form."""
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

        @traigent.optimize(
            evaluation=evaluation,
            configuration_space={"model": ["gpt-4o-mini", "gpt-4o"]},
            objectives=["accuracy"],
        )
        def answer(query: str) -> str:
            return _answer_question(query)

        result = answer.optimize_sync(max_trials=2)

        assert len(result.trials) == 2
        assert result.best_config is not None
