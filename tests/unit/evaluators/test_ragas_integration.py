from __future__ import annotations

import warnings

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample, SimpleScoringEvaluator
from traigent.evaluators.local import LocalEvaluator
from traigent.metrics.ragas_metrics import RAGAS_AVAILABLE

pytestmark = [
    pytest.mark.skipif(
        not RAGAS_AVAILABLE,
        reason="ragas not installed",
    ),
    pytest.mark.filterwarnings("ignore:unclosed file.*:ResourceWarning"),
]


@pytest.mark.asyncio
async def test_local_evaluator_includes_ragas_metrics() -> None:
    dataset = Dataset(
        examples=[
            EvaluationExample(
                input_data={"question": "Who wrote Hamlet?"},
                expected_output="William Shakespeare",
                metadata={
                    "retrieved_contexts": [
                        "Hamlet was written by William Shakespeare."
                    ],
                    "reference_contexts": [
                        "Hamlet was written by William Shakespeare."
                    ],
                },
            )
        ]
    )

    async def answer(question: str) -> str:
        return "William Shakespeare"

    evaluator = LocalEvaluator(
        metrics=["context_precision", "context_recall"], detailed=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        result = await evaluator.evaluate(answer, {}, dataset)

    assert pytest.approx(result.metrics["context_precision"], rel=1e-2) == 1.0
    assert pytest.approx(result.metrics["context_recall"], rel=1e-2) == 1.0


@pytest.mark.asyncio
async def test_simple_scoring_evaluator_augments_with_ragas_metrics() -> None:
    dataset = Dataset(
        examples=[
            EvaluationExample(
                input_data={"question": "Who wrote Hamlet?"},
                expected_output="William Shakespeare",
                metadata={
                    "retrieved_contexts": [
                        "Hamlet was written by William Shakespeare."
                    ],
                    "reference_contexts": [
                        "Hamlet was written by William Shakespeare."
                    ],
                },
            )
        ]
    )

    async def answer(question: str) -> str:
        return "William Shakespeare"

    evaluator = SimpleScoringEvaluator(metrics=["context_precision"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        result = await evaluator.evaluate(answer, {}, dataset)

    assert pytest.approx(result.metrics["context_precision"], rel=1e-2) == 1.0
