from __future__ import annotations

import os
import warnings

import pytest

from traigent.api.types import ExampleResult
from traigent.evaluators.base import EvaluationExample
from traigent.metrics.ragas_metrics import (
    RAGAS_AVAILABLE,
    RagasConfig,
    RagasConfigurationError,
    compute_ragas_metrics,
)

warnings.filterwarnings("ignore", message="unclosed file.*", category=ResourceWarning)

pytestmark = [
    pytest.mark.filterwarnings("ignore:unclosed file.*:ResourceWarning"),
]

os.environ.setdefault("RAGAS_DISABLE_ANALYTICS", "true")


@pytest.mark.skipif(
    not RAGAS_AVAILABLE,
    reason="ragas not installed",
)
def test_compute_ragas_metrics_non_llm_variants() -> None:
    example_results = [
        ExampleResult(
            example_id="example_0",
            input_data={"question": "Who wrote Hamlet?"},
            expected_output="William Shakespeare",
            actual_output="William Shakespeare",
            metrics={},
            execution_time=0.1,
            success=True,
            error_message=None,
            metadata={
                "retrieved_contexts": ["Hamlet was written by William Shakespeare."],
                "reference_contexts": ["Hamlet was written by William Shakespeare."],
            },
        )
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        results = compute_ragas_metrics(
            example_results,
            ["context_precision", "context_recall", "answer_similarity"],
        )

    assert pytest.approx(results["context_precision"], rel=1e-2) == 1.0
    assert pytest.approx(results["context_recall"], rel=1e-2) == 1.0
    assert pytest.approx(results["answer_similarity"], rel=1e-2) == 1.0


@pytest.mark.skipif(
    not RAGAS_AVAILABLE,
    reason="ragas not installed",
)
def test_compute_ragas_metrics_requires_llm() -> None:
    example_results = [
        ExampleResult(
            example_id="example_0",
            input_data={"question": "What is RAG?"},
            expected_output="Retrieval augmented generation",
            actual_output="Retrieval augmented generation",
            metrics={},
            execution_time=0.1,
            success=True,
            error_message=None,
            metadata={
                "retrieved_contexts": [
                    "RAG stands for retrieval augmented generation."
                ],
                "reference_contexts": [
                    "RAG stands for retrieval augmented generation."
                ],
            },
        )
    ]

    with pytest.raises(RagasConfigurationError):
        compute_ragas_metrics(example_results, ["answer_relevancy"])


@pytest.mark.skipif(
    not RAGAS_AVAILABLE,
    reason="ragas not installed",
)
def test_compute_ragas_metrics_builds_example_results_from_dataset() -> None:
    dataset_examples = [
        EvaluationExample(
            input_data={"question": "Who wrote Hamlet?"},
            expected_output="William Shakespeare",
            metadata={
                "retrieved_contexts": ["Hamlet was written by William Shakespeare."],
                "reference_contexts": ["Hamlet was written by William Shakespeare."],
            },
        )
    ]

    outputs = ["William Shakespeare"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        results = compute_ragas_metrics(
            example_results=[],
            metric_names=["context_precision", "context_recall"],
            dataset_examples=dataset_examples,
            outputs=outputs,
            config=RagasConfig(),
        )

    assert pytest.approx(results["context_precision"], rel=1e-2) == 1.0
    assert pytest.approx(results["context_recall"], rel=1e-2) == 1.0
