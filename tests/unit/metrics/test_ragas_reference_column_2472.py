"""#2472: RAGAS ``column_map["reference"]`` is applied, and a row without
``expected_output`` is kept for metrics that do not need ``reference``.

ragas-free: ``SingleTurnSample`` is stubbed to record the fields Traigent
would send to RAGAS.
"""

from __future__ import annotations

from typing import Any

import pytest

import traigent.metrics.ragas_metrics as rm
from traigent.api.types import ExampleResult


class _Sample:
    def __init__(self, **fields: Any) -> None:
        self.fields = fields


@pytest.fixture(autouse=True)
def _stub_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rm, "SingleTurnSample", _Sample)


def _row(expected: Any) -> ExampleResult:
    return ExampleResult(
        example_id="ex0",
        input_data={"prompt": "Summarize the benefits of RAG."},
        expected_output=expected,
        actual_output="MODEL_RESPONSE",
        metrics={},
        execution_time=0.0,
        success=True,
        metadata={
            "reference_answer": "MAPPED_REFERENCE_ANSWER",
            "gold_contexts": ["ctx"],
            "retrieved_contexts": ["ctx"],
        },
    )


def test_mapped_reference_wins_over_expected_output() -> None:
    config = rm.RagasConfig(column_map={"reference": "reference_answer"})
    samples = rm._prepare_samples(
        [_row("EXPECTED_OUTPUT_TEXT")],
        config=config,
        required_columns={"reference", "response"},
    )
    assert [s.fields["reference"] for s in samples] == ["MAPPED_REFERENCE_ANSWER"]


def test_unmapped_reference_still_uses_expected_output() -> None:
    samples = rm._prepare_samples(
        [_row("EXPECTED_OUTPUT_TEXT")],
        config=rm.RagasConfig(),
        required_columns={"reference", "response"},
    )
    assert [s.fields["reference"] for s in samples] == ["EXPECTED_OUTPUT_TEXT"]


def test_row_without_reference_kept_for_context_only_metric() -> None:
    config = rm.RagasConfig(column_map={"reference_contexts": "gold_contexts"})
    samples = rm._prepare_samples(
        [_row(None)],
        config=config,
        required_columns={"retrieved_contexts", "reference_contexts"},
    )
    assert len(samples) == 1
    assert samples[0].fields["reference"] is None


def test_row_without_reference_dropped_when_reference_required() -> None:
    samples = rm._prepare_samples(
        [_row(None)],
        config=rm.RagasConfig(),
        required_columns={"reference", "response"},
    )
    assert samples == []
