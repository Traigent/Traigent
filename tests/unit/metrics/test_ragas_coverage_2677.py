"""RAGAS aggregation must not hide failed rows from the denominator (#2677)."""

from __future__ import annotations

import math

import pytest

import traigent.metrics.ragas_metrics as rm
from traigent.api.types import ExampleResult


class _Sample:
    def __init__(self, **kw):
        self.fields = kw


class _Metric:
    name = "faithfulness"
    required_columns = {"SINGLE_TURN": {"response", "reference"}}


def _row(i, out):
    return ExampleResult(
        example_id=f"ex{i}",
        input_data={"question": f"q{i}"},
        expected_output="ref",
        actual_output=out,
        metrics={},
        execution_time=0.0,
        success=out is not None,
    )


@pytest.fixture
def stub(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(rm, "RAGAS_AVAILABLE", True)
    monkeypatch.setattr(rm, "SingleTurnSample", _Sample)
    monkeypatch.setattr(rm, "EvaluationDataset", list)
    monkeypatch.setattr(rm, "_build_metric", lambda name, config: _Metric())

    def install(values_fn):
        def fake_evaluate(dataset, metrics, **kw):
            seen["rows"] = len(dataset)
            seen["kwargs"] = kw
            return {"faithfulness": values_fn(len(dataset))}

        monkeypatch.setattr(rm, "ragas_evaluate", fake_evaluate)

    install.seen = seen
    return install


def test_agent_failed_rows_count_as_zero_over_full_denominator(stub) -> None:
    stub(lambda n: [0.9] + [math.nan] * (n - 1) if n > 1 else [0.9])
    rows = [_row(0, "answer")] + [_row(i, None) for i in range(1, 10)]
    out = rm.compute_ragas_metrics(rows, ["faithfulness"], config=rm.RagasConfig())
    assert out["faithfulness"] == pytest.approx(0.09)


def test_judge_exceptions_are_not_swallowed_as_nan(stub) -> None:
    stub(lambda n: [0.5] * n)
    rm.compute_ragas_metrics([_row(0, "a")], ["faithfulness"], config=rm.RagasConfig())
    assert stub.seen["kwargs"].get("raise_exceptions") is True


def test_all_nan_raises_instead_of_returning_nan(stub) -> None:
    stub(lambda n: [math.nan] * n)
    rows = [_row(i, "a") for i in range(3)]
    with pytest.raises(rm.RagasConfigurationError):
        rm.compute_ragas_metrics(rows, ["faithfulness"], config=rm.RagasConfig())


def test_fully_scored_batch_unchanged(stub) -> None:
    stub(lambda n: [0.5, 0.7, 0.9, 1.0][:n])
    rows = [_row(i, "a") for i in range(4)]
    out = rm.compute_ragas_metrics(rows, ["faithfulness"], config=rm.RagasConfig())
    assert out["faithfulness"] == pytest.approx(0.775)
