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


def test_nan_answered_row_plus_failed_rows_still_raises(stub) -> None:
    """Review: no defined judge score must fail even with agent-failed rows."""
    stub(lambda n: [math.nan] * n)
    rows = [_row(0, "answer"), _row(1, None)]
    with pytest.raises(rm.RagasConfigurationError):
        rm.compute_ragas_metrics(rows, ["faithfulness"], config=rm.RagasConfig())


def _scored(k: int, n: int):
    return lambda size: [0.8] * k + [math.nan] * (n - k)


def test_one_of_ten_defined_fails_below_default_floor(stub) -> None:
    stub(_scored(1, 10))
    rows = [_row(i, "a") for i in range(10)]
    with pytest.raises(rm.RagasCoverageError) as excinfo:
        rm.compute_ragas_metrics(rows, ["faithfulness"], config=rm.RagasConfig())
    message = str(excinfo.value)
    assert "1 of 10" in message
    assert "q0" not in message and "ref" not in message.replace("rows", "")


def test_six_of_ten_passes_and_reports_coverage_keys(stub) -> None:
    stub(_scored(6, 10))
    rows = [_row(i, "a") for i in range(10)]
    out = rm.compute_ragas_metrics(rows, ["faithfulness"], config=rm.RagasConfig())
    assert out["ragas_faithfulness_rows_scored"] == 6
    assert out["ragas_faithfulness_rows_total"] == 10
    assert isinstance(out["ragas_faithfulness_rows_scored"], int)
    assert out["faithfulness"] == pytest.approx(0.48)


def test_floor_is_configurable_and_zero_disables(stub) -> None:
    stub(_scored(1, 10))
    rows = [_row(i, "a") for i in range(10)]
    out = rm.compute_ragas_metrics(
        rows, ["faithfulness"], config=rm.RagasConfig(min_scored_share=0.0)
    )
    assert out["ragas_faithfulness_rows_scored"] == 1
    with pytest.raises(rm.RagasCoverageError):
        rm.compute_ragas_metrics(
            rows, ["faithfulness"], config=rm.RagasConfig(min_scored_share=0.2)
        )
    out = rm.compute_ragas_metrics(
        rows, ["faithfulness"], config=rm.RagasConfig(min_scored_share=0.1)
    )
    assert out["ragas_faithfulness_rows_total"] == 10


def test_agent_failed_rows_count_as_scored_in_coverage(stub) -> None:
    stub(lambda n: [0.9] + [math.nan] * (n - 1) if n > 1 else [0.9])
    rows = [_row(0, "answer")] + [_row(i, None) for i in range(1, 10)]
    out = rm.compute_ragas_metrics(rows, ["faithfulness"], config=rm.RagasConfig())
    assert out["ragas_faithfulness_rows_scored"] == 10
    assert out["ragas_faithfulness_rows_total"] == 10


def test_invalid_floor_rejected() -> None:
    with pytest.raises(rm.RagasConfigurationError):
        rm.configure_ragas_defaults(min_scored_share=1.5)


def test_compute_metrics_passes_coverage_keys_and_reserves_them(monkeypatch) -> None:
    import traigent.evaluators.base as base
    from traigent.evaluators.base import SimpleScoringEvaluator

    evaluator = SimpleScoringEvaluator(metrics=["context_precision"])
    evaluator._ragas_metric_names = {"context_precision"}
    monkeypatch.setattr(
        base,
        "compute_ragas_metrics",
        lambda **kw: {
            "context_precision": 0.5,
            "ragas_context_precision_rows_scored": 6,
            "ragas_context_precision_rows_total": 10,
        },
    )
    out = evaluator.compute_metrics(["a"], ["b"], [None])
    assert out["ragas_context_precision_rows_scored"] == 6
    assert out["ragas_context_precision_rows_total"] == 10
    reserved = evaluator._evaluator_computable_metric_names()
    assert "ragas_context_precision_rows_scored" in reserved


def test_compute_metrics_fails_closed_on_low_coverage(monkeypatch) -> None:
    import traigent.evaluators.base as base
    from traigent.evaluators.base import SimpleScoringEvaluator
    from traigent.utils.exceptions import EvaluationError

    evaluator = SimpleScoringEvaluator(metrics=["context_precision"])
    evaluator._ragas_metric_names = {"context_precision"}

    def boom(**kw):
        raise rm.RagasCoverageError("scored 1 of 10 rows")

    monkeypatch.setattr(base, "compute_ragas_metrics", boom)
    with pytest.raises(EvaluationError):
        evaluator.compute_metrics(["a"], ["b"], [None])
