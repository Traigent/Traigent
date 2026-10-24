"""Characterization tests for OptimizedFunction._dataset_from_source path lists."""

from types import SimpleNamespace

import pytest

from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import ConfigurationError


def _dataset(*labels: str) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"q": label}, expected_output=label)
            for label in labels
        ],
        name="part",
    )


def _resolve(source):
    owner = SimpleNamespace(
        _combine_dataset_paths=OptimizedFunction._combine_dataset_paths
    )
    return OptimizedFunction._dataset_from_source(owner, source)


def test_path_list_combines_examples_in_order(monkeypatch):
    parts = {"a.jsonl": _dataset("a1", "a2"), "b.jsonl": _dataset("b1")}
    monkeypatch.setattr(Dataset, "from_jsonl", classmethod(lambda c, p: parts[p]))

    combined = _resolve(["a.jsonl", "b.jsonl"])

    assert [e.expected_output for e in combined.examples] == ["a1", "a2", "b1"]
    assert combined.name == "combined_dataset"
    assert combined.description == "Combined dataset from 2 files"


def test_path_list_failure_names_the_failing_path(monkeypatch):
    def _load(_cls, path):
        if path == "bad.jsonl":
            raise ValueError("unreadable")
        return _dataset("ok")

    monkeypatch.setattr(Dataset, "from_jsonl", classmethod(_load))

    with pytest.raises(ConfigurationError, match="bad.jsonl: unreadable"):
        _resolve(["ok.jsonl", "bad.jsonl"])


class _Trial:
    def __init__(self, metadata=None):
        self.trial_id = "t"
        self.duration = 0.0
        self.metadata = metadata or {}


def test_comparison_end_guard_noop_without_trials(monkeypatch):
    import traigent.core.optimized_function as of_mod

    calls = []
    monkeypatch.setattr(
        of_mod, "_guard_cost_objective_without_usage", lambda *a: calls.append("g")
    )
    monkeypatch.setattr(
        of_mod, "_warn_cost_objective_partial_coverage", lambda *a: calls.append("w")
    )
    owner = SimpleNamespace(objectives=["accuracy"])

    OptimizedFunction._apply_comparison_end_guards(owner, [], None)

    assert calls == []


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [("no_usage_captured", "no_usage_captured"), ("timeout", None)],
)
def test_comparison_end_guard_marks_no_usage_stop(monkeypatch, stop_reason, expected):
    import traigent.core.optimized_function as of_mod

    seen = []
    monkeypatch.setattr(
        of_mod,
        "_guard_cost_objective_without_usage",
        lambda view, _objectives: seen.append(
            (list(view.trials), getattr(view, "stop_reason", None))
        ),
    )
    monkeypatch.setattr(
        of_mod, "_warn_cost_objective_partial_coverage", lambda *a: None
    )
    trials = [_Trial(), _Trial()]
    owner = SimpleNamespace(
        objectives=["accuracy"],
        _single_trial_result=lambda trial: SimpleNamespace(
            trials=[trial], stop_reason=None
        ),
    )

    OptimizedFunction._apply_comparison_end_guards(owner, trials, stop_reason)

    assert seen == [(trials, expected)]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_close_comparison_evaluator_awaits_or_calls_close(is_async):
    closed = []

    if is_async:

        async def close():
            closed.append(1)

    else:

        def close():
            closed.append(1)

    await OptimizedFunction._close_comparison_evaluator(SimpleNamespace(close=close))
    await OptimizedFunction._close_comparison_evaluator(None)
    await OptimizedFunction._close_comparison_evaluator(SimpleNamespace())

    assert closed == [1]


@pytest.mark.asyncio
async def test_close_comparison_evaluator_swallows_close_failure():
    def close():
        raise RuntimeError("boom")

    await OptimizedFunction._close_comparison_evaluator(SimpleNamespace(close=close))
