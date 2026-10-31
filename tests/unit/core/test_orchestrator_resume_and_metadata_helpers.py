"""Characterization tests for helpers behind orchestrator result metadata and resume."""

from types import SimpleNamespace

import pytest

import traigent.core.orchestrator as orch_mod
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.identity.examples import result_identity_fields


def _trial(timed_out):
    return SimpleNamespace(metadata={"timed_out_examples": timed_out})


def _build_metadata(monkeypatch, trials):
    monkeypatch.setattr(orch_mod, "policy_from_config", lambda _cfg: None)
    monkeypatch.setattr(orch_mod, "is_offline_requested", lambda _policy: False)
    fake_self = SimpleNamespace(
        _warm_start_from=None,
        _function_descriptor=None,
        traigent_config=None,
        _vendor_attempt_totals=lambda: {},
        _trials=trials,
    )
    return OptimizationOrchestrator._build_result_metadata(
        fake_self, session_summary=None, safeguards_telemetry={}
    )


def test_timed_out_examples_are_summed_over_valid_trial_counts(monkeypatch):
    trials = [
        _trial(2),
        _trial(True),  # bool is not a count
        _trial(0),
        _trial(-1),
        _trial("3"),
        SimpleNamespace(metadata=None),
        _trial(3),
    ]
    metadata = _build_metadata(monkeypatch, trials)
    assert metadata["timed_out_examples"] == 5


def test_timed_out_examples_absent_when_no_trial_timed_out(monkeypatch):
    metadata = _build_metadata(monkeypatch, [_trial(0), SimpleNamespace(metadata={})])
    assert "timed_out_examples" not in metadata


@pytest.mark.parametrize("dataset", [None, Dataset(examples=[], name="empty")])
def test_resume_example_ids_none_without_examples(dataset):
    assert OptimizationOrchestrator._resume_dataset_example_ids(dataset) is None


def test_resume_example_ids_follow_example_order():
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"q": str(i)}, expected_output="x")
            for i in range(3)
        ],
        name="d",
    )
    ids = OptimizationOrchestrator._resume_dataset_example_ids(dataset)
    assert ids is not None
    assert len(ids) == 3
    assert len(set(ids)) == 3
    assert all(isinstance(i, str) for i in ids)
    # Order: position i must carry the id of example i (positional fallback or
    # content id, whichever identity mode is active).
    expected = [
        str(result_identity_fields(ex, f"example_{i}")["example_id"])
        for i, ex in enumerate(dataset.examples)
    ]
    assert ids == expected
