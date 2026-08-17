"""Signals must be correct on the path production actually uses.

``build_example_signals`` (traigent.utils.outcome_signals) is unit-tested against
real ``ExampleResult`` objects, but the SDK's actual wire path never hands it one:
``trial_result_factory`` redacts example results to PLAIN DICTS
(``ExampleResult.to_dict()`` + ``redact_sensitive_data``) before
``metadata_helpers`` builds the per-example measures from them. A field reader
that only understands attribute access (``getattr``) silently returns ``None``
for every field on a dict, so every example in every real run would get the
identical digest of ``null`` and no evaluator-quality signal would ever
differentiate two examples.

These tests drive the REAL production pipeline -- ``build_success_result``
(trial_result_factory) followed by ``build_backend_metadata``
(metadata_helpers) -- exactly as the orchestrator calls them, rather than
calling ``build_example_signals`` directly on an ``ExampleResult``.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from traigent.api.types import ExampleResult
from traigent.config.backend_config import BackendConfig
from traigent.config.types import ExecutionMode, TraigentConfig
from traigent.core.metadata_helpers import build_backend_metadata
from traigent.core.trial_result_factory import build_success_result
from traigent.utils.outcome_signals import build_example_signals


@pytest.fixture(autouse=True)
def _project_api_key(monkeypatch: pytest.MonkeyPatch):
    """The signals are fail-closed on a missing project API key (FIX 5) --
    give the real pipeline one so these tests exercise the signal-emitting
    path, not the (separately tested) fail-closed path.
    """
    monkeypatch.setattr(
        BackendConfig, "get_api_key", classmethod(lambda cls: "wiring-test-project-key")
    )


def _example(idx: int, *, question: str, answer: str, output: str) -> ExampleResult:
    return ExampleResult(
        example_id=f"ex-{idx}",
        input_data={"question": question},
        expected_output=answer,
        actual_output=output,
        metrics={"accuracy": 1.0 if output == answer else 0.0},
        execution_time=0.05,
        success=True,
    )


def _config(*, privacy_enabled: bool = False) -> TraigentConfig:
    config = Mock(spec=TraigentConfig)
    config.execution_mode = "edge_analytics"
    config.minimal_logging = False
    config.privacy_enabled = privacy_enabled
    config.execution_mode_enum = ExecutionMode.LOCAL
    return config


def _measures_from_real_pipeline(example_results: list[ExampleResult]) -> list[dict]:
    """Drive the actual production call chain: eval_result -> TrialResult ->
    backend metadata -> per-example ``measures``."""
    eval_result = Mock()
    eval_result.metrics = {"accuracy": 1.0}
    eval_result.success_rate = 1.0
    eval_result.has_errors = False
    eval_result.outputs = [e.actual_output for e in example_results]
    eval_result.example_results = example_results
    eval_result.successful_examples = len(example_results)
    eval_result.summary_stats = None

    trial_result = build_success_result(
        trial_id="trial_signals",
        evaluation_config={"model": "gpt-4"},
        eval_result=eval_result,
        duration=1.0,
        examples_attempted=len(example_results),
        total_cost=0.01,
        optuna_trial_id=None,
    )

    metadata = build_backend_metadata(trial_result, "accuracy", _config())
    return metadata["measures"]


def test_real_pipeline_gives_each_example_a_distinct_digest_and_a_present_match() -> (
    None
):
    """The regression this PR exists to fix: distinct content -> distinct digests."""
    example_results = [
        _example(0, question="2+2?", answer="4", output="4"),
        _example(1, question="capital of France?", answer="Paris", output="Paris"),
    ]

    measures = _measures_from_real_pipeline(example_results)

    assert len(measures) == 2
    digests = [m["example_digest"] for m in measures]
    assert len(set(digests)) == 2, (
        "every example got the same example_digest -- signals were computed "
        "from the wrong object (see outcome_signals.build_example_signals)"
    )
    output_digests = [m["output_digest"] for m in measures]
    assert len(set(output_digests)) == 2

    for measure in measures:
        assert "verified_match" in measure
        assert measure["verified_match"] == 1.0


def test_real_pipeline_never_emits_the_null_example_collision_digest() -> None:
    """Negative guard: no emitted digest may equal digest(null, null).

    Before the fix, ``getattr(dict, "input_data", None)`` returns ``None`` for
    every real (dict-form) example, so every example collapsed to
    ``example_digest(None, None)``. This must be impossible to reintroduce
    silently.
    """
    null_collision_digest = build_example_signals(
        Mock(
            input_data=None,
            expected_output=None,
            actual_output=None,
            error_message=None,
        )
    )["example_digest"]

    example_results = [
        _example(0, question="2+2?", answer="4", output="4"),
        _example(1, question="capital of France?", answer="Paris", output="Paris"),
        _example(2, question="3*3?", answer="9", output="9"),
    ]

    measures = _measures_from_real_pipeline(example_results)

    for measure in measures:
        assert measure["example_digest"] != null_collision_digest
