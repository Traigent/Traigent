"""Regression tests for opt-in effectuation behavior."""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO

from traigent.api.decorators import optimize
from traigent.api.types import ExampleResult
from traigent.config.context import ConfigurationContext
from traigent.config_generator.agent_classifier import ClassificationResult
from traigent.config_generator.catalog import load_catalog
from traigent.config_generator.subsystems.tvar_recommendations import (
    generate_recommendations,
)
from traigent.config_generator.types import AutoConfigResult
from traigent.effectuation import apply_effectuation, compile_effectuation
from traigent.evaluators.base import EvaluationExample


def test_apply_effectuation_default_disabled_returns_original_callable() -> None:
    def target() -> str:
        return "unchanged"

    assert apply_effectuation(target, {"candidate_count": (1, 3)}) is target


def test_apply_effectuation_enabled_runs_declared_self_consistency_knob() -> None:
    responses = iter(["same", "different", "same"])

    def target() -> str:
        return next(responses)

    wrapped = apply_effectuation(
        target,
        {"candidate_count": (1, 3)},
        enabled=True,
    )

    assert wrapped is not target
    with ConfigurationContext({"candidate_count": 3}):
        assert wrapped() == "same"


def test_compile_effectuation_exposes_aggregate_events() -> None:
    responses = iter(["same", "different", "same"])

    def target() -> str:
        return next(responses)

    application = compile_effectuation(
        target,
        {"candidate_count": (1, 3)},
        enabled=True,
    )

    with ConfigurationContext({"candidate_count": 3}):
        assert application.wrapped_callable() == "same"

    assert application.emit_events() == [
        {
            "strategy": "self_consistency",
            "knob": "candidate_count",
            "n": 3,
            "calls": 3,
            "unique_outputs": 2,
            "aggregator": "majority_vote",
            "passthrough": False,
        }
    ]


def test_apply_effectuation_enabled_skips_manual_structural_knobs() -> None:
    calls: list[int] = []

    def target() -> str:
        calls.append(1)
        return "single"

    wrapped = apply_effectuation(
        target,
        {"schema_context": ["none", "full_ddl_fk"]},
        enabled=True,
    )

    assert wrapped is target
    assert wrapped() == "single"
    assert len(calls) == 1


def test_catalog_executable_and_manual_statuses_are_honest() -> None:
    entries = load_catalog()
    by_name = {entry["name"]: entry for entry in entries}

    assert by_name["candidate_count"]["effectuation_status"] == "executable"
    assert by_name["candidate_count"]["effectuation_strategy"] == "self_consistency"

    manual_names = {
        entry["name"]
        for entry in entries
        if entry["effectuation_status"] == "manual_guidance"
    }
    assert manual_names == {
        "citation_policy",
        "compression_ratio",
        "context_order",
        "context_selection_policy",
        "edit_granularity",
        "retrieval_k",
        "schema_context",
        "evidence_usage",
        "file_view_window",
        "fewshot_selector",
        "generation_path",
        "fewshot_k",
        "patch_review_mode",
        "repair_policy",
        "repo_context_strategy",
        "summary_style",
        "test_selection_strategy",
    }


def test_generate_config_json_recommendation_keys_remain_unchanged() -> None:
    from traigent.cli.generate_config_command import _output_json

    classification = ClassificationResult(
        agent_type="code_gen",
        confidence=0.9,
        source="heuristic",
        reasoning="test",
    )
    rec = next(
        r
        for r in generate_recommendations([], classification=classification)
        if r.name == "candidate_count"
    )

    buf = StringIO()
    with redirect_stdout(buf):
        _output_json(AutoConfigResult(recommendations=(rec,)))

    data = json.loads(buf.getvalue())
    assert set(data["recommendations"][0]) == {
        "name",
        "range_code",
        "category",
        "impact",
        "reasoning",
    }


def test_optimize_effectuation_defaults_off() -> None:
    calls: list[int] = []

    async def evaluator(func, config, example):
        output = func()
        return ExampleResult(
            example_id="example-1",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=output,
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
        )

    @optimize(
        configuration_space={"candidate_count": [3]},
        evaluation={
            "eval_dataset": [
                EvaluationExample(input_data={}, expected_output="answer"),
            ],
            "custom_evaluator": evaluator,
        },
        objectives=["accuracy"],
        algorithm="grid",
        max_trials=1,
    )
    def target() -> str:
        calls.append(1)
        return "answer"

    result = target.optimize_sync(cost_approved=True, progress_bar=False)

    assert calls == [1]
    assert "effectuation_events" not in (result.trials[0].metadata or {})


def test_optimize_effectuation_opt_in_emits_trial_metadata() -> None:
    responses = iter(["answer", "different", "answer"])

    async def evaluator(func, config, example):
        output = func()
        return ExampleResult(
            example_id="example-1",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=output,
            metrics={"accuracy": 1.0 if output == "answer" else 0.0},
            execution_time=0.0,
            success=True,
        )

    @optimize(
        configuration_space={"candidate_count": [3]},
        evaluation={
            "eval_dataset": [
                EvaluationExample(input_data={}, expected_output="answer"),
            ],
            "custom_evaluator": evaluator,
        },
        injection={"effectuation": True},
        objectives=["accuracy"],
        algorithm="grid",
        max_trials=1,
    )
    def target() -> str:
        return next(responses)

    result = target.optimize_sync(cost_approved=True, progress_bar=False)

    events = result.trials[0].metadata["effectuation_events"]
    assert events == [
        {
            "strategy": "self_consistency",
            "knob": "candidate_count",
            "n": 3,
            "calls": 3,
            "unique_outputs": 2,
            "aggregator": "majority_vote",
            "passthrough": False,
        }
    ]
