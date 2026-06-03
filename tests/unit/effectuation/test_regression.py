"""Regression tests for opt-in effectuation behavior."""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO

from traigent.config.context import ConfigurationContext
from traigent.config_generator.agent_classifier import ClassificationResult
from traigent.config_generator.catalog import load_catalog
from traigent.config_generator.subsystems.tvar_recommendations import (
    generate_recommendations,
)
from traigent.config_generator.types import AutoConfigResult
from traigent.effectuation import apply_effectuation


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
        "retrieval_k",
        "schema_context",
        "evidence_usage",
        "fewshot_selector",
        "generation_path",
        "fewshot_k",
        "repair_policy",
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
