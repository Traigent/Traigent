"""Regression tests for #2799/#2831: generate_config must never emit empty,
reversed or degenerate tuned-variable ranges, and LLM replies of the wrong
shape must be rejected instead of crashing."""

from __future__ import annotations

from pathlib import Path

import pytest

from traigent.config_generator import generate_config
from traigent.config_generator.agent_classifier import _llm_classify
from traigent.config_generator.subsystems.tvar_ranges import (
    _heuristic_from_value,
    _parse_llm_range_response,
)
from traigent.config_generator.types import TVarSpec, tvar_spec_problem
from traigent.tuned_variables.detection_strategies import _suggest_range
from traigent.tuned_variables.detection_types import CandidateType

_SOURCE = """
import litellm

def predict(q):
    return litellm.completion(
        model="gpt-4o-mini", messages=[], max_tokens=1800, {kwarg}
    )
"""


def _generate(tmp_path: Path, kwarg: str):
    path = tmp_path / "agent.py"
    path.write_text(_SOURCE.format(kwarg=kwarg))
    return generate_config(path, function_name="predict", enrich=False)


def _assert_all_valid(result) -> None:
    for tvar in result.tvars:
        assert tvar_spec_problem(tvar) is None, tvar


@pytest.mark.parametrize(
    "kwarg", ["num_retries=0", "timeout=0", "max_completion_tokens=0"]
)
def test_zero_literal_never_emits_reversed_range(tmp_path: Path, kwarg: str) -> None:
    result = _generate(tmp_path, kwarg)
    _assert_all_valid(result)
    for tvar in result.tvars:
        if tvar.range_type == "IntRange":
            assert tvar.range_kwargs["low"] < tvar.range_kwargs["high"]


def test_tiny_float_literal_never_emits_empty_range(tmp_path: Path) -> None:
    result = _generate(tmp_path, "request_timeout=0.000001")
    _assert_all_valid(result)


@pytest.mark.parametrize("value", [0, -3])
def test_suggest_range_skips_non_positive_int(value: int) -> None:
    assert _suggest_range(None, CandidateType.NUMERIC_INTEGER, value) is None
    assert _heuristic_from_value("x", value) is None


@pytest.mark.parametrize("value", [-0.5, -2e-06])
def test_negative_float_skipped(value: float) -> None:
    assert _suggest_range(None, CandidateType.NUMERIC_CONTINUOUS, value) is None
    assert _heuristic_from_value("x", value) is None


def test_tiny_float_bracket_is_ordered() -> None:
    for fn in (
        lambda v: _suggest_range(None, CandidateType.NUMERIC_CONTINUOUS, v).kwargs,
        lambda v: _heuristic_from_value("x", v).range_kwargs,
    ):
        kw = fn(1e-6)
        assert 0 < kw["low"] < kw["high"]


def test_pipeline_drops_invalid_spec_with_warning(tmp_path: Path, monkeypatch) -> None:
    bad = TVarSpec("k", "IntRange", {"low": 1, "high": 0}, source="detection")
    monkeypatch.setattr(
        "traigent.config_generator.pipeline.generate_tvar_specs",
        lambda *a, **k: [bad],
    )
    result = _generate(tmp_path, "num_retries=3")
    assert result.tvars == ()
    assert any("Dropped tuned variable 'k'" in w for w in result.warnings)


@pytest.mark.parametrize(
    "spec,fragment",
    [
        (TVarSpec("a", "IntRange", {"low": 1, "high": 0}), "reversed"),
        (TVarSpec("a", "Range", {"low": 0.0, "high": 0.0}), "reversed"),
        (TVarSpec("a", "LogRange", {"low": 0.0, "high": 1.0}), "positive"),
        (TVarSpec("a", "Choices", {"values": []}), "at least two"),
        (TVarSpec("a", "Choices", {"values": ["x"]}), "at least two"),
        (TVarSpec("a", "Range", {"low": "a", "high": 1}), "numbers"),
    ],
)
def test_tvar_spec_problem_rejects(spec: TVarSpec, fragment: str) -> None:
    assert fragment in (tvar_spec_problem(spec) or "")


def test_tvar_spec_problem_accepts_valid() -> None:
    assert tvar_spec_problem(TVarSpec("a", "IntRange", {"low": 1, "high": 2})) is None
    assert tvar_spec_problem(TVarSpec("a", "Choices", {"values": [1, 2]})) is None


@pytest.mark.parametrize(
    "reply",
    [
        '{"range_type": "IntRange", "kwargs": {"low": 5, "high": 1}}',
        '{"range_type": "Choices", "kwargs": {"values": []}}',
        "[1]",
        '"x"',
        "null",
        "42",
    ],
)
def test_llm_range_reply_rejected_without_crash(reply: str) -> None:
    assert _parse_llm_range_response("k", reply) is None


def test_llm_range_reply_valid_still_accepted() -> None:
    spec = _parse_llm_range_response(
        "k", '{"range_type": "IntRange", "kwargs": {"low": 1, "high": 5}}'
    )
    assert spec is not None and spec.range_kwargs == {"low": 1, "high": 5}


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        return self._reply


@pytest.mark.parametrize("reply", ["[1]", '"x"', "null", "42"])
def test_agent_classifier_non_dict_reply_returns_none(reply: str) -> None:
    assert _llm_classify("def f(): pass", _FakeLLM(reply)) is None  # type: ignore[arg-type]
