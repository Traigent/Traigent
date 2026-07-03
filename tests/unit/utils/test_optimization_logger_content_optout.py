"""Tests for the content-logging opt-out (Traigent/Traigent#1069).

TRAIGENT_LOG_EXAMPLE_CONTENT (or `log_example_content=`) lets a user keep ids and
metrics on disk while omitting per-example query/response/expected content, without
enabling full privacy mode.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.utils.optimization_logger import (
    ENV_LOG_EXAMPLE_CONTENT,
    OptimizationLogger,
    _serialize_example_result,
    _should_log_example_content,
)

PROMPT = "SENSITIVE_PROMPT_XYZ"
EXPECTED = "SENSITIVE_EXPECTED_XYZ"


def _example() -> dict:
    return {
        "example_id": "ex1",
        "actual_output": {"query": PROMPT, "response": "the response"},
        "expected_output": EXPECTED,
        "metrics": {"accuracy": 1.0},
        "cost_usd": 0.01,
        "latency_ms": 5.0,
    }


def _logger(tmp_path: Path, **kwargs) -> OptimizationLogger:
    return OptimizationLogger(
        experiment_name="exp",
        session_id="sess12345678",
        execution_mode="local",
        base_path=tmp_path,
        buffer_size=1,
        **kwargs,
    )


def _trial_with_example() -> TrialResult:
    return TrialResult(
        trial_id="t1",
        config={"model": "gpt-4o"},
        metrics={"accuracy": 1.0},
        status=TrialStatus("completed"),
        duration=1.0,
        timestamp=datetime(2026, 6, 2, tzinfo=UTC),
        metadata={"example_results": [_example()]},
    )


# --- env switch resolution ------------------------------------------------


def test_env_default_is_on(monkeypatch):
    monkeypatch.delenv(ENV_LOG_EXAMPLE_CONTENT, raising=False)
    assert _should_log_example_content() is True


@pytest.mark.parametrize("value", ["false", "0", "off", "no", "FALSE", " Off ", ""])
def test_env_disables(monkeypatch, value):
    monkeypatch.setenv(ENV_LOG_EXAMPLE_CONTENT, value)
    assert _should_log_example_content() is False


@pytest.mark.parametrize("value", ["true", "1", "on", "yes"])
def test_env_enables(monkeypatch, value):
    monkeypatch.setenv(ENV_LOG_EXAMPLE_CONTENT, value)
    assert _should_log_example_content() is True


# --- serializer gating ----------------------------------------------------


def test_serializer_omits_content_when_disabled():
    out = _serialize_example_result(_example(), log_content=False)
    assert out["query"] is None
    assert out["response"] is None
    assert out["expected"] is None
    # ids + metrics retained
    assert out["example_id"] == "ex1"
    assert out["accuracy"] == 1.0
    assert out["cost_usd"] == 0.01


def test_serializer_keeps_content_when_enabled():
    out = _serialize_example_result(_example(), log_content=True)
    assert out["query"] == PROMPT
    assert out["expected"] == EXPECTED


# --- constructor honors env / explicit param ------------------------------


def test_constructor_reads_env(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_LOG_EXAMPLE_CONTENT, "false")
    assert _logger(tmp_path).log_example_content is False


def test_explicit_param_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_LOG_EXAMPLE_CONTENT, "false")
    assert _logger(tmp_path, log_example_content=True).log_example_content is True


# --- on-disk behavior (the issue's reproduction) --------------------------


def _all_disk_text(root: Path) -> str:
    return "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in root.rglob("*")
        if p.is_file()
    )


def test_content_absent_on_disk_when_disabled(tmp_path):
    log = _logger(tmp_path / "off", log_example_content=False)
    log.log_trial_result(_trial_with_example())
    log._flush_trial_buffer()
    disk = _all_disk_text(tmp_path / "off")
    assert PROMPT not in disk
    assert EXPECTED not in disk
    # but the example is still recorded (id present)
    assert "ex1" in disk


def test_content_present_on_disk_by_default(tmp_path):
    """Default DX behavior — content IS persisted (the gap #1069 addresses)."""
    log = _logger(tmp_path / "on", log_example_content=True)
    log.log_trial_result(_trial_with_example())
    log._flush_trial_buffer()
    assert PROMPT in _all_disk_text(tmp_path / "on")


def test_log_dir_gitignore_written(tmp_path):
    _logger(tmp_path / "gi")
    assert (
        (tmp_path / "gi" / ".gitignore")
        .read_text(encoding="utf-8")
        .strip()
        .endswith("*")
    )
