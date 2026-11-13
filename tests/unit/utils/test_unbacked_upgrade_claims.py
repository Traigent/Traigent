"""Upgrade hints and comparisons must not state figures the code cannot back."""

from __future__ import annotations

import re
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from traigent.config.types import ExecutionMode, TraigentConfig
from traigent.utils.incentives import IncentiveManager
from traigent.utils.local_analytics import LocalAnalytics

_UNBACKED = re.compile(
    r"\d+(?:\s*-\s*\d+)?\s*x\b"  # speed-up multipliers such as 10x or 3-5x
    r"|\d+\s*-\s*\d+\s*%"  # unsourced percentage ranges such as 60-80%
    r"|unlimited"
    r"|20 trial"
    r"|higher trial allowances"
    r"|\bup to \d+\s*%",
    re.IGNORECASE,
)


@pytest.fixture
def storage_dir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def config(storage_dir: str) -> TraigentConfig:
    return TraigentConfig(
        execution_mode=ExecutionMode.LOCAL.value,
        enable_usage_analytics=True,
        local_storage_path=storage_dir,
    )


def _flatten(value: object) -> list[str]:
    if isinstance(value, dict):
        return [t for v in value.values() for t in _flatten(v)]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in _flatten(v)]
    return [str(value)]


def _assert_clean(text: str) -> None:
    match = _UNBACKED.search(text)
    assert match is None, f"unbacked claim {match.group(0)!r} in: {text!r}"


def test_achievement_banners_carry_no_unbacked_claims(config: TraigentConfig) -> None:
    for name in (
        "first_optimization",
        "optimization_explorer",
        "efficiency_expert",
        "optimization_master",
    ):
        manager = IncentiveManager(config)
        message = manager.show_achievement_unlock(name)
        assert message
        _assert_clean(message)


@pytest.mark.parametrize("sessions", [3, 10, 25])
def test_contextual_hints_carry_no_unbacked_claims(
    config: TraigentConfig, sessions: int
) -> None:
    manager = IncentiveManager(config)
    manager._state["completed_sessions"] = sessions
    manager._state["last_hint"] = None
    hint = manager.get_contextual_hint(
        "storage_info" if sessions >= 10 else "session_complete"
    )
    assert hint
    _assert_clean(hint)


def test_context_sensitive_hints_carry_no_unbacked_claims(
    config: TraigentConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    for context in (
        "large_config_space",
        "multiple_objectives",
        "slow_evaluation",
        "high_trial_count",
    ):
        manager = IncentiveManager(config)
        manager._state["last_hint"] = None
        manager._state["completed_sessions"] = 3
        with patch.object(manager, "should_show_hint", return_value=True):
            manager.show_context_sensitive_hint(context)
    _assert_clean(capsys.readouterr().out)


def test_local_vs_cloud_table_has_no_invented_local_cap(
    config: TraigentConfig,
) -> None:
    table = IncentiveManager(config).get_local_vs_cloud_comparison()
    _assert_clean(table)
    assert "│ 20 " not in table
    parallel = [ln for ln in table.splitlines() if "Parallel" in ln]
    assert not parallel or "❌" not in parallel[0]  # local runs support parallelism


def test_value_proposition_has_no_unbacked_numbers(config: TraigentConfig) -> None:
    manager = IncentiveManager(config)
    manager._state["completed_sessions"] = 10
    manager._state["total_trials"] = 100
    manager.analytics = None
    props = manager.get_upgrade_value_proposition()
    for text in _flatten(props):
        _assert_clean(text)
    assert "hours" not in props["time_savings"]["current_time_investment"]


def test_analytics_cloud_benefits_have_no_unbacked_claims(
    config: TraigentConfig, storage_dir: str
) -> None:
    analytics = LocalAnalytics(config)
    session_id = analytics.storage.create_session(
        function_name="f", optimization_config={"search_space": {"a": [1, 2]}}
    )
    analytics.storage.add_trial_result(session_id, {"a": 1}, 0.5)
    analytics.storage.finalize_session(session_id, "completed")
    data = analytics.get_cloud_incentive_data()
    assert data
    for text in _flatten(data["cloud_benefits"]):
        _assert_clean(text)


def test_high_trial_message_has_no_unbacked_percentage(config: TraigentConfig) -> None:
    analytics = LocalAnalytics(config)
    message = analytics._generate_personalized_message(
        {"total_sessions": 8, "total_trials": 150, "avg_improvement_percent": None}
    )
    assert "150 trials" in message
    _assert_clean(message)


def test_interactive_doc_has_no_unsourced_savings_figure() -> None:
    doc = (
        Path(__file__).resolve().parents[3]
        / "docs/user-guide/interactive_optimization.md"
    )
    text = doc.read_text(encoding="utf-8")
    assert "60-80%" not in text


@pytest.mark.parametrize(
    "stats",
    [
        {"total_sessions": 1, "total_trials": 3, "avg_improvement_percent": None},
        {"total_sessions": 6, "total_trials": 50, "avg_improvement_percent": None},
        {"total_sessions": 25, "total_trials": 300, "avg_improvement_percent": 5},
        {"total_sessions": 6, "total_trials": 50, "avg_improvement_percent": 30.0},
    ],
)
def test_every_personalized_message_branch_is_clean(
    config: TraigentConfig, stats: dict
) -> None:
    _assert_clean(LocalAnalytics(config)._generate_personalized_message(stats))
