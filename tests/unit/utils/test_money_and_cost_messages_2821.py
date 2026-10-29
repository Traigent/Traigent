"""#2821: sub-cent costs must not print as $0.00/$0.01; the results table must
render cost and latency metrics with their own units, never as percentages."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from traigent.core.cost_enforcement import CostEnforcer, CostEnforcerConfig
from traigent.utils.money import format_usd
from traigent.utils.results_table import _render_metric_cell, print_results_table


@pytest.fixture
def unapproved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRAIGENT_COST_APPROVED", raising=False)
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))


@pytest.mark.parametrize(
    ("amount", "expected"),
    [
        (0.0, "$0.00"),
        (0.004, "$0.004"),
        (0.0051, "$0.0051"),
        (0.0104, "$0.0104"),
        (0.01, "$0.01"),
        (0.5, "$0.50"),
        (14.123, "$14.12"),
        (102.74, "$102.74"),
        (0.00004, "$0.00004"),
        (None, "unknown"),
    ],
)
def test_format_usd(amount: float, expected: str) -> None:
    assert format_usd(amount) == expected


def test_sub_cent_limit_refusal_is_not_self_contradictory(
    unapproved: None, capsys: pytest.CaptureFixture[str]
) -> None:
    enforcer = CostEnforcer(CostEnforcerConfig(limit=0.004))
    assert enforcer.check_and_approve(0.0051) is False
    err = capsys.readouterr().err
    assert "$0.0051" in err
    assert "limit $0.004" in err


def test_estimate_just_over_a_cent_limit_shows_both_values(
    unapproved: None, capsys: pytest.CaptureFixture[str]
) -> None:
    enforcer = CostEnforcer(CostEnforcerConfig(limit=0.01))
    assert enforcer.check_and_approve(0.0104) is False
    err = capsys.readouterr().err
    assert "$0.0104 exceeds limit $0.01." in err


def test_cost_and_latency_cells_use_their_units() -> None:
    assert _render_metric_cell("total_cost", 0.0123) == "$0.01230"
    assert _render_metric_cell("cost_per_example_mean", 0.0004) == "$0.00040"
    assert _render_metric_cell("response_time_ms", 850.0) == "850ms"
    assert _render_metric_cell("duration", 1.5) == "1.50s"
    assert _render_metric_cell("accuracy", 0.9) == "90.0%"


def test_missing_latency_renders_n_a_not_zero() -> None:
    assert _render_metric_cell("latency", None) == "n/a"
    assert _render_metric_cell("response_time_ms", None) == "n/a"


def test_results_table_prints_cost_and_latency_without_percent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def trial(model: str, acc: float, cost: float, ms: float, tid: str):
        return SimpleNamespace(
            config={"model": model},
            metrics={"accuracy": acc, "total_cost": cost, "response_time_ms": ms},
            metadata={"successful_examples": 5, "examples_attempted": 5},
            trial_id=tid,
            status="completed",
            is_successful=True,
        )

    trials = [
        trial("big", 0.9, 0.0123, 850.0, "t1"),
        trial("mini", 0.8, 0.0004, 120.0, "t2"),
    ]
    results = MagicMock()
    results.trials = trials
    results.metadata = {}
    results.best_trial_id = None
    results.best_config = trials[0].config
    results.best_score = 0.9
    results.calculate_weighted_scores.return_value = {
        "best_weighted_config": trials[0].config,
        "weighted_scores": [],
    }
    print_results_table(
        results,
        config_space={"model": ["big", "mini"]},
        objectives=["accuracy", "total_cost", "response_time_ms"],
    )
    out = capsys.readouterr().out
    assert "$0.0123" in out and "$0.0004" in out
    assert "850ms" in out and "120ms" in out
    assert "850.0%" not in out and "1.2%" not in out
