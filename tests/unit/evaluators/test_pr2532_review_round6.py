"""Round-6 sol findings on merged PR 2532 (owner ruling A, unknown tracked cost).

Scenario: one priced gpt-4o-mini call plus one unpriced call, with custom
metrics cost=100 and total_cost=200. No finite custom cost may survive in
returned rows, progress payloads (copied at call time) or nested summary stats.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from traigent.evaluators.base import SimpleScoringEvaluator
from traigent.evaluators.local import LocalEvaluator

from tests.unit.evaluators.test_pr2532_review_round5 import (
    _custom_cost_functions,
    _mixed_func,
    _two_examples,
)

_FLOOR = 0.0000135


def _no_custom(metrics: dict, label: str) -> None:
    for key, custom in (("cost", 100.0), ("total_cost", 200.0)):
        value = metrics.get(key)
        assert value != custom, f"{label}: {key}={value}"
        assert value is None or value == pytest.approx(_FLOOR), f"{label}: {key}"


async def test_local_detailed_rows_drop_custom_cost() -> None:
    result = await LocalEvaluator(
        metrics=["accuracy", "cost", "total_cost"],
        detailed=True,
        metric_functions=_custom_cost_functions(),
    ).evaluate(_mixed_func(), {"model": "gpt-4o-mini"}, _two_examples())
    assert result.tracked_spend["unmeasured"] is True
    assert result.example_results
    for n, row in enumerate(result.example_results):
        _no_custom(row.metrics, f"row{n}")


async def test_simple_scoring_progress_payload_copied_at_call_time() -> None:
    copies: list[dict[str, Any]] = []
    evaluator = SimpleScoringEvaluator(
        metric_functions=_custom_cost_functions(), detailed=True
    )
    await evaluator.evaluate(
        _mixed_func(),
        {"model": "gpt-4o-mini"},
        _two_examples(),
        progress_callback=lambda i, payload: copies.append(copy.deepcopy(payload)),
    )
    assert copies
    for n, payload in enumerate(copies):
        _no_custom(payload.get("metrics") or {}, f"progress{n}")


async def test_local_summary_stats_drop_custom_cost() -> None:
    result = await LocalEvaluator(
        metrics=["accuracy", "cost", "total_cost"],
        detailed=True,
        execution_mode="local",
        metric_functions=_custom_cost_functions(),
    ).evaluate(_mixed_func(), {"model": "gpt-4o-mini"}, _two_examples())
    stats = (result.summary_stats or {}).get("metrics", {})
    assert stats.get("cost_unmeasured") == 1.0
    assert "cost" not in stats, f"summary_stats.metrics.cost={stats.get('cost')}"
    total = stats.get("total_cost")
    # total_cost, when present, is the tracker's describe(), never the custom 200.
    assert total is None or total.get("mean") != 200.0
