"""Regression tests: the bare ``latency`` metric is MILLISECONDS on every lane.

Issue #1855. The hybrid lane aggregates per-result ``latency_ms`` into
``metrics["latency"]`` (ms); the local builtin ``_compute_latency`` returned
SECONDS — a 1000x cross-lane disagreement under one metric key, which also
made any unit label in the results table a lie for one of the lanes. The
producers are now unified on ms and the results-table renders the ms label
(the label half is #1859 by IsraelTraigent).
"""

from __future__ import annotations

from dataclasses import dataclass

from traigent.evaluators.base import BaseEvaluator
from traigent.utils.results_table import _format_metric_value


@dataclass
class _FakeExampleResult:
    execution_time: float  # seconds, as recorded by the local lane


class _Probe(BaseEvaluator):
    """Minimal concrete evaluator to reach the builtin metric helpers."""

    async def evaluate(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError


def _compute(evaluator_kwargs: dict) -> float:
    probe = _Probe.__new__(_Probe)  # skip __init__: helper is self-contained
    return probe._compute_latency([], [], [], **evaluator_kwargs)


class TestLatencyUnitMs:
    def test_execution_time_seconds_converted_to_ms(self) -> None:
        results = [_FakeExampleResult(0.5), _FakeExampleResult(1.5)]
        # mean(0.5s, 1.5s) = 1.0s -> 1000 ms. Pre-#1855 this returned 1.0.
        assert _compute({"example_results": results}) == 1000.0

    def test_context_latency_fallback_is_seconds_input(self) -> None:
        assert _compute({"latency": 0.25}) == 250.0

    def test_context_avg_response_time_fallback_is_seconds_input(self) -> None:
        # avg_response_time is seconds (see base.py avg_response_time_seconds).
        assert _compute({"avg_response_time": 2.0}) == 2000.0

    def test_results_table_renders_ms_label(self) -> None:
        # 850 ms must render as ~"850ms", never "850.000s" (#1855/#1859).
        assert _format_metric_value("latency", 850.0) == "850ms"
