"""Round-2 hardening probes for the tuple-return metrics channel (codex review).

These pin the four codex-verified round-2 findings on the
``(output, metrics_dict)`` return channel. They are written RED-first: each
asserts the *fixed* behavior so it fails against the pre-fix code.

* G1 — ``SimpleScoringEvaluator`` caps user keys, then inserts
  ``examples_attempted`` AFTER the cap with no final-union ceiling, so >50 flat
  trial metrics can escape. The fix applies the SAME ``enforce_user_metric_ceiling``
  helper as the local lane as the LAST step before returning trial metrics.
* G2 — ``_unpack_user_metrics`` gated on ``str.isidentifier()``, which accepts
  Unicode identifiers (e.g. ``"π_metric"``) that the ``MeasuresDict`` wire
  contract (``^[a-zA-Z_]\\w*$`` with ``re.ASCII``) rejects. The fix swaps the
  gate to the EXACT ASCII pattern, pinned to ``dtos.py``'s ``KEY_PATTERN`` by a
  consistency test.
* G3 — reserved-key protection missed evaluator-computable names not in the
  static frozenset (RAGAS metrics such as ``context_precision``). The fix
  consults ``RESERVED_METRIC_KEYS`` UNION the evaluator's own metric-registry +
  RAGAS names at every merge site.
* G4 — SDK-internal cost/token keys (``input_cost`` etc.) injected by the local
  lane flow through the shared user-aggregation pass and were skipped with a
  WARNING that *lies* (they are evaluator-internal, not user metrics). The fix
  logs reserved-key skips in the shared aggregator at DEBUG; the per-example
  ``_merge_user_metrics`` (genuinely user dicts) KEEPS the WARNING.
"""

from __future__ import annotations

import logging
import re

import pytest

from traigent.cloud.dtos import MeasuresDict
from traigent.evaluators.base import Dataset, EvaluationExample, SimpleScoringEvaluator
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import (
    USER_METRIC_KEY_PATTERN,
    aggregate_user_custom_metrics,
)
from traigent.knobs.telemetry import TOTAL_MEASURES_CEILING


def _two_example_dataset() -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"text": "q1"}, expected_output="YES"),
            EvaluationExample(input_data={"text": "q2"}, expected_output="YES"),
        ],
        name="tuple_metrics_hardening_round2",
    )


def _local_evaluator(metrics: list[str] | None = None) -> LocalEvaluator:
    return LocalEvaluator(
        metrics=metrics or ["accuracy"],
        detailed=True,
        execution_mode="local",
    )


# ---------------------------------------------------------------------------
# G1: SimpleScoringEvaluator final-union ceiling (examples_attempted added last)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simple_scoring_final_union_capped_after_examples_attempted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """G1 (codex repro): metrics=["accuracy"] + 60 tuple user keys -> <= ceiling.

    Pre-fix: user keys are capped, then ``examples_attempted`` is added AFTER
    with no final-union cap, so ``len(result.metrics) == 51``.
    """

    def score(output: object, expected: object) -> float:
        return 1.0 if output == expected else 0.0

    user_keys = {f"composite_metric_{i}": float(i) for i in range(60)}

    def func(text: str) -> tuple[str, dict[str, float]]:
        return "YES", dict(user_keys)

    evaluator = SimpleScoringEvaluator(
        scoring_function=score,
        metrics=["accuracy"],
        capture_llm_metrics=False,
    )
    with caplog.at_level(logging.WARNING):
        result = await evaluator.evaluate(func, {}, _two_example_dataset())

    # RED before G1: examples_attempted slips past the user-key cap -> 51 keys.
    assert len(result.metrics) <= TOTAL_MEASURES_CEILING
    # examples_attempted is a reserved evaluator key and must survive the cap.
    assert "examples_attempted" in result.metrics
    assert result.metrics["examples_attempted"] == 2
    # The result is wire-valid against the MeasuresDict ceiling.
    assert len(MeasuresDict(dict(result.metrics))) <= MeasuresDict.MAX_KEYS


# ---------------------------------------------------------------------------
# G2: Unicode-identifier bypass closed; ASCII pattern matches MeasuresDict
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unicode_identifier_key_is_not_unpacked() -> None:
    """G2 (codex repro): "π_metric".isidentifier() is True but the wire rejects it.

    Pre-fix: the tuple is unpacked, ``π_metric`` rides into result.metrics, and
    MeasuresDict rejects it at submission (the compat catch then submits
    unvalidated). Fixed: the shape does NOT match, so the raw tuple rides
    through and accuracy is poisoned (output != "YES").
    """

    async def func(text: str) -> tuple[str, dict[str, float]]:
        return "ok", {"π_metric": 1.0}

    result = await _local_evaluator().evaluate(func, {}, _two_example_dataset())

    assert "π_metric" not in result.metrics
    # No unpack -> the raw tuple stays the output -> accuracy is 0.0.
    assert result.metrics["accuracy"] == 0.0


def test_unpack_helper_rejects_unicode_identifier_directly() -> None:
    """G2 (unit): _unpack_user_metrics passes through on a Unicode-identifier key."""
    assert "π_metric".isidentifier() is True  # the bypass premise
    raw = ("ok", {"π_metric": 1.0, "fine_key": 2.0})
    output, user_metrics = SimpleScoringEvaluator._unpack_user_metrics(raw)
    assert output is raw
    assert user_metrics is None


def test_user_metric_pattern_matches_measuresdict_pattern() -> None:
    """G2 (consistency): the evaluator gate and the wire contract agree.

    Imports BOTH the evaluator-side ``USER_METRIC_KEY_PATTERN`` and the cloud
    ``MeasuresDict.KEY_PATTERN`` and asserts they accept/reject the SAME key
    corpus. If the wire pattern drifts, THIS test fails rather than the bypass
    silently reopening.
    """
    accept = ["_x", "ok_key", "metric_123", "abc", "A1_b2"]
    reject = ["π_metric", "a-b", "9a", "", "has space", "éclair"]

    wire = MeasuresDict.KEY_PATTERN
    # The wire contract is ASCII-identifier syntax; the evaluator gate must be
    # at least as strict (it must reject everything the wire rejects). Both must
    # use re.ASCII semantics so \\w does not leak Unicode word chars.
    assert wire.flags & re.ASCII, "MeasuresDict.KEY_PATTERN must be compiled re.ASCII"
    assert USER_METRIC_KEY_PATTERN.flags & re.ASCII

    for key in accept:
        assert USER_METRIC_KEY_PATTERN.match(key), key
        assert wire.match(key), key
    for key in reject:
        assert not USER_METRIC_KEY_PATTERN.match(key), key
        assert not wire.match(key), key


# ---------------------------------------------------------------------------
# G3: reserved-key protection covers evaluator-computable (RAGAS) names
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_cannot_overwrite_evaluator_computable_ragas_name() -> None:
    """G3 (codex repro): metrics=["context_precision"] + tuple {context_precision}.

    Pre-fix: ``context_precision`` is absent from RESERVED_METRIC_KEYS and the
    local merge overwrites it with the user's 0.9, so the user value wins.
    Fixed: the evaluator's metric-registry/RAGAS name is reserved, so the user
    value is skipped and the evaluator-computed value wins.
    """

    async def func(text: str) -> tuple[str, dict[str, float]]:
        return "YES", {"context_precision": 0.9}

    evaluator = _local_evaluator(metrics=["accuracy", "context_precision"])
    result = await evaluator.evaluate(func, {}, _two_example_dataset())

    # The user's 0.9 must NOT win: context_precision is evaluator-computable.
    # RAGAS is unavailable in unit tests, so the evaluator computes 0.0 (the
    # fail-closed default); the key must exist and must not be the user's 0.9.
    assert "context_precision" in result.metrics
    assert result.metrics["context_precision"] != pytest.approx(0.9)


def test_shared_aggregator_honors_extra_reserved_keys() -> None:
    """G3 (unit): aggregate_user_custom_metrics skips extra_reserved names."""
    target: dict[str, float] = {}
    aggregate_user_custom_metrics(
        target,
        [{"context_precision": 0.9, "composite_ok": 1.0}],
        context="g3 unit",
        extra_reserved=frozenset({"context_precision"}),
    )
    assert "context_precision" not in target
    assert target["composite_ok"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# G4: reserved-key skips in the shared aggregator log at DEBUG (not WARNING)
# ---------------------------------------------------------------------------


def test_shared_aggregator_reserved_skip_logs_debug_not_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """G4: SDK-internal cost keys skipped in the shared pass log at DEBUG.

    The shared aggregator receives evaluator-internal keys (input_cost,
    total_cost, ...) injected by the local lane. Skipping them is correct, but
    the WARNING lied ("user metric"). Fixed: reserved skips in the shared
    aggregator log at DEBUG.
    """
    target: dict[str, float] = {}
    with caplog.at_level(logging.DEBUG, logger="traigent.evaluators.metrics_tracker"):
        aggregate_user_custom_metrics(
            target,
            [{"input_cost": 0.01, "total_cost": 0.02, "composite_ok": 1.0}],
            context="g4 unit",
        )

    # The non-reserved key still rides.
    assert target["composite_ok"] == pytest.approx(1.0)
    reserved_records = [
        r for r in caplog.records if "reserved" in r.getMessage().lower()
    ]
    assert reserved_records, "expected a reserved-skip log line"
    # RED before G4: these are WARNINGs. Fixed: DEBUG.
    assert all(r.levelno == logging.DEBUG for r in reserved_records)
    assert not any(
        r.levelno >= logging.WARNING and "reserved" in r.getMessage().lower()
        for r in caplog.records
    )


def test_per_example_user_merge_reserved_skip_keeps_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """G4: the per-example _merge_user_metrics KEEPS the WARNING (real user dict)."""
    evaluator = _local_evaluator(metrics=["accuracy", "success_rate"])
    target: dict[str, float] = {}
    with caplog.at_level(logging.WARNING, logger="traigent.evaluators.base"):
        evaluator._merge_user_metrics(
            target,
            {"success_rate": 0.01, "composite_ok": 1.0},
            context="local lane",
        )
    assert target["composite_ok"] == pytest.approx(1.0)
    assert "success_rate" not in target
    assert any(
        "reserved" in r.getMessage().lower() and r.levelno >= logging.WARNING
        for r in caplog.records
    )
