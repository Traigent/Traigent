"""Regression tests for the SDK orientation + pricing sweep (2026-07).

Supersedes the two NO-GO'd PRs (#1956 tie-break half, #1969 sweep half). One
focused, offline-only test per issue, using reproducers that FAIL against the
old/partial code and pass only with the complete fix:

* #1957 — model pricing uses forward longest-prefix only (no reverse-prefix
  mis-pricing of partial model names).
* #1958 — the constraint fallback-pricing path agrees BY CONSTRUCTION with the
  canonical ``_estimation_cost_from_tokens`` path (no input/output averaging).
* #1959 — ``best_result`` honors the declared objective orientation.
* #1960 — ``_simple_is_better`` (live incumbent primary) honors declared
  orientation.
* #1955 — the LIVE secondary tie-break (``_secondary_tie_breaks_incumbent``)
  honors declared orientation, and the POST-HOC weighted tie-break does too.
* #1962 — ``load_result`` raises the documented ValueError on corrupt metadata.
* #1966 — the TVL constraint AST gate rejects ``**`` (Pow) AND the unbounded
  arbitrary-precision integer math calls (``factorial``/``comb``/``perm``).
* #1967 — retry decision honors RetryableError / NonRetryableError type.
* #1968 — HTTP 408 is treated as retryable and is actually re-attempted.
"""

from __future__ import annotations

import asyncio
import json
import types
from datetime import UTC, datetime
from unittest import mock

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _trial(config: dict, metrics: dict, trial_id: str) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config=config,
        metrics=metrics,
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=datetime.now(UTC),
    )


def _schema(*objectives: ObjectiveDefinition) -> ObjectiveSchema:
    return ObjectiveSchema.from_objectives(list(objectives))


# --------------------------------------------------------------------------- #
# #1957 — forward-prefix only, no reverse mis-pricing
# --------------------------------------------------------------------------- #
def test_1957_forward_prefix_only_no_reverse_mispricing() -> None:
    """``_find_fallback_pricing`` must not reverse-match a bare ``gpt-4``.

    The removed reverse-prefix branch matched ``gpt-4`` against the first
    dict-order ``gpt-4*`` key (``gpt-4o``), mis-pricing it. Forward-only
    matching yields no match for a bare ``gpt-4`` base (no key is a prefix of
    it); the legitimate short alias resolves via MODEL_NAME_ALIASES upstream.
    """
    from traigent.utils.cost_calculator import _find_fallback_pricing

    pricing, matched_key = _find_fallback_pricing("gpt-4")
    assert matched_key != "gpt-4o"
    assert pricing is None and matched_key is None


# --------------------------------------------------------------------------- #
# #1958 — constraint fallback pricing agrees with the canonical path
# --------------------------------------------------------------------------- #
def test_1958_constraint_fallback_agrees_with_canonical_estimation() -> None:
    """The constraint fallback price equals the canonical estimation price.

    Reproducer (the one the old averaging path got WRONG): ``gpt-4`` /
    ``max_tokens=1000``. The canonical path prices the whole budget as input
    tokens -> ``1000 * 10e-6 = $0.010``. The retired averaging path returned
    ``((10e-3 + 30e-3) / 2) * 1 = $0.020``. The constraint must now match the
    canonical value, not the average.
    """
    from traigent.utils.constraints import _try_fallback_pricing
    from traigent.utils.cost_calculator import _estimation_cost_from_tokens

    constraint_val = _try_fallback_pricing("gpt-4", 1000)
    canonical_val = sum(_estimation_cost_from_tokens("gpt-4", 1000, 0, _quiet=True))

    assert constraint_val is not None
    assert constraint_val == pytest.approx(canonical_val, rel=1e-12)
    assert constraint_val == pytest.approx(0.010, rel=1e-9)
    # Guard against a silent return to input/output averaging (the old bug):
    assert constraint_val != pytest.approx(0.020, rel=1e-9)


def test_1958_constraint_resolves_gpt4_alias_and_not_gpt4o() -> None:
    """'gpt-4' prices identically to 'gpt-4-turbo' and NOT to gpt-4o."""
    from traigent.utils.constraints import _try_fallback_pricing

    gpt4 = _try_fallback_pricing("gpt-4", 1000)
    turbo = _try_fallback_pricing("gpt-4-turbo", 1000)
    gpt4o = _try_fallback_pricing("gpt-4o", 1000)

    assert gpt4 is not None
    assert gpt4 == turbo
    assert gpt4 != gpt4o


def test_1958_unknown_model_fails_the_constraint() -> None:
    """An unknown model has no canonical price -> None -> constraint fails."""
    from traigent.utils.constraints import _try_fallback_pricing

    assert _try_fallback_pricing("totally-unknown-model-xyz", 1000) is None


# --------------------------------------------------------------------------- #
# #1959 / #1960 / #1955 — declared-orientation selection (live paths)
# --------------------------------------------------------------------------- #
def _bare_orchestrator(schema: ObjectiveSchema):
    """Build an orchestrator without the heavy __init__.

    Only the attributes exercised by best_result / _simple_is_better /
    _secondary_tie_breaks_incumbent are set.
    """
    from traigent.core.orchestrator import OptimizationOrchestrator

    objective_names = [obj.name for obj in schema.objectives]
    orch = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
    orch.optimizer = types.SimpleNamespace(objectives=objective_names)
    orch.objective_schema = schema
    orch._best_trial_cached = None
    orch._trials = []
    return orch


def test_1959_best_result_honors_declared_minimize() -> None:
    """'brier' misses the name heuristic; declared minimize must pick the lower."""
    orch = _bare_orchestrator(
        _schema(ObjectiveDefinition(name="brier", orientation="minimize", weight=1.0))
    )
    low = _trial({"p": 1}, {"brier": 0.1}, "t_low")
    high = _trial({"p": 2}, {"brier": 0.9}, "t_high")
    orch._trials = [high, low]

    assert orch.best_result is low  # minimize -> lowest brier wins


def test_1960_simple_is_better_honors_declared_minimize_primary() -> None:
    """Live incumbent primary compare must use declared orientation, not name."""
    orch = _bare_orchestrator(
        _schema(ObjectiveDefinition(name="brier", orientation="minimize", weight=1.0))
    )
    incumbent = _trial({"p": 1}, {"brier": 0.5}, "incumbent")
    orch._best_trial_cached = incumbent
    orch._trials = [incumbent]

    better = _trial({"p": 2}, {"brier": 0.2}, "candidate")  # lower is better
    worse = _trial({"p": 3}, {"brier": 0.8}, "worse")

    assert orch._simple_is_better(better) is True
    assert orch._simple_is_better(worse) is False


def test_1955_live_secondary_tie_break_honors_declared_minimize() -> None:
    """Primary tied -> the declared-minimize SECONDARY must not promote a worse
    candidate.

    quality(max) is tied at 0.5; brier(min) is the secondary. Incumbent
    brier=0.2 beats candidate brier=0.8, so the candidate must NOT be promoted.
    Uniform weights keep live tracking on the simple (non-weighted) path so this
    exercises ``_secondary_tie_breaks_incumbent`` directly.
    """
    orch = _bare_orchestrator(
        _schema(
            ObjectiveDefinition(name="quality", orientation="maximize", weight=1.0),
            ObjectiveDefinition(name="brier", orientation="minimize", weight=1.0),
        )
    )
    incumbent = _trial({"p": 1}, {"quality": 0.5, "brier": 0.2}, "incumbent")
    orch._best_trial_cached = incumbent
    orch._trials = [incumbent]

    candidate = _trial({"p": 2}, {"quality": 0.5, "brier": 0.8}, "candidate")
    assert orch._simple_is_better(candidate) is False


def test_1955_posthoc_weighted_tie_break_honors_declared_minimize() -> None:
    """POST-HOC ``calculate_weighted_scores`` breaks weighted ties in the
    declared direction (terminal/post-hoc parity).

    accuracy(max) + token_budget(min) + quality(max), equal weights: wasteful
    and frugal tie at weighted 0.5. The declared-minimize token_budget must
    crown ``frugal`` (smaller budget), matching the terminal selector.
    """
    schema = _schema(
        ObjectiveDefinition(name="accuracy", orientation="maximize", weight=1.0),
        ObjectiveDefinition(name="token_budget", orientation="minimize", weight=1.0),
        ObjectiveDefinition(name="quality", orientation="maximize", weight=1.0),
    )
    wasteful = _trial(
        {"model": "wasteful"},
        {"accuracy": 0.9, "token_budget": 5000.0, "quality": 1.0},
        "wasteful",
    )
    frugal = _trial(
        {"model": "frugal"},
        {"accuracy": 0.9, "token_budget": 500.0, "quality": 0.0},
        "frugal",
    )
    result = OptimizationResult(
        trials=[wasteful, frugal],
        best_config={"model": "frugal"},
        best_score=0.9,
        optimization_id="opt-1955",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy", "token_budget", "quality"],
        algorithm="grid",
        timestamp=datetime.now(UTC),
    )

    weighted = result.calculate_weighted_scores(objective_schema=schema)
    assert weighted["best_weighted_config"] == {"model": "frugal"}


# --------------------------------------------------------------------------- #
# #1962 — load_result raises the documented ValueError on corruption
# --------------------------------------------------------------------------- #
def _optimization_result_stub() -> OptimizationResult:
    return OptimizationResult(
        trials=[],
        best_config={"param": 1},
        best_score=0.0,
        optimization_id="opt-1962",
        duration=1.0,
        convergence_info={"status": "stable"},
        status=OptimizationStatus.COMPLETED,
        objectives=["objective"],
        algorithm="grid_search",
        timestamp=datetime.now(UTC),
        metadata={
            "function_name": "demo",
            "configuration_space": {"param": [0, 1]},
        },
    )


def test_1962_load_result_missing_key_raises_value_error(tmp_path) -> None:
    from traigent.utils.persistence import METADATA_FILE, PersistenceManager

    pm = PersistenceManager(base_dir=tmp_path)
    pm.save_result(_optimization_result_stub(), "corruptme")

    # Corrupt metadata: drop a required key.
    meta_path = tmp_path / "corruptme" / METADATA_FILE
    data = json.loads(meta_path.read_text())
    del data["algorithm"]
    meta_path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="missing 'algorithm'"):
        pm.load_result("corruptme")


# --------------------------------------------------------------------------- #
# #1966 — TVL constraint AST gate rejects Pow and unbounded-integer math calls
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "expr",
    [
        "9 ** 9 > 0",  # ** operator
        "math.factorial(100000000) > 0",  # unbounded int via literal (no **)
        "math.comb(100000000, 5) > 0",
        "math.perm(100000000, 5) > 0",
    ],
)
def test_1966_integer_explosion_constructs_rejected(expr: str) -> None:
    from traigent.tvl.spec_loader import compile_constraint_expression
    from traigent.utils.exceptions import TVLValidationError

    with pytest.raises(TVLValidationError):
        compile_constraint_expression(expr, label="dos")


def test_1966_bounded_float_pow_still_compiles() -> None:
    from traigent.tvl.spec_loader import compile_constraint_expression

    compiled = compile_constraint_expression("math.pow(2, 3) > 0", label="ok")
    assert compiled({}, {}) is True


# --------------------------------------------------------------------------- #
# #1967 — retry decision honors the explicit exception type
# --------------------------------------------------------------------------- #
def test_1967_should_retry_honors_exception_type() -> None:
    from traigent.cloud.resilient_client import ResilientClient
    from traigent.utils.exceptions import NonRetryableError, RetryableError

    client = ResilientClient(max_retries=3)

    # RetryableError whose message carries no "magic" retry token -> still retried.
    assert client.should_retry(RetryableError("something odd happened"), 0) is True

    # NonRetryableError whose message contains a network word -> NOT retried.
    assert (
        client.should_retry(NonRetryableError("Invalid connection string"), 0) is False
    )


# --------------------------------------------------------------------------- #
# #1968 — HTTP 408 is retryable AND actually re-attempted
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, status: int) -> None:
        self.status = status
        self.headers: dict[str, str] = {}

    async def __aenter__(self) -> _FakeResp:
        return self

    async def __aexit__(self, *_a) -> bool:
        return False

    async def json(self) -> dict:
        return {"ok": True}


class _CountingSession:
    """aiohttp.ClientSession stand-in that counts request() invocations."""

    def __init__(self, status: int, counter: list[int]) -> None:
        self._status = status
        self._counter = counter

    async def __aenter__(self) -> _CountingSession:
        return self

    async def __aexit__(self, *_a) -> bool:
        return False

    def request(self, *_a, **_k) -> _FakeResp:
        self._counter[0] += 1
        return _FakeResp(self._status)


def _run_counting_backend_request(
    status: int, counter: list[int], *, max_retries: int
) -> None:
    """Drive one resilient_backend_request; mutate ``counter`` per attempt.

    Raises whatever the request surfaces. ``counter`` is owned by the caller so
    the attempt count survives the propagated exception.
    """
    from traigent.cloud import resilient_client as rc

    client = rc.ResilientClient(max_retries=max_retries)

    async def _no_sleep(*_a, **_k) -> None:
        return None

    with (
        mock.patch(
            "traigent.cloud.client.raise_if_cloud_egress_disabled",
            lambda *a, **k: None,
        ),
        mock.patch(
            "aiohttp.ClientSession",
            lambda *a, **k: _CountingSession(status, counter),
        ),
        mock.patch.object(rc.asyncio, "sleep", _no_sleep),
    ):
        asyncio.run(
            rc.resilient_backend_request(client, "GET", "https://example.test/x")
        )


def test_1968_http_408_is_retryable_and_reattempted() -> None:
    from traigent.utils.exceptions import RetryableError

    # 408 must surface as RetryableError AND be actually retried: with
    # max_retries=1 the request is attempted twice before the error propagates.
    counter = [0]
    with pytest.raises(RetryableError):
        _run_counting_backend_request(408, counter, max_retries=1)
    assert counter[0] == 2, "408 should trigger one real retry (2 attempts)"


def test_1968_http_404_is_not_retried() -> None:
    from traigent.utils.exceptions import NonRetryableError

    # A genuine permanent 4xx (404) raises NonRetryableError on the FIRST
    # attempt and is never re-attempted.
    counter = [0]
    with pytest.raises(NonRetryableError):
        _run_counting_backend_request(404, counter, max_retries=1)
    assert counter[0] == 1, "404 must not be retried (single attempt)"
