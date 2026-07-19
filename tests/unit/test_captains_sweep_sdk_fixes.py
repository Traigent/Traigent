"""Regression tests for the captains-leader SDK sweep (2026-07).

Covers, one focused test per issue, offline-only (no LLM/API cost):

* #1957 — model pricing uses forward longest-prefix only (no reverse-prefix
  mis-pricing of partial model names).
* #1958 — the constraint fallback-pricing path applies MODEL_NAME_ALIASES so
  it agrees with the canonical estimation path.
* #1959 — ``best_result`` honors the declared objective orientation.
* #1960 — ``_simple_is_better`` (live incumbent) honors declared orientation.
* #1962 — ``load_result`` raises the documented ValueError on corrupt metadata.
* #1966 — the TVL constraint AST gate rejects ``**`` (Pow) to block
  integer-explosion DoS.
* #1967 — retry decision honors RetryableError / NonRetryableError type.
* #1968 — HTTP 408 is treated as retryable, not a permanent 4xx.
"""

from __future__ import annotations

import asyncio
import json
import types
from datetime import UTC, datetime
from unittest import mock

import pytest

from traigent.api.types import OptimizationStatus, TrialResult, TrialStatus


# --------------------------------------------------------------------------- #
# #1957 / #1958 — model pricing exactness
# --------------------------------------------------------------------------- #
def test_1957_forward_prefix_only_no_reverse_mispricing() -> None:
    """cost_calculator._find_fallback_pricing must not reverse-match 'gpt-4'.

    The removed reverse-prefix branch matched 'gpt-4' against the first
    dict-order 'gpt-4*' key (gpt-4o), mis-pricing it. Forward-only matching
    yields no match for a bare 'gpt-4' base (no key is a prefix of it).
    """
    from traigent.utils.cost_calculator import _find_fallback_pricing

    pricing, matched_key = _find_fallback_pricing("gpt-4")
    assert matched_key != "gpt-4o"
    assert pricing is None and matched_key is None


def test_1957_1958_constraint_path_prices_gpt4_as_turbo() -> None:
    """The constraint fallback path resolves 'gpt-4' to gpt-4-turbo pricing.

    #1958 adds the MODEL_NAME_ALIASES resolution; combined with the #1957
    forward-only fix, 'gpt-4' now prices identically to 'gpt-4-turbo' and NOT
    to gpt-4o.
    """
    from traigent.utils.constraints import _try_fallback_pricing

    gpt4 = _try_fallback_pricing("gpt-4", 1000)
    turbo = _try_fallback_pricing("gpt-4-turbo", 1000)
    gpt4o = _try_fallback_pricing("gpt-4o", 1000)

    assert gpt4 is not None
    assert gpt4 == turbo
    assert gpt4 != gpt4o


def test_1958_estimation_and_constraint_paths_agree_for_gpt4() -> None:
    """Constraint path pricing matches the canonical estimation path for 'gpt-4'."""
    from traigent.utils.constraints import _try_fallback_pricing
    from traigent.utils.cost_calculator import _estimation_cost_from_tokens

    # Canonical estimation for 1000 in + 0 out, expressed per-1k input-equivalent.
    in_cost, out_cost = _estimation_cost_from_tokens("gpt-4", 500, 500)
    canonical_avg_per_1k = in_cost + out_cost  # 500+500 tokens -> per-1k avg basis
    constraint_val = _try_fallback_pricing("gpt-4", 1000)
    assert constraint_val is not None
    assert canonical_avg_per_1k == pytest.approx(constraint_val, rel=1e-9)


# --------------------------------------------------------------------------- #
# #1959 / #1960 — declared-orientation selection
# --------------------------------------------------------------------------- #
def _minimize_schema(name: str):
    from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

    return ObjectiveSchema.from_objectives(
        [ObjectiveDefinition(name=name, orientation="minimize", weight=1.0)]
    )


def _trial(metric_name: str, value: float, trial_id: str) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={"p": 1},
        metrics={metric_name: value},
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=datetime.now(UTC),
    )


def _bare_orchestrator(objective_name: str):
    """Build an orchestrator instance without the heavy __init__.

    Only the attributes exercised by best_result / _simple_is_better are set.
    """
    from traigent.core.orchestrator import OptimizationOrchestrator

    orch = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
    orch.optimizer = types.SimpleNamespace(objectives=[objective_name])
    orch.objective_schema = _minimize_schema(objective_name)
    orch._best_trial_cached = None
    orch._trials = []
    return orch


def test_1959_best_result_honors_declared_minimize() -> None:
    """'brier' misses the name heuristic; declared minimize must pick the lower."""
    orch = _bare_orchestrator("brier")
    low = _trial("brier", 0.1, "t_low")
    high = _trial("brier", 0.9, "t_high")
    orch._trials = [high, low]

    best = orch.best_result
    assert best is low  # minimize -> lowest brier wins


def test_1960_simple_is_better_honors_declared_minimize() -> None:
    """Live incumbent compare must use declared orientation, not the name."""
    orch = _bare_orchestrator("brier")
    incumbent = _trial("brier", 0.5, "incumbent")
    orch._best_trial_cached = incumbent
    orch._trials = [incumbent]

    better = _trial("brier", 0.2, "candidate")  # lower is better for minimize
    worse = _trial("brier", 0.8, "worse")

    assert orch._simple_is_better(better) is True
    assert orch._simple_is_better(worse) is False


# --------------------------------------------------------------------------- #
# #1962 — load_result raises the documented ValueError on corruption
# --------------------------------------------------------------------------- #
def test_1962_load_result_missing_key_raises_value_error(tmp_path) -> None:
    from traigent.utils.persistence import METADATA_FILE, PersistenceManager

    pm = PersistenceManager(base_dir=tmp_path)
    result = OptimizationResult_stub()
    name = pm.save_result(result, "corruptme")

    # Corrupt metadata: drop a required key.
    meta_path = tmp_path / "corruptme" / METADATA_FILE
    data = json.loads(meta_path.read_text())
    del data["algorithm"]
    meta_path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="missing 'algorithm'"):
        pm.load_result("corruptme")

    assert name  # sanity: save returned a path


def OptimizationResult_stub():
    from traigent.api.types import OptimizationResult

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


# --------------------------------------------------------------------------- #
# #1966 — TVL constraint AST gate rejects Pow
# --------------------------------------------------------------------------- #
def test_1966_pow_rejected_in_constraint_expression() -> None:
    from traigent.tvl.spec_loader import (
        TVLValidationError,
        compile_constraint_expression,
    )

    with pytest.raises(TVLValidationError):
        compile_constraint_expression("9 ** 9 > 0", label="dos")

    # A bounded, whitelisted alternative still compiles.
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
    assert (
        client.should_retry(RetryableError("something odd happened"), attempt=0) is True
    )

    # NonRetryableError whose message contains a network word -> NOT retried.
    assert (
        client.should_retry(NonRetryableError("Invalid connection string"), attempt=0)
        is False
    )


# --------------------------------------------------------------------------- #
# #1968 — HTTP 408 is retryable
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


class _FakeSession:
    def __init__(self, status: int) -> None:
        self._status = status

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_a) -> bool:
        return False

    def request(self, *_a, **_k) -> _FakeResp:
        return _FakeResp(self._status)


def _run_backend_request(status: int):
    from traigent.cloud import resilient_client as rc

    client = rc.ResilientClient(max_retries=0)  # no retry sleeps

    with (
        mock.patch(
            "traigent.cloud.client.raise_if_cloud_egress_disabled",
            lambda *a, **k: None,
        ),
        mock.patch("aiohttp.ClientSession", lambda *a, **k: _FakeSession(status)),
    ):
        return asyncio.run(
            rc.resilient_backend_request(client, "GET", "https://example.test/x")
        )


def test_1968_http_408_is_retryable() -> None:
    from traigent.utils.exceptions import NonRetryableError, RetryableError

    # 408 must surface as a RetryableError (transient), not NonRetryableError.
    with pytest.raises(RetryableError):
        _run_backend_request(408)

    # A genuine permanent 4xx (404) still raises NonRetryableError.
    with pytest.raises(NonRetryableError):
        _run_backend_request(404)
