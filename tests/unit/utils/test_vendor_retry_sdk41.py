"""SDK-41: strict retry settings, per-call vendor retry, fault hook, accounting."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from traigent.core.trial_lifecycle import TrialLifecycle
from traigent.testing import inject_vendor_failures
from traigent.utils.exceptions import (
    ConfigurationError,
    InsufficientFundsError,
    RateLimitError,
    ServiceUnavailableError,
)
from traigent.utils.langchain_interceptor import (
    capture_scope,
    clear_captured_responses,
)
from traigent.utils.vendor_retry import (
    classify_retryable,
    resolve_vendor_retry_settings,
    summarize_bucket,
)

RETRIES = "TRAIGENT_VENDOR_MAX_RETRIES"
BACKOFF = "TRAIGENT_VENDOR_RETRY_BACKOFF"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv(BACKOFF, "0")
    monkeypatch.delenv(RETRIES, raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")
    clear_captured_responses()
    yield
    clear_captured_responses()


# ---------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, (0, 1.0)),
        ({RETRIES: "", BACKOFF: "  "}, (0, 1.0)),
        ({RETRIES: "0"}, (0, 1.0)),
        ({RETRIES: "3", BACKOFF: "0"}, (3, 0.0)),
        ({RETRIES: " 2 ", BACKOFF: "0.5"}, (2, 0.5)),
        ({BACKOFF: "2"}, (0, 2.0)),
    ],
)
def test_settings_good(env, expected):
    s = resolve_vendor_retry_settings(env)
    assert (s.max_retries, s.base_backoff_seconds) == expected


@pytest.mark.parametrize("bad", ["abc", "-1", "2.7", "1e3", "nan", "inf", "1e309"])
def test_retries_bad(bad):
    with pytest.raises(ConfigurationError, match=RETRIES):
        resolve_vendor_retry_settings({RETRIES: bad})


@pytest.mark.parametrize("bad", ["abc", "-1", "nan", "inf", "-inf", "1e309"])
def test_backoff_bad(bad):
    with pytest.raises(ConfigurationError, match=BACKOFF):
        resolve_vendor_retry_settings({BACKOFF: bad})


def test_orchestrator_reads_strictly(monkeypatch):
    from traigent.core.orchestrator import OptimizationOrchestrator

    monkeypatch.setenv(RETRIES, "abc")
    with pytest.raises(ConfigurationError):
        OptimizationOrchestrator._vendor_retry_settings()
    monkeypatch.setenv(RETRIES, "2")
    assert OptimizationOrchestrator._vendor_retry_settings() == (2, 0.0)


# ------------------------------------------------------------ classification


def test_classify_retryable():
    assert classify_retryable(RateLimitError("x")) is not None
    assert classify_retryable(ServiceUnavailableError("x")) is not None
    assert classify_retryable(InsufficientFundsError("402")) is None
    assert classify_retryable(ValueError("boom")) is None
    assert classify_retryable(RateLimitError("429 insufficient_quota")) is None


# ------------------------------------------------------- interceptor retries


def _response(model="gpt-4o"):
    r = MagicMock()
    r.model = model
    r.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return r


def _make_litellm(sync_effects):
    mod = types.ModuleType("litellm")
    effects = list(sync_effects)
    calls = {"n": 0}

    def _next():
        calls["n"] += 1
        e = effects.pop(0) if effects else _response()
        if isinstance(e, Exception):
            raise e
        return e

    mod.completion = lambda *a, **k: _next()

    async def _acompletion(*a, **k):
        return _next()

    mod.acompletion = _acompletion
    mod.pre_call_rules = []
    mod.utils = types.SimpleNamespace(function_setup=lambda f, *a, **k: (None, k))
    mod._traigent_patched_completion = False
    mod._traigent_patched_acompletion = False
    mod.calls = calls
    return mod


@pytest.fixture
def mock_off():
    with patch(
        "traigent.integrations.utils.mock_adapter.MockAdapter.is_mock_enabled",
        return_value=False,
    ):
        yield


def _patched(effects):
    from traigent.utils.litellm_interceptor import patch_litellm_for_metadata_capture

    mod = _make_litellm(effects)
    patcher = patch.dict("sys.modules", {"litellm": mod})
    patcher.start()
    patch_litellm_for_metadata_capture()
    return mod, patcher


def test_sync_retry_then_success(mock_off, monkeypatch):
    monkeypatch.setenv(RETRIES, "2")
    mod, p = _patched([RateLimitError("429"), ServiceUnavailableError("503")])
    try:
        with capture_scope() as bucket:
            out = mod.completion(model="gpt-4o")
        assert out.model == "gpt-4o"
        assert mod.calls["n"] == 3
        s = summarize_bucket(bucket)
        assert s["vendor_call_attempts"] == 3
        assert s["vendor_call_retries"] == 2
        assert s["vendor_call_unknown_usage_attempts"] == 2
        assert bucket.vendor_attempts[-1]["usage"]["total_tokens"] == 15
        assert bucket.vendor_attempts[0]["usage"] is None
        assert bucket.vendor_attempts[0]["usage_known"] is False
    finally:
        p.stop()


def test_sync_budget_exhausted_propagates(mock_off, monkeypatch):
    monkeypatch.setenv(RETRIES, "1")
    mod, p = _patched([RateLimitError("429")] * 3)
    try:
        with capture_scope() as bucket, pytest.raises(RateLimitError):
            mod.completion(model="gpt-4o")
        assert mod.calls["n"] == 2
        assert len(bucket.vendor_attempts) == 2
    finally:
        p.stop()


def test_non_retryable_not_retried(mock_off, monkeypatch):
    monkeypatch.setenv(RETRIES, "5")
    mod, p = _patched([InsufficientFundsError("402 payment required")])
    try:
        with capture_scope() as bucket, pytest.raises(InsufficientFundsError):
            mod.completion(model="gpt-4o")
        assert mod.calls["n"] == 1
        assert len(bucket.vendor_attempts) == 1
    finally:
        p.stop()


def test_default_zero_unchanged(mock_off):
    mod, p = _patched([RateLimitError("429")])
    try:
        with pytest.raises(RateLimitError):
            mod.completion(model="gpt-4o")
        assert mod.calls["n"] == 1
    finally:
        p.stop()


@pytest.mark.asyncio
async def test_async_retry_then_success(mock_off, monkeypatch):
    monkeypatch.setenv(RETRIES, "2")
    mod, p = _patched([RateLimitError("429")])
    try:
        with capture_scope() as bucket:
            out = await mod.acompletion(model="gpt-4o")
        assert out.model == "gpt-4o"
        assert mod.calls["n"] == 2
        assert len(bucket.vendor_attempts) == 2
    finally:
        p.stop()


@pytest.mark.asyncio
async def test_async_budget_exhausted_and_non_retryable(mock_off, monkeypatch):
    monkeypatch.setenv(RETRIES, "1")
    mod, p = _patched([ServiceUnavailableError("503")] * 2)
    try:
        with pytest.raises(ServiceUnavailableError):
            await mod.acompletion(model="gpt-4o")
        assert mod.calls["n"] == 2
    finally:
        p.stop()
    mod, p = _patched([ValueError("bad")])
    try:
        with pytest.raises(ValueError):
            await mod.acompletion(model="gpt-4o")
        assert mod.calls["n"] == 1
    finally:
        p.stop()


def test_retry_after_is_respected(mock_off, monkeypatch):
    monkeypatch.setenv(RETRIES, "1")
    monkeypatch.setenv(BACKOFF, "9")
    mod, p = _patched([RateLimitError("429", retry_after=0.25)])
    try:
        with patch("time.sleep") as sleep, capture_scope() as bucket:
            mod.completion(model="gpt-4o")
        sleep.assert_called_once_with(0.25)
        assert bucket.vendor_attempts[0]["backoff_seconds"] == 0.25
    finally:
        p.stop()


# --------------------------------------------- success criterion (mock mode)


def test_ten_calls_one_injected_failure_reports_eleven_attempts(monkeypatch):
    monkeypatch.setenv(RETRIES, "1")
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    mod, p = _patched([])
    try:
        with capture_scope() as bucket:
            with inject_vendor_failures(1, per_call=False):
                for _ in range(10):
                    mod.completion(model="gpt-4o-mini")
        assert mod.calls["n"] == 0  # mock mode: provider never reached
        s = summarize_bucket(bucket)
        assert s["vendor_call_attempts"] == 11
        assert s["vendor_call_retries"] == 1
        assert len(s["vendor_call_attempt_records"]) == 11
        assert s["vendor_call_unknown_usage_attempts"] == 1

        result = types.SimpleNamespace(metadata={})
        TrialLifecycle._apply_vendor_attempts(result, bucket)
        assert result.metadata["vendor_call_attempts"] == 11
    finally:
        p.stop()


def test_injection_without_retries_propagates(monkeypatch):
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    mod, p = _patched([])
    try:
        with inject_vendor_failures(1, kind="service_unavailable"):
            with pytest.raises(ServiceUnavailableError):
                mod.completion(model="m")
    finally:
        p.stop()


def test_hook_per_call_pool(monkeypatch):
    monkeypatch.setenv(RETRIES, "2")
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    mod, p = _patched([])
    try:
        with capture_scope() as bucket, inject_vendor_failures(2):
            for _ in range(3):
                mod.completion(model="m")
        assert len(bucket.vendor_attempts) == 9
    finally:
        p.stop()


def test_hook_refused_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(RuntimeError, match="production"):
        with inject_vendor_failures(1):
            pass


def test_hook_validates_args():
    with pytest.raises(ValueError):
        with inject_vendor_failures(0):
            pass
    with pytest.raises(ValueError):
        with inject_vendor_failures(1, kind="nope"):
            pass


def test_hook_inactive_no_effect(monkeypatch):
    monkeypatch.setenv(RETRIES, "1")
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    mod, p = _patched([])
    try:
        with capture_scope() as bucket:
            for _ in range(3):
                mod.completion(model="m")
        assert len(bucket.vendor_attempts) == 3
        assert all(a["outcome"] == "success" for a in bucket.vendor_attempts)
        with inject_vendor_failures(1):
            pass
        with capture_scope() as bucket2:
            mod.completion(model="m")
        assert len(bucket2.vendor_attempts) == 1
    finally:
        p.stop()


def test_orchestrator_aggregates_attempt_totals():
    from traigent.core.orchestrator import OptimizationOrchestrator

    t1 = types.SimpleNamespace(
        trial_id="a",
        metadata={
            "vendor_call_attempts": 3,
            "vendor_call_retries": 1,
            "vendor_call_unknown_usage_attempts": 1,
            "vendor_call_attempt_records": [
                {"attempt": 1},
                {"attempt": 2},
                {"attempt": 3},
            ],
        },
    )
    t2 = types.SimpleNamespace(trial_id="b", metadata={})
    orch = types.SimpleNamespace(_trials=[t1, t2])
    totals = OptimizationOrchestrator._vendor_attempt_totals(orch)
    assert totals["vendor_call_attempts"] == 3
    assert totals["vendor_call_retries"] == 1
    assert [r["trial_id"] for r in totals["vendor_call_attempt_records"]] == ["a"] * 3
    assert (
        OptimizationOrchestrator._vendor_attempt_totals(
            types.SimpleNamespace(_trials=[t2])
        )
        == {}
    )


# ------------------------------------------------- other trial exit paths


def _lifecycle_with(side_effect):
    from unittest.mock import AsyncMock

    orch = MagicMock()
    orch._abandoned_vendor_attempts = []
    orch._content_identity_run = None
    orch._constraints_post_eval = []
    orch.evaluator.evaluate = AsyncMock(side_effect=side_effect)
    return TrialLifecycle(orch), orch


async def _run_exit_path(exc):
    from unittest.mock import AsyncMock
    from traigent.utils.langchain_interceptor import record_vendor_attempt

    async def evaluate(*a, **k):
        # Two attempts (one failed+retried) made inside the evaluator's own
        # nested scope, like the real LocalEvaluator.
        with capture_scope():
            record_vendor_attempt(
                {"retried": True, "usage_known": False, "outcome": "error"}
            )
            record_vendor_attempt({"usage_known": True, "outcome": "success"})
        raise exc

    lifecycle, orch = _lifecycle_with(evaluate)
    lifecycle._collect_workflow_span = MagicMock()
    lifecycle._build_evaluate_kwargs = MagicMock(return_value={})
    lifecycle._finalize_budget_lease = MagicMock(return_value=None)
    lifecycle._score_surrogate = AsyncMock(return_value=None)
    coro = lifecycle._execute_trial_with_tracing(
        func=lambda: None,
        dataset=MagicMock(),
        trial_id="t1",
        backend_trial_id=None,
        evaluation_config={},
        start_time=0.0,
        optuna_trial_id=None,
        progress_callback=None,
        progress_state={},
        lease=None,
        span=MagicMock(),
    )
    return lifecycle, orch, coro


@pytest.mark.asyncio
@pytest.mark.parametrize("which", ["pruned", "constraint", "generic"])
async def test_attempts_attached_on_non_success_exits(which):
    from traigent.utils.exceptions import TrialPrunedError, TVLConstraintError

    exc = {
        "pruned": TrialPrunedError("p", step=1),
        "constraint": TVLConstraintError("c"),
        "generic": RuntimeError("boom"),
    }[which]
    _, _, coro = await _run_exit_path(exc)
    result = await coro
    assert result.metadata["vendor_call_attempts"] == 2
    assert result.metadata["vendor_call_retries"] == 1


@pytest.mark.asyncio
async def test_vendor_pause_path_keeps_attempts_in_run_totals():
    from traigent.core.orchestrator import OptimizationOrchestrator
    from traigent.utils.exceptions import VendorPauseError

    _, orch, coro = await _run_exit_path(RateLimitError("429"))
    with pytest.raises(VendorPauseError):
        await coro
    (entry,) = orch._abandoned_vendor_attempts
    assert entry["trial_id"] == "t1" and entry["vendor_call_attempts"] == 2
    probe = types.SimpleNamespace(_trials=[], _abandoned_vendor_attempts=[entry])
    totals = OptimizationOrchestrator._vendor_attempt_totals(probe)
    assert totals["vendor_call_attempts"] == 2


# ----------------------------------------------------------- hook policy


@pytest.mark.parametrize(
    "env",
    [
        {"TRAIGENT_ENV": "production"},
        {"TRAIGENT_ENVIRONMENT": "prod"},
        {},  # nothing set: fail closed
        {"ENVIRONMENT": "banana"},  # unknown label: fail closed
    ],
)
def test_hook_refused_by_central_policy(monkeypatch, env):
    for key in ("ENVIRONMENT", "TRAIGENT_ENV", "TRAIGENT_ENVIRONMENT"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match="refused"):
        with inject_vendor_failures(1):
            pass


def test_hook_allowed_for_test_environment(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")
    with inject_vendor_failures(1):
        pass


def test_hook_refuses_at_firing_if_environment_turns_production(monkeypatch):
    from traigent.testing.vendor_faults import maybe_raise_injected_failure

    with inject_vendor_failures(1):
        monkeypatch.setenv("ENVIRONMENT", "production")
        with pytest.raises(RuntimeError, match="refused"):
            maybe_raise_injected_failure(0)


@pytest.mark.asyncio
async def test_hook_is_context_local():
    import asyncio

    from traigent.testing.vendor_faults import maybe_raise_injected_failure

    started = asyncio.Event()
    release = asyncio.Event()
    seen = {}

    async def unrelated():  # created OUTSIDE the with-block's context
        started.set()
        await release.wait()
        try:
            maybe_raise_injected_failure(0)
            seen["raised"] = False
        except Exception:
            seen["raised"] = True

    task = asyncio.create_task(unrelated())
    await started.wait()
    with inject_vendor_failures(1):
        release.set()
        await task
        with pytest.raises(RateLimitError):
            maybe_raise_injected_failure(0)
    assert seen["raised"] is False


def test_hook_not_visible_to_plain_thread():
    import threading

    from traigent.testing.vendor_faults import maybe_raise_injected_failure

    out = []

    def worker():
        try:
            maybe_raise_injected_failure(0)
            out.append("ok")
        except Exception:
            out.append("raised")

    with inject_vendor_failures(1):
        t = threading.Thread(target=worker)
        t.start()
        t.join()
    assert out == ["ok"]


# --------------------------------------------------------- classification


class _Http(Exception):
    def __init__(self, msg, status=None, headers=None, **kw):
        super().__init__(msg)
        self.status_code = status
        self.headers = headers
        for k, v in kw.items():
            setattr(self, k, v)


def test_status_has_precedence_over_message():
    assert classify_retryable(_Http("server_error: overloaded", 500)) is None
    assert classify_retryable(_Http("400 rate limit exceeded quoted", 400)) is None
    assert classify_retryable(_Http("anything", 429)) is not None
    assert classify_retryable(_Http("anything", 503)) is not None
    # no status: message classification still applies
    assert classify_retryable(Exception("Rate limit reached")) is not None


def test_error_type_is_closed():
    from traigent.utils.vendor_retry import ERROR_TYPES, _error_type

    for exc in (_Http("x", 500), RateLimitError("r"), ValueError("v"), _Http("y", 429)):
        assert _error_type(exc) in ERROR_TYPES
    assert _error_type(ValueError("v")) == "other"


# ---------------------------------------------------------------- waiting


def test_vendor_wait_forms():
    from email.utils import format_datetime
    from datetime import UTC, datetime, timedelta

    from traigent.utils.vendor_retry import vendor_wait_seconds

    assert vendor_wait_seconds(_Http("x", 429, {"retry-after": "7"})) == 7.0
    assert vendor_wait_seconds(_Http("x", 429, {"Retry-After-Ms": "1500"})) == 1.5
    assert vendor_wait_seconds(_Http("x", 429, retry_after=3)) == 3.0
    resp = types.SimpleNamespace(headers={"retry-after": "4"})
    assert vendor_wait_seconds(_Http("x", 429, response=resp)) == 4.0
    lit = _Http("x", 429, litellm_response_headers={"retry-after": "5"})
    assert vendor_wait_seconds(lit) == 5.0
    when = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    got = vendor_wait_seconds(_Http("x", 429, {"retry-after": when}))
    assert 20 <= got <= 31
    assert vendor_wait_seconds(_Http("x", 429, {"retry-after": "junk"})) is None
    assert vendor_wait_seconds(_Http("x", 429)) is None


def test_wait_over_cap_stops_instead_of_retrying(mock_off, monkeypatch):
    monkeypatch.setenv(RETRIES, "3")
    exc = _Http("429", 429, {"retry-after": "500"})
    mod, p = _patched([exc])
    try:
        with patch("time.sleep") as sleep, capture_scope() as bucket:
            with pytest.raises(_Http):
                mod.completion(model="gpt-4o")
        sleep.assert_not_called()
        assert mod.calls["n"] == 1
        assert bucket.vendor_attempts[0]["retried"] is False
    finally:
        p.stop()


def test_retry_after_ms_header_used_for_sleep(mock_off, monkeypatch):
    monkeypatch.setenv(RETRIES, "1")
    mod, p = _patched([_Http("429", 429, {"retry-after-ms": "250"})])
    try:
        with patch("time.sleep") as sleep:
            mod.completion(model="gpt-4o")
        sleep.assert_called_once_with(0.25)
    finally:
        p.stop()


def test_http_500_not_retried(mock_off, monkeypatch):
    monkeypatch.setenv(RETRIES, "3")
    mod, p = _patched([_Http("server_error", 500)])
    try:
        with pytest.raises(_Http):
            mod.completion(model="gpt-4o")
        assert mod.calls["n"] == 1
    finally:
        p.stop()


# ----------------------------------------------------------------- streams


class _Chunk:
    def __init__(self, usage=None):
        self.usage = usage


class _SyncStream:
    def __init__(self, fail_after=None):
        self.n = 0
        self.fail_after = fail_after

    def __iter__(self):
        return self

    def __next__(self):
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            raise RateLimitError("429 mid stream")
        if self.n > 3:
            raise StopIteration
        return _Chunk(
            MagicMock(prompt_tokens=1, completion_tokens=2, total_tokens=3)
            if self.n == 3
            else None
        )


class _AsyncStream:
    def __init__(self, fail_after=None):
        self.inner = _SyncStream(fail_after)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.inner)
        except StopIteration:
            raise StopAsyncIteration from None


def test_stream_success_recorded_only_on_exhaustion(mock_off, monkeypatch):
    monkeypatch.setenv(RETRIES, "1")
    mod, p = _patched([_SyncStream()])
    try:
        with capture_scope() as bucket:
            stream = mod.completion(model="gpt-4o", stream=True)
            # counted at open, but not finished until consumed
            assert [a["outcome"] for a in bucket.vendor_attempts] == ["in_progress"]
            assert len(list(stream)) == 3
        assert len(bucket.vendor_attempts) == 1
        rec = bucket.vendor_attempts[0]
        assert rec["outcome"] == "success" and rec["usage"]["total_tokens"] == 3
    finally:
        p.stop()


def test_stream_failure_mid_consumption_recorded_not_retried(mock_off, monkeypatch):
    monkeypatch.setenv(RETRIES, "3")
    mod, p = _patched([_SyncStream(fail_after=1)])
    try:
        with capture_scope() as bucket:
            stream = mod.completion(model="gpt-4o", stream=True)
            with pytest.raises(RateLimitError):
                list(stream)
        assert mod.calls["n"] == 1
        (rec,) = bucket.vendor_attempts
        assert rec["outcome"] == "error" and rec["retried"] is False
        assert rec["usage_known"] is False
    finally:
        p.stop()


@pytest.mark.asyncio
async def test_async_stream_failure_recorded_not_retried(mock_off, monkeypatch):
    monkeypatch.setenv(RETRIES, "3")
    mod, p = _patched([_AsyncStream(fail_after=1)])
    try:
        with capture_scope() as bucket:
            stream = await mod.acompletion(model="gpt-4o", stream=True)
            with pytest.raises(RateLimitError):
                async for _ in stream:
                    pass
        assert mod.calls["n"] == 1
        (rec,) = bucket.vendor_attempts
        assert rec["outcome"] == "error"
    finally:
        p.stop()


def test_stream_open_failure_is_retried(mock_off, monkeypatch):
    monkeypatch.setenv(RETRIES, "1")
    mod, p = _patched([RateLimitError("429"), _SyncStream()])
    try:
        with capture_scope() as bucket:
            list(mod.completion(model="gpt-4o", stream=True))
        assert mod.calls["n"] == 2
        assert [a["outcome"] for a in bucket.vendor_attempts] == ["error", "success"]
    finally:
        p.stop()


# -------------------------------------------------------------- truncation


def test_totals_survive_record_truncation(monkeypatch):
    import traigent.utils.langchain_interceptor as li

    monkeypatch.setattr(li, "_MAX_VENDOR_ATTEMPT_RECORDS", 3)
    with capture_scope() as bucket:
        for i in range(10):
            li.record_vendor_attempt(
                {"retried": i % 2 == 0, "usage_known": i % 2 == 1, "outcome": "x"}
            )
    s = summarize_bucket(bucket)
    assert s["vendor_call_attempts"] == 10
    assert s["vendor_call_retries"] == 5
    assert s["vendor_call_unknown_usage_attempts"] == 5
    assert len(s["vendor_call_attempt_records"]) == 3
    assert s["vendor_call_attempt_records_truncated"] is True


def test_nested_scope_merges_into_parent():
    from traigent.utils.langchain_interceptor import record_vendor_attempt

    with capture_scope() as outer:
        with capture_scope():
            record_vendor_attempt({"retried": True, "usage_known": False})
        record_vendor_attempt({"usage_known": True})
    assert outer.vendor_attempt_total == 2 and outer.vendor_retry_total == 1


# -------------------------------------------------- honest cost / e2e run


def test_retried_failure_marks_trial_cost_unmeasured_keeping_floor():
    from traigent.utils.langchain_interceptor import record_vendor_attempt
    from traigent.utils.trial_costs import is_cost_unmeasured

    with capture_scope() as bucket:
        record_vendor_attempt({"retried": True, "usage_known": False})
        record_vendor_attempt({"usage_known": True})
    result = types.SimpleNamespace(metadata={}, metrics={"cost": 0.25})
    TrialLifecycle._apply_vendor_attempts(result, bucket)
    assert is_cost_unmeasured(result.metrics)
    assert result.metrics["cost"] == 0.25  # known spend kept as the floor

    with capture_scope() as clean:
        record_vendor_attempt({"usage_known": True})
    result2 = types.SimpleNamespace(metadata={}, metrics={"cost": 0.25})
    TrialLifecycle._apply_vendor_attempts(result2, clean)
    assert not is_cost_unmeasured(result2.metrics)


def _run_real_optimize(tmp_path, monkeypatch, *, strict, inject=True):
    import asyncio
    import json

    import litellm  # noqa: F401  (installed in dev env)

    import traigent

    root = tmp_path / "ds"
    root.mkdir()
    ds = root / "d.jsonl"
    ds.write_text(
        "\n".join(
            json.dumps({"input": {"q": f"q{i}"}, "output": "mock"}) for i in range(10)
        )
    )
    monkeypatch.chdir(root)
    for k, v in {
        "TRAIGENT_MOCK_LLM": "true",
        "TRAIGENT_OFFLINE_MODE": "true",
        "TRAIGENT_DATASET_ROOT": str(root),
        "TRAIGENT_RESULTS_FOLDER": str(tmp_path / "results"),
        "TRAIGENT_VENDOR_MAX_RETRIES": "1",
        "TRAIGENT_STRICT_COST_ACCOUNTING": "true" if strict else "false",
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("TRAIGENT_REQUIRE_COST_TRACKING", raising=False)

    @traigent.optimize(
        eval_dataset=str(ds),
        objectives=["accuracy", "cost"],
        configuration_space={"model": ["gpt-4o-mini", "gpt-4o"]},
        injection_mode="parameter",
    )
    def agent(q, config=None):
        import litellm as ll

        r = ll.completion(
            model=config["model"], messages=[{"role": "user", "content": q}]
        )
        return r.choices[0].message.content

    async def go():
        if inject:
            with inject_vendor_failures(1, per_call=False):
                return await agent.optimize(max_trials=1, algorithm="grid")
        return await agent.optimize(max_trials=1, algorithm="grid")

    return asyncio.run(go())


def test_e2e_optimize_mock_mode_one_injected_failure_non_strict(tmp_path, monkeypatch):
    from traigent.utils.trial_costs import is_cost_unmeasured

    result = _run_real_optimize(tmp_path, monkeypatch, strict=False)
    assert result.metadata["vendor_call_attempts"] == 11
    assert result.metadata["vendor_call_retries"] == 1
    assert len(result.metadata["vendor_call_attempt_records"]) == 11
    trial = result.trials[0]
    assert trial.metadata["vendor_call_attempts"] == 11
    assert is_cost_unmeasured(trial.metrics)


def test_e2e_optimize_strict_cost_raises(tmp_path, monkeypatch):
    from traigent.core.cost_enforcement import CostTrackingRequiredError

    with pytest.raises(Exception) as info:
        _run_real_optimize(tmp_path, monkeypatch, strict=True)
    chain = []
    exc = info.value
    while exc is not None and exc not in chain:
        chain.append(exc)
        exc = exc.__cause__ or exc.__context__
    assert any(isinstance(e, CostTrackingRequiredError) for e in chain)


def test_e2e_without_injection_is_unchanged(tmp_path, monkeypatch):
    from traigent.utils.trial_costs import is_cost_unmeasured

    result = _run_real_optimize(tmp_path, monkeypatch, strict=True, inject=False)
    assert result.metadata["vendor_call_attempts"] == 10
    assert result.metadata["vendor_call_retries"] == 0
    assert not is_cost_unmeasured(result.trials[0].metrics)


# ------------------------------------------------------------ review round 2


@pytest.mark.asyncio
async def test_outer_retry_honours_wait_cap_and_header_forms():
    from traigent.core.exception_handler import VendorErrorCategory
    from traigent.core.orchestrator import OptimizationOrchestrator
    from traigent.utils.exceptions import VendorPauseError

    def make(error):
        o = MagicMock()
        o._vendor_retry_counts = {}
        o._vendor_retry_settings = OptimizationOrchestrator._vendor_retry_settings
        o._maybe_auto_retry_vendor_error = (
            OptimizationOrchestrator._maybe_auto_retry_vendor_error.__get__(o)
        )
        return o, VendorPauseError(
            "rl", original_error=error, category=VendorErrorCategory.RATE_LIMIT
        )

    import os

    os.environ[RETRIES] = "2"
    try:
        o, exc = make(RateLimitError("429"))
        exc.original_error.headers = {"retry-after": "500"}
        assert await o._maybe_auto_retry_vendor_error(exc) is None
        assert o._vendor_retry_counts == {}  # budget not consumed by a refusal

        o, exc = make(RateLimitError("429"))
        exc.original_error.headers = {"retry-after-ms": "10"}
        with patch("asyncio.sleep") as sleep:
            sleep.return_value = None
            assert await o._maybe_auto_retry_vendor_error(exc) == "continue"
        assert sleep.call_args.args[0] == 0.01
    finally:
        os.environ.pop(RETRIES, None)


def test_sync_stream_close_early_recorded_once(mock_off, monkeypatch):
    mod, p = _patched([_SyncStream()])
    try:
        with capture_scope() as bucket:
            stream = mod.completion(model="gpt-4o", stream=True)
            next(stream)
            stream.close()
            stream.close()
        (rec,) = bucket.vendor_attempts
        assert rec["outcome"] == "closed_early" and rec["usage_known"] is False
    finally:
        p.stop()


@pytest.mark.asyncio
async def test_async_stream_aclose_and_cancellation(mock_off):
    import asyncio

    mod, p = _patched([_AsyncStream(), _AsyncStream()])
    try:
        with capture_scope() as bucket:
            stream = await mod.acompletion(model="gpt-4o", stream=True)
            await stream.__anext__()
            await stream.aclose()
            await stream.aclose()
            assert [a["outcome"] for a in bucket.vendor_attempts] == ["closed_early"]

            stream2 = await mod.acompletion(model="gpt-4o", stream=True)

            async def boom():
                raise asyncio.CancelledError

            stream2._stream.__anext__ = boom
            with pytest.raises(asyncio.CancelledError):
                await stream2.__anext__()
            assert bucket.vendor_attempts[-1]["outcome"] == "cancelled"
            assert len(bucket.vendor_attempts) == 2
    finally:
        p.stop()


def test_stream_records_into_originating_bucket(mock_off):
    mod, p = _patched([_SyncStream(), _SyncStream()])
    try:
        with capture_scope() as opener:
            s1 = mod.completion(model="gpt-4o", stream=True)
            s2 = mod.completion(model="gpt-4o", stream=True)
        # consumed in a different scope, and outside any scope
        with capture_scope() as other:
            list(s1)
        list(s2)
        assert other.vendor_attempt_total == 0
        assert opener.vendor_attempt_total == 2
    finally:
        p.stop()


def test_late_stream_in_nested_scope_forwards_to_parent(mock_off):
    mod, p = _patched([_SyncStream()])
    try:
        with capture_scope() as outer:
            with capture_scope():
                s = mod.completion(model="gpt-4o", stream=True)
            list(s)  # inner scope already merged and closed
        assert outer.vendor_attempt_total == 1
    finally:
        p.stop()


def test_sibling_nested_scopes_get_distinct_call_ids():
    from traigent.utils.langchain_interceptor import (
        begin_vendor_call,
        record_vendor_attempt,
    )

    with capture_scope() as outer:
        for _ in range(2):
            with capture_scope():
                idx = begin_vendor_call()
                record_vendor_attempt({"call_index": idx, "usage_known": True})
        idx = begin_vendor_call()
        record_vendor_attempt({"call_index": idx, "usage_known": True})
    ids = [r["call_index"] for r in outer.vendor_attempts]
    assert len(set(ids)) == 3
    assert outer.vendor_attempt_total == 3


# ------------------------------------------------------------ review round 3


def test_concurrent_worker_recording_during_scope_close_lands_once():
    import contextvars
    import threading

    from traigent.utils.langchain_interceptor import record_vendor_attempt

    barrier = threading.Barrier(2)
    done = threading.Event()

    with capture_scope() as outer:
        with capture_scope():
            ctx = contextvars.copy_context()

            def worker():
                barrier.wait()  # released together with the scope close
                ctx.run(record_vendor_attempt, {"usage_known": True})
                done.set()

            t = threading.Thread(target=worker)
            t.start()
            barrier.wait()
        # inner scope has closed; the worker may record before or after
        t.join()
    assert done.is_set()
    assert outer.vendor_attempt_total == 1
    assert len(outer.vendor_attempts) == 1


@pytest.mark.asyncio
async def test_async_cancel_between_pulls_still_counts_one_attempt(mock_off):
    mod, p = _patched([_AsyncStream()])
    try:
        with capture_scope() as bucket:
            stream = await mod.acompletion(model="gpt-4o", stream=True)
            await stream.__anext__()
            # the consumer is cancelled in its loop body: the proxy is never
            # asked again and never closed
        s = summarize_bucket(bucket)
        assert s["vendor_call_attempts"] == 1
        assert s["vendor_call_incomplete_attempts"] == 1
        assert s["vendor_call_unknown_usage_attempts"] == 1
        assert s["vendor_call_attempt_records"][0]["outcome"] == "incomplete"

        result = types.SimpleNamespace(metadata={}, metrics={"cost": 0.1})
        TrialLifecycle._apply_vendor_attempts(result, bucket)
        from traigent.utils.trial_costs import is_cost_unmeasured

        assert is_cost_unmeasured(result.metrics)
    finally:
        p.stop()


def test_late_call_in_ended_scope_gets_unique_id(mock_off):
    mod, p = _patched([])
    try:
        with capture_scope() as outer:
            with capture_scope() as inner:
                mod.completion(model="gpt-4o")
            # a worker still holding the ended inner scope's context
            import contextvars

            from traigent.utils.langchain_interceptor import _capture_scope

            ctx = contextvars.copy_context()
            ctx.run(_capture_scope.set, inner)
            ctx.run(mod.completion, model="gpt-4o")
            mod.completion(model="gpt-4o")
        ids = [a["call_index"] for a in outer.vendor_attempts]
        assert len(ids) == 3 and len(set(ids)) == 3
    finally:
        p.stop()


# ------------------------------------------------------------ review round 4


def test_trial_scopes_inside_an_outer_scope_own_separate_ledgers():
    from traigent.utils.langchain_interceptor import (
        begin_vendor_call,
        record_vendor_attempt,
    )

    with capture_scope() as outer:
        with capture_scope(ledger_owner=True) as t1:
            with capture_scope():  # the evaluator's nested scope
                record_vendor_attempt(
                    {
                        "call_index": begin_vendor_call(),
                        "retried": True,
                        "usage_known": False,
                    }
                )
        with capture_scope(ledger_owner=True) as t2:
            with capture_scope():
                record_vendor_attempt(
                    {"call_index": begin_vendor_call(), "usage_known": True}
                )
    s1, s2 = summarize_bucket(t1), summarize_bucket(t2)
    assert s1["vendor_call_attempts"] == 1 and s1["vendor_call_retries"] == 1
    assert s2["vendor_call_attempts"] == 1 and s2["vendor_call_retries"] == 0
    assert outer.vendor_attempt_total == 0

    # a sibling's retry does not flag the other trial's cost
    from traigent.utils.trial_costs import is_cost_unmeasured

    r1 = types.SimpleNamespace(metadata={}, metrics={"cost": 0.1})
    r2 = types.SimpleNamespace(metadata={}, metrics={"cost": 0.1})
    TrialLifecycle._apply_vendor_attempts(r1, t1)
    TrialLifecycle._apply_vendor_attempts(r2, t2)
    assert is_cost_unmeasured(r1.metrics) and not is_cost_unmeasured(r2.metrics)


@pytest.mark.asyncio
async def test_parallel_trials_in_outer_scope_report_own_attempts_and_sum():
    import asyncio

    from traigent.core.orchestrator import OptimizationOrchestrator
    from traigent.utils.langchain_interceptor import (
        begin_vendor_call,
        record_vendor_attempt,
    )

    async def trial(n_attempts, retried):
        async with capture_scope(ledger_owner=True) as bucket:
            async with capture_scope():
                for i in range(n_attempts):
                    await asyncio.sleep(0)
                    record_vendor_attempt(
                        {
                            "call_index": begin_vendor_call(),
                            "retried": retried and i == 0,
                            "usage_known": not (retried and i == 0),
                        }
                    )
        result = types.SimpleNamespace(
            trial_id=str(n_attempts), metadata={}, metrics={}
        )
        TrialLifecycle._apply_vendor_attempts(result, bucket)
        return result

    with capture_scope():
        a, b = await asyncio.gather(trial(2, True), trial(3, False))
    assert a.metadata["vendor_call_attempts"] == 2
    assert b.metadata["vendor_call_attempts"] == 3
    assert "cost_unmeasured" in a.metrics and "cost_unmeasured" not in b.metrics
    totals = OptimizationOrchestrator._vendor_attempt_totals(
        types.SimpleNamespace(_trials=[a, b], _abandoned_vendor_attempts=[])
    )
    assert totals["vendor_call_attempts"] == 5
    assert totals["vendor_call_retries"] == 1
