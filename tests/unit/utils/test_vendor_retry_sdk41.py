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
    summarize_attempts,
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
        s = summarize_attempts(bucket.vendor_attempts)
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
        s = summarize_attempts(bucket.vendor_attempts)
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
