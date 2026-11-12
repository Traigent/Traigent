"""Retry-After handling and structured error classification (#3010, #3011)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from unittest.mock import patch

import pytest

from traigent.cloud.resilient_client import ErrorType, ResilientClient
from traigent.utils.exceptions import RateLimitError, RetryableError
from traigent.utils.retry import RetryConfig, RetryHandler
from traigent.utils.retry_after import (
    MAX_RETRY_AFTER_SECONDS,
    parse_retry_after,
    retry_after_from_headers,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _StatusError(RuntimeError):
    def __init__(self, msg: str, status_code: int) -> None:
        super().__init__(msg)
        self.status_code = status_code


class TestParser:
    def test_delta_seconds_and_http_date(self):
        assert parse_retry_after("30") == 30.0
        date = format_datetime(NOW + timedelta(seconds=45), usegmt=True)
        assert parse_retry_after(date, now=lambda: NOW) == 45.0

    @pytest.mark.parametrize(
        "bad", [None, "", "abc", "-1", "nan", "inf", True, float("nan")]
    )
    def test_invalid_is_none(self, bad):
        assert parse_retry_after(bad) is None

    def test_not_clamped(self):
        assert parse_retry_after("100000") == 100000.0

    def test_headers_case_insensitive_lookup(self):
        assert retry_after_from_headers({"retry-after": "7"}) == 7.0
        assert retry_after_from_headers(None) is None


class TestClassification:
    def test_login_5xx_text_with_authentication_is_server(self):
        err = RuntimeError("Backend authentication failed with HTTP 503")
        assert ResilientClient().classify_error(err) is ErrorType.SERVER

    def test_preset_is_not_reset(self):
        err = RuntimeError("invalid preset value")
        assert ResilientClient().classify_error(err) is not ErrorType.NETWORK
        assert not ResilientClient().should_retry(err, 0)

    def test_structured_403_on_retryable_type_is_not_retried(self):
        class Boom(RetryableError):
            status_code = 403

        assert not ResilientClient().should_retry(Boom("x"), 0)

    @pytest.mark.parametrize("status", [401, 403])
    def test_structured_auth_status_never_retried(self, status):
        err = _StatusError("connection reset", status)
        assert not ResilientClient().should_retry(err, 0)

    def test_refresh_429_retry_after_is_used(self):
        client = ResilientClient(
            max_retries=3, base_delay=1.0, max_delay=10.0, jitter_factor=0
        )
        assert client.calculate_delay(0, RateLimitError("x", retry_after=7)) == 7.0
        date = format_datetime(datetime.now(UTC) + timedelta(seconds=8), usegmt=True)
        assert (
            6.0
            <= client.calculate_delay(0, RateLimitError("x", retry_after=date))
            <= 9.0
        )

    def test_retry_after_beyond_ceiling_stops_retry(self):
        client = ResilientClient(max_retries=3, max_delay=10.0)
        assert not client.should_retry(RateLimitError("x", retry_after=30), 0)
        assert client.should_retry(RateLimitError("x", retry_after=5), 0)


class TestRetryHandler:
    def _handler(self):
        return RetryHandler(
            RetryConfig(max_attempts=3, initial_delay=0.1, max_delay=2.0, jitter=False)
        )

    def test_honours_retry_after_above_backoff_ceiling_without_sleeping(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] < 2:
                raise RateLimitError("slow down", retry_after=30)
            return "ok"

        with patch("traigent.utils.retry.time.sleep") as sleep:
            assert self._handler().execute(fn) == "ok"
        sleep.assert_called_once_with(30.0)

    def test_http_date_is_honoured(self):
        date = format_datetime(datetime.now(UTC) + timedelta(seconds=20), usegmt=True)
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] < 2:
                raise RateLimitError("slow down", retry_after=date)
            return "ok"

        with patch("traigent.utils.retry.time.sleep") as sleep:
            self._handler().execute(fn)
        assert 18.0 <= sleep.call_args.args[0] <= 21.0

    def test_ask_beyond_ceiling_raises_without_retry(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise RateLimitError("slow down", retry_after=MAX_RETRY_AFTER_SECONDS + 1)

        with patch("traigent.utils.retry.time.sleep") as sleep:
            outcome = self._handler().execute_with_result(fn)
        assert outcome.success is False
        assert isinstance(outcome.error, RateLimitError)
        assert calls["n"] == 1
        sleep.assert_not_called()

    def test_async_ask_beyond_ceiling_raises_without_retry(self):
        calls = {"n": 0}

        async def fn():
            calls["n"] += 1
            raise RateLimitError("slow down", retry_after=999)

        with patch("traigent.utils.retry.asyncio.sleep") as sleep:
            with pytest.raises(RateLimitError):
                asyncio.run(self._handler().execute_async(fn))
        assert calls["n"] == 1
        sleep.assert_not_called()


class TestSubmitAndFinalize:
    def test_submit_delay_honours_date_and_stops_beyond_ceiling(self):
        from traigent.cloud.trial_operations import _submit_retry_delay

        class R:
            headers = {"Retry-After": "30"}

        assert _submit_retry_delay(R, 0) == 30.0
        R.headers = {"Retry-After": "600"}
        assert _submit_retry_delay(R, 0) is None
        R.headers = {}
        assert _submit_retry_delay(R, 1) == 1.0

    def test_finalize_does_not_retry_403(self):
        from traigent.core.backend_session_manager import _is_transient_finalize_error

        assert not _is_transient_finalize_error(_StatusError("denied", 403))
        assert _is_transient_finalize_error(_StatusError("busy", 429))

    def test_hybrid_transport_parses_http_date(self):
        import httpx

        from traigent.hybrid.http_transport import HTTPTransport

        date = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
        got = HTTPTransport._parse_retry_after(httpx.Headers({"Retry-After": date}))
        assert got is not None and 28.0 <= got <= 31.0
