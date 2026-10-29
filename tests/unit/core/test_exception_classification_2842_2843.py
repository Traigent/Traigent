"""#2842/#2843: user-code errors must not read as provider auth/quota failures."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from traigent.core.exception_handler import (
    VendorErrorCategory,
    classify_systematic_provider_failure,
    classify_vendor_error,
)
from traigent.invokers.local import LocalInvoker
from traigent.utils.error_handler import APIKeyError


class _HttpError(Exception):
    def __init__(self, status: int, text: str = "failure") -> None:
        super().__init__(text)
        self.status_code = status


@pytest.mark.parametrize(
    "text",
    [
        "Function call failed: PermissionError: [Errno 13] Permission denied: '/data/spider.sqlite'",
        "Function call failed: OperationalError: no such column: quota_used",
        "Function call failed: table 'forbidden_words' not found",
        "KeyError: 'api_key_hash'",
        "Max tokens exceeded for batch 4290",
    ],
)
def test_agent_error_text_is_not_a_provider_failure(text: str) -> None:
    assert classify_vendor_error(RuntimeError(text)) is None
    assert classify_systematic_provider_failure(text) is None


def test_local_exception_types_are_never_text_matched() -> None:
    assert classify_vendor_error(PermissionError("permission denied")) is None
    assert classify_vendor_error(sqlite3.OperationalError("quota exceeded")) is None
    assert classify_vendor_error(KeyError("unauthorized")) is None


@pytest.mark.parametrize(
    ("exc", "category"),
    [
        (_HttpError(429), VendorErrorCategory.RATE_LIMIT),
        (_HttpError(401), VendorErrorCategory.AUTHENTICATION),
        (_HttpError(503), VendorErrorCategory.SERVICE_UNAVAILABLE),
        (RuntimeError("Error code: 429 - rate limit reached"), VendorErrorCategory.RATE_LIMIT),
        (RuntimeError("You exceeded your current quota"), VendorErrorCategory.QUOTA_EXHAUSTED),
        (RuntimeError("Error code: 401 invalid_api_key"), VendorErrorCategory.AUTHENTICATION),
        (RuntimeError("rate_limit_exceeded"), VendorErrorCategory.RATE_LIMIT),
    ],
)
def test_real_provider_failures_still_classify(exc, category) -> None:
    assert classify_vendor_error(exc) is category


@pytest.mark.parametrize(
    "exc",
    [
        sqlite3.OperationalError("no such table: authentication"),
        KeyError("api_key"),
        ValueError("authentication header missing in tool schema"),
    ],
)
def test_user_exception_text_is_not_an_api_key_failure(exc: Exception) -> None:
    from traigent.core.exception_handler import is_provider_api_key_failure

    assert is_provider_api_key_failure(exc) is False


def test_structured_auth_signals_are_api_key_failures() -> None:
    from traigent.core.exception_handler import is_provider_api_key_failure

    assert is_provider_api_key_failure(_HttpError(401)) is True
    assert is_provider_api_key_failure(_HttpError(403)) is True

    class AuthenticationError(Exception):
        pass

    assert is_provider_api_key_failure(AuthenticationError("bad")) is True


def test_real_sdk_authentication_errors_still_abort() -> None:
    from traigent.core.exception_handler import is_provider_api_key_failure

    openai = pytest.importorskip("openai")
    anthropic = pytest.importorskip("anthropic")
    import httpx

    response = httpx.Response(401, request=httpx.Request("POST", "https://x.test"))
    assert is_provider_api_key_failure(
        openai.AuthenticationError("nope", response=response, body=None)
    )
    assert is_provider_api_key_failure(
        anthropic.AuthenticationError("nope", response=response, body=None)
    )


def test_local_invoker_returns_per_example_failure_for_sql_error() -> None:
    def agent(text: str) -> str:
        conn = sqlite3.connect(":memory:")
        return str(conn.execute("select api_key from authentication").fetchall())

    result = asyncio.run(LocalInvoker().invoke(agent, {}, {"text": "x"}))
    assert result.is_successful is False
    assert "no such table" in (result.error or "")


def test_local_invoker_still_aborts_on_provider_auth_failure() -> None:
    def agent(text: str) -> str:
        raise _HttpError(401, "bad credentials")

    with pytest.raises(APIKeyError):
        asyncio.run(LocalInvoker().invoke(agent, {}, {"text": "x"}))
