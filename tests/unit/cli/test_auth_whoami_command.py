"""Tests for `traigent auth whoami` status classification."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from typing import Any

import pytest
from click.testing import CliRunner

from traigent.cli import auth_commands


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int,
        json_payload: dict[str, Any] | None = None,
        text_payload: str = "",
    ) -> None:
        self.status = status
        self._json_payload = json_payload or {}
        self._text_payload = text_payload

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def json(self, content_type: str | None = None) -> dict[str, Any]:
        return self._json_payload

    async def text(self) -> str:
        return self._text_payload


class _FakeSession:
    last_post_kwargs = None

    def __init__(
        self,
        *,
        response: _FakeResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._error = error

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def get(self, url: str, headers: dict[str, str]) -> _FakeResponse:
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response

    def post(
        self, url: str, headers: dict[str, str] | None = None, **kwargs: Any
    ) -> _FakeResponse:
        if self._error is not None:
            raise self._error
        assert self._response is not None
        type(self).last_post_kwargs = {
            "url": url,
            "headers": headers,
            **kwargs,
        }
        return self._response


def _install_fake_aiohttp(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: _FakeResponse | None = None,
    error: Exception | None = None,
) -> Any:
    _FakeSession.last_post_kwargs = None
    fake_module = types.SimpleNamespace()

    class _ClientError(Exception):
        pass

    fake_module.ClientError = _ClientError
    fake_module.ClientTimeout = lambda total=15: types.SimpleNamespace(total=total)

    def _client_session(**kwargs: Any) -> _FakeSession:
        assert kwargs.get("trust_env") is True
        return _FakeSession(response=response, error=error)

    fake_module.ClientSession = _client_session
    monkeypatch.setitem(sys.modules, "aiohttp", fake_module)
    return fake_module


def _run_whoami(
    monkeypatch: pytest.MonkeyPatch, api_key: str | None = "tg_test_key"
) -> Any:
    monkeypatch.setattr(
        auth_commands.BackendConfig,
        "get_backend_api_url",
        staticmethod(lambda: "http://localhost:5000/api/v1"),
    )
    runner = CliRunner()
    args = ["whoami"] if api_key is None else ["whoami", api_key]
    return runner.invoke(auth_commands.auth, args)


def test_whoami_valid_key_200(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(
            status=200,
            json_payload={
                "valid": True,
                "data": {
                    "email": "dev@traigent.ai",
                    "name": "Dev User",
                    "organization": "Traigent",
                },
            },
        ),
    )

    result = _run_whoami(monkeypatch)
    assert result.exit_code == 0
    assert "✅ Valid" in result.output
    assert "Category" in result.output
    assert "authenticated" in result.output


def test_whoami_posts_json_payload_to_validate_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = "sk_" + "a" * 43  # pragma: allowlist secret
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(status=200, json_payload={"valid": True, "data": {}}),
    )

    result = _run_whoami(monkeypatch, api_key=api_key)

    assert result.exit_code == 0
    assert _FakeSession.last_post_kwargs is not None
    assert _FakeSession.last_post_kwargs["json"] == {"api_key": api_key}
    assert (
        _FakeSession.last_post_kwargs["headers"]["Content-Type"] == "application/json"
    )


def test_whoami_uses_env_api_key_when_argument_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = "tg_" + "a" * 43
    monkeypatch.setenv("TRAIGENT_API_KEY", api_key)
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(status=200, json_payload={"valid": True, "data": {}}),
    )

    result = _run_whoami(monkeypatch, api_key=None)

    assert result.exit_code == 0
    assert _FakeSession.last_post_kwargs is not None
    assert _FakeSession.last_post_kwargs["json"] == {"api_key": api_key}


def test_whoami_requires_argument_or_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)

    result = _run_whoami(monkeypatch, api_key=None)

    assert result.exit_code == 1
    assert "Missing API key" in result.output
    assert "TRAIGENT_API_KEY" in result.output


@pytest.mark.parametrize("prefix", ["tg_", "uk_", "sk_", "ak_", "tk_"])
def test_whoami_accepts_backend_issued_prefixes(
    monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(
            status=200,
            json_payload={
                "valid": True,
                "data": {"email": "dev@traigent.ai"},
            },
        ),
    )

    key = prefix + "a" * 43
    result = _run_whoami(monkeypatch, api_key=key)
    assert result.exit_code == 0
    assert "✅ Valid" in result.output


# This test changed deliberately (#1775). It previously asserted that BOTH 401 and 403
# print "Invalid or unauthorized API key" under category "authentication" -- i.e. it
# pinned the exact collapse the issue reports. #1754 / PR #1762 split these on the
# session path; the CLI kept the collapse, so an insufficient-scope 403 told the user
# to rotate a perfectly valid key. The parametrisation now carries the EXPECTED
# distinction rather than asserting the two are the same.
@pytest.mark.parametrize(
    "status,expected_fragment,expected_category",
    [
        (401, "Invalid or expired API key", "authentication"),
        (403, "lacks the required scope", "authorization"),
    ],
)
def test_whoami_auth_failures_classified(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    expected_fragment: str,
    expected_category: str,
    plain: Callable[[str], str],
) -> None:
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(status=status, text_payload="unauthorized"),
    )

    result = _run_whoami(monkeypatch)
    output = plain(result.output)
    assert result.exit_code == 1
    assert expected_fragment in output
    assert "Category:" in output
    assert expected_category in output
    assert f"HTTP status: {status}" in output


def test_whoami_403_from_the_edge_is_not_reported_as_a_scope_problem(
    monkeypatch: pytest.MonkeyPatch, plain: Callable[[str], str]
) -> None:
    """A Cloudflare 403 never reached Traigent, so neither key nor scope is at fault."""
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(
            status=403,
            text_payload="Attention Required! | Cloudflare (error code: 1010)",
        ),
    )

    output = plain(_run_whoami(monkeypatch).output)

    assert "edge" in output.lower()
    assert "lacks the required scope" not in output


def test_whoami_404_backend_mismatch(
    monkeypatch: pytest.MonkeyPatch, plain: Callable[[str], str]
) -> None:
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(status=404, text_payload="not found"),
    )

    result = _run_whoami(monkeypatch)
    output = plain(result.output)
    assert result.exit_code == 1
    assert "Backend endpoint mismatch" in output
    assert "backend_endpoint_mismatch" in output
    assert "TRAIGENT_BACKEND_URL / TRAIGENT_API_URL" in output


@pytest.mark.parametrize(
    ("status", "category", "message_fragment"),
    [
        (408, "timeout", "Backend request timed out"),
        (409, "backend_conflict", "Backend reported a request conflict"),
        (429, "rate_limited", "Backend rate limit exceeded"),
        (500, "server_error", "Backend server error"),
        (503, "server_error", "Backend server error"),
    ],
)
def test_whoami_extended_status_classification(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    category: str,
    message_fragment: str,
    plain: Callable[[str], str],
) -> None:
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(status=status, text_payload="simulated backend failure"),
    )

    result = _run_whoami(monkeypatch)
    output = plain(result.output)
    assert result.exit_code == 1
    assert message_fragment in output
    assert category in output
    assert f"HTTP status: {status}" in output


def test_whoami_connectivity_error(
    monkeypatch: pytest.MonkeyPatch,
    plain: Callable[[str], str],
) -> None:
    fake_aiohttp = _install_fake_aiohttp(monkeypatch)

    def _client_session(**kwargs: Any) -> _FakeSession:
        assert kwargs.get("trust_env") is True
        return _FakeSession(error=fake_aiohttp.ClientError("connection refused"))

    fake_aiohttp.ClientSession = _client_session

    result = _run_whoami(monkeypatch)
    output = plain(result.output)
    assert result.exit_code == 1
    assert "Cannot reach backend to validate API key" in output
    assert "connectivity_error" in output


def test_whoami_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
    plain: Callable[[str], str],
) -> None:
    _install_fake_aiohttp(monkeypatch, error=TimeoutError("timed out"))

    result = _run_whoami(monkeypatch)
    output = plain(result.output)
    assert result.exit_code == 1
    assert "Cannot reach backend to validate API key" in output
    assert "connectivity_error" in output
