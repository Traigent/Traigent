"""Tests for `traigent auth whoami` status classification."""

from __future__ import annotations

import sys
import types
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
    fake_module.ClientSession = lambda timeout=None: _FakeSession(  # noqa: ARG005
        response=response,
        error=error,
    )
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


@pytest.mark.parametrize("status", [401, 403])
def test_whoami_auth_failures_classified(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(status=status, text_payload="unauthorized"),
    )

    result = _run_whoami(monkeypatch)
    assert result.exit_code == 1
    assert "Invalid or unauthorized API key" in result.output
    assert "Category:" in result.output
    assert "authentication" in result.output
    assert f"HTTP status: {status}" in result.output


def test_whoami_404_backend_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(status=404, text_payload="not found"),
    )

    result = _run_whoami(monkeypatch)
    assert result.exit_code == 1
    assert "Backend endpoint mismatch" in result.output
    assert "backend_endpoint_mismatch" in result.output
    assert "TRAIGENT_BACKEND_URL / TRAIGENT_API_URL" in result.output


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
) -> None:
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(status=status, text_payload="simulated backend failure"),
    )

    result = _run_whoami(monkeypatch)
    assert result.exit_code == 1
    assert message_fragment in result.output
    assert category in result.output
    assert f"HTTP status: {status}" in result.output


def test_whoami_connectivity_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_aiohttp = _install_fake_aiohttp(monkeypatch)
    fake_aiohttp.ClientSession = lambda timeout=None: _FakeSession(  # noqa: ARG005
        error=fake_aiohttp.ClientError("connection refused")
    )

    result = _run_whoami(monkeypatch)
    assert result.exit_code == 1
    assert "Cannot reach backend to validate API key" in result.output
    assert "connectivity_error" in result.output


def test_whoami_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_aiohttp(monkeypatch, error=TimeoutError("timed out"))

    result = _run_whoami(monkeypatch)
    assert result.exit_code == 1
    assert "Cannot reach backend to validate API key" in result.output
    assert "connectivity_error" in result.output
