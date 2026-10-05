"""Tests for `traigent auth whoami`: key resolution and status classification."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from traigent.cli import auth_commands
from traigent.cloud import credential_manager

ARGV_WARNING = (
    "Warning: passing an API key as a command-line argument exposes it in the "
    "process list and shell history. Run 'traigent auth whoami' without KEY instead."
)
MISSING_KEY_HINT = (
    "Set TRAIGENT_API_KEY in your environment, or run 'traigent auth login' "
    "to store credentials. If you already logged in, set "
    "TRAIGENT_MASTER_PASSWORD so the stored credentials can be unlocked."
)
REDACTED = "<redacted API key>"


def _leaked_fragments(key: str, output: str, prefix: str = "tg_") -> set[str]:
    """Every 8-character window of the key body that appears in the output.

    Whitespace is removed first, so a key Rich wrapped across lines still counts.
    """
    squashed = "".join(output.split())
    body = key[len(prefix) :]
    return {
        body[i : i + 8] for i in range(len(body) - 7) if body[i : i + 8] in squashed
    }


def _flat(output: str) -> str:
    """Collapse whitespace, so a message Rich wrapped at 80 columns reads as one."""
    return " ".join(output.split())


def _synthetic_key(label: str, prefix: str = "tg_") -> str:
    """Build a well-formed, non-repeating test key without a secret literal."""
    return prefix + hashlib.sha256(label.encode()).hexdigest()[:43]


class _FakeSecureStore:
    """Stands in for the encrypted credential store the CLI login writes to."""

    def __init__(self, payload: dict[str, Any] | None) -> None:
        self._serialized = None if payload is None else json.dumps(payload)

    def get(self, name: str, check_env: bool = True) -> str | None:
        if name == credential_manager.SECURE_CLI_CREDENTIAL_NAME:
            return self._serialized
        return None


def _use_secure_store(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any] | None
) -> None:
    store = _FakeSecureStore(payload)
    monkeypatch.setattr(
        credential_manager, "get_secure_credential_store", lambda: store
    )
    monkeypatch.setattr(auth_commands, "get_secure_credential_store", lambda: store)


@pytest.fixture(autouse=True)
def _isolate_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Keep every test away from the real environment key and ~/.traigent."""
    for name in (
        "TRAIGENT_API_KEY",
        "TRAIGENT_ALLOW_PLAINTEXT_CREDENTIALS",
        "TRAIGENT_DEV_API_KEY",
        "TRAIGENT_DEV_MODE",
        "TRAIGENT_GENERATE_MOCKS",
    ):
        monkeypatch.delenv(name, raising=False)
    legacy_file = tmp_path / ".traigent" / "credentials.json"
    monkeypatch.setattr(credential_manager, "CREDENTIALS_FILE", legacy_file)
    _use_secure_store(monkeypatch, None)
    return legacy_file


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int,
        json_payload: dict[str, Any] | None = None,
        text_payload: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self._json_payload = json_payload or {}
        self._text_payload = text_payload
        # Real aiohttp responses always carry headers, and the 403 classifier reads
        # them (cf-ray/cf-mitigated are headers, not body text). A double without
        # them cannot exercise that path.
        self.headers = headers or {}

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
    for name in ("get_backend_api_url", "get_cloud_api_url"):
        monkeypatch.setattr(
            auth_commands.BackendConfig,
            name,
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
    api_key = "tg_" + "a" * 43  # pragma: allowlist secret
    monkeypatch.setenv("TRAIGENT_API_KEY", api_key)
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(status=200, json_payload={"valid": True, "data": {}}),
    )

    result = _run_whoami(monkeypatch, api_key=None)

    assert result.exit_code == 0
    assert _FakeSession.last_post_kwargs is not None
    assert _FakeSession.last_post_kwargs["json"] == {"api_key": api_key}


def test_whoami_help_never_shows_a_key_in_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = CliRunner().invoke(auth_commands.auth, ["whoami", "--help"])

    assert result.exit_code == 0
    assert "tg_" not in result.output
    invocations = [
        line.split("traigent auth whoami", 1)[1]
        for line in result.output.splitlines()
        if "traigent auth whoami" in line
    ]
    assert invocations, result.output
    # Every example is the bare command: nothing typed after it can be a key.
    assert all(rest.strip() == "" for rest in invocations), invocations
    # Click rewraps help paragraphs to the terminal, so compare word sequences.
    words = " ".join(result.output.split())
    assert "process list" in words
    assert "shell history" in words


def test_whoami_without_argument_validates_stored_credentials(
    monkeypatch: pytest.MonkeyPatch, plain: Callable[[str], str]
) -> None:
    stored_key = _synthetic_key("stored")
    _use_secure_store(
        monkeypatch,
        {"api_key": stored_key, "backend_url": "http://localhost:5000"},
    )
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(status=200, json_payload={"valid": True, "data": {}}),
    )

    result = _run_whoami(monkeypatch, api_key=None)
    output = plain(result.output)

    assert result.exit_code == 0, output
    assert _FakeSession.last_post_kwargs is not None
    assert _FakeSession.last_post_kwargs["json"] == {"api_key": stored_key}
    assert "API key source: stored CLI credentials" in output
    assert stored_key not in output
    assert stored_key[3:11] not in output


def test_whoami_env_key_reports_source_and_wins_over_stored(
    monkeypatch: pytest.MonkeyPatch, plain: Callable[[str], str]
) -> None:
    env_key = _synthetic_key("env")
    monkeypatch.setenv("TRAIGENT_API_KEY", env_key)
    _use_secure_store(monkeypatch, {"api_key": _synthetic_key("stored")})
    stored_lookups: list[None] = []

    def _stored_lookup(cls: type) -> dict[str, Any] | None:
        stored_lookups.append(None)
        return {"api_key": _synthetic_key("stored")}

    monkeypatch.setattr(
        credential_manager.CredentialManager,
        "_load_cli_credentials",
        classmethod(_stored_lookup),
    )
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(status=200, json_payload={"valid": True, "data": {}}),
    )

    result = _run_whoami(monkeypatch, api_key=None)
    output = plain(result.output)

    assert result.exit_code == 0, output
    assert _FakeSession.last_post_kwargs is not None
    assert _FakeSession.last_post_kwargs["json"] == {"api_key": env_key}
    assert "API key source: TRAIGENT_API_KEY" in output
    assert env_key not in output
    assert stored_lookups == []


def test_whoami_positional_key_warns_once_and_reports_source(
    monkeypatch: pytest.MonkeyPatch, plain: Callable[[str], str]
) -> None:
    argv_key = _synthetic_key("argv")
    monkeypatch.setenv("TRAIGENT_API_KEY", _synthetic_key("env"))
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(status=200, json_payload={"valid": True, "data": {}}),
    )

    result = _run_whoami(monkeypatch, api_key=argv_key)
    output = plain(result.output)

    assert result.exit_code == 0, output
    assert output.count(ARGV_WARNING) == 1
    assert "API key source: command-line argument" in output
    assert _FakeSession.last_post_kwargs is not None
    assert _FakeSession.last_post_kwargs["json"] == {"api_key": argv_key}
    assert argv_key not in output


def test_whoami_invalid_positional_key_warns_and_sends_nothing(
    monkeypatch: pytest.MonkeyPatch, plain: Callable[[str], str]
) -> None:
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(status=200, json_payload={"valid": True, "data": {}}),
    )

    result = _run_whoami(monkeypatch, api_key="not-a-valid-key")
    output = plain(result.output)

    assert result.exit_code == 1
    assert output.count(ARGV_WARNING) == 1
    assert "Invalid API key format" in output
    assert _FakeSession.last_post_kwargs is None


@pytest.mark.parametrize(
    "stored_state",
    ["nothing_stored", "jwt_only_session", "ignored_plaintext_file"],
)
def test_whoami_without_a_resolvable_key_fails_without_a_request(
    monkeypatch: pytest.MonkeyPatch,
    plain: Callable[[str], str],
    _isolate_credentials: Path,
    stored_state: str,
) -> None:
    if stored_state == "jwt_only_session":
        _use_secure_store(monkeypatch, {"jwt_token": "header.payload.signature"})
    elif stored_state == "ignored_plaintext_file":
        # Plaintext credentials are ignored unless explicitly allowed for migration.
        _isolate_credentials.parent.mkdir(parents=True)
        _isolate_credentials.write_text(
            json.dumps({"api_key": _synthetic_key("plaintext")})
        )
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(status=200, json_payload={"valid": True, "data": {}}),
    )

    result = _run_whoami(monkeypatch, api_key=None)
    output = plain(result.output)

    assert result.exit_code == 1
    assert "Missing API key" in output
    assert MISSING_KEY_HINT in output
    assert "as an argument" not in output
    assert _FakeSession.last_post_kwargs is None


def test_whoami_reports_the_development_mode_source(
    monkeypatch: pytest.MonkeyPatch, plain: Callable[[str], str]
) -> None:
    """With dev mode explicitly on and nothing else, the dev key is the one checked."""
    dev_key = _synthetic_key("dev")
    monkeypatch.setenv("TRAIGENT_DEV_MODE", "true")
    monkeypatch.setenv("TRAIGENT_DEV_API_KEY", dev_key)
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(status=200, json_payload={"valid": True, "data": {}}),
    )

    result = _run_whoami(monkeypatch, api_key=None)
    output = plain(result.output)

    assert result.exit_code == 0, output
    assert _FakeSession.last_post_kwargs is not None
    assert _FakeSession.last_post_kwargs["json"] == {"api_key": dev_key}
    assert "API key source: TRAIGENT_DEV_API_KEY (development mode)" in output
    assert not _leaked_fragments(dev_key, output)


def test_whoami_invalid_env_key_still_reports_its_source(
    monkeypatch: pytest.MonkeyPatch, plain: Callable[[str], str]
) -> None:
    """A malformed key shadowing a stored one must still say where it came from."""
    env_key = _synthetic_key("wrong-variable", prefix="sk-proj-")
    monkeypatch.setenv("TRAIGENT_API_KEY", env_key)
    _use_secure_store(monkeypatch, {"api_key": _synthetic_key("stored")})
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(status=200, json_payload={"valid": True, "data": {}}),
    )

    result = _run_whoami(monkeypatch, api_key=None)
    output = plain(result.output)

    assert result.exit_code == 1
    assert "API key source: TRAIGENT_API_KEY" in output
    assert "Invalid API key format" in output
    assert output.index("API key source:") < output.index("Invalid API key format")
    assert _FakeSession.last_post_kwargs is None
    assert not _leaked_fragments(env_key, output, prefix="sk-proj-")


def test_whoami_redacts_a_key_echoed_in_a_422_body(
    monkeypatch: pytest.MonkeyPatch, plain: Callable[[str], str]
) -> None:
    """A validation error that echoes the request's api_key must not print it."""
    stored_key = _synthetic_key("stored-422")
    _use_secure_store(monkeypatch, {"api_key": stored_key})
    body = json.dumps(
        {
            "detail": [
                {
                    "type": "string_too_short",
                    "loc": ["body", "api_key"],
                    "msg": "rejected",
                    "input": stored_key,
                }
            ]
        }
    )
    assert len(body) < 220  # the whole key sits inside the displayed preview
    _install_fake_aiohttp(
        monkeypatch, response=_FakeResponse(status=422, text_payload=body)
    )

    result = _run_whoami(monkeypatch, api_key=None)
    output = plain(result.output)

    assert result.exit_code == 1
    assert "Category: backend_response_error" in output
    assert "HTTP status: 422" in output
    assert not _leaked_fragments(stored_key, output)
    assert REDACTED in _flat(output)


@pytest.mark.parametrize("suffix", ['"', "\\", "é"])
@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_whoami_redacts_json_escaped_key_echo(
    monkeypatch: pytest.MonkeyPatch,
    plain: Callable[[str], str],
    suffix: str,
    ensure_ascii: bool,
) -> None:
    key_value = _synthetic_key("escaped-echo") + suffix
    monkeypatch.setenv("TRAIGENT_API_KEY", key_value)
    body = json.dumps({"detail": [{"input": key_value}]}, ensure_ascii=ensure_ascii)
    _install_fake_aiohttp(
        monkeypatch, response=_FakeResponse(status=422, text_payload=body)
    )
    result = _run_whoami(monkeypatch, api_key=None)
    output = plain(result.output)
    assert result.exit_code == 1
    assert "HTTP status: 422" in output
    assert not _leaked_fragments(key_value, output)
    assert REDACTED in _flat(output)


def test_whoami_redacts_a_key_straddling_the_preview_cut(
    monkeypatch: pytest.MonkeyPatch, plain: Callable[[str], str]
) -> None:
    """Redaction runs before the 220-character cut, so no key prefix survives it."""
    env_key = _synthetic_key("straddle")
    monkeypatch.setenv("TRAIGENT_API_KEY", env_key)
    padding = "x" * 200
    body = json.dumps({"detail": padding + env_key})
    start = body.index(env_key)
    assert start < 220 < start + len(env_key)
    _install_fake_aiohttp(
        monkeypatch, response=_FakeResponse(status=422, text_payload=body)
    )

    result = _run_whoami(monkeypatch, api_key=None)
    output = plain(result.output)

    assert result.exit_code == 1
    assert "Category: backend_response_error" in output
    assert not _leaked_fragments(env_key, output)
    assert env_key[:8] not in "".join(output.split())


def test_whoami_redacted_403_preview_keeps_full_body_classification(
    monkeypatch: pytest.MonkeyPatch, plain: Callable[[str], str]
) -> None:
    """The preview is redacted; the classifier still reads past the 220-char cut."""
    env_key = _synthetic_key("edge")
    monkeypatch.setenv("TRAIGENT_API_KEY", env_key)
    body = f"<html>{env_key} {'.' * 300} cloudflare</html>"
    _install_fake_aiohttp(
        monkeypatch, response=_FakeResponse(status=403, text_payload=body)
    )

    result = _run_whoami(monkeypatch, api_key=None)
    output = plain(result.output)

    assert result.exit_code == 1
    assert "edge_blocked" in output
    assert not _leaked_fragments(env_key, output)
    assert REDACTED in _flat(output)


def test_whoami_redacts_a_key_in_an_exception_message(
    monkeypatch: pytest.MonkeyPatch, plain: Callable[[str], str]
) -> None:
    env_key = _synthetic_key("exception")
    monkeypatch.setenv("TRAIGENT_API_KEY", env_key)
    _install_fake_aiohttp(
        monkeypatch, error=TimeoutError(f"timed out sending api_key={env_key}")
    )

    result = _run_whoami(monkeypatch, api_key=None)
    output = plain(result.output)

    assert result.exit_code == 1
    assert "connectivity_error" in output
    assert not _leaked_fragments(env_key, output)
    assert REDACTED in _flat(output)


def test_whoami_redacts_a_key_in_success_metadata(
    monkeypatch: pytest.MonkeyPatch, plain: Callable[[str], str]
) -> None:
    env_key = _synthetic_key("metadata")
    monkeypatch.setenv("TRAIGENT_API_KEY", env_key)
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(
            status=200,
            json_payload={
                "valid": True,
                "data": {"key_name": f"laptop {env_key}", "user_id": 7},
            },
        ),
    )

    result = _run_whoami(monkeypatch, api_key=None)
    output = plain(result.output)

    assert result.exit_code == 0, output
    assert "✅ Valid" in output
    assert "authenticated" in output
    assert not _leaked_fragments(env_key, output)
    assert REDACTED in _flat(output)


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
#
# The 403 body is now a realistic Traigent API error rather than the bare string
# "unauthorized". That matters: review of PR #2107 found that classifying a 403 by
# ruling OUT a Cloudflare-specific signal list defaulted every unrecognised body to
# "authorization", whose remediation asserts the key is valid and a scope is
# missing -- a confidently wrong instruction for an AWS WAF or Akamai block. The
# classifier now keys on the POSITIVE Traigent signal, so the fixture has to be one.
# `test_whoami_403_that_matches_nothing_is_reported_as_indeterminate` below covers
# what the old fixture actually represented.
@pytest.mark.parametrize(
    "status,text_payload,expected_fragment,expected_category",
    [
        (401, "unauthorized", "Invalid or expired API key", "authentication"),
        (
            403,
            '{"detail":"API key lacks scope experiment.write"}',
            "lacks the required scope",
            "authorization",
        ),
    ],
)
def test_whoami_auth_failures_classified(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    text_payload: str,
    expected_fragment: str,
    expected_category: str,
    plain: Callable[[str], str],
) -> None:
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(status=status, text_payload=text_payload),
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


def test_whoami_403_that_matches_nothing_is_reported_as_indeterminate(
    monkeypatch: pytest.MonkeyPatch,
    plain: Callable[[str], str],
) -> None:
    """A 403 we cannot attribute must not claim the key merely lacks a scope.

    This is what the old `text_payload="unauthorized"` fixture actually was: a body
    matching neither the edge vocabulary nor a Traigent error shape. It used to fall
    through to "authorization" and print "Grant the scope rather than rotating the
    key" -- stated with full confidence about a request that may never have reached
    Traigent at all.
    """
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(status=403, text_payload="unauthorized"),
    )

    result = _run_whoami(monkeypatch)
    output = plain(result.output)

    assert result.exit_code == 1
    assert "forbidden_indeterminate" in output
    assert "Grant the scope rather than rotating the key" not in output
    # Rich wraps the hint across lines, so match a fragment that cannot straddle one.
    assert "do not rotate" in output


def test_whoami_403_with_a_cloudflare_header_is_an_edge_block(
    monkeypatch: pytest.MonkeyPatch,
    plain: Callable[[str], str],
) -> None:
    """cf-ray is a HEADER; it was previously only ever searched for in the body."""
    _install_fake_aiohttp(
        monkeypatch,
        response=_FakeResponse(
            status=403,
            text_payload="<html>Attention Required</html>",
            headers={"CF-RAY": "8abc123def456"},  # pragma: allowlist secret
        ),
    )

    result = _run_whoami(monkeypatch)
    output = plain(result.output)

    assert "edge_blocked" in output
    assert "did not reach Traigent" in output
