"""Tests for Traigent device-code authentication."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from traigent.cli import auth_commands
from traigent.cli.auth_commands import SECURE_CLI_CREDENTIAL_NAME, TraigentAuthCLI
from traigent.security.credentials import CredentialType

API_KEY = "sk_" + "a" * 43  # pragma: allowlist secret
DEVICE_CODE = "A" * 43
DEVICE_LOGIN_CLIENT_ID = "traigent-sdk-cli"
DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
DEVICE_AUTHORIZE_ENVELOPE_REQUIRED_KEYS = ["success", "message", "data"]
DEVICE_AUTHORIZE_DATA_REQUIRED_KEYS = [
    "device_code",
    "user_code",
    "verification_uri",
    "verification_uri_complete",
    "expires_in",
    "interval",
]


class _FakeSecureStore:
    def __init__(self, *, fail_on_set: bool = False) -> None:
        self.fail_on_set = fail_on_set
        self.saved: tuple[str, str, CredentialType, dict[str, str] | None] | None = None

    def get(self, name: str, check_env: bool = True) -> str | None:
        assert name == SECURE_CLI_CREDENTIAL_NAME
        assert check_env is False
        return None

    def set(
        self,
        name: str,
        value: str,
        credential_type: CredentialType,
        metadata: dict[str, str] | None = None,
    ) -> None:
        if self.fail_on_set:
            raise auth_commands.SecurityError("secure store unavailable")
        self.saved = (name, value, credential_type, metadata)

    def delete_secure(self, name: str) -> bool:
        assert name == SECURE_CLI_CREDENTIAL_NAME
        return True


class _FakeResponse:
    def __init__(
        self,
        status: int,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self.payload = payload
        self.headers = headers or {}

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def text(self) -> str:
        return json.dumps(self.payload)


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse | BaseException]) -> None:
        self.responses = responses
        self.post_calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        if url.endswith("/auth/device/authorize"):
            assert kwargs.get("json") == {"client_id": DEVICE_LOGIN_CLIENT_ID}
        elif url.endswith("/auth/device/token"):
            assert kwargs.get("json") == {
                "grant_type": DEVICE_CODE_GRANT_TYPE,
                "device_code": DEVICE_CODE,
                "client_id": DEVICE_LOGIN_CLIENT_ID,
            }
        self.post_calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError(f"Unexpected POST: {url}")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _authorization_data(interval: int = 1, expires_in: int = 600) -> dict[str, Any]:
    return {
        "device_code": DEVICE_CODE,
        "user_code": "BCDF-2345",
        "verification_uri": "https://portal.example.test/device",
        "verification_uri_complete": "https://portal.example.test/device?user_code=BCDF-2345",
        "expires_in": expires_in,
        "interval": interval,
    }


def _authorize_payload(interval: int = 1, expires_in: int = 600) -> dict[str, Any]:
    return {
        "success": True,
        "message": "Device authorization created",
        "data": _authorization_data(interval=interval, expires_in=expires_in),
    }


# Required by TraigentSchema#94 device_token_response_schema.json.
DEVICE_TOKEN_SUCCESS_ENVELOPE_REQUIRED_KEYS = ["success", "message", "data"]
DEVICE_TOKEN_SUCCESS_REQUIRED_KEYS = [
    "api_key",
    "tenant_id",
    "project_id",
    "user",
    "subscription_tier",
    "quota",
]


def _success_payload() -> dict[str, Any]:
    return {
        "success": True,
        "message": "Device token issued",
        "data": {
            "api_key": API_KEY,
            "tenant_id": "tenant_123",
            "project_id": "project_456",
            "user": {"id": "user_1", "email": "dev@example.test"},
            "subscription_tier": "free",
            "quota": {"trial_limit": 25, "api_call_limit": 1000},
        },
    }


def _error_payload(error: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": False,
        "message": error.replace("_", " "),
        "error": error,
        "error_code": error,
    }
    if details is not None:
        payload["details"] = details
    return payload


def _rate_limit_response(retry_after: str | None = None) -> _FakeResponse:
    headers = {"Retry-After": retry_after} if retry_after is not None else None
    return _FakeResponse(
        429,
        {
            "success": False,
            "message": "Rate limit exceeded",
            "error": "rate_limit_exceeded",
        },
        headers=headers,
    )


def _install_device_test_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    responses: list[_FakeResponse | BaseException],
    *,
    fail_secure_store: bool = False,
) -> tuple[_FakeSecureStore, _FakeSession]:
    store = _FakeSecureStore(fail_on_set=fail_secure_store)
    session = _FakeSession(responses)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(auth_commands, "TRAIGENT_CONFIG_DIR", tmp_path / ".traigent")
    monkeypatch.setattr(
        auth_commands, "CREDENTIALS_FILE", tmp_path / ".traigent" / "credentials.json"
    )
    monkeypatch.setattr(auth_commands, "get_secure_credential_store", lambda: store)
    assert auth_commands.aiohttp is not None
    monkeypatch.setattr(
        auth_commands.aiohttp,
        "ClientSession",
        lambda *args, **kwargs: session,
    )
    return store, session


def _make_cli(
    backend_url_override: str | None = "https://backend.example.test",
) -> TraigentAuthCLI:
    return TraigentAuthCLI(backend_url_override=backend_url_override)


def test_device_authorize_schema_contract_pins_success_envelope() -> None:
    cli = TraigentAuthCLI.__new__(TraigentAuthCLI)

    payload = _authorize_payload()
    assert list(payload.keys()) == DEVICE_AUTHORIZE_ENVELOPE_REQUIRED_KEYS
    assert list(payload["data"].keys()) == DEVICE_AUTHORIZE_DATA_REQUIRED_KEYS

    validated = cli._validate_device_authorize_payload(payload)

    assert validated["device_code"] == DEVICE_CODE
    assert validated["user_code"] == "BCDF-2345"

    with pytest.raises(auth_commands.AuthenticationError, match="success missing"):
        cli._validate_device_authorize_payload(_authorization_data())


def test_device_token_success_schema_contract_pins_subscription_tier() -> None:
    cli = TraigentAuthCLI.__new__(TraigentAuthCLI)
    cli.backend_url = "https://backend.example.test"
    payload = _success_payload()
    assert list(payload.keys()) == DEVICE_TOKEN_SUCCESS_ENVELOPE_REQUIRED_KEYS
    token_data = payload["data"]
    assert isinstance(token_data, dict)
    assert list(token_data.keys()) == DEVICE_TOKEN_SUCCESS_REQUIRED_KEYS

    validated = cli._validate_device_token_success(_success_payload())

    assert validated["subscription_tier"] == "free"

    missing_subscription_tier = _success_payload()
    missing_data = missing_subscription_tier["data"]
    assert isinstance(missing_data, dict)
    missing_data.pop("subscription_tier")

    with pytest.raises(auth_commands.AuthenticationError, match="subscription_tier"):
        cli._validate_device_token_success(missing_subscription_tier)


def test_device_flow_happy_path_persists_credentials_secure_only_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, session = _install_device_test_fakes(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, _authorize_payload()),
            _FakeResponse(200, _success_payload()),
        ],
    )

    result = asyncio.run(_make_cli().device_login(sleep=lambda _seconds: _noop_sleep()))

    assert result is True
    assert store.saved is not None
    name, serialized, credential_type, metadata = store.saved
    assert name == SECURE_CLI_CREDENTIAL_NAME
    assert credential_type is CredentialType.TOKEN
    assert metadata == {"source": "traigent-cli"}
    saved = json.loads(serialized)
    assert saved["api_key"] == API_KEY
    assert saved["tenant_id"] == "tenant_123"
    assert saved["project_id"] == "project_456"
    assert saved["backend_url"] == "https://backend.example.test"

    assert not (tmp_path / ".env").exists()

    assert session.post_calls[0]["url"].endswith("/auth/device/authorize")
    assert session.post_calls[0]["json"] == {"client_id": DEVICE_LOGIN_CLIENT_ID}
    assert session.post_calls[1]["url"].endswith("/auth/device/token")
    assert session.post_calls[1]["json"] == {
        "grant_type": DEVICE_CODE_GRANT_TYPE,
        "device_code": DEVICE_CODE,
        "client_id": DEVICE_LOGIN_CLIENT_ID,
    }
    captured = capsys.readouterr()
    assert API_KEY not in captured.out
    assert API_KEY not in captured.err


@pytest.mark.parametrize(
    ("backend_url", "expected_api_base"),
    [
        ("http://localhost:5000", "http://localhost:5000/api/v1"),
        ("http://localhost:5000/api/v1", "http://localhost:5000/api/v1"),
    ],
)
def test_device_flow_composes_authorize_and_token_urls_from_api_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend_url: str,
    expected_api_base: str,
) -> None:
    _store, session = _install_device_test_fakes(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, _authorize_payload()),
            _FakeResponse(200, _success_payload()),
        ],
    )

    result = asyncio.run(
        _make_cli(backend_url_override=backend_url).device_login(
            sleep=lambda _seconds: _noop_sleep()
        )
    )

    assert result is True
    assert [call["url"] for call in session.post_calls] == [
        f"{expected_api_base}/auth/device/authorize",
        f"{expected_api_base}/auth/device/token",
    ]


def test_device_flow_env_api_url_precedence_banner_matches_transport_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", "http://localhost:5000")
    monkeypatch.setenv("TRAIGENT_API_URL", "https://api.example.test/custom/v1")
    _store, session = _install_device_test_fakes(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, _authorize_payload()),
            _FakeResponse(200, _success_payload()),
        ],
    )

    result = asyncio.run(
        _make_cli(backend_url_override=None).device_login(
            sleep=lambda _seconds: _noop_sleep()
        )
    )

    assert result is True
    assert [call["url"] for call in session.post_calls] == [
        "https://api.example.test/custom/v1/auth/device/authorize",
        "https://api.example.test/custom/v1/auth/device/token",
    ]
    output = capsys.readouterr().out
    assert "Authenticating with: https://api.example.test/custom/v1" in output
    assert "Authenticating with: http://localhost:5000" not in output


def test_device_flow_env_file_consent_writes_with_0600(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _store, _session = _install_device_test_fakes(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, _authorize_payload()),
            _FakeResponse(200, _success_payload()),
        ],
    )

    result = asyncio.run(
        _make_cli().device_login(
            save_env_file=True,
            sleep=lambda _seconds: _noop_sleep(),
        )
    )

    assert result is True
    env_file = tmp_path / ".env"
    assert env_file.exists()
    assert env_file.stat().st_mode & 0o777 == 0o600
    env_text = env_file.read_text(encoding="utf-8")
    assert f"TRAIGENT_API_KEY={API_KEY}" in env_text
    assert "TRAIGENT_TENANT_ID=tenant_123" in env_text
    assert "TRAIGENT_PROJECT_ID=project_456" in env_text
    assert "TRAIGENT_BACKEND_URL=https://backend.example.test" in env_text


def test_device_flow_secure_store_failure_without_explicit_flag_skips_env_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store, _session = _install_device_test_fakes(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, _authorize_payload()),
            _FakeResponse(200, _success_payload()),
        ],
        fail_secure_store=True,
    )

    result = asyncio.run(
        _make_cli().device_login(
            save_env_file=True,
            sleep=lambda _seconds: _noop_sleep(),
        )
    )

    assert result is False
    assert store.saved is None
    assert not (tmp_path / ".env").exists()


def test_env_file_rewrite_replaces_export_prefixed_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "export TRAIGENT_API_KEY=sk_stale",  # pragma: allowlist secret
                "TRAIGENT_API_KEY=sk_duplicate",  # pragma: allowlist secret
                "OTHER=value",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _make_cli()._save_api_key_to_env_file(
        API_KEY,
        tenant_id="tenant_123",
        project_id="project_456",
        backend_url="https://backend.example.test",
    )

    env_text = env_file.read_text(encoding="utf-8")
    assert "sk_stale" not in env_text
    assert "sk_duplicate" not in env_text
    assert env_text.count("TRAIGENT_API_KEY=") == 1
    assert f"export TRAIGENT_API_KEY={API_KEY}" in env_text


def test_device_flow_pending_then_slow_down_honors_authoritative_interval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _store, _session = _install_device_test_fakes(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, _authorize_payload(interval=1)),
            _FakeResponse(400, _error_payload("authorization_pending")),
            _FakeResponse(400, _error_payload("slow_down", {"interval": 7})),
            _FakeResponse(200, _success_payload()),
        ],
    )
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    result = asyncio.run(_make_cli().device_login(sleep=fake_sleep))

    assert result is True
    assert sleeps == [1, 7]


def test_device_poll_rate_limit_retry_after_continues_and_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _store, session = _install_device_test_fakes(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, _authorize_payload(interval=1)),
            _rate_limit_response(retry_after="2"),
            _FakeResponse(200, _success_payload()),
        ],
    )
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    result = asyncio.run(_make_cli().device_login(sleep=fake_sleep))

    assert result is True
    assert sleeps == [2]
    token_calls = [
        call for call in session.post_calls if call["url"].endswith("/auth/device/token")
    ]
    assert len(token_calls) == 2


def test_device_poll_rate_limit_retry_after_still_honors_expiry_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _store, session = _install_device_test_fakes(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, _authorize_payload(interval=1, expires_in=3)),
            _rate_limit_response(retry_after="10"),
            _FakeResponse(200, _success_payload()),
        ],
    )
    now = 0.0
    sleeps: list[float] = []

    def fake_clock() -> float:
        return now

    async def fake_sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    result = asyncio.run(
        _make_cli().device_login(sleep=fake_sleep, clock=fake_clock)
    )

    assert result is False
    assert sleeps == [3]
    token_calls = [
        call for call in session.post_calls if call["url"].endswith("/auth/device/token")
    ]
    assert len(token_calls) == 1


def test_device_poll_caps_sleep_and_timeout_to_expiry_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _store, session = _install_device_test_fakes(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, _authorize_payload(interval=10, expires_in=3)),
            _FakeResponse(400, _error_payload("authorization_pending")),
        ],
    )
    now = 0.0
    sleeps: list[float] = []

    def fake_clock() -> float:
        return now

    async def fake_sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    result = asyncio.run(
        _make_cli().device_login(sleep=fake_sleep, clock=fake_clock)
    )

    assert result is False
    assert sleeps == [3]
    token_call = session.post_calls[1]
    assert token_call["url"].endswith("/auth/device/token")
    assert token_call["timeout"].total == 3


def test_device_poll_transport_error_retry_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, session = _install_device_test_fakes(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, _authorize_payload(interval=1)),
            OSError("network one"),
            OSError("network two"),
            OSError("network three"),
            _FakeResponse(200, _success_payload()),
        ],
    )
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    result = asyncio.run(_make_cli().device_login(sleep=fake_sleep))

    assert result is False
    assert store.saved is None
    assert sleeps == [1, 1]
    token_calls = [
        call for call in session.post_calls if call["url"].endswith("/auth/device/token")
    ]
    assert len(token_calls) == 3
    assert "failed after 3 consecutive transport errors" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("error_name", "message"),
    [
        ("access_denied", "denied"),
        ("expired_token", "expired"),
    ],
)
def test_device_flow_terminal_errors_exit_clearly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    error_name: str,
    message: str,
) -> None:
    store, _session = _install_device_test_fakes(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, _authorize_payload()),
            _FakeResponse(400, _error_payload(error_name)),
        ],
    )

    result = asyncio.run(_make_cli().device_login(sleep=lambda _seconds: _noop_sleep()))

    assert result is False
    assert store.saved is None
    assert message in capsys.readouterr().out


def test_device_flow_malformed_success_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store, _session = _install_device_test_fakes(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, _authorize_payload()),
            _FakeResponse(
                200,
                {
                    "success": True,
                    "data": {"api_key": API_KEY, "tenant_id": "tenant_123"},
                },
            ),
        ],
    )

    result = asyncio.run(_make_cli().device_login(sleep=lambda _seconds: _noop_sleep()))

    assert result is False
    assert store.saved is None


def test_device_authorize_error_reports_status_and_backend_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, _session = _install_device_test_fakes(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(
                400,
                {
                    "success": False,
                    "message": "Validation error",
                    "error": "client_id: Missing data for required field.",
                },
            ),
        ],
    )

    result = asyncio.run(_make_cli().device_login(sleep=lambda _seconds: _noop_sleep()))

    assert result is False
    assert store.saved is None
    output = capsys.readouterr().out
    assert "HTTP 400" in output
    assert "Validation error" in output


async def _noop_sleep() -> None:
    return None
