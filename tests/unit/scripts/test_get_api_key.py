"""Tests for scripts/auth/get_api_key.py helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


def _load_get_api_key_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "scripts" / "auth" / "get_api_key.py"
    spec = importlib.util.spec_from_file_location("get_api_key_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_backend_url_strips_trailing_slash() -> None:
    module = _load_get_api_key_module()

    assert (
        module.normalize_backend_url("https://api.traigent.ai/")
        == "https://api.traigent.ai"
    )


def test_normalize_backend_url_allows_localhost_http() -> None:
    module = _load_get_api_key_module()

    assert (
        module.normalize_backend_url("http://localhost:5000/")
        == "http://localhost:5000"
    )


@pytest.mark.parametrize(
    ("backend_url", "error"),
    [
        ("ftp://api.traigent.ai", "must start with http:// or https://"),
        (
            "https://api.traigent.ai?token=1",
            "must not include params, query strings, or fragments",
        ),
        ("https://", "must include a hostname"),
        ("http://api.traigent.ai", "Non-local backend URLs must use https"),
        (
            "https://user:pass@api.traigent.ai",  # pragma: allowlist secret
            "must not include embedded credentials",
        ),
        ("https://192.168.1.10", "must not target private or loopback IPs"),
    ],
)
def test_normalize_backend_url_rejects_unsafe_values(
    backend_url: str, error: str
) -> None:
    module = _load_get_api_key_module()

    with pytest.raises(ValueError, match=error):
        module.normalize_backend_url(backend_url)


def test_main_quiet_prints_generated_key(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Quiet mode is the script contract for programmatic key capture."""
    module = _load_get_api_key_module()

    monkeypatch.setattr(
        module,
        "get_api_key",
        lambda email, password, backend_url, verbose=True: "tg_generated_key",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "get_api_key.py",
            "--email",
            "user@example.com",
            "--backend-url",
            "https://api.traigent.ai",
            "--quiet",
        ],
    )
    monkeypatch.setenv(
        "TRAIGENT_AUTH_PASSWORD",
        "test-password",  # pragma: allowlist secret
    )

    module.main()

    assert capfd.readouterr().out == "tg_generated_key\n"


def test_main_rejects_password_cli_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passwords must not be accepted through argv."""
    module = _load_get_api_key_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "get_api_key.py",
            "--email",
            "user@example.com",
            "--password",
            "test-password",  # pragma: allowlist secret
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == 2


def test_main_reads_password_from_stdin(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Automation can provide a password through stdin instead of argv."""
    module = _load_get_api_key_module()

    monkeypatch.setattr(
        module,
        "get_api_key",
        lambda email, password, backend_url, verbose=True: "tg_generated_key"
        if password == "stdin-password"  # pragma: allowlist secret
        else "wrong",
    )
    monkeypatch.setattr(
        sys.stdin,
        "readline",
        lambda: "stdin-password\n",  # pragma: allowlist secret
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "get_api_key.py",
            "--email",
            "user@example.com",
            "--password-stdin",
            "--quiet",
        ],
    )

    module.main()

    assert capfd.readouterr().out == "tg_generated_key\n"


def test_verbose_responses_are_redacted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verbose diagnostics must not print JWTs or generated API keys."""
    module = _load_get_api_key_module()
    jwt = "jwt.secret.token"  # pragma: allowlist secret
    api_key = "tg_secret_key"  # pragma: allowlist secret

    responses = [
        Mock(
            status_code=200,
            headers={"Authorization": "Bearer " + jwt},
            json=lambda: {
                "success": True,
                "data": {
                    "access_token": jwt,
                    "refresh_token": "refresh-secret",  # pragma: allowlist secret
                },
            },
        ),
        Mock(
            status_code=201,
            headers={},
            json=lambda: {"data": {"key": api_key}},
        ),
    ]
    monkeypatch.setattr(
        module.requests, "post", lambda *args, **kwargs: responses.pop(0)
    )

    assert (
        module.get_api_key(
            "user@example.com",
            "test-password",  # pragma: allowlist secret
            "https://api.traigent.ai",
            verbose=True,
        )
        == api_key
    )

    output = capsys.readouterr().out
    assert jwt not in output
    assert api_key not in output
    assert "refresh-secret" not in output
    assert '"access_token": "***"' in output
    assert '"key": "***"' in output
