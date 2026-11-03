"""#2844: structured HTTP status drives transient classification; timeout is enforced."""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from traigent.providers.validation import (
    ProviderValidator,
    _is_transient_error,
    _run_with_timeout,
)


class _StatusError(Exception):
    def __init__(self, code: int | None, text: str = "boom") -> None:
        super().__init__(text)
        self.code = code


def _google_status(exc: Exception):
    client = Mock()
    client.models.list.side_effect = exc
    genai = SimpleNamespace(Client=Mock(return_value=client))
    mods = {"google": SimpleNamespace(genai=genai), "google.genai": genai}
    with patch.dict(sys.modules, mods):
        fake_env = {"GOOGLE_API_KEY": "fake-google-key"}  # pragma: allowlist secret
        with patch.dict("os.environ", fake_env):
            return ProviderValidator(timeout=5)._validate_google()


@pytest.mark.parametrize("code", [408, 429, 500, 503])
def test_google_transient_status_does_not_block(code: int) -> None:
    status = _google_status(_StatusError(code, "RESOURCE_EXHAUSTED"))
    assert status.valid is True
    assert "unverified" in status.message.lower()


def test_google_real_genai_error_classes_429_and_503() -> None:
    errors = pytest.importorskip("google.genai.errors")
    for exc in (
        errors.ClientError(429, {"error": {"status": "RESOURCE_EXHAUSTED"}}),
        errors.ServerError(503, {"error": {"status": "UNAVAILABLE"}}),
    ):
        assert _google_status(exc).valid is True


def test_google_statusless_unknown_error_still_blocks() -> None:
    assert _google_status(RuntimeError("weird")).valid is False


def test_google_client_error_400_is_not_transient() -> None:
    assert _is_transient_error(_StatusError(400)) is False


def test_run_with_timeout_returns_at_the_deadline() -> None:
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        _run_with_timeout(lambda: time.sleep(3), 0.5, "Google")
    assert time.monotonic() - start < 1.5
