"""#3164: malformed approval expiry denies; it never raises and never grants."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from traigent.core.ci_approval import (
    _parse_expiry_utc,
    _validate_hmac_token,
    _validate_legacy_token,
    check_ci_approval,
)
from traigent.runtime import manifest
from traigent.utils.exceptions import OptimizationError

_FUTURE = datetime.now(UTC) + timedelta(hours=2)
_BAD_EXPIRIES = [
    pytest.param(1893456000, id="epoch-int"),
    pytest.param(1893456000.5, id="epoch-float"),
    pytest.param(None, id="null"),
    pytest.param(True, id="bool"),
    pytest.param(["2030-01-01T00:00:00Z"], id="list"),
    pytest.param(
        (_FUTURE.replace(tzinfo=None)).isoformat(timespec="seconds"), id="naive"
    ),
    pytest.param(_FUTURE.strftime("%Y-%m-%d"), id="bare-date"),
    pytest.param("not-a-date", id="garbage"),
    pytest.param("", id="empty"),
]


def _config(storage: Path) -> MagicMock:
    config = MagicMock()
    config.is_local_mode.return_value = True
    config.get_local_storage_path.return_value = str(storage)
    return config


def _env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != "TRAIGENT_RUN_APPROVED"}


def _run_gate(storage: Path) -> None:
    with (
        patch("traigent.core.ci_approval._is_ci_environment", return_value=True),
        patch.dict(os.environ, _env(), clear=True),
    ):
        check_ci_approval(_config(storage))


@pytest.mark.parametrize("bad", _BAD_EXPIRIES)
def test_parser_rejects_everything_but_aware_strings(bad: object) -> None:
    with pytest.raises(ValueError):
        _parse_expiry_utc(bad)


@pytest.mark.parametrize("good", ["Z", "+00:00", "+05:30", "-08:00"])
def test_parser_accepts_aware_timestamps(good: str) -> None:
    text = _FUTURE.strftime("%Y-%m-%dT%H:%M:%S") + good
    parsed = _parse_expiry_utc(text)
    assert parsed.tzinfo is not None


@pytest.mark.parametrize("bad", _BAD_EXPIRIES)
def test_legacy_token_bad_expiry_denied_without_exception(bad: object) -> None:
    assert _validate_legacy_token({"approved_by": "t", "expires_at": bad}) is False


@pytest.mark.parametrize("bad", _BAD_EXPIRIES)
def test_hmac_token_bad_expiry_denied_without_exception(
    bad: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_APPROVAL_SECRET", "s3cret")
    # Sign whatever the token carries so only the expiry parse can decide.
    payload = f"alice|{bad}|n1".encode()
    sig = base64.b64encode(hmac.new(b"s3cret", payload, hashlib.sha256).digest())
    token = {
        "approver": "alice",
        "expires_iso": bad,
        "nonce": "n1",
        "signature": sig.decode(),
    }
    assert _validate_hmac_token(token) is False


@pytest.mark.parametrize("bad", _BAD_EXPIRIES)
@pytest.mark.parametrize("kind", ["legacy", "hmac"])
def test_gate_gives_guided_message_not_traceback(
    bad: object, kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_APPROVAL_SECRET", "s3cret")
    if kind == "legacy":
        token: dict = {"approved_by": "alice", "expires_at": bad}
    else:
        payload = f"alice|{bad}|n1".encode()
        sig = base64.b64encode(hmac.new(b"s3cret", payload, hashlib.sha256).digest())
        token = {
            "approver": "alice",
            "expires_iso": bad,
            "nonce": "n1",
            "signature": sig.decode(),
        }
    (tmp_path / "approval.token").write_text(json.dumps(token), encoding="utf-8")
    with pytest.raises(OptimizationError) as info:
        _run_gate(tmp_path)  # AttributeError before the fix for non-str expiry
    assert "CI/CD Approval Required" in str(info.value)


@pytest.mark.parametrize("payload", ["123", "null", '"x"', "[1]", "{}"])
def test_gate_malformed_token_shapes_deny(tmp_path: Path, payload: str) -> None:
    (tmp_path / "approval.token").write_text(payload, encoding="utf-8")
    with pytest.raises(OptimizationError, match="CI/CD Approval Required"):
        _run_gate(tmp_path)


def test_gate_valid_aware_legacy_token_still_grants(tmp_path: Path) -> None:
    stamp = _FUTURE.strftime("%Y-%m-%dT%H:%M:%SZ")
    (tmp_path / "approval.token").write_text(
        json.dumps({"approved_by": "alice", "expires_at": stamp}), encoding="utf-8"
    )
    _run_gate(tmp_path)  # must not raise


def test_gate_expired_aware_token_denied(tmp_path: Path) -> None:
    stamp = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    (tmp_path / "approval.token").write_text(
        json.dumps({"approved_by": "alice", "expires_at": stamp}), encoding="utf-8"
    )
    with pytest.raises(OptimizationError, match="CI/CD Approval Required"):
        _run_gate(tmp_path)


# --- runtime manifest -------------------------------------------------------------


_NOW = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "bad",
    [
        "2030-01-02T00:00:00",
        "2030-01-02",
        "garbage",
        "",
        None,
        1893456000,
        ["x"],
    ],
)
def test_runtime_malformed_expiry_is_expired_without_exception(bad: object) -> None:
    assert manifest.approval_expired({"expires_at": bad}, _NOW) is True


def test_runtime_missing_expiry_is_expired() -> None:
    assert manifest.approval_expired({}, _NOW) is True


def test_runtime_valid_aware_future_not_expired() -> None:
    future = (_NOW + timedelta(hours=1)).isoformat()
    assert manifest.approval_expired({"expires_at": future}, _NOW) is False


def test_runtime_valid_aware_past_expired() -> None:
    past = (_NOW - timedelta(hours=1)).isoformat()
    assert manifest.approval_expired({"expires_at": past}, _NOW) is True


def test_runtime_naive_now_denies_rather_than_raising() -> None:
    future = (_NOW + timedelta(hours=1)).isoformat()
    naive_now = _NOW.replace(tzinfo=None)
    assert manifest.approval_expired({"expires_at": future}, naive_now) is True


@pytest.mark.parametrize("sig", ["é", 5, None, ["x"]])
def test_hmac_non_string_or_non_ascii_signature_denies(
    sig: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_APPROVAL_SECRET", "s3cret")
    stamp = _FUTURE.strftime("%Y-%m-%dT%H:%M:%SZ")
    token = {"approver": "a", "expires_iso": stamp, "nonce": "n", "signature": sig}
    assert _validate_hmac_token(token) is False


@pytest.mark.parametrize("suffix", ["z", " ", "\n"])
def test_parser_rejects_lowercase_z_and_padding(suffix: str) -> None:
    base = _FUTURE.strftime("%Y-%m-%dT%H:%M:%S")
    text = base + suffix if suffix == "z" else " " + base + "Z" + suffix
    with pytest.raises(ValueError):
        _parse_expiry_utc(text)
