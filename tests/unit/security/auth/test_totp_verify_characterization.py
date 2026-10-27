"""Characterization of TOTPAuthProvider.verify_totp_code replay/format policy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

pytest.importorskip("pyotp")

import pyotp  # noqa: E402

from traigent.security.auth import totp as totp_module  # noqa: E402
from traigent.security.auth.totp import TOTPAuthProvider  # noqa: E402

SECRET = "JBSWY3DPEHPK3PXP"  # pragma: allowlist secret


class _RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def warning(self, msg, *a, **k):
        self.calls.append(("warning", msg))

    def info(self, msg, *a, **k):
        self.calls.append(("info", msg))

    def error(self, msg, *a, **k):
        self.calls.append(("error", msg))


@pytest.fixture
def log(monkeypatch: pytest.MonkeyPatch) -> _RecordingLogger:
    rec = _RecordingLogger()
    monkeypatch.setattr(totp_module, "logger", rec)
    return rec


class _VerifyCalls(list):
    result = True


@pytest.fixture
def verify_calls(monkeypatch: pytest.MonkeyPatch) -> _VerifyCalls:
    calls = _VerifyCalls()

    def fake_verify(self, code, valid_window=0, **k):
        calls.append((code, valid_window))
        return calls.result

    monkeypatch.setattr(pyotp.TOTP, "verify", fake_verify)
    return calls


@pytest.fixture
def provider() -> TOTPAuthProvider:
    return TOTPAuthProvider()


class TestFormatDenials:
    @pytest.mark.parametrize(
        "code", ["", None, "12345", "123456789", "12345a", "abcdef", " 123456"]
    )
    def test_bad_code_denied_before_verify(self, provider, log, verify_calls, code):
        assert provider.verify_totp_code(code, SECRET, "u") is False
        assert log.calls == [("warning", "Invalid TOTP code format")]
        assert verify_calls == []

    @pytest.mark.parametrize("code", ["123456", "1234567", "12345678"])
    def test_code_length_boundaries_accepted(self, provider, log, verify_calls, code):
        assert provider.verify_totp_code(code, SECRET, "u") is True
        assert verify_calls == [(code, 1)]

    @pytest.mark.parametrize("code", ["١٢٣٤٥٦", "123456\n", "123456 ", "\n123456"])
    def test_non_ascii_digits_and_whitespace_denied_without_verifying(
        self, provider, log, verify_calls, code
    ):
        assert provider.verify_totp_code(code, SECRET, "u") is False
        assert log.calls == [("warning", "Invalid TOTP code format")]
        assert verify_calls == []

    def test_fullwidth_digits_normalized_for_verification(
        self, provider, log, verify_calls
    ):
        assert provider.verify_totp_code("１２３４５６", SECRET, "u") is True
        assert verify_calls == [("123456", 1)]
        assert provider._used_codes["u"] == {"123456"}

    def test_equivalent_code_forms_share_validation_state(
        self, provider, log, verify_calls
    ):
        assert provider.verify_totp_code("123456", SECRET, "u") is True
        log.calls.clear()
        assert provider.verify_totp_code("１２３４５６", SECRET, "u") is False
        assert log.calls == [("warning", "TOTP code replay attempt for user u")]
        assert provider.verify_totp_code("123456", SECRET, "u") is False
        assert len(verify_calls) == 1

    def test_equivalent_code_forms_share_state_regardless_of_first_form(
        self, provider, log, verify_calls
    ):
        assert provider.verify_totp_code("１２３４５６", SECRET, "u") is True
        assert provider.verify_totp_code("123456", SECRET, "u") is False
        assert provider.verify_totp_code("１２３４５６", SECRET, "u") is False
        assert len(verify_calls) == 1

    @pytest.mark.parametrize("secret", ["", None, "abc", "JBSW1", "JBSW=A", "JBSW Y"])
    def test_bad_secret_denied_before_verify(self, provider, log, verify_calls, secret):
        assert provider.verify_totp_code("123456", secret, "u") is False
        assert log.calls == [("warning", "Invalid TOTP secret format")]
        assert verify_calls == []

    def test_code_format_checked_before_secret(self, provider, log, verify_calls):
        assert provider.verify_totp_code("bad", "bad", "u") is False
        assert log.calls == [("warning", "Invalid TOTP code format")]


class TestReplayPolicy:
    def test_success_then_same_code_is_replay_denied(self, provider, log, verify_calls):
        assert provider.verify_totp_code("123456", SECRET, "alice") is True
        assert log.calls == [("info", "TOTP verification successful for user alice")]
        log.calls.clear()
        assert provider.verify_totp_code("123456", SECRET, "alice") is False
        assert log.calls == [("warning", "TOTP code replay attempt for user alice")]
        assert len(verify_calls) == 1

    def test_replay_is_per_user(self, provider, log, verify_calls):
        assert provider.verify_totp_code("123456", SECRET, "alice") is True
        assert provider.verify_totp_code("123456", SECRET, "bob") is True

    def test_replay_denied_even_when_expiry_passed_check_order(
        self, provider, log, verify_calls
    ):
        provider.verify_totp_code("123456", SECRET, "alice")
        provider._code_expiry["alice"] = datetime.now(UTC) - timedelta(seconds=1)
        log.calls.clear()
        assert provider.verify_totp_code("123456", SECRET, "alice") is False
        assert log.calls == [("warning", "TOTP code replay attempt for user alice")]
        # replay branch returns before expired codes are pruned
        assert provider._used_codes["alice"] == {"123456"}

    def test_expired_history_cleared_allows_reuse(self, provider, log, verify_calls):
        provider.verify_totp_code("123456", SECRET, "alice")
        provider._code_expiry["alice"] = datetime.now(UTC) - timedelta(seconds=1)
        assert provider.verify_totp_code("654321", SECRET, "alice") is True
        assert provider._used_codes["alice"] == {"654321"}

    def test_unexpired_history_kept(self, provider, log, verify_calls):
        provider.verify_totp_code("123456", SECRET, "alice")
        provider._code_expiry["alice"] = datetime.now(UTC) + timedelta(seconds=60)
        assert provider.verify_totp_code("654321", SECRET, "alice") is True
        assert provider._used_codes["alice"] == {"123456", "654321"}

    def test_history_without_expiry_entry_not_cleared(
        self, provider, log, verify_calls
    ):
        provider._used_codes["alice"] = {"111111"}
        assert provider.verify_totp_code("654321", SECRET, "alice") is True
        assert provider._used_codes["alice"] == {"111111", "654321"}

    def test_failed_verification_not_recorded(self, provider, log, verify_calls):
        verify_calls.result = False
        assert provider.verify_totp_code("123456", SECRET, "alice") is False
        assert log.calls == [("warning", "TOTP verification failed for user alice")]
        assert "alice" not in provider._used_codes
        assert "alice" not in provider._code_expiry
        verify_calls.result = True
        assert provider.verify_totp_code("123456", SECRET, "alice") is True


class TestWindowAndRecording:
    @pytest.mark.parametrize(
        "window,expected", [(0, 0), (1, 1), (2, 2), (3, 2), (50, 2)]
    )
    def test_window_capped_at_two(self, provider, log, verify_calls, window, expected):
        provider.verify_totp_code("123456", SECRET, "u", window=window)
        assert verify_calls == [("123456", expected)]

    def test_expiry_set_ninety_seconds_ahead(self, provider, log, verify_calls):
        before = datetime.now(UTC)
        provider.verify_totp_code("123456", SECRET, "u")
        after = datetime.now(UTC)
        expiry = provider._code_expiry["u"]
        assert before + timedelta(seconds=90) <= expiry <= after + timedelta(seconds=90)

    def test_history_of_ten_is_kept(self, provider, log, verify_calls):
        provider._used_codes["u"] = {f"10000{i}" for i in range(9)}
        assert provider.verify_totp_code("123456", SECRET, "u") is True
        assert len(provider._used_codes["u"]) == 10

    def test_history_over_ten_trimmed_to_five(self, provider, log, verify_calls):
        old = {f"10000{i}" for i in range(10)}
        provider._used_codes["u"] = set(old)
        assert provider.verify_totp_code("123456", SECRET, "u") is True
        kept = provider._used_codes["u"]
        assert len(kept) == 5
        assert kept <= old | {"123456"}


class TestExceptionBoundary:
    def test_verify_exception_denied_and_logged(self, provider, log, monkeypatch):
        def boom(self, code, valid_window=0, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(pyotp.TOTP, "verify", boom)
        assert provider.verify_totp_code("123456", SECRET, "u") is False
        assert log.calls == [("error", "TOTP verification error: boom")]
        assert "u" not in provider._used_codes

    def test_corrupt_expiry_state_denied_fail_closed(self, provider, log, verify_calls):
        provider._used_codes["u"] = {"111111"}
        provider._code_expiry["u"] = 5  # type: ignore[assignment]
        assert provider.verify_totp_code("123456", SECRET, "u") is False
        assert len(log.calls) == 1
        assert log.calls[0][0] == "error"
        assert log.calls[0][1].startswith("TOTP verification error: ")
        assert verify_calls == []
