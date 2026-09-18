"""Tests for the public mock-mode disable counterpart.

Traigent/Traigent#1646: ``traigent.testing`` had an enable-only public
surface (``enable_mock_mode_for_quickstart``) with no supported way to
turn the process-local flag back off within the same interpreter. A
same-process "mock test-run, then real run" flow would silently keep
using canned mock responses for the "real" run. ``disable_mock_mode()``
is the public, documented off switch.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from traigent import testing as traigent_testing


@pytest.fixture(autouse=True)
def reset_mock_mode_flag() -> Iterator[None]:
    traigent_testing._reset_for_tests()
    yield
    traigent_testing._reset_for_tests()


def test_disable_mock_mode_is_public() -> None:
    """The issue's core ask: a public counterpart to the enable function."""
    assert "disable_mock_mode" in traigent_testing.__all__
    assert callable(traigent_testing.disable_mock_mode)


def test_disable_turns_off_a_previously_enabled_flag() -> None:
    traigent_testing.enable_mock_mode_for_quickstart()
    assert traigent_testing.is_mock_mode_enabled() is True

    traigent_testing.disable_mock_mode()

    assert traigent_testing.is_mock_mode_enabled() is False


def test_disable_is_idempotent_when_already_off() -> None:
    assert traigent_testing.is_mock_mode_enabled() is False

    traigent_testing.disable_mock_mode()  # must not raise
    traigent_testing.disable_mock_mode()  # second no-op call

    assert traigent_testing.is_mock_mode_enabled() is False


def test_disable_has_no_production_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unlike enable(), disable() must work even with ENVIRONMENT=production —
    turning mock mode OFF can never substitute a fake result for a real one."""
    traigent_testing.enable_mock_mode_for_quickstart()
    monkeypatch.setenv("ENVIRONMENT", "production")

    traigent_testing.disable_mock_mode()  # must not raise

    assert traigent_testing.is_mock_mode_enabled() is False


def test_disable_logs_info_on_deactivation(caplog: pytest.LogCaptureFixture) -> None:
    traigent_testing.enable_mock_mode_for_quickstart()

    with caplog.at_level(logging.INFO, logger="traigent.testing"):
        traigent_testing.disable_mock_mode()

    assert any("INACTIVE" in record.message for record in caplog.records), caplog.text


def test_disable_when_never_enabled_does_not_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="traigent.testing"):
        traigent_testing.disable_mock_mode()

    assert not any("INACTIVE" in record.message for record in caplog.records)


def test_same_process_test_run_then_real_run_no_longer_stays_fake() -> None:
    """Reproduces the issue's exact scenario: mock test-run, then a
    same-process "real" run must not silently keep using mocked responses
    once ``disable_mock_mode()`` is called in between."""
    traigent_testing.enable_mock_mode_for_quickstart()
    assert traigent_testing.is_mock_mode_enabled() is True  # the "test run"

    traigent_testing.disable_mock_mode()

    assert traigent_testing.is_mock_mode_enabled() is False  # the "real run"


def test_reenabling_after_disable_logs_activation_warning_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """After disable(), the activation-logged latch resets so a subsequent
    enable() in the same process is visible in logs again, not silently
    swallowed as a no-op repeat."""
    traigent_testing.enable_mock_mode_for_quickstart()
    traigent_testing.disable_mock_mode()

    with caplog.at_level(logging.WARNING, logger="traigent.testing"):
        traigent_testing.enable_mock_mode_for_quickstart()

    assert any("ACTIVE" in record.message for record in caplog.records)
