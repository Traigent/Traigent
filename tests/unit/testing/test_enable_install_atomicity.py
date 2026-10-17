"""enable_mock_mode_for_quickstart() publishes enablement only after install.

Review finding (PR 2532): ``_enabled`` was set before the interceptors were
installed, so a concurrent caller returned before interception existed and an
install failure left enablement latched.
"""

from __future__ import annotations

import threading

import pytest

import traigent.testing as testing


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    testing._reset_for_tests()
    yield
    testing._reset_for_tests()


def test_install_failure_does_not_latch_enablement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom() -> None:
        raise ImportError("install failed")

    monkeypatch.setattr(testing, "_install_mock_interceptors", boom)
    with pytest.raises(ImportError):
        testing.enable_mock_mode_for_quickstart()
    assert testing.is_mock_mode_enabled() is False

    installed: list[int] = []
    monkeypatch.setattr(
        testing, "_install_mock_interceptors", lambda: installed.append(1)
    )
    testing.enable_mock_mode_for_quickstart()  # retried, not a latched no-op
    assert installed == [1]
    assert testing.is_mock_mode_enabled() is True


def test_concurrent_caller_does_not_return_before_install_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    finished_install = threading.Event()

    def slow_install() -> None:
        started.set()
        assert release.wait(10)
        finished_install.set()

    monkeypatch.setattr(testing, "_install_mock_interceptors", slow_install)
    first = threading.Thread(target=testing.enable_mock_mode_for_quickstart)
    first.start()
    assert started.wait(10)

    seen_at_return: list[bool] = []

    def second_call() -> None:
        testing.enable_mock_mode_for_quickstart()
        seen_at_return.append(finished_install.is_set())

    second = threading.Thread(target=second_call)
    second.start()
    second.join(0.5)  # must still be waiting for the first install
    assert second.is_alive() or seen_at_return == [True]
    assert testing.is_mock_mode_enabled() is False  # not published mid-install
    release.set()
    first.join(10)
    second.join(10)
    assert seen_at_return == [True]
    assert testing.is_mock_mode_enabled() is True
