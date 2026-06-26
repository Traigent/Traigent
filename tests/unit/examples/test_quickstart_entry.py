"""Tests for the quickstart env-configuration helper.

Guards the current quickstart contract: the demo "just works" - when a
``TRAIGENT_API_KEY`` is set results sync to the portal (with LLM calls
still mocked for the packaged demo), and without a key the demo still
runs but prints results locally instead of syncing.

* When the quickstart runs without a ``TRAIGENT_API_KEY``, the user must
  receive an explicit notice that results print locally and that setting
  ``TRAIGENT_API_KEY`` with ``experiments:write`` and running the positive
  publish-and-verify example syncs a mock run to the portal.
* The helper must force ``TRAIGENT_OFFLINE_MODE=true`` when a portal key is
  absent so stored credentials cannot leak a keyless quickstart run to the
  backend. When a key is present, it must not override the caller's explicit
  offline-mode choice.
* The helper no longer touches ``TRAIGENT_MOCK_LLM`` — mock mode is
  activated in code via :func:`traigent.testing.enable_mock_mode_for_quickstart`.
"""

from __future__ import annotations

import pytest

from traigent.examples.quickstart._env import configure_quickstart_env


@pytest.fixture
def empty_env() -> dict[str, str]:
    return {}


def test_warns_when_no_api_key(
    empty_env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Without a portal key the notice must tell the user the demo still
    runs (results print locally) and point to the positive portal-sync
    example."""
    configure_quickstart_env(empty_env)

    stderr = capsys.readouterr().err
    assert "TRAIGENT_API_KEY" in stderr
    # New contract: notice explains the demo runs locally and syncing is
    # opt-in via a portal key; the legacy "offline" wording is gone.
    assert "locally" in stderr.lower()
    assert "sync" in stderr.lower()
    assert "publish_and_verify" in stderr
    assert "offline" not in stderr.lower()
    assert empty_env["TRAIGENT_OFFLINE_MODE"] == "true"
    assert (
        empty_env["OPENAI_API_KEY"] == "mock-key-for-demos"
    )  # pragma: allowlist secret


def test_no_warning_when_api_key_present(
    capsys: pytest.CaptureFixture[str],
) -> None:
    env: dict[str, str] = {"TRAIGENT_API_KEY": "real-key"}

    configure_quickstart_env(env)

    stderr = capsys.readouterr().err
    assert "TRAIGENT_API_KEY" not in stderr
    assert "TRAIGENT_OFFLINE_MODE" not in env
    assert env["OPENAI_API_KEY"] == "mock-key-for-demos"  # pragma: allowlist secret


def test_does_not_set_legacy_mock_env_var(empty_env: dict[str, str]) -> None:
    configure_quickstart_env(empty_env)
    # Legacy env-var-based activation is gone; mock mode is enabled in
    # code via traigent.testing.enable_mock_mode_for_quickstart().
    assert "TRAIGENT_MOCK_LLM" not in empty_env


def test_forces_offline_when_no_api_key(
    empty_env: dict[str, str],
) -> None:
    """Without a portal key, the helper must force offline execution."""
    empty_env["TRAIGENT_OFFLINE_MODE"] = "false"

    configure_quickstart_env(empty_env)

    assert empty_env["TRAIGENT_OFFLINE_MODE"] == "true"


def test_does_not_touch_offline_when_api_key_present(
    empty_env: dict[str, str],
) -> None:
    """When a portal key IS set, the user wants results synced; this
    helper must not override their offline-mode choice."""
    empty_env["TRAIGENT_API_KEY"] = "real-key"  # pragma: allowlist secret
    empty_env["TRAIGENT_OFFLINE_MODE"] = "false"

    configure_quickstart_env(empty_env)

    assert empty_env["TRAIGENT_OFFLINE_MODE"] == "false"


def test_preserves_existing_openai_key(empty_env: dict[str, str]) -> None:
    empty_env["OPENAI_API_KEY"] = "real-openai-key"  # pragma: allowlist secret

    configure_quickstart_env(empty_env)

    assert empty_env["OPENAI_API_KEY"] == "real-openai-key"  # pragma: allowlist secret
