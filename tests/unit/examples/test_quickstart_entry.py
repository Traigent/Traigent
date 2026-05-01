"""Tests for the quickstart env-configuration helper.

Guards two invariants:

* When the quickstart runs without a ``TRAIGENT_API_KEY``, the user must
  receive an explicit notice before the script forces
  ``TRAIGENT_OFFLINE_MODE=true`` (otherwise the offline short-circuit
  suppresses the downstream "No API key found" warning — original PR #664
  review).
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
    configure_quickstart_env(empty_env)

    stderr = capsys.readouterr().err
    assert "TRAIGENT_API_KEY" in stderr
    assert "offline" in stderr.lower()
    assert empty_env["TRAIGENT_OFFLINE_MODE"] == "true"
    assert empty_env["OPENAI_API_KEY"] == "mock-key-for-demos"  # pragma: allowlist secret


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


def test_overrides_offline_false_when_no_api_key(
    empty_env: dict[str, str],
) -> None:
    """Codex review (round 3) finding #4: without a portal API key the
    SDK can't sync anyway, so an explicit ``TRAIGENT_OFFLINE_MODE=false``
    plus no key would produce noisy auth errors. We force offline in
    that case (override, not setdefault)."""
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
