from __future__ import annotations

from pathlib import Path

import pytest

from traigent.examples.tutorial_bootstrap import (
    configure_tutorial_mock_mode,
    should_auto_enable_mock_mode,
)
from traigent.testing import _reset_for_tests, is_mock_mode_enabled


@pytest.fixture(autouse=True)
def reset_mock_mode() -> None:
    _reset_for_tests()
    yield
    _reset_for_tests()


def test_auto_enables_mock_mode_when_provider_key_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env: dict[str, str] = {}

    enabled = configure_tutorial_mock_mode(
        provider_env_keys=("ANTHROPIC_API_KEY",),
        tutorial_name="Simple Prompt Optimization",
        results_base=tmp_path,
        env=env,
    )

    assert enabled is True
    assert is_mock_mode_enabled() is True
    assert env["TRAIGENT_OFFLINE_MODE"] == "true"
    assert env["HOME"] == str(tmp_path)
    assert env["TRAIGENT_RESULTS_FOLDER"] == str(tmp_path / ".traigent_local")
    assert "TRAIGENT_MOCK_LLM" not in env
    assert "local mock mode" in capsys.readouterr().err


def test_mock_tutorial_defaults_offline_even_with_portal_key() -> None:
    env = {"TRAIGENT_API_KEY": "test-traigent-key"}  # pragma: allowlist secret

    enabled = configure_tutorial_mock_mode(
        provider_env_keys=("ANTHROPIC_API_KEY",),
        tutorial_name="Simple Prompt Optimization",
        env=env,
    )

    assert enabled is True
    assert env["TRAIGENT_OFFLINE_MODE"] == "true"
    assert "TRAIGENT_API_KEY" not in env


def test_explicit_offline_false_allows_mock_tutorial_sync() -> None:
    env = {
        "TRAIGENT_API_KEY": "test-traigent-key",  # pragma: allowlist secret
        "TRAIGENT_OFFLINE_MODE": "false",
    }

    enabled = configure_tutorial_mock_mode(
        provider_env_keys=("ANTHROPIC_API_KEY",),
        tutorial_name="Simple Prompt Optimization",
        env=env,
    )

    assert enabled is True
    assert env["TRAIGENT_OFFLINE_MODE"] == "false"


def test_provider_key_keeps_real_mode() -> None:
    env = {"ANTHROPIC_API_KEY": "test-anthropic-key"}  # pragma: allowlist secret

    enabled = configure_tutorial_mock_mode(
        provider_env_keys=("ANTHROPIC_API_KEY",),
        tutorial_name="Simple Prompt Optimization",
        env=env,
    )

    assert enabled is False
    assert is_mock_mode_enabled() is False
    assert "TRAIGENT_OFFLINE_MODE" not in env


@pytest.mark.parametrize("value", ["false", "0", "no", "off"])
def test_explicit_mock_false_is_respected(value: str) -> None:
    env = {"TRAIGENT_MOCK_LLM": value}

    assert should_auto_enable_mock_mode(("ANTHROPIC_API_KEY",), env=env) is False


@pytest.mark.parametrize("value", ["true", "1", "yes", "on"])
def test_explicit_mock_true_is_respected_even_with_provider_key(value: str) -> None:
    env = {
        "TRAIGENT_MOCK_LLM": value,
        "ANTHROPIC_API_KEY": "test-anthropic-key",  # pragma: allowlist secret
    }

    assert should_auto_enable_mock_mode(("ANTHROPIC_API_KEY",), env=env) is True


def test_production_guard_still_blocks_auto_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")

    with pytest.raises(RuntimeError, match="ENVIRONMENT=production"):
        configure_tutorial_mock_mode(
            provider_env_keys=("ANTHROPIC_API_KEY",),
            tutorial_name="Simple Prompt Optimization",
            env={},
        )
