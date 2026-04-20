"""Tests for the quickstart env-configuration helper.

Guards the invariant raised in PR #664 review: when the quickstart runs in mock
mode without a ``TRAIGENT_API_KEY``, the user must receive an explicit notice
before the script silently forces ``TRAIGENT_OFFLINE_MODE=true`` (the offline
short-circuit otherwise suppresses the downstream "No API key found" warning).
"""

from __future__ import annotations

import pytest

from traigent.examples.quickstart._env import configure_quickstart_env


@pytest.fixture
def empty_env() -> dict[str, str]:
    return {}


def test_warns_when_no_api_key_in_mock_mode(
    empty_env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    configure_quickstart_env(empty_env)

    stderr = capsys.readouterr().err
    assert "TRAIGENT_API_KEY" in stderr
    assert "offline" in stderr.lower()
    assert empty_env["TRAIGENT_OFFLINE_MODE"] == "true"
    assert empty_env["TRAIGENT_MOCK_LLM"] == "true"


def test_no_warning_when_api_key_present(
    capsys: pytest.CaptureFixture[str],
) -> None:
    env: dict[str, str] = {"TRAIGENT_API_KEY": "real-key"}

    configure_quickstart_env(env)

    stderr = capsys.readouterr().err
    assert "TRAIGENT_API_KEY" not in stderr
    assert "TRAIGENT_OFFLINE_MODE" not in env
    assert env["TRAIGENT_MOCK_LLM"] == "true"


def test_respects_explicit_mock_llm_false(
    capsys: pytest.CaptureFixture[str],
) -> None:
    env: dict[str, str] = {"TRAIGENT_MOCK_LLM": "false"}

    configure_quickstart_env(env)

    stderr = capsys.readouterr().err
    assert stderr == ""
    assert "TRAIGENT_OFFLINE_MODE" not in env
    assert "OPENAI_API_KEY" not in env


@pytest.mark.parametrize("truthy", ["1", "true", "YES", "On"])
def test_accepts_all_truthy_mock_llm_values(truthy: str) -> None:
    env: dict[str, str] = {"TRAIGENT_MOCK_LLM": truthy}

    configure_quickstart_env(env)

    assert env["OPENAI_API_KEY"] == "mock-key-for-demos"  # pragma: allowlist secret
    assert env["TRAIGENT_OFFLINE_MODE"] == "true"


def test_preserves_existing_offline_mode(
    empty_env: dict[str, str],
) -> None:
    empty_env["TRAIGENT_OFFLINE_MODE"] = "false"

    configure_quickstart_env(empty_env)

    assert empty_env["TRAIGENT_OFFLINE_MODE"] == "false"
