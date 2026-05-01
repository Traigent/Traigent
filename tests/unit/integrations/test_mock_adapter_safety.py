"""Safety guarantees for mock-mode activation.

These tests guard the original prod incident: a stray ``TRAIGENT_MOCK_LLM``
env var must NOT silently swap real LLM calls for canned mock responses.
Mock mode is process-local and only flips on after an explicit in-code
call to :func:`traigent.testing.enable_mock_mode_for_quickstart`.

Also covers Codex review finding #5: every LLM-interceptor call site must
read the same flag. We check all four (LangChain OpenAI, LangChain
Anthropic, LiteLLM sync, LiteLLM async) via ``MockAdapter.is_mock_enabled``,
which is what each interceptor consults at runtime.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from traigent import testing as traigent_testing
from traigent.integrations.utils.mock_adapter import MockAdapter
from traigent.utils.env_config import is_mock_llm


@pytest.fixture(autouse=True)
def reset_mock_mode_flag() -> Iterator[None]:
    traigent_testing._reset_for_tests()
    yield
    traigent_testing._reset_for_tests()


@pytest.fixture(autouse=True)
def _restore_environ() -> Iterator[None]:
    """Snapshot ``os.environ`` and restore it after the test.

    Codex review (round 3) finding #2: tests that call
    ``importlib.reload(env_config)`` trigger ``load_dotenv()`` at module
    init, which mutates ``os.environ`` *outside* of monkeypatch's
    tracking. Without this fixture, vars added by a ``.env`` file leak
    into subsequent tests — including ``ENVIRONMENT=production``, which
    flipped a downstream test result.
    """
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


def test_env_var_does_not_enable_in_code_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-code flag must remain false when only the env var is set —
    the API call is the only thing that flips :func:`is_mock_mode_enabled`.

    The env var still triggers the SDK's mock-aware behavior in
    non-production (see ``test_env_var_works_in_dev_blocks_in_prod``),
    but that path is gated by environment, not by this flag."""
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

    assert traigent_testing.is_mock_mode_enabled() is False, (
        "enable_mock_mode_for_quickstart() is the only way to flip the "
        "in-code flag — env var must not touch it."
    )


def test_env_var_works_in_dev_blocks_in_prod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward-compat path: ``TRAIGENT_MOCK_LLM=true`` activates mock
    mode in dev/test (so existing fixtures keep working) but the env-var
    branch is hard-skipped when ``ENVIRONMENT=production``. This is the
    surviving guard against the original prod incident."""
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

    monkeypatch.setenv("ENVIRONMENT", "development")
    # Re-evaluate the cached _is_production_env in env_config — it's set
    # at import time; re-import to refresh.
    import importlib

    from traigent.utils import env_config as _ec

    importlib.reload(_ec)
    assert _ec.is_mock_llm() is True, "dev env must honor env-var fallback"

    monkeypatch.setenv("ENVIRONMENT", "production")
    # Production env-var presence is now hard-blocked at import; the
    # import itself must raise. This is why the sweeping prod-time
    # incident is no longer reachable.
    with pytest.raises(OSError, match="production"):
        importlib.reload(_ec)

    # Restore for downstream tests.
    monkeypatch.setenv("ENVIRONMENT", "development")
    importlib.reload(_ec)


def test_in_code_api_enables_mock_for_every_interceptor() -> None:
    """Codex finding #5: all four interceptor call sites must consult the
    same flag. ``MockAdapter.is_mock_enabled`` is what each interceptor
    actually calls — verify every provider name flips on together."""
    assert traigent_testing.is_mock_mode_enabled() is False

    traigent_testing.enable_mock_mode_for_quickstart()

    assert traigent_testing.is_mock_mode_enabled() is True
    assert is_mock_llm() is True
    for provider in ("openai", "anthropic", "litellm", "azure_openai", "gemini"):
        assert MockAdapter.is_mock_enabled(provider) is True, (
            f"Provider {provider} did not see mock mode after explicit "
            "API call — interceptor sites are not centralized."
        )


def test_activation_is_idempotent_and_logs_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """First activation logs a single mandatory WARN. Subsequent calls
    are no-ops and must NOT spam logs (Codex finding #2)."""
    import logging

    with caplog.at_level(logging.WARNING, logger="traigent.testing"):
        traigent_testing.enable_mock_mode_for_quickstart()
        traigent_testing.enable_mock_mode_for_quickstart()
        traigent_testing.enable_mock_mode_for_quickstart()

    activation_warns = [
        r for r in caplog.records if "mock mode is now ACTIVE" in r.getMessage()
    ]
    assert len(activation_warns) == 1, (
        f"Expected exactly one mandatory activation WARN, got "
        f"{len(activation_warns)} — log spam will hide the signal"
    )


def test_in_code_api_is_blocked_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex review #3: ``enable_mock_mode_for_quickstart`` itself must
    be blocked in production, not just the env-var path. Otherwise a
    misconfigured deployment that runs example scripts in prod could
    silently activate mock mode."""
    monkeypatch.setenv("ENVIRONMENT", "production")

    with pytest.raises(RuntimeError, match="production"):
        traigent_testing.enable_mock_mode_for_quickstart()

    assert traigent_testing.is_mock_mode_enabled() is False, (
        "Mock mode must NOT be enabled when the in-code API was rejected"
    )


def test_dotenv_late_load_does_not_bypass_prod_guard(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex review #1 regression test: a ``.env`` file that lands
    ``ENVIRONMENT=production`` and ``TRAIGENT_MOCK_LLM=true`` into
    ``os.environ`` AFTER the first guard runs must still trigger the
    OSError on the post-dotenv pass."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ENVIRONMENT=production\nTRAIGENT_MOCK_LLM=true\n",  # pragma: allowlist secret
        encoding="utf-8",
    )

    # Drop any pre-existing values so the dotenv load is the path that
    # introduces them. ``load_dotenv`` is non-overriding by default —
    # which is the exact reason a manual ``os.environ.update`` would not
    # reproduce this scenario.
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)

    from dotenv import load_dotenv

    from traigent.utils.env_config import _check_mock_llm_prod_guard

    # First pass (no .env loaded yet) — guard sees clean env, no-op.
    _check_mock_llm_prod_guard()

    # Simulate the SDK's own dotenv load, then re-run the guard exactly
    # as env_config.py does at module import.
    load_dotenv(env_file)
    with pytest.raises(OSError, match="production"):
        _check_mock_llm_prod_guard()


def test_reset_for_tests_clears_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """The private test helper must fully reset both the flag and the
    once-only log gate, so tests that re-activate get the WARN again."""
    # The repo's autouse conftest fixture sets TRAIGENT_MOCK_LLM=true for
    # every test. Clear it here so we can prove the in-code flag is the
    # only thing keeping mock mode on; otherwise the env-var fallback
    # would mask whether the reset actually worked.
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)

    traigent_testing.enable_mock_mode_for_quickstart()
    assert traigent_testing.is_mock_mode_enabled() is True

    traigent_testing._reset_for_tests()

    assert traigent_testing.is_mock_mode_enabled() is False
    for provider in ("openai", "anthropic", "litellm"):
        assert MockAdapter.is_mock_enabled(provider) is False
