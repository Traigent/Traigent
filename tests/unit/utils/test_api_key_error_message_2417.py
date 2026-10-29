"""APIKeyError built from a provider failure: readable fix line, mock-aware
hint (#2417)."""

from collections.abc import Iterator

import pytest

from traigent import testing as traigent_testing
from traigent.utils.error_handler import APIKeyError


@pytest.fixture(autouse=True)
def reset_mock_mode_flag() -> Iterator[None]:
    # The in-code mock flag leaks from other tests in the same xdist worker;
    # is_mock_llm() reads it, so the mock-aware hint would flip under us.
    traigent_testing._reset_for_tests()
    yield
    traigent_testing._reset_for_tests()


def test_message_is_not_treated_as_key_name(monkeypatch):
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    err = APIKeyError.from_provider_error(RuntimeError("Missing credentials"))
    assert err.message.startswith("API key error detected")
    assert err.fix is not None
    assert "API key error detected" not in err.fix
    assert "to your .env file" not in err.fix
    assert "TRAIGENT_MOCK_LLM=true" in err.fix


def test_hint_omits_mock_advice_when_mock_already_on(monkeypatch):
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    err = APIKeyError.from_provider_error(RuntimeError("Missing credentials"))
    assert "TRAIGENT_MOCK_LLM=true" not in err.fix
    assert "already on" in err.fix
    assert "placeholder key" in err.fix


def test_key_name_form_unchanged():
    err = APIKeyError("OPENAI_API_KEY")
    assert err.message == "Missing or invalid API key: OPENAI_API_KEY"
    assert (
        err.fix == "Add OPENAI_API_KEY to your .env file or set as environment variable"
    )


def test_error_never_echoes_exception_text(monkeypatch):
    for mock in ("true", "false"):
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", mock)
        exc = RuntimeError(
            "Incorrect API key provided: sk-SECRETMARKER123 for prompt "
            "PROMPTMARKER-hello reading /home/x/data.csv"
        )
        err = APIKeyError.from_provider_error(exc)
        shown = f"{err} {err.message} {err.fix}"
        for marker in ("sk-SECRETMARKER123", "PROMPTMARKER", "/home/x/data.csv"):
            assert marker not in shown
        assert "RuntimeError" in err.message
        assert "category: auth" in err.message


def test_original_exception_stays_chained_at_call_site():
    from traigent.evaluators.base import SimpleScoringEvaluator

    class AuthenticationError(Exception):
        """Stand-in for a provider SDK auth error (decided by class, #2842)."""

    exc = AuthenticationError("authentication failed sk-SECRETMARKER123")
    try:
        SimpleScoringEvaluator._check_api_key_error(None, exc)
    except APIKeyError as err:
        assert err.__cause__ is exc
        assert "sk-SECRETMARKER123" not in str(err)
    else:
        raise AssertionError("expected APIKeyError")


def test_category_is_closed_set():
    for text, cat in (
        ("permission denied", "permission"),
        ("missing key", "missing-key"),
        ("something odd", "unknown"),
    ):
        err = APIKeyError.from_provider_error(RuntimeError(text))
        assert f"category: {cat}" in err.message
