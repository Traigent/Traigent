"""Credential-error hint is mock-aware and does not repeat itself (#2417).

Before the fix the hint always said "use TRAIGENT_MOCK_LLM=true for testing",
even with mock mode already on, and ``APIKeyError`` built its fix line as
``Add <the whole message> to your .env file``, so the provider error appeared
twice.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from traigent import testing as traigent_testing
from traigent.evaluators.base import BaseEvaluator
from traigent.invokers.local import LocalInvoker
from traigent.utils.error_handler import APIKeyError, provider_credential_error

_PROVIDER_MESSAGE = (
    "Missing credentials. Please pass an `api_key`, or set the "
    "`OPENAI_API_KEY` environment variable."
)


@pytest.fixture(autouse=True)
def _mock_flag_reset(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # The suite conftest turns the legacy env var on; each test picks its mode.
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    traigent_testing._reset_for_tests()
    yield
    traigent_testing._reset_for_tests()


def _evaluator_error() -> APIKeyError:
    with pytest.raises(APIKeyError) as info:
        BaseEvaluator._check_api_key_error(None, RuntimeError(_PROVIDER_MESSAGE))  # type: ignore[arg-type]
    return info.value


def _invoker_error() -> APIKeyError:
    def agent(question: str) -> str:
        raise RuntimeError(_PROVIDER_MESSAGE)

    invoker = LocalInvoker()
    with pytest.raises(APIKeyError) as info:
        asyncio.run(invoker.invoke(agent, {}, {"question": "q"}))
    return info.value


def test_hint_under_mock_mode_explains_the_bypass() -> None:
    traigent_testing.enable_mock_mode_for_quickstart()
    text = str(_evaluator_error())
    assert "TRAIGENT_MOCK_LLM" not in text
    assert "Mock mode is already active" in text
    for option in ("LiteLLM/LangChain", "stub the client", "placeholder key"):
        assert option in text


def test_hint_without_mock_mode_names_the_in_code_helper() -> None:
    text = str(_evaluator_error())
    assert "enable_mock_mode_for_quickstart()" in text
    assert "TRAIGENT_MOCK_LLM" not in text


@pytest.mark.parametrize("mock_on", [False, True])
def test_provider_message_appears_once_and_both_sites_agree(mock_on: bool) -> None:
    if mock_on:
        traigent_testing.enable_mock_mode_for_quickstart()
    from_evaluator = _evaluator_error()
    from_invoker = _invoker_error()
    assert str(from_evaluator).count(_PROVIDER_MESSAGE) == 1
    assert str(from_evaluator) == str(from_invoker)


def test_non_credential_error_is_not_converted() -> None:
    assert provider_credential_error(RuntimeError("division by zero")) is None


def test_key_name_contract_is_unchanged() -> None:
    error = APIKeyError("OPENAI_API_KEY")
    assert error.message == "Missing or invalid API key: OPENAI_API_KEY"
    assert error.fix == (
        "Add OPENAI_API_KEY to your .env file or set as environment variable"
    )
