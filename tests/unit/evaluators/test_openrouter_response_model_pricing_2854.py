"""Traigent#2854: an OpenRouter response reports its model WITHOUT the
``openrouter/`` prefix; that name must not be priced when the requested id
(or the prefixed form of the returned one) resolves.

Keyless: LiteLLM ``mock_response`` calls priced by litellm's bundled table.
"""

from __future__ import annotations

import logging

import pytest

litellm = pytest.importorskip("litellm")

from traigent.evaluators.local import LocalEvaluator  # noqa: E402
from traigent.utils.response_model import resolve_pricing_model  # noqa: E402

OPENROUTER_IDS = [
    "openrouter/anthropic/claude-haiku-4.5",
    "openrouter/meta-llama/llama-3.3-70b-instruct",
    "openrouter/mistralai/mistral-small-3.2-24b-instruct",
]


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)


@pytest.mark.parametrize("requested", OPENROUTER_IDS)
def test_prefix_stripped_response_model_is_priced_via_openrouter(
    requested: str,
) -> None:
    returned = requested.removeprefix("openrouter/")
    assert resolve_pricing_model(returned, requested) == requested


def test_response_model_that_resolves_on_its_own_is_kept() -> None:
    # gpt-4o-mini resolves with or without the prefix: the response wins.
    assert (
        resolve_pricing_model("gpt-4o-mini-2024-07-18", "openrouter/openai/gpt-4o-mini")
        == "gpt-4o-mini-2024-07-18"
    )


def test_non_openrouter_request_never_rewrites_the_response_model() -> None:
    assert (
        resolve_pricing_model("anthropic/claude-haiku-4.5", "claude-haiku-4-5")
        == "anthropic/claude-haiku-4.5"
    )
    assert resolve_pricing_model("anthropic/claude-haiku-4.5", None) == (
        "anthropic/claude-haiku-4.5"
    )


def test_no_candidate_resolves_returns_response_model_so_the_warning_is_real() -> None:
    assert (
        resolve_pricing_model(
            "acme/unreleased-model", "openrouter/acme/unreleased-model"
        )
        == "acme/unreleased-model"
    )


@pytest.mark.parametrize("requested", OPENROUTER_IDS)
def test_priced_call_has_no_unknown_model_warning(
    requested: str, caplog: pytest.LogCaptureFixture
) -> None:
    response = litellm.completion(
        model=requested,
        messages=[{"role": "user", "content": "q"}],
        mock_response="paris",
    )
    assert not response.model.startswith("openrouter/")  # the premise of #2854
    expected = litellm.completion_cost(completion_response=response)

    with caplog.at_level(logging.WARNING):
        metrics = LocalEvaluator._price_one_call(
            response,
            agent_model=requested,
            is_agent_call=True,
            requested_model=requested,
        )

    assert "Unknown model" not in caplog.text
    assert metrics.cost.total_cost == pytest.approx(expected)
    assert not metrics.cost.unpriced
