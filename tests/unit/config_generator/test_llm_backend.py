"""Tests for config_generator.llm_backend."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from traigent.config_generator.llm_backend import (
    BudgetExhausted,
    ConfigGenLLM,
    LiteLLMBackend,
    NoOpLLMBackend,
    UnmeasurableCostError,
)


class TestNoOpLLMBackend:
    def test_complete_returns_empty(self) -> None:
        backend = NoOpLLMBackend()
        assert backend.complete("test prompt") == ""

    def test_calls_made_zero(self) -> None:
        backend = NoOpLLMBackend()
        backend.complete("test")
        assert backend.calls_made == 0

    def test_total_cost_zero(self) -> None:
        backend = NoOpLLMBackend()
        assert backend.total_cost_usd == 0.0

    def test_implements_protocol(self) -> None:
        backend = NoOpLLMBackend()
        assert isinstance(backend, ConfigGenLLM)


class TestLiteLLMBackend:
    def test_budget_enforcement(self) -> None:
        backend = LiteLLMBackend(budget_usd=0.01)
        backend._spent_usd = 0.01
        with pytest.raises(BudgetExhausted, match="Budget"):
            backend.complete("test")

    def test_litellm_not_installed(self) -> None:
        backend = LiteLLMBackend(budget_usd=1.0)
        with patch.dict("sys.modules", {"litellm": None}):
            with pytest.raises(BudgetExhausted, match="not installed"):
                backend.complete("test")

    def test_successful_call(self) -> None:
        backend = LiteLLMBackend(budget_usd=1.0)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"

        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            result = backend.complete("test prompt", max_tokens=512)

        assert result == "test response"
        assert backend.calls_made == 1
        assert backend.total_cost_usd > 0
        mock_completion.assert_called_once()

    def test_cost_tracking(self) -> None:
        backend = LiteLLMBackend(budget_usd=1.0)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 1000
        mock_usage.completion_tokens = 500

        mock_choice = MagicMock()
        mock_choice.message.content = "response"

        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        with patch("litellm.completion", return_value=mock_response):
            backend.complete("prompt 1")
            backend.complete("prompt 2")

        assert backend.calls_made == 2
        assert backend.total_cost_usd > 0

    def test_null_content_returns_empty(self) -> None:
        backend = LiteLLMBackend(budget_usd=1.0)

        mock_choice = MagicMock()
        mock_choice.message.content = None

        mock_response = MagicMock()
        mock_response.usage = MagicMock(prompt_tokens=1, completion_tokens=0)
        mock_response.choices = [mock_choice]

        with patch("litellm.completion", return_value=mock_response):
            result = backend.complete("test")

        assert result == ""

    def test_implements_protocol(self) -> None:
        backend = LiteLLMBackend()
        assert isinstance(backend, ConfigGenLLM)

    def test_default_model(self) -> None:
        backend = LiteLLMBackend()
        assert backend._model == "gpt-4o-mini"

    def test_cost_tracking_is_thread_safe(self) -> None:
        """Concurrent completions should preserve call and cost accounting."""
        backend = LiteLLMBackend(budget_usd=1.0)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50

        mock_choice = MagicMock()
        mock_choice.message.content = "thread-safe"

        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        barrier = threading.Barrier(2)

        def _complete() -> None:
            barrier.wait()
            backend.complete("prompt")

        with patch("litellm.completion", return_value=mock_response):
            threads = [threading.Thread(target=_complete) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        expected_cost = 2 * ((100 * 0.00000015) + (50 * 0.0000006))
        assert backend.calls_made == 2
        assert backend.total_cost_usd == pytest.approx(expected_cost)


def _response(prompt_tokens: object, completion_tokens: object) -> MagicMock:
    choice = MagicMock()
    choice.message.content = "ok"
    response = MagicMock()
    response.usage = MagicMock(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    response.choices = [choice]
    return response


class TestBudgetSafety2928:
    @pytest.mark.parametrize(
        "bad", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, True, "1"]
    )
    def test_invalid_budget_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError, match="budget_usd"):
            LiteLLMBackend(budget_usd=bad)  # type: ignore[arg-type]

    def test_priced_with_configured_model_not_gpt4o_mini(self) -> None:
        """A pricier model must stop at the cap using its own rate."""
        import litellm

        expected = sum(
            litellm.cost_per_token(
                model="gpt-4o", prompt_tokens=100_000, completion_tokens=10_000
            )
        )
        mini = sum(
            litellm.cost_per_token(
                model="gpt-4o-mini", prompt_tokens=100_000, completion_tokens=10_000
            )
        )
        assert expected > 10 * mini
        backend = LiteLLMBackend(model="gpt-4o", budget_usd=1.0)
        with patch("litellm.completion", return_value=_response(100_000, 10_000)):
            backend.complete("p")
            assert backend.total_cost_usd == pytest.approx(expected)
            assert backend.calls_made == 1

    def test_single_call_cannot_overshoot_budget(self) -> None:
        """Worst-case cost of the next call is checked before sending it."""
        backend = LiteLLMBackend(model="gpt-4o", budget_usd=0.01)
        with patch("litellm.completion") as completion:
            with pytest.raises(BudgetExhausted, match="would be exceeded"):
                backend.complete("x" * 4000, max_tokens=1024)
        completion.assert_not_called()
        assert backend.calls_made == 0

    def test_unpriced_model_fails_closed_before_calling(self) -> None:
        backend = LiteLLMBackend(model="no-such-model-xyz", budget_usd=1.0)
        with patch("litellm.completion") as completion:
            with pytest.raises(UnmeasurableCostError, match="no-such-model-xyz"):
                backend.complete("p")
        completion.assert_not_called()
        assert backend.total_cost_usd == 0.0

    @pytest.mark.parametrize("usage", [None, "missing"])
    def test_missing_usage_is_not_free(self, usage: object) -> None:
        backend = LiteLLMBackend(budget_usd=1.0)
        response = _response(1, 1)
        response.usage = (
            None
            if usage is None
            else MagicMock(prompt_tokens=None, completion_tokens=None)
        )
        with patch("litellm.completion", return_value=response):
            with pytest.raises(UnmeasurableCostError, match="usage"):
                backend.complete("p")
            with pytest.raises(BudgetExhausted):
                backend.complete("p")
