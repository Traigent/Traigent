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
        backend = LiteLLMBackend(budget_usd=0.0)
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
        mock_response.usage = None
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
