"""Tests for mock adapter utilities.

Note: ``MockAdapter`` previously consulted ``TRAIGENT_MOCK_LLM`` and a set
of provider-specific ``*_MOCK`` environment variables to decide whether
to swap real LLM calls for canned mock text. That env-toggle was removed
because a stray env var in production caused real calls to be silently
replaced with mock data. ``is_mock_enabled`` now always returns False;
tests that need mock responses must patch the LLM client directly or
call ``MockAdapter.get_mock_response`` from inside an explicit
``unittest.mock.patch``.
"""

import os
from unittest.mock import patch

from traigent.integrations.utils.mock_adapter import MockAdapter, MockResponse


class TestMockResponse:
    """Tests for MockResponse dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        response = MockResponse()
        assert response.text == "This is a mock response for testing."
        assert response.model == "mock-model"
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 20
        assert response.total_tokens == 30
        assert response.finish_reason == "stop"
        assert response.response_id == "mock-response-id"

    def test_custom_values(self) -> None:
        """Test custom values override defaults."""
        response = MockResponse(
            text="Custom text",
            model="custom-model",
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
        )
        assert response.text == "Custom text"
        assert response.model == "custom-model"
        assert response.prompt_tokens == 100
        assert response.completion_tokens == 200
        assert response.total_tokens == 300


class TestMockAdapterIsMockEnabled:
    """``is_mock_enabled`` always returns False (env-toggle removed)."""

    def test_global_mock_env_is_ignored(self) -> None:
        """TRAIGENT_MOCK_LLM=true must NOT enable mock mode."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "true"}, clear=True):
            assert MockAdapter.is_mock_enabled("openai") is False
            assert MockAdapter.is_mock_enabled("anthropic") is False
            assert MockAdapter.is_mock_enabled("unknown") is False

    def test_global_mock_env_1_is_ignored(self) -> None:
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "1"}, clear=True):
            assert MockAdapter.is_mock_enabled("openai") is False

    def test_global_mock_env_yes_is_ignored(self) -> None:
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "yes"}, clear=True):
            assert MockAdapter.is_mock_enabled("openai") is False

    def test_provider_specific_env_is_ignored(self) -> None:
        """Provider-specific ``*_MOCK`` env vars must NOT enable mock mode."""
        with patch.dict(os.environ, {"OPENAI_MOCK": "true"}, clear=True):
            assert MockAdapter.is_mock_enabled("openai") is False
            assert MockAdapter.is_mock_enabled("anthropic") is False

    def test_no_env_returns_false(self) -> None:
        """With no env vars, is_mock_enabled returns False."""
        with patch.dict(os.environ, {}, clear=True):
            assert MockAdapter.is_mock_enabled("openai") is False
            assert MockAdapter.is_mock_enabled("anthropic") is False

    def test_case_insensitive_provider_still_false(self) -> None:
        with patch.dict(os.environ, {"OPENAI_MOCK": "true"}, clear=True):
            assert MockAdapter.is_mock_enabled("OpenAI") is False
            assert MockAdapter.is_mock_enabled("OPENAI") is False


class TestMockAdapterGetMockResponse:
    """``get_mock_response`` works for tests that invoke it explicitly."""

    def test_openai_mock_response(self) -> None:
        """Test OpenAI mock response structure."""
        response = MockAdapter.get_mock_response("openai")
        assert "id" in response
        assert response["object"] == "chat.completion"
        assert "choices" in response
        assert len(response["choices"]) == 1
        assert response["choices"][0]["message"]["role"] == "assistant"
        assert "usage" in response

    def test_azure_openai_uses_openai_format(self) -> None:
        """Test Azure OpenAI uses same format as OpenAI."""
        response = MockAdapter.get_mock_response("azure_openai")
        assert response["object"] == "chat.completion"
        assert "choices" in response

    def test_anthropic_mock_response(self) -> None:
        """Test Anthropic mock response structure."""
        response = MockAdapter.get_mock_response("anthropic")
        assert response["type"] == "message"
        assert response["role"] == "assistant"
        assert "content" in response
        assert len(response["content"]) == 1
        assert response["content"][0]["type"] == "text"
        assert "usage" in response

    def test_gemini_mock_response(self) -> None:
        """Test Gemini mock response structure."""
        response = MockAdapter.get_mock_response("gemini")
        assert "candidates" in response
        assert len(response["candidates"]) == 1
        assert response["candidates"][0]["content"]["role"] == "model"
        assert "parts" in response["candidates"][0]["content"]
        assert "usage_metadata" in response

    def test_cohere_mock_response(self) -> None:
        """Test Cohere mock response structure."""
        response = MockAdapter.get_mock_response("cohere")
        assert "id" in response
        assert "text" in response
        assert "meta" in response
        assert "billed_units" in response["meta"]

    def test_unknown_provider_uses_generic(self) -> None:
        """Test unknown provider uses generic format."""
        response = MockAdapter.get_mock_response("unknown_provider")
        assert "text" in response
        assert "model" in response
        assert "usage" in response

    def test_custom_response_text(self) -> None:
        """Test custom response text is used."""
        response = MockAdapter.get_mock_response(
            "openai", response_text="Custom response"
        )
        assert response["choices"][0]["message"]["content"] == "Custom response"

    def test_custom_model(self) -> None:
        """Test custom model is used."""
        response = MockAdapter.get_mock_response("openai", model="gpt-4-turbo")
        assert response["model"] == "gpt-4-turbo"

    def test_model_from_kwargs(self) -> None:
        """Test model extracted from kwargs."""
        response = MockAdapter.get_mock_response(
            "openai", model=None, model_kwarg="gpt-3.5-turbo"
        )
        # Should use MockResponse.model as default since model=None
        assert response["model"] == "mock-model"


class TestBuildMockMethods:
    """Tests for individual mock builder methods."""

    def test_build_openai_mock_structure(self) -> None:
        """Test OpenAI mock structure is complete."""
        data = MockResponse()
        result = MockAdapter._build_openai_mock(data)

        assert result["id"] == data.response_id
        assert result["object"] == "chat.completion"
        assert "created" in result
        assert result["model"] == data.model
        assert result["choices"][0]["index"] == 0
        assert result["choices"][0]["finish_reason"] == data.finish_reason
        assert result["usage"]["prompt_tokens"] == data.prompt_tokens
        assert result["usage"]["completion_tokens"] == data.completion_tokens
        assert result["usage"]["total_tokens"] == data.total_tokens

    def test_build_anthropic_mock_structure(self) -> None:
        """Test Anthropic mock structure is complete."""
        data = MockResponse()
        result = MockAdapter._build_anthropic_mock(data)

        assert result["id"] == data.response_id
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == data.model
        assert result["stop_reason"] == data.finish_reason
        assert result["usage"]["input_tokens"] == data.prompt_tokens
        assert result["usage"]["output_tokens"] == data.completion_tokens

    def test_build_gemini_mock_structure(self) -> None:
        """Test Gemini mock structure is complete."""
        data = MockResponse()
        result = MockAdapter._build_gemini_mock(data)

        candidate = result["candidates"][0]
        assert candidate["content"]["role"] == "model"
        assert candidate["content"]["parts"][0]["text"] == data.text
        assert candidate["finish_reason"] == "STOP"
        assert candidate["index"] == 0
        assert result["usage_metadata"]["prompt_token_count"] == data.prompt_tokens
        assert (
            result["usage_metadata"]["candidates_token_count"] == data.completion_tokens
        )
        assert result["usage_metadata"]["total_token_count"] == data.total_tokens

    def test_build_cohere_mock_structure(self) -> None:
        """Test Cohere mock structure is complete."""
        data = MockResponse()
        result = MockAdapter._build_cohere_mock(data)

        assert result["id"] == data.response_id
        assert result["text"] == data.text
        assert result["generation_id"] == data.response_id
        assert result["meta"]["billed_units"]["input_tokens"] == data.prompt_tokens
        assert result["meta"]["billed_units"]["output_tokens"] == data.completion_tokens

    def test_build_generic_mock_structure(self) -> None:
        """Test generic mock structure is complete."""
        data = MockResponse()
        result = MockAdapter._build_generic_mock(data)

        assert result["text"] == data.text
        assert result["model"] == data.model
        assert result["usage"]["prompt_tokens"] == data.prompt_tokens
        assert result["usage"]["completion_tokens"] == data.completion_tokens
        assert result["usage"]["total_tokens"] == data.total_tokens
