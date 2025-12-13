"""Tests for response wrapper utilities."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from traigent.integrations.utils.response_wrapper import (
    LLMResponse,
    _extract_anthropic_metadata,
    _extract_cohere_metadata,
    _extract_gemini_metadata,
    _extract_generic_metadata,
    _extract_openai_metadata,
    extract_response_metadata,
)


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic LLMResponse creation."""
        response = LLMResponse(text="Hello", raw=None)
        assert response.text == "Hello"
        assert response.raw is None
        assert response.usage is None
        assert response.model is None

    def test_full_creation(self) -> None:
        """Test LLMResponse with all fields."""
        response = LLMResponse(
            text="Hello",
            raw={"key": "value"},
            usage={"total_tokens": 100},
            model="gpt-4",
            provider="openai",
            latency_ms=150.5,
            metadata={"finish_reason": "stop"},
        )
        assert response.text == "Hello"
        assert response.raw == {"key": "value"}
        assert response.usage == {"total_tokens": 100}
        assert response.model == "gpt-4"
        assert response.provider == "openai"
        assert response.latency_ms == 150.5
        assert response.metadata == {"finish_reason": "stop"}


class TestExtractResponseMetadata:
    """Tests for the main extract_response_metadata function."""

    def test_routes_to_openai(self) -> None:
        """Test OpenAI provider routes correctly."""
        mock_response = MagicMock()
        mock_response.choices = []
        result = extract_response_metadata(mock_response, "openai")
        assert result.provider == "openai"

    def test_routes_to_azure_openai(self) -> None:
        """Test Azure OpenAI uses OpenAI format."""
        mock_response = MagicMock()
        mock_response.choices = []
        result = extract_response_metadata(mock_response, "azure_openai")
        assert result.provider == "azure_openai"

    def test_routes_to_anthropic(self) -> None:
        """Test Anthropic provider routes correctly."""
        mock_response = MagicMock()
        mock_response.content = []
        result = extract_response_metadata(mock_response, "anthropic")
        assert result.provider == "anthropic"

    def test_routes_to_gemini(self) -> None:
        """Test Gemini provider routes correctly."""
        mock_response = MagicMock()
        mock_response.text = "Hello"
        result = extract_response_metadata(mock_response, "gemini")
        assert result.provider == "gemini"

    def test_routes_to_cohere(self) -> None:
        """Test Cohere provider routes correctly."""
        mock_response = MagicMock()
        mock_response.text = "Hello"
        result = extract_response_metadata(mock_response, "cohere")
        assert result.provider == "cohere"

    def test_unknown_provider_uses_generic(self) -> None:
        """Test unknown provider uses generic extractor."""
        mock_response = MagicMock()
        mock_response.text = "Hello"
        result = extract_response_metadata(mock_response, "unknown_provider")
        assert result.provider == "unknown_provider"


class TestExtractOpenAIMetadata:
    """Tests for OpenAI metadata extraction."""

    def test_extracts_text_from_choices(self) -> None:
        """Test text extraction from choices."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello world"
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model = "gpt-4"
        mock_response.id = "resp-123"

        result = _extract_openai_metadata(mock_response, "openai")
        assert result.text == "Hello world"
        assert result.model == "gpt-4"
        assert result.metadata["finish_reason"] == "stop"
        assert result.metadata["response_id"] == "resp-123"

    def test_extracts_usage(self) -> None:
        """Test usage extraction."""
        mock_response = MagicMock()
        mock_response.choices = []
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = None
        del mock_response.id

        result = _extract_openai_metadata(mock_response, "openai")
        assert result.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }

    def test_handles_text_attribute(self) -> None:
        """Test handling of text attribute instead of message."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        del mock_choice.message
        mock_choice.text = "Completion text"
        mock_choice.finish_reason = "length"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        del mock_response.model
        del mock_response.id

        result = _extract_openai_metadata(mock_response, "openai")
        assert result.text == "Completion text"

    def test_empty_choices(self) -> None:
        """Test handling of empty choices."""
        mock_response = MagicMock()
        mock_response.choices = []
        mock_response.usage = None
        del mock_response.model
        del mock_response.id

        result = _extract_openai_metadata(mock_response, "openai")
        assert result.text == ""


class TestExtractAnthropicMetadata:
    """Tests for Anthropic metadata extraction."""

    def test_extracts_text_from_content_blocks(self) -> None:
        """Test text extraction from content blocks."""
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "Hello from Claude"
        mock_response.content = [mock_block]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 25
        mock_response.model = "claude-3-opus"
        mock_response.stop_reason = "end_turn"
        mock_response.id = "msg-123"

        result = _extract_anthropic_metadata(mock_response, "anthropic")
        assert result.text == "Hello from Claude"
        assert result.model == "claude-3-opus"
        assert result.metadata["stop_reason"] == "end_turn"

    def test_extracts_usage(self) -> None:
        """Test usage extraction with calculated total."""
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 25
        del mock_response.model
        del mock_response.stop_reason
        del mock_response.id

        result = _extract_anthropic_metadata(mock_response, "anthropic")
        assert result.usage["input_tokens"] == 10
        assert result.usage["output_tokens"] == 25
        assert result.usage["total_tokens"] == 35

    def test_handles_dict_content_blocks(self) -> None:
        """Test handling of dict content blocks."""
        mock_response = MagicMock()
        mock_response.content = [{"text": "Dict text block"}]
        mock_response.usage = None
        del mock_response.model
        del mock_response.stop_reason
        del mock_response.id

        result = _extract_anthropic_metadata(mock_response, "anthropic")
        assert result.text == "Dict text block"


class TestExtractGeminiMetadata:
    """Tests for Gemini metadata extraction."""

    def test_extracts_text_property(self) -> None:
        """Test direct text property access."""
        mock_response = MagicMock()
        mock_response.text = "Gemini response"
        mock_response.usage_metadata = None

        result = _extract_gemini_metadata(mock_response, "gemini")
        assert result.text == "Gemini response"

    def test_extracts_from_candidates(self) -> None:
        """Test text extraction from candidates."""
        mock_response = MagicMock()
        del mock_response.text
        mock_part = MagicMock()
        mock_part.text = "From candidates"
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_candidate.finish_reason = "STOP"
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        result = _extract_gemini_metadata(mock_response, "gemini")
        assert result.text == "From candidates"
        assert result.metadata["finish_reason"] == "STOP"

    def test_extracts_usage_metadata(self) -> None:
        """Test usage metadata extraction."""
        mock_response = MagicMock()
        mock_response.text = "Hello"
        mock_response.usage_metadata.prompt_token_count = 5
        mock_response.usage_metadata.candidates_token_count = 10
        mock_response.usage_metadata.total_token_count = 15

        result = _extract_gemini_metadata(mock_response, "gemini")
        assert result.usage == {
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15,
        }


class TestExtractCohereMetadata:
    """Tests for Cohere metadata extraction."""

    def test_extracts_text(self) -> None:
        """Test text extraction."""
        mock_response = MagicMock()
        mock_response.text = "Cohere response"
        mock_response.meta = None
        mock_response.id = "resp-123"

        result = _extract_cohere_metadata(mock_response, "cohere")
        assert result.text == "Cohere response"
        assert result.metadata["response_id"] == "resp-123"

    def test_extracts_from_generations(self) -> None:
        """Test text extraction from generations."""
        mock_response = MagicMock()
        del mock_response.text
        mock_gen = MagicMock()
        mock_gen.text = "Generated text"
        mock_response.generations = [mock_gen]
        mock_response.meta = None
        del mock_response.id

        result = _extract_cohere_metadata(mock_response, "cohere")
        assert result.text == "Generated text"

    def test_extracts_billed_units(self) -> None:
        """Test billed units extraction."""
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.meta.billed_units.input_tokens = 10
        mock_response.meta.billed_units.output_tokens = 20
        del mock_response.id

        result = _extract_cohere_metadata(mock_response, "cohere")
        assert result.usage["input_tokens"] == 10
        assert result.usage["output_tokens"] == 20
        assert result.usage["total_tokens"] == 30


class TestExtractGenericMetadata:
    """Tests for generic metadata extraction."""

    def test_extracts_text_attribute(self) -> None:
        """Test text attribute extraction."""
        mock_response = MagicMock()
        mock_response.text = "Generic text"

        result = _extract_generic_metadata(mock_response, "unknown")
        assert result.text == "Generic text"

    def test_extracts_content_attribute(self) -> None:
        """Test content attribute extraction."""
        mock_response = MagicMock()
        del mock_response.text
        mock_response.content = "Content text"

        result = _extract_generic_metadata(mock_response, "unknown")
        assert result.text == "Content text"

    def test_extracts_output_attribute(self) -> None:
        """Test output attribute extraction."""
        mock_response = MagicMock()
        del mock_response.text
        del mock_response.content
        mock_response.output = "Output text"

        result = _extract_generic_metadata(mock_response, "unknown")
        assert result.text == "Output text"

    def test_string_response(self) -> None:
        """Test string response handled directly."""
        result = _extract_generic_metadata("String response", "unknown")
        assert result.text == "String response"

    def test_no_text_attributes(self) -> None:
        """Test handling when no text attributes found."""

        @dataclass
        class NoTextResponse:
            data: int = 123

        result = _extract_generic_metadata(NoTextResponse(), "unknown")
        assert result.text == ""
