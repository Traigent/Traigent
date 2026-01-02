"""Tests for message coercion utilities."""

from traigent.integrations.utils.message_coercion import (
    _extract_text_from_parts,
    _normalize_message_dict,
    coerce_messages,
    coerce_to_anthropic_format,
    coerce_to_gemini_format,
    coerce_to_openai_format,
)


class TestCoerceMessages:
    """Tests for the main coerce_messages function."""

    def test_none_returns_empty_list(self) -> None:
        """Test that None input returns empty list."""
        assert coerce_messages(None) == []

    def test_single_string(self) -> None:
        """Test single string input."""
        result = coerce_messages("Hello")
        assert result == [{"role": "user", "content": "Hello"}]

    def test_single_string_custom_role(self) -> None:
        """Test single string with custom default role."""
        result = coerce_messages("Hello", default_role="assistant")
        assert result == [{"role": "assistant", "content": "Hello"}]

    def test_list_of_strings(self) -> None:
        """Test list of string inputs."""
        result = coerce_messages(["Hello", "World"])
        assert result == [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ]

    def test_list_of_dicts_already_formatted(self) -> None:
        """Test list of properly formatted dicts."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = coerce_messages(messages)
        assert result == messages

    def test_single_dict(self) -> None:
        """Test single dict input."""
        result = coerce_messages({"role": "user", "content": "Hello"})
        assert result == [{"role": "user", "content": "Hello"}]

    def test_dict_without_role(self) -> None:
        """Test dict without role gets default role."""
        result = coerce_messages({"content": "Hello"})
        assert result == [{"role": "user", "content": "Hello"}]

    def test_mixed_list(self) -> None:
        """Test list with mixed string and dict elements."""
        result = coerce_messages(
            [
                "Plain text",
                {"role": "assistant", "content": "Response"},
            ]
        )
        assert result == [
            {"role": "user", "content": "Plain text"},
            {"role": "assistant", "content": "Response"},
        ]

    def test_non_string_converted_to_string(self) -> None:
        """Test non-string elements are converted."""
        result = coerce_messages([123, 456])
        assert result == [
            {"role": "user", "content": "123"},
            {"role": "user", "content": "456"},
        ]

    def test_fallback_converts_to_string(self) -> None:
        """Test fallback for unknown types."""
        result = coerce_messages(12345)
        assert result == [{"role": "user", "content": "12345"}]


class TestNormalizeMessageDict:
    """Tests for _normalize_message_dict function."""

    def test_role_preserved(self) -> None:
        """Test that existing role is preserved."""
        result = _normalize_message_dict({"role": "assistant", "content": "Hi"})
        assert result["role"] == "assistant"

    def test_default_role_added(self) -> None:
        """Test that default role is added when missing."""
        result = _normalize_message_dict({"content": "Hi"})
        assert result["role"] == "user"

    def test_custom_default_role(self) -> None:
        """Test custom default role."""
        result = _normalize_message_dict({"content": "Hi"}, default_role="system")
        assert result["role"] == "system"

    def test_text_alias(self) -> None:
        """Test 'text' is aliased to 'content'."""
        result = _normalize_message_dict({"text": "Hello"})
        assert result["content"] == "Hello"
        assert "text" not in result

    def test_parts_alias(self) -> None:
        """Test Gemini 'parts' format is handled."""
        result = _normalize_message_dict({"parts": [{"text": "Hello"}]})
        assert result["content"] == "Hello"

    def test_empty_content_when_missing(self) -> None:
        """Test empty content when no content found."""
        result = _normalize_message_dict({"role": "user"})
        assert result["content"] == ""


class TestExtractTextFromParts:
    """Tests for _extract_text_from_parts function."""

    def test_string_input(self) -> None:
        """Test string input passed through."""
        assert _extract_text_from_parts("Hello") == "Hello"

    def test_list_of_strings(self) -> None:
        """Test list of strings joined."""
        result = _extract_text_from_parts(["Hello", "World"])
        assert result == "Hello World"

    def test_list_of_text_dicts(self) -> None:
        """Test list of text dicts."""
        result = _extract_text_from_parts([{"text": "Hello"}, {"text": "World"}])
        assert result == "Hello World"

    def test_list_of_content_dicts(self) -> None:
        """Test list of content dicts."""
        result = _extract_text_from_parts([{"content": "Hello"}])
        assert result == "Hello"

    def test_non_sequence_converted(self) -> None:
        """Test non-sequence is converted to string."""
        result = _extract_text_from_parts(123)
        assert result == "123"

    def test_none_returns_empty(self) -> None:
        """Test None returns empty string."""
        result = _extract_text_from_parts(None)
        assert result == ""


class TestCoerceToOpenAIFormat:
    """Tests for coerce_to_openai_format function."""

    def test_string_input(self) -> None:
        """Test string input."""
        result = coerce_to_openai_format("Hello")
        assert result == [{"role": "user", "content": "Hello"}]

    def test_preserves_user_role(self) -> None:
        """Test that default role is always user for OpenAI."""
        result = coerce_to_openai_format(["Message 1", "Message 2"])
        assert all(msg["role"] == "user" for msg in result)


class TestCoerceToAnthropicFormat:
    """Tests for coerce_to_anthropic_format function."""

    def test_extracts_system_message(self) -> None:
        """Test system message is extracted separately."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        system, conversation = coerce_to_anthropic_format(messages)
        assert system == "You are helpful"
        assert conversation == [{"role": "user", "content": "Hello"}]

    def test_no_system_message(self) -> None:
        """Test when no system message present."""
        messages = [{"role": "user", "content": "Hello"}]
        system, conversation = coerce_to_anthropic_format(messages)
        assert system is None
        assert conversation == messages

    def test_multiple_system_messages_concatenated(self) -> None:
        """Test multiple system messages are concatenated."""
        messages = [
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
            {"role": "user", "content": "Hello"},
        ]
        system, conversation = coerce_to_anthropic_format(messages)
        assert system == "Rule 1\nRule 2"

    def test_string_input(self) -> None:
        """Test string input returns no system message."""
        system, conversation = coerce_to_anthropic_format("Hello")
        assert system is None
        assert conversation == [{"role": "user", "content": "Hello"}]


class TestCoerceToGeminiFormat:
    """Tests for coerce_to_gemini_format function."""

    def test_single_user_message_returns_string(self) -> None:
        """Test single user message returns just the string."""
        result = coerce_to_gemini_format("Hello")
        assert result == "Hello"

    def test_multi_turn_returns_content_list(self) -> None:
        """Test multi-turn conversation returns content list."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = coerce_to_gemini_format(messages)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_assistant_role_converted_to_model(self) -> None:
        """Test 'assistant' role is converted to 'model'."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = coerce_to_gemini_format(messages)
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "model"

    def test_content_converted_to_parts(self) -> None:
        """Test content is wrapped in parts format."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = coerce_to_gemini_format(messages)
        assert result[0]["parts"] == [{"text": "Hello"}]
        assert result[1]["parts"] == [{"text": "Hi"}]

    def test_non_user_single_message_returns_list(self) -> None:
        """Test single assistant message returns list not string."""
        messages = [{"role": "assistant", "content": "Hello"}]
        result = coerce_to_gemini_format(messages)
        assert isinstance(result, list)
