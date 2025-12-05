"""Unit tests for the ParameterNormalizer class.

These tests validate cross-framework parameter normalization functionality.
"""

from __future__ import annotations

import pytest

from traigent.integrations.utils.parameter_normalizer import (
    Framework,
    ParameterAlias,
    ParameterNormalizer,
    get_normalizer,
    normalize_params,
)


# =============================================================================
# Framework Enum Tests
# =============================================================================


@pytest.mark.unit
class TestFrameworkEnum:
    """Test Framework enum functionality."""

    def test_framework_values(self) -> None:
        """Test all framework values are defined."""
        assert Framework.TRAIGENT.value == "traigent"
        assert Framework.OPENAI.value == "openai"
        assert Framework.ANTHROPIC.value == "anthropic"
        assert Framework.LANGCHAIN.value == "langchain"
        assert Framework.GEMINI.value == "gemini"
        assert Framework.BEDROCK.value == "bedrock"

    def test_framework_iteration(self) -> None:
        """Test iterating over all frameworks."""
        frameworks = list(Framework)
        assert len(frameworks) >= 10
        assert Framework.OPENAI in frameworks
        assert Framework.ANTHROPIC in frameworks


# =============================================================================
# ParameterAlias Tests
# =============================================================================


@pytest.mark.unit
class TestParameterAlias:
    """Test ParameterAlias dataclass functionality."""

    def test_alias_creation(self) -> None:
        """Test creating a parameter alias."""
        alias = ParameterAlias(
            canonical="model",
            aliases={
                Framework.OPENAI: "model",
                Framework.LANGCHAIN: "model_name",
            },
            description="Model identifier",
        )
        assert alias.canonical == "model"
        assert alias.get_for_framework(Framework.OPENAI) == "model"
        assert alias.get_for_framework(Framework.LANGCHAIN) == "model_name"

    def test_alias_fallback_to_canonical(self) -> None:
        """Test that missing framework falls back to canonical name."""
        alias = ParameterAlias(
            canonical="temperature",
            aliases={Framework.OPENAI: "temperature"},
        )
        # ANTHROPIC not in aliases, should return canonical
        assert alias.get_for_framework(Framework.ANTHROPIC) == "temperature"


# =============================================================================
# ParameterNormalizer Core Tests
# =============================================================================


@pytest.mark.unit
class TestParameterNormalizerCore:
    """Test ParameterNormalizer core functionality."""

    def test_normalizer_initialization(self) -> None:
        """Test normalizer initializes with registry."""
        normalizer = ParameterNormalizer()
        canonical_names = normalizer.get_canonical_names()
        assert "model" in canonical_names
        assert "max_tokens" in canonical_names
        assert "temperature" in canonical_names
        assert "stop" in canonical_names
        assert "stream" in canonical_names

    def test_get_all_aliases(self) -> None:
        """Test getting all aliases for a canonical parameter."""
        normalizer = ParameterNormalizer()
        aliases = normalizer.get_all_aliases("model")

        assert Framework.OPENAI in aliases
        assert Framework.LANGCHAIN in aliases
        assert aliases[Framework.OPENAI] == "model"
        assert aliases[Framework.LANGCHAIN] == "model_name"
        assert aliases[Framework.BEDROCK] == "model_id"

    def test_is_known_parameter(self) -> None:
        """Test parameter recognition."""
        normalizer = ParameterNormalizer()

        # Canonical names should be known
        assert normalizer.is_known_parameter("model") is True
        assert normalizer.is_known_parameter("max_tokens") is True

        # Framework-specific names should be known
        assert normalizer.is_known_parameter("model_name") is True
        assert normalizer.is_known_parameter("model_id") is True
        assert normalizer.is_known_parameter("max_output_tokens") is True
        assert normalizer.is_known_parameter("streaming") is True

        # Unknown parameters
        assert normalizer.is_known_parameter("unknown_param") is False
        assert normalizer.is_known_parameter("foobar") is False


# =============================================================================
# Conversion Tests: to_canonical
# =============================================================================


@pytest.mark.unit
class TestToCanonical:
    """Test conversion to canonical TraiGent format."""

    def test_openai_to_canonical(self) -> None:
        """Test converting OpenAI parameters to canonical."""
        normalizer = ParameterNormalizer()
        params = {
            "model": "gpt-4",
            "max_tokens": 100,
            "temperature": 0.7,
            "stop": ["\n"],
            "stream": True,
        }
        result = normalizer.to_canonical(params, Framework.OPENAI)

        assert result["model"] == "gpt-4"
        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.7
        assert result["stop"] == ["\n"]
        assert result["stream"] is True

    def test_langchain_to_canonical(self) -> None:
        """Test converting LangChain parameters to canonical."""
        normalizer = ParameterNormalizer()
        params = {
            "model_name": "gpt-4",
            "max_tokens": 100,
            "streaming": True,
        }
        result = normalizer.to_canonical(params, Framework.LANGCHAIN)

        assert result["model"] == "gpt-4"
        assert result["max_tokens"] == 100
        assert result["stream"] is True

    def test_gemini_to_canonical(self) -> None:
        """Test converting Gemini parameters to canonical."""
        normalizer = ParameterNormalizer()
        params = {
            "model_name": "gemini-pro",
            "max_output_tokens": 200,
            "temperature": 0.5,
            "stop_sequences": ["END"],
        }
        result = normalizer.to_canonical(params, Framework.GEMINI)

        assert result["model"] == "gemini-pro"
        assert result["max_tokens"] == 200
        assert result["temperature"] == 0.5
        assert result["stop"] == ["END"]

    def test_bedrock_to_canonical(self) -> None:
        """Test converting Bedrock parameters to canonical."""
        normalizer = ParameterNormalizer()
        params = {
            "model_id": "anthropic.claude-3-sonnet",
            "max_tokens": 150,
            "stop_sequences": ["Human:"],
        }
        result = normalizer.to_canonical(params, Framework.BEDROCK)

        assert result["model"] == "anthropic.claude-3-sonnet"
        assert result["max_tokens"] == 150
        assert result["stop"] == ["Human:"]

    def test_anthropic_to_canonical(self) -> None:
        """Test converting Anthropic parameters to canonical."""
        normalizer = ParameterNormalizer()
        params = {
            "model": "claude-3-opus",
            "max_tokens": 1000,
            "stop_sequences": ["\n\nHuman:"],
            "system": "You are a helpful assistant.",
        }
        result = normalizer.to_canonical(params, Framework.ANTHROPIC)

        assert result["model"] == "claude-3-opus"
        assert result["max_tokens"] == 1000
        assert result["stop"] == ["\n\nHuman:"]
        assert result["system"] == "You are a helpful assistant."

    def test_unknown_params_passthrough(self) -> None:
        """Test unknown parameters pass through in non-strict mode."""
        normalizer = ParameterNormalizer()
        params = {
            "model": "gpt-4",
            "custom_param": "value",
            "another_custom": 123,
        }
        result = normalizer.to_canonical(params, Framework.OPENAI)

        assert result["model"] == "gpt-4"
        assert result["custom_param"] == "value"
        assert result["another_custom"] == 123

    def test_unknown_params_strict_mode(self) -> None:
        """Test strict mode raises error for unknown parameters."""
        normalizer = ParameterNormalizer()
        params = {
            "model": "gpt-4",
            "unknown_param": "value",
        }
        with pytest.raises(ValueError, match="Unknown parameter 'unknown_param'"):
            normalizer.to_canonical(params, Framework.OPENAI, strict=True)


# =============================================================================
# Conversion Tests: from_canonical
# =============================================================================


@pytest.mark.unit
class TestFromCanonical:
    """Test conversion from canonical to framework-specific format."""

    def test_canonical_to_openai(self) -> None:
        """Test converting canonical to OpenAI format."""
        normalizer = ParameterNormalizer()
        params = {
            "model": "gpt-4",
            "max_tokens": 100,
            "stream": True,
        }
        result = normalizer.from_canonical(params, Framework.OPENAI)

        assert result["model"] == "gpt-4"
        assert result["max_tokens"] == 100
        assert result["stream"] is True

    def test_canonical_to_langchain(self) -> None:
        """Test converting canonical to LangChain format."""
        normalizer = ParameterNormalizer()
        params = {
            "model": "gpt-4",
            "max_tokens": 100,
            "stream": True,
        }
        result = normalizer.from_canonical(params, Framework.LANGCHAIN)

        assert result["model_name"] == "gpt-4"
        assert result["max_tokens"] == 100
        assert result["streaming"] is True

    def test_canonical_to_gemini(self) -> None:
        """Test converting canonical to Gemini format."""
        normalizer = ParameterNormalizer()
        params = {
            "model": "gemini-pro",
            "max_tokens": 200,
            "stop": ["END"],
            "system": "Be helpful",
        }
        result = normalizer.from_canonical(params, Framework.GEMINI)

        assert result["model_name"] == "gemini-pro"
        assert result["max_output_tokens"] == 200
        assert result["stop_sequences"] == ["END"]
        assert result["system_instruction"] == "Be helpful"

    def test_canonical_to_bedrock(self) -> None:
        """Test converting canonical to Bedrock format."""
        normalizer = ParameterNormalizer()
        params = {
            "model": "anthropic.claude-3-sonnet",
            "max_tokens": 150,
            "stop": ["Human:"],
        }
        result = normalizer.from_canonical(params, Framework.BEDROCK)

        assert result["model_id"] == "anthropic.claude-3-sonnet"
        assert result["max_tokens"] == 150
        assert result["stop_sequences"] == ["Human:"]

    def test_canonical_to_huggingface(self) -> None:
        """Test converting canonical to HuggingFace format."""
        normalizer = ParameterNormalizer()
        params = {
            "model": "meta-llama/Llama-2-7b",
            "max_tokens": 256,
            "frequency_penalty": 0.5,
        }
        result = normalizer.from_canonical(params, Framework.HUGGINGFACE)

        assert result["model_id"] == "meta-llama/Llama-2-7b"
        assert result["max_new_tokens"] == 256
        assert result["repetition_penalty"] == 0.5


# =============================================================================
# Direct Conversion Tests
# =============================================================================


@pytest.mark.unit
class TestDirectConversion:
    """Test direct framework-to-framework conversion."""

    def test_langchain_to_openai(self) -> None:
        """Test converting LangChain to OpenAI format."""
        normalizer = ParameterNormalizer()
        params = {
            "model_name": "gpt-4",
            "streaming": True,
            "max_tokens": 100,
        }
        result = normalizer.convert(params, Framework.LANGCHAIN, Framework.OPENAI)

        assert result["model"] == "gpt-4"
        assert result["stream"] is True
        assert result["max_tokens"] == 100

    def test_openai_to_gemini(self) -> None:
        """Test converting OpenAI to Gemini format."""
        normalizer = ParameterNormalizer()
        params = {
            "model": "gpt-4",
            "max_tokens": 100,
            "stop": ["\n"],
        }
        result = normalizer.convert(params, Framework.OPENAI, Framework.GEMINI)

        assert result["model_name"] == "gpt-4"
        assert result["max_output_tokens"] == 100
        assert result["stop_sequences"] == ["\n"]

    def test_anthropic_to_bedrock(self) -> None:
        """Test converting Anthropic to Bedrock format."""
        normalizer = ParameterNormalizer()
        params = {
            "model": "claude-3-opus",
            "max_tokens": 500,
            "stop_sequences": ["Human:"],
        }
        result = normalizer.convert(params, Framework.ANTHROPIC, Framework.BEDROCK)

        assert result["model_id"] == "claude-3-opus"
        assert result["max_tokens"] == 500
        assert result["stop_sequences"] == ["Human:"]

    def test_gemini_to_langchain(self) -> None:
        """Test converting Gemini to LangChain format."""
        normalizer = ParameterNormalizer()
        params = {
            "model_name": "gemini-pro",
            "max_output_tokens": 200,
            "stop_sequences": ["END"],
        }
        result = normalizer.convert(params, Framework.GEMINI, Framework.LANGCHAIN)

        assert result["model_name"] == "gemini-pro"
        assert result["max_tokens"] == 200
        assert result["stop"] == ["END"]


# =============================================================================
# normalize_kwargs Tests
# =============================================================================


@pytest.mark.unit
class TestNormalizeKwargs:
    """Test normalize_kwargs method for auto-detection."""

    def test_normalize_mixed_params_to_openai(self) -> None:
        """Test normalizing mixed parameter names to OpenAI format."""
        normalizer = ParameterNormalizer()
        kwargs = {
            "model_name": "gpt-4",  # LangChain style
            "max_output_tokens": 100,  # Gemini style
            "streaming": True,  # LangChain style
            "custom_param": "value",  # Unknown
        }
        result = normalizer.normalize_kwargs(kwargs, Framework.OPENAI)

        assert result["model"] == "gpt-4"
        assert result["max_tokens"] == 100
        assert result["stream"] is True
        assert result["custom_param"] == "value"

    def test_normalize_mixed_params_to_bedrock(self) -> None:
        """Test normalizing mixed parameter names to Bedrock format."""
        normalizer = ParameterNormalizer()
        kwargs = {
            "model": "claude-3",  # OpenAI style
            "max_tokens": 100,
            "stop": ["\n"],  # OpenAI style
        }
        result = normalizer.normalize_kwargs(kwargs, Framework.BEDROCK)

        assert result["model_id"] == "claude-3"
        assert result["max_tokens"] == 100
        assert result["stop_sequences"] == ["\n"]


# =============================================================================
# Convenience Function Tests
# =============================================================================


@pytest.mark.unit
class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_get_normalizer_singleton(self) -> None:
        """Test get_normalizer returns singleton instance."""
        normalizer1 = get_normalizer()
        normalizer2 = get_normalizer()
        assert normalizer1 is normalizer2

    def test_normalize_params_function(self) -> None:
        """Test normalize_params convenience function."""
        params = {
            "model_name": "gpt-4",
            "streaming": True,
        }
        result = normalize_params(params, Framework.LANGCHAIN, Framework.OPENAI)

        assert result["model"] == "gpt-4"
        assert result["stream"] is True

    def test_normalize_params_with_strings(self) -> None:
        """Test normalize_params with string framework names."""
        params = {
            "model_name": "gpt-4",
            "streaming": True,
        }
        result = normalize_params(params, "langchain", "openai")

        assert result["model"] == "gpt-4"
        assert result["stream"] is True

    def test_normalize_params_invalid_framework(self) -> None:
        """Test normalize_params raises error for invalid framework."""
        params = {"model": "gpt-4"}
        with pytest.raises(ValueError, match="Unknown framework"):
            normalize_params(params, "invalid_framework", "openai")


# =============================================================================
# Framework String Conversion Tests
# =============================================================================


@pytest.mark.unit
class TestFrameworkStringConversion:
    """Test framework string to enum conversion."""

    def test_get_framework_from_string_valid(self) -> None:
        """Test converting valid framework strings."""
        assert (
            ParameterNormalizer.get_framework_from_string("openai") == Framework.OPENAI
        )
        assert (
            ParameterNormalizer.get_framework_from_string("OPENAI") == Framework.OPENAI
        )
        assert (
            ParameterNormalizer.get_framework_from_string("langchain")
            == Framework.LANGCHAIN
        )
        assert (
            ParameterNormalizer.get_framework_from_string("azure_openai")
            == Framework.AZURE_OPENAI
        )
        assert (
            ParameterNormalizer.get_framework_from_string("azure-openai")
            == Framework.AZURE_OPENAI
        )

    def test_get_framework_from_string_invalid(self) -> None:
        """Test invalid framework strings return None."""
        assert ParameterNormalizer.get_framework_from_string("invalid") is None
        assert ParameterNormalizer.get_framework_from_string("") is None


# =============================================================================
# Edge Cases and Special Scenarios
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_params(self) -> None:
        """Test handling empty parameter dicts."""
        normalizer = ParameterNormalizer()
        result = normalizer.to_canonical({}, Framework.OPENAI)
        assert result == {}

        result = normalizer.from_canonical({}, Framework.OPENAI)
        assert result == {}

    def test_none_values_preserved(self) -> None:
        """Test None values are preserved during conversion."""
        normalizer = ParameterNormalizer()
        params = {
            "model": "gpt-4",
            "temperature": None,
            "max_tokens": None,
        }
        result = normalizer.to_canonical(params, Framework.OPENAI)

        assert result["model"] == "gpt-4"
        assert result["temperature"] is None
        assert result["max_tokens"] is None

    def test_complex_values_preserved(self) -> None:
        """Test complex values (lists, dicts) are preserved."""
        normalizer = ParameterNormalizer()
        params = {
            "model": "gpt-4",
            "stop": ["\n", "END", "STOP"],
            "tools": [{"name": "search", "description": "Search the web"}],
        }
        result = normalizer.to_canonical(params, Framework.OPENAI)

        assert result["stop"] == ["\n", "END", "STOP"]
        assert result["tools"] == [{"name": "search", "description": "Search the web"}]

    def test_cohere_special_params(self) -> None:
        """Test Cohere special parameter names (p, k)."""
        normalizer = ParameterNormalizer()

        # From Cohere to canonical
        params = {"model": "command", "p": 0.9, "k": 50}
        result = normalizer.to_canonical(params, Framework.COHERE)
        assert result["top_p"] == 0.9
        assert result["top_k"] == 50

        # From canonical to Cohere
        canonical = {"model": "command", "top_p": 0.9, "top_k": 50}
        result = normalizer.from_canonical(canonical, Framework.COHERE)
        assert result["model"] == "command"
        assert result["p"] == 0.9
        assert result["k"] == 50

    def test_huggingface_special_params(self) -> None:
        """Test HuggingFace special parameter names."""
        normalizer = ParameterNormalizer()

        # max_tokens -> max_new_tokens, frequency_penalty -> repetition_penalty
        canonical = {
            "model": "llama-7b",
            "max_tokens": 256,
            "frequency_penalty": 1.2,
        }
        result = normalizer.from_canonical(canonical, Framework.HUGGINGFACE)

        assert result["model_id"] == "llama-7b"
        assert result["max_new_tokens"] == 256
        assert result["repetition_penalty"] == 1.2
