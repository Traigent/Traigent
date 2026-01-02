"""Tests for LlamaIndex plugin integration.

This module tests the LlamaIndex plugin functionality including:
- Plugin initialization and metadata
- Parameter mappings
- Validation rules
- Override application
- Target class and method identification
"""

import pytest

from traigent.config.types import TraigentConfig
from traigent.integrations.llms.llamaindex_plugin import LlamaIndexPlugin
from traigent.utils.exceptions import TraigentError


class TestLlamaIndexPlugin:
    """Test suite for LlamaIndex plugin."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = LlamaIndexPlugin()

    def test_plugin_metadata(self):
        """Test plugin metadata is correctly set."""
        assert self.plugin.metadata.name == "llamaindex"
        assert self.plugin.metadata.version == "1.0.0"
        assert "llama_index" in self.plugin.metadata.supported_packages
        assert "llama-index" in self.plugin.metadata.supported_packages

    def test_target_classes(self):
        """Test that target classes are properly defined."""
        target_classes = self.plugin.get_target_classes()

        # Check core LLM classes
        assert "llama_index.llms.openai.OpenAI" in target_classes
        assert "llama_index.llms.anthropic.Anthropic" in target_classes
        assert "llama_index.llms.cohere.Cohere" in target_classes
        assert "llama_index.llms.huggingface.HuggingFaceLLM" in target_classes

        # Check legacy imports
        assert "llama_index.llms.OpenAI" in target_classes
        assert "llama_index.llms.Anthropic" in target_classes

        # Check embedding models
        assert "llama_index.embeddings.openai.OpenAIEmbedding" in target_classes
        assert (
            "llama_index.embeddings.huggingface.HuggingFaceEmbedding" in target_classes
        )

        # Check service context
        assert "llama_index.core.ServiceContext" in target_classes
        assert "llama_index.ServiceContext" in target_classes

        # Check chat engines
        assert "llama_index.chat_engine.SimpleChatEngine" in target_classes
        # Check retrievers
        assert "llama_index.core.retrievers.BaseRetriever" in target_classes
        assert (
            "llama_index.indices.vector_store.retrievers.VectorIndexRetriever"
            in target_classes
        )

    def test_target_methods(self):
        """Test that target methods are properly mapped."""
        target_methods = self.plugin.get_target_methods()

        # Check OpenAI LLM methods
        openai_methods = target_methods.get("llama_index.llms.openai.OpenAI", [])
        assert "complete" in openai_methods
        assert "acomplete" in openai_methods
        assert "chat" in openai_methods
        assert "achat" in openai_methods
        assert "stream_complete" in openai_methods
        assert "stream_chat" in openai_methods

        # Check embedding methods
        embedding_methods = target_methods.get(
            "llama_index.embeddings.openai.OpenAIEmbedding", []
        )
        assert "get_embedding" in embedding_methods
        assert "get_query_embedding" in embedding_methods
        assert "get_text_embedding" in embedding_methods

        # Check service context methods
        service_methods = target_methods.get("llama_index.core.ServiceContext", [])
        assert "from_defaults" in service_methods
        # Check retriever methods
        retriever_methods = target_methods.get(
            "llama_index.core.retrievers.BaseRetriever", []
        )
        assert "retrieve" in retriever_methods
        assert "aretrieve" in retriever_methods

    def test_parameter_mappings(self):
        """Test parameter mappings from Traigent to LlamaIndex."""
        mappings = self.plugin.get_parameter_mappings()

        # Core LLM parameters
        assert mappings["model"] == "model"
        assert mappings["temperature"] == "temperature"
        assert mappings["max_tokens"] == "max_tokens"
        assert mappings["top_p"] == "top_p"
        assert mappings["frequency_penalty"] == "frequency_penalty"
        assert mappings["presence_penalty"] == "presence_penalty"

        # Additional parameters
        assert mappings["top_k"] == "top_k"
        assert mappings["stream"] == "stream"
        assert mappings["system_prompt"] == "system_prompt"

        # Embedding parameters
        assert mappings["embedding_model"] == "model_name"
        assert mappings["embed_batch_size"] == "embed_batch_size"
        assert mappings["chunk_size"] == "chunk_size"

        # Retrieval parameters
        assert mappings["similarity_top_k"] == "similarity_top_k"
        assert mappings["similarity_cutoff"] == "similarity_cutoff"

        # Response synthesis parameters
        assert mappings["response_mode"] == "response_mode"
        assert mappings["streaming"] == "streaming"
        # Response formatting
        assert mappings["response_format"] == "response_format"

    def test_validation_rules(self):
        """Test validation rules for parameters."""
        # Test valid configurations
        valid_config = TraigentConfig(
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            custom_params={
                "top_k": 50,
                "similarity_top_k": 10,
                "chunk_size": 512,
                "response_mode": "refine",
            },
        )

        # Should not raise any exceptions
        assert self.plugin.validate_config(valid_config) is True

        # Test invalid temperature (using custom_params to bypass TraigentConfig validation)
        invalid_temp_config = TraigentConfig(custom_params={"temperature": 3.0})
        with pytest.raises(TraigentError) as exc_info:
            self.plugin.validate_config(invalid_temp_config)
        assert "temperature" in str(exc_info.value)
        assert "above maximum" in str(exc_info.value)

        # Test invalid max_tokens
        invalid_tokens_config = TraigentConfig(custom_params={"max_tokens": 0})
        with pytest.raises(TraigentError) as exc_info:
            self.plugin.validate_config(invalid_tokens_config)
        assert "max_tokens" in str(exc_info.value)
        assert "below minimum" in str(exc_info.value)

        # Test invalid response_mode
        invalid_mode_config = TraigentConfig(
            custom_params={"response_mode": "invalid_mode"}
        )
        with pytest.raises(TraigentError) as exc_info:
            self.plugin.validate_config(invalid_mode_config)
        assert "response_mode" in str(exc_info.value)
        assert "not in allowed values" in str(exc_info.value)

    def test_apply_overrides_basic(self):
        """Test basic parameter override application."""
        original_kwargs = {"some_other_param": "value"}

        config = TraigentConfig(
            model="gpt-4", temperature=0.5, max_tokens=2000, top_p=0.8
        )

        overridden = self.plugin.apply_overrides(original_kwargs, config)

        # Check overrides were applied
        assert overridden["model"] == "gpt-4"
        assert overridden["model_name"] == "gpt-4"  # Should also set model_name
        assert overridden["temperature"] == 0.5
        assert overridden["max_tokens"] == 2000
        assert (
            overridden["max_tokens_to_sample"] == 2000
        )  # Should also set Anthropic variation
        assert overridden["top_p"] == 0.8

        # Check original params preserved
        assert overridden["some_other_param"] == "value"

    def test_apply_overrides_with_transformations(self):
        """Test parameter transformations during override."""
        original_kwargs = {}

        config = TraigentConfig(
            model="claude-3",
            max_tokens=1000,
            custom_params={"stream": True, "top_k": 40},
        )

        overridden = self.plugin.apply_overrides(original_kwargs, config)

        # Check transformations
        assert overridden["model"] == "claude-3"
        assert overridden["model_name"] == "claude-3"  # Auto-set model_name
        assert overridden["max_tokens"] == 1000
        assert overridden["max_tokens_to_sample"] == 1000  # Anthropic variation
        assert overridden["stream"] is True
        assert overridden["streaming"] is True  # Stream variation
        assert overridden["top_k"] == 40

    def test_apply_overrides_with_custom_params(self):
        """Test custom parameter handling."""
        original_kwargs = {"existing": "param"}

        config = TraigentConfig(
            model="gpt-4",
            custom_params={
                "system_prompt": "You are a helpful assistant",
                "similarity_top_k": 5,
                "chunk_size": 1024,
                "response_mode": "tree_summarize",
            },
        )

        overridden = self.plugin.apply_overrides(original_kwargs, config)

        assert overridden["model"] == "gpt-4"
        assert overridden["system_prompt"] == "You are a helpful assistant"
        assert overridden["similarity_top_k"] == 5
        assert overridden["chunk_size"] == 1024
        assert overridden["response_mode"] == "tree_summarize"
        assert overridden["existing"] == "param"

    def test_apply_overrides_response_format(self):
        """Response formatting should pass through."""
        config = TraigentConfig(
            model="gpt-4",
            custom_params={"response_format": {"type": "json_object"}},
        )

        overridden = self.plugin.apply_overrides({}, config)

        assert overridden["response_format"] == {"type": "json_object"}

    def test_apply_overrides_with_dict_config(self):
        """Plugins should accept raw dict configuration payloads."""
        original_kwargs = {}
        config_payload = {
            "model": "gpt-4",
            "stream": True,
            "max_tokens": 512,
        }

        overridden = self.plugin.apply_overrides(original_kwargs, config_payload)

        assert overridden["model"] == "gpt-4"
        assert overridden["max_tokens"] == 512
        assert overridden["stream"] is True
        assert overridden["streaming"] is True

    def test_validate_model_name(self):
        """Test custom model name validation."""
        # Valid model names
        valid_names = [
            "gpt-4",
            "claude-3-opus-20240229",
            "meta-llama/Llama-2-7b-hf",
            "text-davinci-003",
            "gpt-4-turbo-preview",
        ]

        for name in valid_names:
            errors = self.plugin._validate_model_name("model", name)
            assert errors == [], f"Valid model name '{name}' was rejected"

        # Invalid model names
        invalid_configs = [
            ("", "cannot be empty"),
            (123, "must be a string"),
            ("model!@#$", "invalid characters"),
            (["model"], "must be a string"),
        ]

        for value, expected_error in invalid_configs:
            errors = self.plugin._validate_model_name("model", value)
            assert len(errors) > 0, f"Invalid value '{value}' was not rejected"
            assert any(expected_error in error for error in errors)

    def test_plugin_enable_disable(self):
        """Test plugin enable/disable functionality."""
        assert self.plugin.enabled is True

        self.plugin.disable()
        assert self.plugin.enabled is False

        # Overrides should not be applied when disabled
        original_kwargs = {}
        config = TraigentConfig(model="gpt-4")
        overridden = self.plugin.apply_overrides(original_kwargs, config)
        assert overridden == original_kwargs  # No changes when disabled

        self.plugin.enable()
        assert self.plugin.enabled is True

        # Overrides should work again
        overridden = self.plugin.apply_overrides(original_kwargs, config)
        assert overridden["model"] == "gpt-4"

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with None values
        config = TraigentConfig(
            model=None,
            temperature=None,
            custom_params={"top_k": None, "response_mode": None},
        )

        # Should handle None values gracefully
        original = {"existing": "value"}
        overridden = self.plugin.apply_overrides(original, config)
        assert "model" not in overridden  # None values are filtered out
        assert "temperature" not in overridden
        assert "top_k" not in overridden
        assert "response_mode" not in overridden
        assert overridden["existing"] == "value"

        # Test with empty config
        empty_config = TraigentConfig()
        overridden = self.plugin.apply_overrides(original, empty_config)
        assert overridden == original  # No changes with empty config

        # Test with boundary values
        boundary_config = TraigentConfig(
            temperature=0.0,  # Min value
            max_tokens=1,  # Min value
            top_p=1.0,  # Max value
            custom_params={
                "similarity_top_k": 100,  # Max value
                "chunk_size": 8192,  # Max value
                "alpha": 0.0,  # Min value
            },
        )

        # Should accept boundary values
        assert self.plugin.validate_config(boundary_config) is True
