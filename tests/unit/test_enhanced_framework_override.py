#!/usr/bin/env python3
"""Tests for enhanced framework override system with streaming and tool support."""

import pytest

from traigent.config.context import set_config
from traigent.config.types import TraigentConfig
from traigent.integrations.framework_override import (
    FrameworkOverrideManager,
    disable_framework_overrides,
    enable_framework_overrides,
    override_all_platforms,
    override_context,
    register_framework_mapping,
)


class MockOpenAIClient:
    """Mock OpenAI client for testing."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.chat = self.ChatCompletions(self)

    class ChatCompletions:
        def __init__(self, client):
            self.client = client
            self.completions = self.Completions(client)

        class Completions:
            def __init__(self, client):
                self.client = client

            def create(self, **kwargs):
                return {
                    "client_init": self.client.init_kwargs,
                    "call_kwargs": kwargs,
                    "response": "mock response",
                }


class MockAnthropicClient:
    """Mock Anthropic client for testing."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.messages = self.Messages(self)

    class Messages:
        def __init__(self, client):
            self.client = client

        def create(self, **kwargs):
            return {
                "client_init": self.client.init_kwargs,
                "call_kwargs": kwargs,
                "response": "mock anthropic response",
            }

        async def stream(self, **kwargs):
            """Mock async streaming method."""
            for chunk in ["chunk1", "chunk2", "chunk3"]:
                yield {
                    "client_init": self.client.init_kwargs,
                    "call_kwargs": kwargs,
                    "chunk": chunk,
                }


class MockLangChainLLM:
    """Mock LangChain LLM for testing."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs

    def invoke(self, prompt, **kwargs):
        return {
            "init_kwargs": self.init_kwargs,
            "invoke_kwargs": kwargs,
            "prompt": prompt,
            "response": "mock langchain response",
        }

    def stream(self, prompt, **kwargs):
        for chunk in ["stream1", "stream2"]:
            yield {
                "init_kwargs": self.init_kwargs,
                "stream_kwargs": kwargs,
                "prompt": prompt,
                "chunk": chunk,
            }


class TestFrameworkOverrideManager:
    """Test the enhanced FrameworkOverrideManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = FrameworkOverrideManager()

    def test_init(self):
        """Test FrameworkOverrideManager initialization."""
        assert self.manager._parameter_mappings is not None
        assert self.manager._method_mappings is not None
        assert len(self.manager._parameter_mappings) > 0
        assert len(self.manager._method_mappings) > 0

    def test_parameter_mappings_include_new_platforms(self):
        """Test that parameter mappings include new platforms."""
        mappings = self.manager._parameter_mappings

        # Check Anthropic mappings
        assert "anthropic.Anthropic" in mappings
        assert "anthropic.AsyncAnthropic" in mappings
        assert "langchain_anthropic.ChatAnthropic" in mappings

        # Check Cohere mappings
        assert "cohere.Client" in mappings
        assert "cohere.AsyncClient" in mappings

        # Check HuggingFace mappings
        assert "transformers.pipeline" in mappings
        assert "transformers.AutoModelForCausalLM" in mappings

    def test_method_mappings_include_streaming_and_tools(self):
        """Test that method mappings include streaming and tool support."""
        mappings = self.manager._method_mappings

        # Check OpenAI method mappings
        openai_methods = mappings.get("openai.OpenAI", {})
        assert "completions.create" in openai_methods
        assert "chat.completions.create" in openai_methods
        assert "stream" in openai_methods.get("chat.completions.create", [])
        assert "tools" in openai_methods.get("chat.completions.create", [])

        # Check Anthropic method mappings
        anthropic_methods = mappings.get("anthropic.Anthropic", {})
        assert "messages.create" in anthropic_methods
        assert "messages.stream" in anthropic_methods
        assert "tools" in anthropic_methods.get("messages.create", [])

    def test_register_framework_target(self):
        """Test registering custom framework targets."""
        custom_mapping = {"model": "custom_model", "temp": "temperature"}
        self.manager.register_framework_target("custom.Framework", custom_mapping)

        assert "custom.Framework" in self.manager._parameter_mappings
        assert self.manager._parameter_mappings["custom.Framework"] == custom_mapping

    def test_mock_override_application(self):
        """Test registering mock classes with override manager."""
        # First register parameter mappings for the mock classes
        self.manager.register_framework_target(
            "MockOpenAI", {"model": "model", "temperature": "temperature"}
        )
        self.manager.register_framework_target(
            "MockLangChain", {"model": "model", "temperature": "temperature"}
        )

        # Test that we can register mock classes
        mock_classes = {
            "MockOpenAI": MockOpenAIClient,
            "MockLangChain": MockLangChainLLM,
        }

        # This should not raise any errors
        self.manager.override_mock_classes(mock_classes)

        # Verify that the mock classes are registered in the active overrides
        assert "MockOpenAI" in self.manager._active_overrides
        assert "MockLangChain" in self.manager._active_overrides

        # Verify that the registered classes are the ones we provided
        assert self.manager._active_overrides["MockOpenAI"] == MockOpenAIClient
        assert self.manager._active_overrides["MockLangChain"] == MockLangChainLLM


class TestMethodOverrides:
    """Test method-level parameter overrides."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = FrameworkOverrideManager()

    def test_openai_method_override(self):
        """Test OpenAI method parameter override (simplified test)."""
        # This test verifies that method mappings are defined correctly
        # The actual method override functionality requires real framework classes

        # Check that method mappings exist for OpenAI
        openai_methods = self.manager._method_mappings.get("openai.OpenAI", {})
        assert "completions.create" in openai_methods
        assert "chat.completions.create" in openai_methods

        # Check that the expected parameters are in the mappings
        chat_params = openai_methods.get("chat.completions.create", [])
        assert "model" in chat_params
        assert "temperature" in chat_params
        assert "stream" in chat_params
        assert "tools" in chat_params

    def test_anthropic_method_override(self):
        """Test Anthropic method parameter override (simplified test)."""
        # Check that method mappings exist for Anthropic
        anthropic_methods = self.manager._method_mappings.get("anthropic.Anthropic", {})
        assert "messages.create" in anthropic_methods
        assert "messages.stream" in anthropic_methods

        # Check that the expected parameters are in the mappings
        create_params = anthropic_methods.get("messages.create", [])
        assert "model" in create_params
        assert "temperature" in create_params
        assert "max_tokens" in create_params
        assert "stream" in create_params
        assert "tools" in create_params

    @pytest.mark.asyncio
    async def test_anthropic_streaming_override(self):
        """Test Anthropic streaming method parameter mapping."""
        # Check that async method mappings exist
        anthropic_methods = self.manager._method_mappings.get(
            "anthropic.AsyncAnthropic", {}
        )
        assert "messages.create" in anthropic_methods
        assert "messages.stream" in anthropic_methods

        # Check that streaming parameters are supported
        stream_params = anthropic_methods.get("messages.stream", [])
        assert "model" in stream_params
        assert "temperature" in stream_params
        assert "max_tokens" in stream_params

    def test_langchain_method_override(self):
        """Test LangChain method parameter mapping."""
        # Check that method mappings exist for LangChain
        langchain_methods = self.manager._method_mappings.get(
            "langchain_openai.ChatOpenAI", {}
        )
        assert "invoke" in langchain_methods
        assert "stream" in langchain_methods
        assert "astream" in langchain_methods

        # Check that the expected parameters are in the mappings
        invoke_params = langchain_methods.get("invoke", [])
        assert "model" in invoke_params
        assert "temperature" in invoke_params
        assert "max_tokens" in invoke_params
        assert "streaming" in invoke_params


class TestOverrideContext:
    """Test context manager for temporary overrides."""

    def test_context_manager_scope(self):
        """Test that overrides only apply within context."""
        # Set up config
        config = TraigentConfig(model="test-model", temperature=0.5)
        set_config(config)

        # Outside context - no overrides (need to ensure no overrides are active)
        from traigent.integrations.framework_override import disable_framework_overrides

        disable_framework_overrides()

        # Create fresh mock class for testing
        class TestMockClient:
            def __init__(self, **kwargs):
                self.init_kwargs = kwargs

        # Outside context - no overrides
        client_outside = TestMockClient(model="original")
        assert client_outside.init_kwargs.get("model") == "original"

        # Inside context - overrides applied
        with override_context(
            ["traigent.integrations.framework_override.TestMockClient"]
        ):
            # This won't actually work because TestMockClient is not a real module
            # But we can test that the context manager works
            pass

        # Outside context again - overrides removed
        client_outside_again = TestMockClient(model="original")
        assert client_outside_again.init_kwargs.get("model") == "original"


class TestConvenienceFunctions:
    """Test convenience functions for framework overrides."""

    def test_enable_disable_framework_overrides(self):
        """Test enable/disable framework overrides functions."""
        # Test enable - returns FrameworkOverrideManager
        result1 = enable_framework_overrides(["openai.OpenAI"])
        assert isinstance(result1, FrameworkOverrideManager)

        # Test disable - returns None
        result2 = disable_framework_overrides()
        assert result2 is None

    def test_override_all_platforms(self):
        """Test override_all_platforms function."""
        result1 = override_all_platforms()
        result2 = disable_framework_overrides()
        assert result1 is None  # Function returns None
        assert result2 is None  # Function returns None

    def test_register_framework_mapping(self):
        """Test register_framework_mapping function."""
        custom_mapping = {"model": "custom_model"}
        register_framework_mapping("test.Framework", custom_mapping)

        # Verify it was registered
        from traigent.integrations.framework_override import _framework_override_manager

        assert "test.Framework" in _framework_override_manager._parameter_mappings
        assert (
            _framework_override_manager._parameter_mappings["test.Framework"]
            == custom_mapping
        )


class TestParameterMappings:
    """Test parameter mapping functionality."""

    def test_anthropic_parameter_mapping(self):
        """Test Anthropic parameter mappings."""
        manager = FrameworkOverrideManager()
        anthropic_mapping = manager._parameter_mappings["anthropic.Anthropic"]

        # Check basic parameters
        assert anthropic_mapping["model"] == "model"
        assert anthropic_mapping["temperature"] == "temperature"
        assert anthropic_mapping["max_tokens"] == "max_tokens_to_sample"

        # Check Anthropic-specific parameters
        assert anthropic_mapping["top_p"] == "top_p"
        assert anthropic_mapping["top_k"] == "top_k"
        assert anthropic_mapping["stop_sequences"] == "stop_sequences"

        # Check new features
        assert anthropic_mapping["stream"] == "stream"
        assert anthropic_mapping["tools"] == "tools"
        assert anthropic_mapping["system_prompt"] == "system"

    def test_cohere_parameter_mapping(self):
        """Test Cohere parameter mappings."""
        manager = FrameworkOverrideManager()
        cohere_mapping = manager._parameter_mappings["cohere.Client"]

        # Check basic parameters
        assert cohere_mapping["model"] == "model"
        assert cohere_mapping["temperature"] == "temperature"
        assert cohere_mapping["max_tokens"] == "max_tokens"

        # Check Cohere-specific parameter names
        assert cohere_mapping["top_p"] == "p"
        assert cohere_mapping["top_k"] == "k"

        # Check new features
        assert cohere_mapping["stream"] == "stream"
        assert cohere_mapping["tools"] == "tools"

    def test_huggingface_parameter_mapping(self):
        """Test HuggingFace parameter mappings."""
        manager = FrameworkOverrideManager()
        hf_mapping = manager._parameter_mappings["transformers.pipeline"]

        # Check basic parameters
        assert hf_mapping["model"] == "model"
        assert hf_mapping["temperature"] == "temperature"
        assert hf_mapping["max_tokens"] == "max_new_tokens"

        # Check HuggingFace-specific parameters
        assert hf_mapping["top_p"] == "top_p"
        assert hf_mapping["top_k"] == "top_k"
        assert hf_mapping["stop_sequences"] == "stop"

        # Check streaming support
        assert hf_mapping["stream"] == "streamer"


class TestIntegrationCompatibility:
    """Test compatibility with existing integrations."""

    def test_backwards_compatibility(self):
        """Test that existing functionality still works."""
        # Test original OpenAI override still works
        manager = FrameworkOverrideManager()
        openai_mapping = manager._parameter_mappings["openai.OpenAI"]

        # Original parameters should still be there
        assert openai_mapping["model"] == "model"
        assert openai_mapping["temperature"] == "temperature"
        assert openai_mapping["max_tokens"] == "max_tokens"
        assert openai_mapping["top_p"] == "top_p"

    def test_langchain_integration_enhanced(self):
        """Test enhanced LangChain integration."""
        manager = FrameworkOverrideManager()

        # Check original LangChain mappings still exist
        assert "langchain.llms.OpenAI" in manager._parameter_mappings
        assert "langchain_openai.ChatOpenAI" in manager._parameter_mappings

        # Check new LangChain mappings added
        assert "langchain_anthropic.ChatAnthropic" in manager._parameter_mappings

        # Check method mappings for LangChain
        langchain_methods = manager._method_mappings.get(
            "langchain_openai.ChatOpenAI", {}
        )
        assert "invoke" in langchain_methods
        assert "stream" in langchain_methods
        assert "astream" in langchain_methods


if __name__ == "__main__":
    pytest.main([__file__])
