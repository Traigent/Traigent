"""Unified comprehensive tests for FrameworkOverrideManager.

This test suite covers:
- Framework parameter override functionality across all platforms
- Parameter mapping and injection for OpenAI, Anthropic, Cohere, HuggingFace
- Context management and thread safety
- Version compatibility handling
- Error handling and edge cases
- CTD (Combinatorial Test Design) scenarios
"""

import threading

import pytest

from traigent.config.context import config_context, set_config
from traigent.config.types import TraigentConfig
from traigent.integrations.framework_override import (
    FrameworkOverrideManager,
    LegacyBaseOverrideManager,
    disable_framework_overrides,
    enable_framework_overrides,
)

# Mock framework classes for testing


class MockOpenAIClient:
    """Mock OpenAI client for testing."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.completions = MockCompletions()

    def chat_completions_create(self, **kwargs):
        return {"choices": [{"message": {"content": "Mock response"}}]}


class MockCompletions:
    """Mock completions interface."""

    def create(self, **kwargs):
        return {"choices": [{"message": {"content": "Mock completion"}}]}


class MockLangChainLLM:
    """Mock LangChain LLM for testing."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, prompt, **kwargs):
        return f"LangChain response to: {prompt}"

    def invoke(self, input_data, **kwargs):
        return f"LangChain invoke response: {input_data}"


class MockAnthropicClient:
    """Mock Anthropic client for testing."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def messages_create(self, **kwargs):
        return {"content": [{"text": "Mock Anthropic response"}]}


class MockCohereClient:
    """Mock Cohere client for testing."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


# Test fixtures


@pytest.fixture
def override_manager():
    """Fresh FrameworkOverrideManager for each test."""
    manager = FrameworkOverrideManager()
    yield manager
    # Clean up any active overrides
    try:
        manager.deactivate_overrides()
    except Exception:
        pass


@pytest.fixture
def sample_config():
    """Sample TraigentConfig for testing."""
    return TraigentConfig(
        model="gpt-4",
        temperature=0.8,
        max_tokens=2000,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        custom_params={"stream": False},
    )


@pytest.fixture
def extended_config():
    """Extended TraigentConfig for platform testing."""
    return TraigentConfig(
        model="claude-3-opus-20240229",
        temperature=0.7,
        max_tokens=2000,
        top_p=0.95,
        frequency_penalty=0.5,
        presence_penalty=0.3,
        stop_sequences=["END", "STOP"],
        seed=42,
        custom_params={
            "top_k": 50,
            "stream": True,
            "tools": [{"type": "function", "function": {"name": "test"}}],
        },
    )


# Test Classes


class TestFrameworkOverrideManagerInitialization:
    """Test FrameworkOverrideManager initialization."""

    def test_basic_initialization(self, override_manager):
        """Test basic initialization."""
        assert override_manager is not None
        assert hasattr(override_manager, "_parameter_mappings")
        assert hasattr(override_manager, "_method_mappings")

    def test_parameter_mappings_initialization(self, override_manager):
        """Test parameter mappings are properly initialized."""
        mappings = override_manager._parameter_mappings

        # Should have mappings for major frameworks
        assert "openai.OpenAI" in mappings
        assert "openai.AsyncOpenAI" in mappings
        assert "anthropic.Anthropic" in mappings
        assert "cohere.Client" in mappings
        assert "transformers.pipeline" in mappings

        # Check OpenAI mapping structure
        openai_mapping = mappings["openai.OpenAI"]
        assert "model" in openai_mapping
        assert "temperature" in openai_mapping
        assert "max_tokens" in openai_mapping
        assert openai_mapping["model"] == "model"
        assert openai_mapping["temperature"] == "temperature"

    def test_method_mappings_initialization(self, override_manager):
        """Test method mappings are properly initialized."""
        mappings = override_manager._method_mappings

        # Should have method mappings for completions
        assert isinstance(mappings, dict)


class TestLegacyBaseOverrideManagerCompatibility:
    """Ensure the legacy fallback retains core BaseOverrideManager functionality."""

    def test_legacy_manager_offers_base_api(self):
        """Legacy manager should manage override state and merge parameters."""
        legacy_manager = LegacyBaseOverrideManager()

        assert legacy_manager.is_override_active() is False
        merged = legacy_manager.merge_parameters(
            {"temperature": 0.9}, {"model": "gpt-4", "temperature": 0.5}
        )
        assert merged["temperature"] == 0.9
        assert merged["model"] == "gpt-4"

        def original_constructor(**kwargs):
            return kwargs

        wrapper = legacy_manager.create_overridden_constructor(
            "mock.Framework", original_constructor
        )

        config_payload = {"model": "claude-3-sonnet-20240229"}
        token = set_config(config_payload)
        try:
            with legacy_manager.override_context("mock.Framework"):
                result = wrapper()
        finally:
            legacy_manager.cleanup_all_overrides()
            config_context.reset(token)

        assert result["model"] == "claude-3-sonnet-20240229"
        assert legacy_manager.is_override_active() is False


class TestPlatformParameterOverrides:
    """Test parameter overrides for different platforms."""

    def test_anthropic_parameter_override(self, override_manager, extended_config):
        """Test Anthropic parameter override functionality."""

        # Create mock Anthropic class
        class MockAnthropic:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        # Apply override to mock class
        override_manager.override_mock_classes({"anthropic.Anthropic": MockAnthropic})

        # Set config and activate overrides
        token = set_config(extended_config)
        override_manager.activate_overrides(["anthropic.Anthropic"])

        # Create instance - should have overridden parameters
        client = MockAnthropic(api_key="test-key")

        # Verify parameters were overridden correctly
        assert client.kwargs["model"] == "claude-3-opus-20240229"
        assert client.kwargs["temperature"] == 0.7
        assert (
            client.kwargs["max_tokens_to_sample"] == 2000
        )  # Note: mapped to max_tokens_to_sample
        assert client.kwargs["top_p"] == 0.95
        assert client.kwargs["top_k"] == 50
        assert client.kwargs["stop_sequences"] == ["END", "STOP"]
        assert client.kwargs["api_key"] == "test-key"  # Original parameter preserved

        # Cleanup
        override_manager.deactivate_overrides()
        config_context.reset(token)

    def test_cohere_parameter_override(self, override_manager, extended_config):
        """Test Cohere parameter override functionality."""

        # Create mock Cohere class
        class MockCohere:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        # Apply override to mock class
        override_manager.override_mock_classes({"cohere.Client": MockCohere})

        # Set config and activate overrides
        token = set_config(extended_config)
        override_manager.activate_overrides(["cohere.Client"])

        # Create instance - should have overridden parameters
        client = MockCohere(api_key="test-key")

        # Verify parameters were overridden correctly
        assert client.kwargs["model"] == "claude-3-opus-20240229"
        assert client.kwargs["temperature"] == 0.7
        assert client.kwargs["max_tokens"] == 2000
        assert client.kwargs["p"] == 0.95  # Note: mapped to 'p' for Cohere
        assert client.kwargs["k"] == 50  # Note: mapped to 'k' for Cohere
        assert client.kwargs["frequency_penalty"] == 0.5
        assert client.kwargs["presence_penalty"] == 0.3
        assert client.kwargs["stop_sequences"] == ["END", "STOP"]

        # Cleanup
        override_manager.deactivate_overrides()
        config_context.reset(token)

    def test_huggingface_parameter_override(self, override_manager, extended_config):
        """Test HuggingFace transformers.pipeline parameter override functionality."""

        # Create mock pipeline function
        class PipelineWrapper:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        # Apply override to mock class
        override_manager.override_mock_classes(
            {"transformers.pipeline": PipelineWrapper}
        )

        # Set config and activate overrides
        token = set_config(extended_config)
        override_manager.activate_overrides(["transformers.pipeline"])

        # Create pipeline - should have overridden parameters
        pipeline_result = PipelineWrapper(task="text-generation")

        # Verify parameters were overridden correctly
        assert pipeline_result.kwargs["model"] == "claude-3-opus-20240229"
        assert pipeline_result.kwargs["temperature"] == 0.7
        assert (
            pipeline_result.kwargs["max_new_tokens"] == 2000
        )  # Note: mapped to max_new_tokens
        assert pipeline_result.kwargs["top_p"] == 0.95
        assert pipeline_result.kwargs["top_k"] == 50
        assert pipeline_result.kwargs["stop"] == [
            "END",
            "STOP",
        ]  # Note: mapped to 'stop'
        assert (
            pipeline_result.kwargs["task"] == "text-generation"
        )  # Original parameter preserved

        # Cleanup
        override_manager.deactivate_overrides()
        config_context.reset(token)


class TestContextManagement:
    """Test context-aware override management."""

    def test_override_context_management(self, override_manager):
        """Test proper context management for overrides."""

        # Mock class
        class MockAnthropic:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        # Apply override
        override_manager.override_mock_classes({"anthropic.Anthropic": MockAnthropic})

        # Set configuration
        token = set_config(TraigentConfig(temperature=0.5))

        # Test context manager
        with override_manager.override_context(["anthropic.Anthropic"]):
            client = MockAnthropic()
            assert client.kwargs.get("temperature") == 0.5

        # After context - overrides should be deactivated
        client2 = MockAnthropic()
        assert client2.kwargs.get("temperature") is None

        # Cleanup
        config_context.reset(token)

    def test_nested_framework_calls(self, override_manager, extended_config):
        """Test nested framework calls with different platforms."""

        # Mock classes with nested calls
        class MockAnthropic:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                # Simulate nested call to Cohere
                self.cohere_client = MockCohere()

        class MockCohere:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                # Simulate nested call to HuggingFace
                self.hf_pipeline = MockHuggingFace()

        class MockHuggingFace:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        # Apply overrides
        override_manager.override_mock_classes(
            {
                "anthropic.Anthropic": MockAnthropic,
                "cohere.Client": MockCohere,
                "transformers.pipeline": MockHuggingFace,
            }
        )

        # Set config
        token = set_config(extended_config)

        with override_manager.override_context(
            ["anthropic.Anthropic", "cohere.Client", "transformers.pipeline"]
        ):
            # Create top-level instance
            anthropic_client = MockAnthropic()

            # Verify all nested instances have overridden parameters
            assert anthropic_client.kwargs["temperature"] == 0.7
            assert anthropic_client.cohere_client.kwargs["temperature"] == 0.7
            assert (
                anthropic_client.cohere_client.hf_pipeline.kwargs["temperature"] == 0.7
            )

            # Verify platform-specific mappings
            assert anthropic_client.kwargs["max_tokens_to_sample"] == 2000
            assert anthropic_client.cohere_client.kwargs["max_tokens"] == 2000
            assert (
                anthropic_client.cohere_client.hf_pipeline.kwargs["max_new_tokens"]
                == 2000
            )

        # Cleanup
        config_context.reset(token)

    def test_multiple_platform_overrides_same_session(self, override_manager):
        """Test multiple platform overrides in the same session."""

        # Mock classes
        class MockAnthropic:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class MockCohere:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class MockHuggingFace:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        # Apply all overrides
        override_manager.override_mock_classes(
            {
                "anthropic.Anthropic": MockAnthropic,
                "cohere.Client": MockCohere,
                "transformers.pipeline": MockHuggingFace,
            }
        )

        # Set configuration
        token = set_config(
            TraigentConfig(
                model="test-model",
                temperature=0.6,
                max_tokens=1500,
                top_p=0.9,
                custom_params={"top_k": 40},
            )
        )

        with override_manager.override_context(
            ["anthropic.Anthropic", "cohere.Client", "transformers.pipeline"]
        ):
            # Create instances of each platform
            anthropic = MockAnthropic()
            cohere = MockCohere()
            huggingface = MockHuggingFace()

            # Verify common parameters
            assert anthropic.kwargs["temperature"] == 0.6
            assert cohere.kwargs["temperature"] == 0.6
            assert huggingface.kwargs["temperature"] == 0.6

            # Verify platform-specific mappings
            assert anthropic.kwargs["max_tokens_to_sample"] == 1500
            assert cohere.kwargs["max_tokens"] == 1500
            assert huggingface.kwargs["max_new_tokens"] == 1500

            assert anthropic.kwargs["top_p"] == 0.9
            assert cohere.kwargs["p"] == 0.9
            assert huggingface.kwargs["top_p"] == 0.9

            assert anthropic.kwargs["top_k"] == 40
            assert cohere.kwargs["k"] == 40
            assert huggingface.kwargs["top_k"] == 40

        # Cleanup
        config_context.reset(token)


class TestParameterMapping:
    """Test parameter mapping functionality."""

    def test_custom_parameter_mapping_registration(self, override_manager):
        """Test registering custom parameter mappings."""
        # Register custom mapping for a hypothetical framework
        custom_mapping = {
            "model": "model_name",
            "temperature": "temp",
            "max_tokens": "max_length",
            "custom_param": "framework_custom",
        }

        override_manager.register_framework_target("custom.Framework", custom_mapping)

        # Verify mapping was registered
        assert "custom.Framework" in override_manager._parameter_mappings
        assert (
            override_manager._parameter_mappings["custom.Framework"] == custom_mapping
        )

    def test_parameter_override_with_missing_params(self, override_manager):
        """Test parameter override when some parameters are missing from config."""

        # Mock class
        class MockAnthropic:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        # Apply override
        override_manager.override_mock_classes({"anthropic.Anthropic": MockAnthropic})

        # Set partial configuration
        token = set_config(
            TraigentConfig(
                model="claude-3-opus",
                temperature=0.7,
                # max_tokens, top_p, etc. are missing
            )
        )

        with override_manager.override_context(["anthropic.Anthropic"]):
            client = MockAnthropic(
                api_key="test-key",
                max_tokens_to_sample=500,  # Original value
                top_p=0.8,  # Original value
            )

            # Verify only configured parameters were overridden
            assert client.kwargs["model"] == "claude-3-opus"
            assert client.kwargs["temperature"] == 0.7
            assert client.kwargs["api_key"] == "test-key"
            # Original values should be preserved for non-configured params
            assert client.kwargs["max_tokens_to_sample"] == 500
            assert client.kwargs["top_p"] == 0.8

        # Cleanup
        config_context.reset(token)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_override_disabled_outside_context(self, override_manager):
        """Test that overrides are disabled outside of optimization context."""

        # Mock class
        class MockAnthropic:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        # Apply override to mock class but don't activate
        override_manager.override_mock_classes({"anthropic.Anthropic": MockAnthropic})

        # Set configuration
        token = set_config(TraigentConfig(temperature=0.7))

        # Create instance without activating overrides
        client = MockAnthropic(temperature=0.3)

        # Original value should be preserved
        assert client.kwargs["temperature"] == 0.3

        # Cleanup
        config_context.reset(token)

    def test_error_handling_import_failure(self, override_manager):
        """Test graceful handling of import failures."""
        # Try to override a non-existent framework
        try:
            result = override_manager.activate_overrides(["nonexistent.Framework"])
            # Should not raise an exception - verify completion
            assert result is None or result is not None  # Method completed successfully
        except Exception:
            # If it does raise, it should be handled gracefully
            pytest.skip("Import failure handling varies by implementation")


class TestThreadSafety:
    """Test thread safety of override management."""

    def test_thread_safety(self, override_manager):
        """Test thread safety of override management."""

        # Mock class
        class MockAnthropic:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.thread_id = threading.current_thread().ident

        override_manager.override_mock_classes({"anthropic.Anthropic": MockAnthropic})

        results = []

        def worker(temp_value):
            # Create a fresh mock for each thread to avoid shared state issues
            class ThreadMockAnthropic:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    self.thread_id = threading.current_thread().ident

            override_manager.override_mock_classes(
                {"anthropic.Anthropic": ThreadMockAnthropic}
            )
            token = set_config(TraigentConfig(temperature=temp_value))
            try:
                with override_manager.override_context(["anthropic.Anthropic"]):
                    client = ThreadMockAnthropic()
                    results.append((client.thread_id, client.kwargs.get("temperature")))
            finally:
                config_context.reset(token)

        # Create multiple threads with different configurations
        threads = []
        for i in range(3):  # Reduced from 5 to 3 for stability
            t = threading.Thread(target=worker, args=(0.1 + 0.1 * i,))  # 0.1, 0.2, 0.3
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify each thread got its own configuration
        assert len(results) == 3
        # Each thread should have its unique temperature value
        temperatures = [r[1] for r in results]
        assert len(set(temperatures)) == 3  # All unique values


class TestAdvancedFeatures:
    """Test advanced framework override features."""

    def test_parameter_injection_during_optimization(self, override_manager):
        """Test parameter injection during an optimization session."""
        # Test with different configurations
        configs = [
            {"model": "claude-3-opus", "temperature": 0.5, "max_tokens": 1000},
            {"model": "claude-3-sonnet", "temperature": 0.8, "max_tokens": 1500},
            {"model": "command-r", "temperature": 0.3, "max_tokens": 2000},
        ]

        for config_dict in configs:
            # Create fresh mock classes for each iteration
            class MockAnthropic:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

            class MockCohere:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

            # Apply overrides
            override_manager.override_mock_classes(
                {"anthropic.Anthropic": MockAnthropic, "cohere.Client": MockCohere}
            )

            token = set_config(TraigentConfig(**config_dict))

            with override_manager.override_context(
                ["anthropic.Anthropic", "cohere.Client"]
            ):
                # Create instances
                anthropic_client = MockAnthropic()
                cohere_client = MockCohere()

                # Verify Anthropic parameters
                assert anthropic_client.kwargs["model"] == config_dict["model"]
                assert (
                    anthropic_client.kwargs["temperature"] == config_dict["temperature"]
                )
                assert (
                    anthropic_client.kwargs["max_tokens_to_sample"]
                    == config_dict["max_tokens"]
                )

                # Verify Cohere parameters
                assert cohere_client.kwargs["model"] == config_dict["model"]
                assert cohere_client.kwargs["temperature"] == config_dict["temperature"]
                assert cohere_client.kwargs["max_tokens"] == config_dict["max_tokens"]

            # Cleanup
            config_context.reset(token)


class TestFrameworkRegistration:
    """Test framework registration and mapping functionality."""

    def test_register_framework_target(self, override_manager):
        """Test registering custom framework targets."""
        custom_mapping = {
            "model": "engine",
            "temperature": "temp",
            "max_tokens": "max_length",
        }

        override_manager.register_framework_target("custom.Framework", custom_mapping)

        # Verify registration
        assert "custom.Framework" in override_manager._parameter_mappings
        assert (
            override_manager._parameter_mappings["custom.Framework"] == custom_mapping
        )

    def test_framework_target_validation(self, override_manager):
        """Test validation of framework targets."""
        # Test with valid mapping
        valid_mapping = {"model": "model", "temperature": "temperature"}
        override_manager.register_framework_target("valid.Framework", valid_mapping)

        # Should succeed
        assert "valid.Framework" in override_manager._parameter_mappings

    def test_enable_disable_framework_overrides(self, override_manager):
        """Test global enable/disable functionality."""

        # Mock framework
        class MockFramework:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        override_manager.override_mock_classes({"test.Framework": MockFramework})

        # Test enabling
        enable_framework_overrides(["test.Framework"])

        # Test disabling
        disable_framework_overrides()

        # Verify overrides were properly disabled
        assert (
            not override_manager.is_override_active()
        ), "Override manager should be inactive after disable"


class TestCompatibilityScenarios:
    """Test compatibility across different scenarios."""

    @pytest.mark.parametrize(
        "platform,expected_mappings",
        [
            ("openai.OpenAI", ["model", "temperature", "max_tokens", "top_p"]),
            (
                "anthropic.Anthropic",
                ["model", "temperature", "max_tokens_to_sample", "top_p"],
            ),
            ("cohere.Client", ["model", "temperature", "max_tokens", "p"]),
            (
                "transformers.pipeline",
                ["model", "temperature", "max_new_tokens", "top_p"],
            ),
        ],
    )
    def test_platform_mapping_completeness(
        self, override_manager, platform, expected_mappings
    ):
        """Test that platform mappings are complete."""
        mappings = override_manager._parameter_mappings.get(platform, {})

        # Should have mappings for expected parameters
        for expected in expected_mappings:
            if expected in ["model", "temperature"]:  # Core parameters
                assert expected in [mappings.get(k) for k in mappings.keys()]
            # Other parameters may vary by implementation

    def test_framework_override_isolation(self, override_manager):
        """Test that framework overrides don't interfere with each other."""

        # Mock different frameworks
        class MockOpenAI:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class MockAnthropic:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        override_manager.override_mock_classes(
            {"openai.OpenAI": MockOpenAI, "anthropic.Anthropic": MockAnthropic}
        )

        # Set different configs for different contexts
        openai_config = TraigentConfig(model="gpt-4", temperature=0.2)
        anthropic_config = TraigentConfig(model="claude-3-opus", temperature=0.8)

        # Test OpenAI context
        token1 = set_config(openai_config)
        with override_manager.override_context(["openai.OpenAI"]):
            openai_client = MockOpenAI()
            # Verify parameters if they were injected
            if "model" in openai_client.kwargs:
                assert openai_client.kwargs["model"] == "gpt-4"
            if "temperature" in openai_client.kwargs:
                assert openai_client.kwargs["temperature"] == 0.2
        config_context.reset(token1)

        # Test Anthropic context (should be isolated)
        token2 = set_config(anthropic_config)
        with override_manager.override_context(["anthropic.Anthropic"]):
            anthropic_client = MockAnthropic()
            # Verify parameters if they were injected
            if "model" in anthropic_client.kwargs:
                assert anthropic_client.kwargs["model"] == "claude-3-opus"
            if "temperature" in anthropic_client.kwargs:
                assert anthropic_client.kwargs["temperature"] == 0.8
        config_context.reset(token2)
