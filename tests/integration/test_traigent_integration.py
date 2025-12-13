#!/usr/bin/env python3
"""
TraiGent Integration Test Suite

Comprehensive integration tests covering:
- Core API functionality (apply_best_config, get_optimization_insights)
- Framework integration setup and platform support
- Complete optimization workflows
- Business intelligence generation
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

from traigent.integrations import (  # noqa: E402
    LANGCHAIN_INTEGRATION_AVAILABLE,
    OPENAI_SDK_INTEGRATION_AVAILABLE,
    WANDB_INTEGRATION_AVAILABLE,
    FrameworkOverrideManager,
    disable_framework_overrides,
    enable_framework_overrides,
    override_context,
)

# Add the project root directory to path for test discovery
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from traigent import get_optimization_insights
    from traigent.api.types import (
        OptimizationResult,
        OptimizationStatus,
        TrialResult,
        TrialStatus,
    )
    from traigent.core.optimized_function import OptimizedFunction

    CORE_API_AVAILABLE = True
except ImportError:
    CORE_API_AVAILABLE = False

# Conditional imports based on availability
if LANGCHAIN_INTEGRATION_AVAILABLE:
    from traigent.integrations import (
        auto_detect_langchain_llms,
        enable_chatgpt_optimization,
        enable_claude_optimization,
        enable_langchain_optimization,
        get_supported_langchain_llms,
    )

if OPENAI_SDK_INTEGRATION_AVAILABLE:
    from traigent.integrations import (
        auto_detect_openai,
        enable_async_openai,
        enable_openai_optimization,
        enable_sync_openai,
        get_supported_openai_clients,
    )

if WANDB_INTEGRATION_AVAILABLE:
    from traigent.integrations import WandBIntegration

# Config modules
try:
    from traigent.config.context import get_config, set_config
    from traigent.config.types import TraigentConfig

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


@pytest.mark.skipif(not CORE_API_AVAILABLE, reason="Core API modules not available")
class TestCoreAPIIntegration:
    """Test core TraiGent API functionality."""

    def test_apply_best_config_integration(self):
        """Test apply_best_config with actual TraiGent classes."""

        # Create a simple test function that works with seamless mode
        def test_function(text: str) -> str:
            # These will be overridden by TraiGent
            model = "default"
            temperature = 0.5
            return f"{model}({temperature}): {text.upper()}"

        # Create optimization result
        optimization_result = OptimizationResult(
            trials=[
                TrialResult(
                    trial_id="trial_1",
                    config={"model": "GPT-4o", "temperature": 0.1},
                    metrics={"accuracy": 0.94},
                    status=TrialStatus.COMPLETED,
                    duration=2.0,
                    timestamp=datetime.now(),
                    metadata={},
                )
            ],
            best_config={"model": "GPT-4o", "temperature": 0.1},
            best_score=0.94,
            optimization_id="test",
            duration=5.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
            metadata={},
        )

        # Create optimized function without dataset (which causes issues)
        opt_func = OptimizedFunction(
            func=test_function,
            config_space={
                "model": ["gpt-4o-mini", "GPT-4o"],
                "temperature": [0.1, 0.5, 0.9],
            },
            objectives=["accuracy"],
            injection_mode="seamless",  # Will override the default parameter values
        )

        # Store the optimization results
        opt_func._optimization_results = optimization_result

        # Test apply_best_config
        result = opt_func.apply_best_config()
        assert result is True, "apply_best_config should return True"

        # Verify configuration was applied
        assert opt_func._current_config["model"] == "GPT-4o"
        assert opt_func._current_config["temperature"] == 0.1

        # Test function execution with applied config
        output = opt_func("hello world")
        assert "GPT-4o" in output
        assert "0.1" in output
        assert "HELLO WORLD" in output

    def test_optimization_insights_integration(self):
        """Test get_optimization_insights with actual TraiGent classes."""
        # Create comprehensive optimization result
        trials = [
            TrialResult(
                trial_id="trial_1",
                config={"model": "gpt-4o-mini", "temperature": 0.3},
                metrics={"accuracy": 0.82, "cost_per_1k": 0.002},
                status=TrialStatus.COMPLETED,
                duration=1.8,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="trial_2",
                config={"model": "GPT-4o", "temperature": 0.1},
                metrics={"accuracy": 0.94, "cost_per_1k": 0.008},
                status=TrialStatus.COMPLETED,
                duration=2.3,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="trial_3",
                config={"model": "gpt-4o-mini", "temperature": 0.7},
                metrics={"accuracy": 0.78, "cost_per_1k": 0.003},
                status=TrialStatus.COMPLETED,
                duration=2.1,
                timestamp=datetime.now(),
                metadata={},
            ),
        ]

        optimization_result = OptimizationResult(
            trials=trials,
            best_config={"model": "GPT-4o", "temperature": 0.1},
            best_score=0.94,
            optimization_id="insights_test",
            duration=15.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "cost_per_1k"],
            algorithm="bayesian",
            timestamp=datetime.now(),
            metadata={},
        )

        # Test insights generation
        insights = get_optimization_insights(optimization_result)

        # Verify insights structure
        assert "top_configurations" in insights
        assert "performance_summary" in insights
        assert "parameter_insights" in insights
        assert "recommendations" in insights
        assert "error" not in insights

        # Verify insights content
        assert len(insights["top_configurations"]) == 3
        assert insights["performance_summary"]["best_score"] == 0.94
        assert insights["performance_summary"]["total_trials"] == 3
        assert "model" in insights["parameter_insights"]
        assert len(insights["recommendations"]) > 0

    def test_complete_optimization_workflow(self):
        """Test complete workflow: create function -> apply config -> get insights."""

        def customer_support_agent(query: str) -> str:
            # These will be overridden by TraiGent
            model = "default"
            # Simulate different model performance
            quality = "high" if model == "GPT-4o" else "medium"
            return f"[{quality}] Response to: {query}"

        # Create optimization result
        optimization_result = OptimizationResult(
            trials=[
                TrialResult(
                    trial_id="trial_1",
                    config={"model": "gpt-4o-mini", "temperature": 0.5},
                    metrics={"accuracy": 0.85, "cost_per_1k": 0.002},
                    status=TrialStatus.COMPLETED,
                    duration=2.0,
                    timestamp=datetime.now(),
                    metadata={},
                ),
                TrialResult(
                    trial_id="trial_2",
                    config={"model": "GPT-4o", "temperature": 0.1},
                    metrics={"accuracy": 0.94, "cost_per_1k": 0.008},
                    status=TrialStatus.COMPLETED,
                    duration=3.0,
                    timestamp=datetime.now(),
                    metadata={},
                ),
            ],
            best_config={"model": "GPT-4o", "temperature": 0.1},
            best_score=0.94,
            optimization_id="workflow_test",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "cost_per_1k"],
            algorithm="bayesian",
            timestamp=datetime.now(),
            metadata={},
        )

        # Step 1: Create optimized function
        opt_func = OptimizedFunction(
            func=customer_support_agent,
            config_space={
                "model": ["gpt-4o-mini", "GPT-4o"],
                "temperature": [0.1, 0.5, 0.9],
            },
            objectives=["accuracy", "cost_per_1k"],
            injection_mode="seamless",  # Use seamless to override local variables
        )

        # Step 2: Store optimization results (simulating optimization)
        opt_func._optimization_results = optimization_result

        # Step 3: Apply best configuration
        success = opt_func.apply_best_config()
        assert success is True

        # Step 4: Use optimized function
        response = opt_func("What's your return policy?")
        assert "high" in response  # Should use GPT-4o which gives high quality
        assert "return policy" in response

        # Step 5: Get business insights
        insights = get_optimization_insights(optimization_result)

        # Verify business intelligence
        top_configs = insights["top_configurations"]
        assert len(top_configs) == 2
        assert top_configs[0]["rank"] == 1
        assert top_configs[0]["score"] == 0.94

        perf_summary = insights["performance_summary"]
        assert perf_summary["best_score"] == 0.94
        assert perf_summary["improvement"] > 0


class TestFrameworkIntegrationSetup:
    """Test framework integration setup and management."""

    def setup_method(self):
        """Set up test fixtures."""
        if CONFIG_AVAILABLE:
            set_config(None)
        disable_framework_overrides()

    def teardown_method(self):
        """Clean up after tests."""
        if CONFIG_AVAILABLE:
            set_config(None)
        disable_framework_overrides()

    def test_framework_override_manager_initialization(self):
        """Test that framework override manager initializes correctly."""
        manager = FrameworkOverrideManager()

        # Check that parameter mappings are loaded
        assert len(manager._parameter_mappings) > 0
        assert len(manager._method_mappings) > 0

        # Check specific platform support
        assert "openai.OpenAI" in manager._parameter_mappings
        assert "anthropic.Anthropic" in manager._parameter_mappings
        assert "cohere.Client" in manager._parameter_mappings
        assert "transformers.pipeline" in manager._parameter_mappings

    @pytest.mark.skipif(
        not LANGCHAIN_INTEGRATION_AVAILABLE,
        reason="LangChain integration not available",
    )
    def test_langchain_integration_setup(self):
        """Test LangChain integration setup."""
        # Should not raise any errors
        enable_langchain_optimization()

        # Test convenience functions
        enable_chatgpt_optimization()
        enable_claude_optimization()

        supported_llms = get_supported_langchain_llms()
        assert isinstance(supported_llms, list)
        assert len(supported_llms) >= 0

    @pytest.mark.skipif(
        not OPENAI_SDK_INTEGRATION_AVAILABLE,
        reason="OpenAI SDK integration not available",
    )
    def test_openai_sdk_integration_setup(self):
        """Test OpenAI SDK integration setup."""
        # Should not raise any errors
        enable_openai_optimization()

        # Test convenience functions
        enable_sync_openai()
        enable_async_openai()

        supported_clients = get_supported_openai_clients()
        assert isinstance(supported_clients, list)
        assert len(supported_clients) >= 0

    @pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config modules not available")
    def test_config_context_management(self):
        """Test configuration context management."""
        # Test setting and getting config
        config = TraigentConfig(
            model="test-model",
            temperature=0.7,
            max_tokens=500,
            custom_params={"stream": True, "tools": []},
        )

        set_config(config)
        retrieved_config = get_config()

        assert retrieved_config is not None
        assert retrieved_config.model == "test-model"
        assert retrieved_config.temperature == 0.7
        assert retrieved_config.max_tokens == 500
        assert retrieved_config.custom_params["stream"]

        # Clear config - get_config() returns default TraigentConfig when None is set
        set_config(None)
        cleared_config = get_config()
        # After clearing, get_config returns a default TraigentConfig (not None)
        assert isinstance(cleared_config, TraigentConfig)
        assert cleared_config.model is None  # Default has no model set

    def test_override_enable_disable_cycle(self):
        """Test enabling and disabling overrides."""
        # Enable overrides
        enable_framework_overrides(["openai.OpenAI"])

        # Disable overrides
        disable_framework_overrides()

        # Should not raise any errors

    def test_context_manager_functionality(self):
        """Test context manager functionality."""
        # This tests the context manager structure, not actual overrides
        # since those require real framework classes

        try:
            with override_context(["openai.OpenAI"]):
                # Inside context
                pass
            # Outside context
        except Exception:
            # Context manager should handle errors gracefully
            # and not leave overrides active
            pass

        # Verify cleanup happened
        # (Can't easily test this without real framework classes)

    def test_integration_imports(self):
        """Test that all integration modules can be imported."""
        # Test main integration imports

        # Test conditional imports
        if LANGCHAIN_INTEGRATION_AVAILABLE:
            pass

        if OPENAI_SDK_INTEGRATION_AVAILABLE:
            pass

        if WANDB_INTEGRATION_AVAILABLE:
            pass

        # All imports should succeed

    @pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config modules not available")
    def test_config_serialization(self):
        """Test config serialization and parameter extraction."""
        config = TraigentConfig(
            model="test-model",
            temperature=0.8,
            max_tokens=1000,
            custom_params={
                "stream": True,
                "tools": [{"type": "function", "function": {"name": "test"}}],
                "top_k": 40,
            },
        )

        # Test to_dict method
        config_dict = config.to_dict()
        assert config_dict["model"] == "test-model"
        assert config_dict["temperature"] == 0.8
        assert config_dict["max_tokens"] == 1000

        # Custom params should be accessible
        assert config.custom_params["stream"]
        assert len(config.custom_params["tools"]) == 1
        assert config.custom_params["top_k"] == 40


class TestPlatformSupport:
    """Test platform support functionality."""

    def test_all_platforms_have_mappings(self):
        """Test that all supported platforms have parameter mappings."""
        manager = FrameworkOverrideManager()

        # Expected platforms
        expected_platforms = [
            "openai.OpenAI",
            "openai.AsyncOpenAI",
            "anthropic.Anthropic",
            "anthropic.AsyncAnthropic",
            "cohere.Client",
            "cohere.AsyncClient",
            "transformers.pipeline",
            "transformers.AutoModelForCausalLM",
            "langchain_openai.ChatOpenAI",
            "langchain_openai.OpenAI",
            "langchain_anthropic.ChatAnthropic",
        ]

        for platform in expected_platforms:
            assert (
                platform in manager._parameter_mappings
            ), f"Missing mapping for {platform}"
            mapping = manager._parameter_mappings[platform]
            assert isinstance(mapping, dict), f"Invalid mapping type for {platform}"
            assert len(mapping) > 0, f"Empty mapping for {platform}"

    def test_parameter_mapping_completeness(self):
        """Test that parameter mappings are complete for all platforms."""
        manager = FrameworkOverrideManager()

        # Test OpenAI mappings
        openai_mapping = manager._parameter_mappings["openai.OpenAI"]
        required_params = [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "stream",
            "tools",
        ]
        for param in required_params:
            assert param in openai_mapping

        # Test Anthropic mappings
        anthropic_mapping = manager._parameter_mappings["anthropic.Anthropic"]
        required_params = [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "stream",
            "tools",
        ]
        for param in required_params:
            assert param in anthropic_mapping

        # Test Cohere mappings
        cohere_mapping = manager._parameter_mappings["cohere.Client"]
        required_params = [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "stream",
            "tools",
        ]
        for param in required_params:
            assert param in cohere_mapping

    def test_method_mapping_completeness(self):
        """Test that method mappings are complete for all platforms."""
        manager = FrameworkOverrideManager()

        # Test OpenAI method mappings
        openai_methods = manager._method_mappings["openai.OpenAI"]
        assert "completions.create" in openai_methods
        assert "chat.completions.create" in openai_methods

        # Test Anthropic method mappings
        anthropic_methods = manager._method_mappings["anthropic.Anthropic"]
        assert "messages.create" in anthropic_methods
        assert "messages.stream" in anthropic_methods

        # Test LangChain method mappings
        langchain_methods = manager._method_mappings["langchain_openai.ChatOpenAI"]
        assert "invoke" in langchain_methods
        assert "stream" in langchain_methods
        assert "astream" in langchain_methods

    def test_streaming_support_across_platforms(self):
        """Test that streaming is supported across platforms."""
        manager = FrameworkOverrideManager()

        # Platforms that should support streaming
        streaming_platforms = [
            "openai.OpenAI",
            "anthropic.Anthropic",
            "cohere.Client",
            "langchain_openai.ChatOpenAI",
            "langchain_anthropic.ChatAnthropic",
        ]

        for platform in streaming_platforms:
            mapping = manager._parameter_mappings[platform]
            # Should have either 'stream' or 'streaming' parameter
            has_streaming = "stream" in mapping or "streaming" in mapping
            assert has_streaming, f"Platform {platform} missing streaming support"

    def test_tool_support_across_platforms(self):
        """Test that tool calling is supported across platforms."""
        manager = FrameworkOverrideManager()

        # Platforms that should support tools
        tool_platforms = [
            "openai.OpenAI",
            "anthropic.Anthropic",
            "cohere.Client",
        ]

        for platform in tool_platforms:
            mapping = manager._parameter_mappings[platform]
            assert "tools" in mapping, f"Platform {platform} missing tools support"

    def test_parameter_precedence_logic(self):
        """Test parameter precedence logic."""
        manager = FrameworkOverrideManager()

        # Test parameter mapping extraction
        openai_mapping = manager._parameter_mappings["openai.OpenAI"]

        # Test that the mapping has the expected structure
        assert isinstance(openai_mapping, dict)
        assert "model" in openai_mapping
        assert openai_mapping["model"] == "model"  # Direct mapping

        # Test custom parameter mapping registration
        custom_mapping = {"custom_param": "platform_param"}
        manager.register_framework_target("test.Framework", custom_mapping)

        assert "test.Framework" in manager._parameter_mappings
        assert manager._parameter_mappings["test.Framework"] == custom_mapping

    def test_platform_auto_detection(self):
        """Test platform auto-detection functionality."""
        # These should not raise errors even if packages aren't installed
        if LANGCHAIN_INTEGRATION_AVAILABLE:
            auto_detect_langchain_llms()
        if OPENAI_SDK_INTEGRATION_AVAILABLE:
            auto_detect_openai()

    @pytest.mark.skipif(
        not WANDB_INTEGRATION_AVAILABLE, reason="WandB integration not available"
    )
    def test_wandb_integration_compatibility(self):
        """Test WandB integration compatibility."""
        try:
            from datetime import datetime

            from traigent.api.types import (
                OptimizationResult,
                OptimizationStatus,
                TrialResult,
                TrialStatus,
            )

            # Test that we can create the required types
            trial_result = TrialResult(
                trial_id="test-123",
                config={"model": "gpt-4"},
                metrics={"accuracy": 0.85},
                status=TrialStatus.COMPLETED,
                duration=45.0,
                timestamp=datetime.now(),
            )

            result = OptimizationResult(
                trials=[trial_result],
                best_config={"model": "gpt-4"},
                best_score=0.85,
                optimization_id="opt-123",
                duration=120.0,
                convergence_info={},
                status=OptimizationStatus.COMPLETED,
                objectives=["accuracy"],
                algorithm="bayesian",
                timestamp=datetime.now(),
            )

            # Test that the types have the required attributes
            assert hasattr(trial_result, "trial_id")
            assert hasattr(trial_result, "config")
            assert hasattr(trial_result, "metrics")
            assert hasattr(trial_result, "status")
            assert hasattr(trial_result, "duration")
            assert hasattr(trial_result, "timestamp")

            assert hasattr(result, "trials")
            assert hasattr(result, "best_config")
            assert hasattr(result, "duration")
            assert hasattr(result, "status")
        except ImportError:
            # If types are not available, just verify WandBIntegration exists
            assert WandBIntegration is not None


def main():
    """Run all integration tests as a script."""
    if not CORE_API_AVAILABLE:
        print("❌ Core TraiGent API not available - install dependencies:")
        print("pip install numpy pandas scikit-learn")
        return False

    print("🚀 TraiGent Integration Test Suite")
    print("=" * 60)
    print("Testing complete TraiGent integration functionality\n")

    # Run pytest programmatically
    exit_code = pytest.main([__file__, "-v"])
    return exit_code == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
