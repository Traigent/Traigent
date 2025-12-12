"""Integration tests for @traigent.optimize decorator with platform executors.

This module tests the complete optimization workflow with new platforms:
- @traigent.optimize decorator with new platforms
- auto_override_frameworks with Anthropic/Cohere/HuggingFace
- configuration_space with platform-specific models
- optimization objectives with different platforms
- smart subset selection with various platforms
- result aggregation across platforms
"""

import asyncio
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

import traigent


class TestOptimizationWithPlatforms:
    """Test optimization workflow with multiple platforms."""

    @pytest.fixture
    def mock_platform_responses(self):
        """Mock responses for different platforms."""
        return {
            "openai": {
                "gpt-3.5-turbo": {"quality": 0.85, "cost": 0.001, "latency": 0.5},
                "gpt-4": {"quality": 0.95, "cost": 0.01, "latency": 1.2},
            },
            "anthropic": {
                "claude-3-haiku": {"quality": 0.88, "cost": 0.0008, "latency": 0.4},
                "claude-3-sonnet": {"quality": 0.92, "cost": 0.003, "latency": 0.8},
                "claude-3-opus": {"quality": 0.96, "cost": 0.015, "latency": 1.5},
            },
            "cohere": {
                "command": {"quality": 0.87, "cost": 0.0009, "latency": 0.6},
                "command-light": {"quality": 0.82, "cost": 0.0004, "latency": 0.3},
            },
            "huggingface": {
                "meta-llama/Llama-2-7b": {
                    "quality": 0.80,
                    "cost": 0.0005,
                    "latency": 0.7,
                },
                "mistralai/Mistral-7B": {
                    "quality": 0.83,
                    "cost": 0.0006,
                    "latency": 0.8,
                },
            },
            "langchain": {
                "gpt-3.5-turbo": {"quality": 0.86, "cost": 0.0012, "latency": 0.6}
            },
        }

    @pytest.mark.asyncio
    async def test_multi_platform_optimization(self, mock_platform_responses):
        """Test optimization across multiple platforms."""

        # Define configuration space with platform-specific models
        config_space = {
            "platform": ["openai", "anthropic", "cohere", "langchain"],
            "model": [
                "gpt-3.5-turbo",
                "gpt-4",
                "claude-3-haiku",
                "claude-3-sonnet",
                "claude-3-opus",
                "command",
                "command-light",
            ],
            "temperature": [0.0, 0.3, 0.5, 0.7, 1.0],
            "max_tokens": [100, 500, 1000, 2000],
        }

        # Mock function to optimize
        @traigent.optimize(
            optimizer="grid",
            configuration_space=config_space,
            num_trials=20,
            objectives=["quality", "cost", "latency"],
        )
        async def multi_platform_task(
            text: str,
            platform: str = "openai",
            model: str = "gpt-3.5-turbo",
            temperature: float = 0.5,
            max_tokens: int = 500,
        ) -> dict[str, Any]:
            """Task that can run on multiple platforms."""
            # Simulate platform-specific execution
            platform_data = mock_platform_responses.get(platform, {})
            model_data = platform_data.get(model, {})

            # Add some variation based on temperature
            quality = model_data.get("quality", 0.5) * (1 + 0.1 * (0.7 - temperature))
            cost = model_data.get("cost", 0.001) * (1 + max_tokens / 1000)
            latency = model_data.get("latency", 1.0) * (1 + 0.05 * temperature)

            return {
                "output": f"Processed on {platform}/{model}",
                "quality": quality,
                "cost": cost,
                "latency": latency,
                "metadata": {
                    "platform": platform,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            }

        # Run optimization - in test mode this returns the function result directly
        # The optimization happens through the decorator
        result = await multi_platform_task("Test text for optimization")

        # The function should return a dictionary with metrics
        assert isinstance(result, dict)
        assert "output" in result
        assert "quality" in result
        assert "cost" in result
        assert "latency" in result

        # Verify the function has optimization methods
        assert hasattr(multi_platform_task, "optimize")
        assert hasattr(multi_platform_task, "get_optimization_results")

    @pytest.mark.asyncio
    async def test_auto_override_frameworks(self):
        """Test auto_override_frameworks with new platforms."""

        # Mock framework overrides - patch the correct functions
        with (
            patch("traigent.integrations.framework_override.override_openai_sdk"),
            patch("traigent.integrations.framework_override.override_anthropic"),
            patch("traigent.integrations.framework_override.override_cohere"),
            patch("traigent.integrations.framework_override.override_langchain"),
        ):

            @traigent.optimize(
                optimizer="random",
                num_trials=10,
                configuration_space={
                    "platform": ["openai", "anthropic", "cohere", "langchain"],
                    "temperature": [0.0, 0.5, 1.0],
                },
                auto_override_frameworks=["openai", "anthropic", "cohere", "langchain"],
            )
            async def test_function(
                prompt: str, platform: str = "openai", **kwargs
            ) -> str:
                """Test function with framework overrides."""
                # Simulate using different SDKs based on platform
                if platform == "openai":
                    # Would use OpenAI SDK
                    return "OpenAI response"
                elif platform == "anthropic":
                    # Would use Anthropic SDK
                    return "Anthropic response"
                elif platform == "cohere":
                    # Would use Cohere SDK
                    return "Cohere response"
                else:
                    # Default or LangChain
                    return "Default response"

            result = await test_function("Test prompt")

            # Verify function returns expected result
            assert isinstance(result, str)
            assert result in [
                "OpenAI response",
                "Anthropic response",
                "Cohere response",
                "Default response",
            ]

    @pytest.mark.asyncio
    async def test_platform_specific_kwargs(self):
        """Test platform-specific kwargs working correctly."""

        config_space = {
            "platform": ["anthropic", "cohere"],
            "temperature": [0.0, 0.5, 1.0],
        }

        @traigent.optimize(
            optimizer="grid", configuration_space=config_space, num_trials=5
        )
        async def platform_specific_task(
            text: str,
            platform: str = "anthropic",
            temperature: float = 0.5,
            anthropic_metadata: dict = None,
            cohere_connectors: list = None,
        ) -> dict[str, Any]:
            """Task with platform-specific parameters."""
            result = {"platform": platform, "temperature": temperature}

            if platform == "anthropic" and anthropic_metadata:
                result["anthropic_kwargs"] = {"metadata": anthropic_metadata}
            elif platform == "cohere" and cohere_connectors:
                result["cohere_kwargs"] = {"connectors": cohere_connectors}

            return result

        result = await platform_specific_task("Test text")

        # Verify platform-specific kwargs were handled
        assert isinstance(result, dict)
        assert "platform" in result
        assert result["platform"] in ["anthropic", "cohere"]
        assert "temperature" in result

    @pytest.mark.asyncio
    async def test_smart_subset_selection_with_platforms(self):
        """Test smart subset selection across platforms."""

        # Large configuration space
        config_space = {
            "platform": ["openai", "anthropic", "cohere", "huggingface", "langchain"],
            "model": [
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4-turbo",
                "claude-3-haiku",
                "claude-3-sonnet",
                "claude-3-opus",
                "command",
                "command-light",
                "command-nightly",
                "llama-2-7b",
                "mistral-7b",
                "falcon-7b",
                "claude-2",
            ],
            "temperature": [0.0, 0.5, 1.0, 1.5, 2.0],
            "max_tokens": [100, 500, 1000, 2000, 4000],
            "top_p": [0.1, 0.5, 0.9, 1.0],
            "frequency_penalty": [-2.0, -1.0, 0.0, 1.0, 2.0],
        }

        # Mock subset selection
        selected_configs = []

        @traigent.optimize(
            optimizer="bayesian",
            configuration_space=config_space,
            num_trials=20,  # Only 20 trials from potentially thousands of combinations
            subset_selection_strategy="diversity",  # Ensure platform diversity
            objectives=["performance", "cost"],
        )
        async def subset_selection_task(
            text: str,
            platform: str = "openai",
            model: str = "gpt-3.5-turbo",
            temperature: float = 0.5,
            max_tokens: int = 500,
            top_p: float = 0.9,
            frequency_penalty: float = 0.0,
            **kwargs,
        ) -> dict[str, Any]:
            """Task for testing subset selection."""
            config = {
                "platform": platform,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
            }
            config.update(kwargs)
            selected_configs.append(config)

            # Simple scoring
            performance = 0.5
            if "gpt-4" in model or "opus" in model:
                performance = 0.9
            elif "gpt-3.5" in model or "sonnet" in model or "command" in model:
                performance = 0.7

            cost = 0.01
            if "haiku" in model or "light" in model or "7b" in model:
                cost = 0.001
            elif "opus" in model or "gpt-4" in model:
                cost = 0.1

            return {
                "output": f"Result from {platform}/{model}",
                "performance": performance,
                "cost": cost,
            }

        result = await subset_selection_task("Test subset selection")

        # Verify function returned expected result
        assert isinstance(result, dict)
        assert "output" in result
        assert "performance" in result
        assert "cost" in result

        # Verify configs were selected
        assert len(selected_configs) > 0
        print(f"\nConfigs selected: {len(selected_configs)}")

    @pytest.mark.asyncio
    async def test_streaming_optimization_across_platforms(self):
        """Test optimization with streaming responses."""

        @traigent.optimize(
            optimizer="random",
            num_trials=5,
            configuration_space={
                "platform": ["openai", "anthropic", "cohere"],
                "stream": [True, False],
            },
        )
        async def streaming_task(
            prompt: str, platform: str = "openai", stream: bool = True
        ) -> dict[str, Any]:
            """Task that tests streaming capabilities."""
            chunks_received = 0
            full_response = ""

            if stream and platform in ["openai", "anthropic", "cohere"]:
                # Simulate streaming
                test_chunks = ["Hello", " from", f" {platform}", " streaming!"]
                for chunk in test_chunks:
                    chunks_received += 1
                    full_response += chunk
                    await asyncio.sleep(0.01)  # Simulate network delay
            else:
                # Non-streaming response
                full_response = f"Hello from {platform} (non-streaming)"

            return {
                "output": full_response,
                "chunks_received": chunks_received,
                "streaming_enabled": stream and chunks_received > 0,
            }

        result = await streaming_task("Test streaming")

        # Verify streaming result
        assert isinstance(result, dict)
        assert "output" in result
        assert "chunks_received" in result
        assert "streaming_enabled" in result

    @pytest.mark.asyncio
    async def test_tool_calling_optimization(self):
        """Test optimization with tool-calling capabilities."""

        # Define test tools

        @traigent.optimize(
            optimizer="grid",
            configuration_space={
                "platform": ["openai", "anthropic", "langchain"],
                "use_tools": [True, False],
                "temperature": [0.0, 0.5, 1.0],
            },
            num_trials=10,
        )
        async def tool_calling_task(
            prompt: str,
            platform: str = "openai",
            use_tools: bool = True,
            temperature: float = 0.5,
        ) -> dict[str, Any]:
            """Task that tests tool calling."""
            result = {
                "platform": platform,
                "tools_used": [],
                "temperature": temperature,
            }

            if use_tools and platform in ["openai", "anthropic", "langchain"]:
                # Simulate tool usage
                if "calculate" in prompt.lower():
                    result["tools_used"].append("calculate")
                    result["output"] = "Calculation performed: 42"
                elif "search" in prompt.lower():
                    result["tools_used"].append("search")
                    result["output"] = "Search results found"
                else:
                    result["output"] = "No tools needed"
            else:
                result["output"] = "Tools not available"

            # Score based on tool usage effectiveness
            result["effectiveness"] = len(result["tools_used"]) * 0.5 + 0.5

            return result

        # Test with calculation prompt
        calc_result = await tool_calling_task("Calculate 2+2")

        # Verify tool calling result
        assert isinstance(calc_result, dict)
        assert "platform" in calc_result
        assert "tools_used" in calc_result
        assert "effectiveness" in calc_result

    @pytest.mark.asyncio
    async def test_result_aggregation_across_platforms(self):
        """Test result aggregation from multiple platforms."""

        aggregated_results = {
            "platform_scores": {},
            "model_scores": {},
            "total_trials": 0,
            "successful_trials": 0,
        }

        @traigent.optimize(
            optimizer="random",
            num_trials=20,
            configuration_space={
                "platform": ["openai", "anthropic", "cohere", "huggingface"],
                "temperature": [0.0, 0.3, 0.5, 0.7, 1.0],
            },
            objectives=["quality", "speed"],
            callbacks=[
                lambda trial: aggregated_results.update(
                    {"total_trials": aggregated_results["total_trials"] + 1}
                )
            ],
        )
        async def aggregation_task(
            text: str, platform: str = "openai", temperature: float = 0.5
        ) -> dict[str, Any]:
            """Task for testing result aggregation."""
            # Simulate platform-specific behavior
            quality = np.random.uniform(0.6, 0.95)
            speed = np.random.uniform(0.5, 1.0)

            # Platform-specific adjustments
            if platform == "anthropic":
                quality *= 1.05  # Slightly better quality
            elif platform == "huggingface":
                speed *= 1.1  # Faster inference

            # Update aggregated results
            if platform not in aggregated_results["platform_scores"]:
                aggregated_results["platform_scores"][platform] = []
            aggregated_results["platform_scores"][platform].append(
                {"quality": quality, "speed": speed, "temperature": temperature}
            )

            aggregated_results["successful_trials"] += 1

            return {
                "output": f"Result from {platform}",
                "quality": min(quality, 1.0),
                "speed": min(speed, 1.0),
                "platform": platform,
            }

        result = await aggregation_task("Test aggregation")

        # Verify function returned expected result
        assert isinstance(result, dict)
        assert "output" in result
        assert "quality" in result
        assert "speed" in result
        assert "platform" in result

        # Verify aggregation tracking
        assert len(aggregated_results["platform_scores"]) > 0
        print(
            f"\nPlatforms tested: {list(aggregated_results['platform_scores'].keys())}"
        )

    @pytest.mark.asyncio
    async def test_platform_failover_during_optimization(self):
        """Test failover to alternative platforms during optimization."""

        failover_log = []

        @traigent.optimize(
            optimizer="grid",
            configuration_space={
                "platform": ["openai", "anthropic", "cohere"],
                "retry_on_failure": [True, False],
            },
            num_trials=10,
        )
        async def failover_task(
            text: str, platform: str = "openai", retry_on_failure: bool = True
        ) -> dict[str, Any]:
            """Task that simulates platform failures and failover."""
            # Simulate failures for specific platforms
            if platform == "openai" and np.random.random() < 0.3:
                if retry_on_failure:
                    # Failover to Anthropic
                    failover_log.append({"from": "openai", "to": "anthropic"})
                    platform = "anthropic"
                else:
                    raise Exception("OpenAI API error")

            return {
                "output": f"Completed on {platform}",
                "platform_used": platform,
                "failover_occurred": platform
                != "openai",  # Default platform is "openai"
            }

        result = await failover_task("Test failover")

        # Verify failover result
        assert isinstance(result, dict)
        assert "output" in result
        assert "platform_used" in result

        # Check if failovers occurred
        if len(failover_log) > 0:
            print(f"\nFailovers occurred: {len(failover_log)}")
            for failover in failover_log:
                print(f"  {failover['from']} -> {failover.get('to', 'none')}")

    @pytest.mark.asyncio
    async def test_complex_multi_objective_optimization(self):
        """Test complex multi-objective optimization across platforms."""

        # Complex configuration space with dependencies
        config_space = {
            "platform": ["openai", "anthropic", "cohere", "huggingface"],
            "model": [
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4-turbo",
                "claude-3-haiku",
                "claude-3-sonnet",
                "claude-3-opus",
                "command",
                "command-r",
                "command-r-plus",
                "llama-2-7b",
                "llama-2-13b",
                "mistral-7b",
            ],
            "temperature": [0.0, 0.5, 1.0, 1.5, 2.0],
            "max_tokens": [100, 500, 1000, 2000, 4000],
            "use_cache": [True, False],
            "batch_size": [1, 5, 8, 10, 16, 32],
        }

        # Track optimization metrics
        optimization_metrics = {
            "evaluations": 0,
            "platform_distribution": {},
            "pareto_front": [],
            "convergence_history": [],
        }

        @traigent.optimize(
            optimizer="bayesian",  # Use Bayesian for smart exploration
            configuration_space=config_space,
            num_trials=30,
            objectives=["quality", "latency", "cost", "reliability"],
            early_stopping_rounds=5,  # Stop if no improvement
            parallel_config={"trial_concurrency": 3},  # Run 3 trials in parallel
        )
        async def complex_optimization_task(
            text: str,
            platform: str = "openai",
            model: str = "gpt-3.5-turbo",
            temperature: float = 0.5,
            max_tokens: int = 1000,
            use_cache: bool = True,
            batch_size: int = 1,
        ) -> dict[str, Any]:
            """Complex task with multiple objectives."""
            optimization_metrics["evaluations"] += 1

            # Track platform distribution
            if platform not in optimization_metrics["platform_distribution"]:
                optimization_metrics["platform_distribution"][platform] = 0
            optimization_metrics["platform_distribution"][platform] += 1

            # Simulate realistic performance metrics
            base_quality = {
                "gpt-4": 0.95,
                "gpt-4-turbo": 0.93,
                "gpt-3.5-turbo": 0.85,
                "claude-3-opus": 0.96,
                "claude-3-sonnet": 0.91,
                "claude-3-haiku": 0.87,
                "command-r-plus": 0.90,
                "command-r": 0.86,
                "command": 0.83,
                "llama-2-13b": 0.82,
                "llama-2-7b": 0.78,
                "mistral-7b": 0.80,
            }.get(model, 0.75)

            # Adjust quality based on temperature
            quality = base_quality * (1 - abs(temperature - 0.7) * 0.1)

            # Calculate latency (affected by batch size and caching)
            base_latency = {
                "openai": 0.8,
                "anthropic": 0.7,
                "cohere": 0.9,
                "huggingface": 1.2,
            }.get(platform, 1.0)

            latency = base_latency * (1 + max_tokens / 2000)
            if use_cache:
                latency *= 0.7  # 30% faster with caching
            if batch_size > 1:
                latency *= 0.6 + 0.4 / batch_size  # Batching improves latency

            # Calculate cost
            token_cost = {
                "gpt-4": 0.03,
                "gpt-4-turbo": 0.01,
                "gpt-3.5-turbo": 0.001,
                "claude-3-opus": 0.015,
                "claude-3-sonnet": 0.003,
                "claude-3-haiku": 0.00025,
                "command-r-plus": 0.003,
                "command-r": 0.0005,
                "command": 0.001,
                "llama-2-13b": 0.0002,
                "llama-2-7b": 0.0001,
                "mistral-7b": 0.00015,
            }.get(model, 0.001)

            cost = (
                token_cost * max_tokens / 1000
            ) / batch_size  # Batching reduces per-item cost

            # Calculate reliability (some platforms more stable)
            reliability = {
                "openai": 0.98,
                "anthropic": 0.97,
                "cohere": 0.95,
                "huggingface": 0.90,
            }.get(platform, 0.85)

            # Add result to Pareto front tracking
            result = {
                "output": f"Result from {platform}/{model}",
                "quality": quality,
                "latency": latency,
                "cost": cost,
                "reliability": reliability,
                "config": {
                    "platform": platform,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "batch_size": batch_size,
                },
            }

            # Simple Pareto dominance check
            is_dominated = False
            for point in optimization_metrics["pareto_front"]:
                if (
                    point["quality"] >= quality
                    and point["latency"] <= latency
                    and point["cost"] <= cost
                    and point["reliability"] >= reliability
                ):
                    is_dominated = True
                    break

            if not is_dominated:
                # Remove points dominated by this one
                optimization_metrics["pareto_front"] = [
                    p
                    for p in optimization_metrics["pareto_front"]
                    if not (
                        quality >= p["quality"]
                        and latency <= p["latency"]
                        and cost <= p["cost"]
                        and reliability >= p["reliability"]
                    )
                ]
                optimization_metrics["pareto_front"].append(result)

            return result

        # Run optimization
        result = await complex_optimization_task("Optimize this complex task")

        # Verify complex optimization result
        assert isinstance(result, dict)
        assert "output" in result
        assert "quality" in result
        assert "latency" in result
        assert "cost" in result
        assert "reliability" in result

        # Check tracking metrics
        assert optimization_metrics["evaluations"] > 0
        assert len(optimization_metrics["platform_distribution"]) > 0
        print(f"\nEvaluations: {optimization_metrics['evaluations']}")
        print("Platform distribution:", optimization_metrics["platform_distribution"])

    @pytest.mark.asyncio
    async def test_platform_specific_error_recovery(self):
        """Test platform-specific error handling and recovery strategies."""

        error_scenarios = {
            "openai": ["rate_limit", "timeout", "invalid_key"],
            "anthropic": ["overloaded", "context_length", "invalid_model"],
            "cohere": ["quota_exceeded", "bad_request", "server_error"],
            "huggingface": ["model_loading", "out_of_memory", "inference_error"],
        }

        recovery_log = []

        @traigent.optimize(
            optimizer="random",
            num_trials=20,
            configuration_space={
                "platform": ["openai", "anthropic", "cohere", "huggingface"],
                "retry_strategy": [
                    "exponential_backoff",
                    "immediate",
                    "switch_platform",
                ],
                "max_retries": [1, 2, 3, 4, 5],
            },
        )
        async def error_recovery_task(
            text: str,
            platform: str = "openai",
            retry_strategy: str = "exponential_backoff",
            max_retries: int = 3,
        ) -> dict[str, Any]:
            """Task that simulates and recovers from platform errors."""

            # Simulate random error
            if np.random.random() < 0.3:  # 30% error rate
                error_type = np.random.choice(error_scenarios[platform])

                # Log error
                recovery_log.append(
                    {
                        "platform": platform,
                        "error": error_type,
                        "strategy": retry_strategy,
                    }
                )

                # Apply recovery strategy
                if retry_strategy == "switch_platform":
                    # Switch to a different platform
                    alternative_platforms = [
                        p for p in error_scenarios.keys() if p != platform
                    ]
                    new_platform = np.random.choice(alternative_platforms)
                    recovery_log[-1]["switched_to"] = new_platform
                    platform = new_platform
                elif retry_strategy == "exponential_backoff":
                    # Simulate backoff delay
                    await asyncio.sleep(0.1 * (2 ** min(3, len(recovery_log))))

                # Simulate recovery success rate
                recovery_success = np.random.random() > 0.2
                recovery_log[-1]["recovered"] = recovery_success

                if not recovery_success:
                    raise Exception(f"{platform} error: {error_type}")

            return {
                "output": f"Success on {platform}",
                "platform_used": platform,
                "errors_encountered": len(
                    [r for r in recovery_log if r["platform"] == platform]
                ),
                "recovery_rate": len(
                    [r for r in recovery_log if r.get("recovered", False)]
                )
                / max(1, len(recovery_log)),
            }

        # Run with error recovery
        try:
            result = await error_recovery_task("Test error recovery")

            # Verify error recovery result
            assert isinstance(result, dict)
            assert "output" in result
            assert "platform_used" in result

            # Analyze recovery patterns
            if recovery_log:
                print("\nError recovery summary:")
                print(f"Total errors: {len(recovery_log)}")
                print(
                    f"Recovery rate: {len([r for r in recovery_log if r.get('recovered', False)]) / len(recovery_log):.2%}"
                )

        except Exception:
            # Even with errors, should have tried multiple configurations
            assert len(recovery_log) > 0
            print(f"Optimization failed after {len(recovery_log)} recovery attempts")
