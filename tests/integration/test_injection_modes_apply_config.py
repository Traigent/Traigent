#!/usr/bin/env python3
"""
Test script to verify apply_best_config works correctly with Seamless, Parameter, and Context injection modes.

Note: Attribute injection mode was removed in v2.x due to thread-safety issues.
"""

from datetime import datetime

from traigent import optimize
from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.config.types import TraigentConfig


# Test function for Seamless mode
@optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],
    },
    injection_mode="seamless",  # or just omit since it's the default
    objectives=["accuracy"],
)
def seamless_qa_agent(question: str) -> str:
    """Q&A agent using seamless injection - Tuned Variables defined in function body."""
    # These will be overridden by Traigent during optimization
    model = "gpt-3.5-turbo"  # Will be replaced with values from config space
    temperature = 0.7  # Will be replaced with values from config space

    # Simulate different model behaviors
    if model == "gpt-4o":
        quality = "excellent"
    elif model == "gpt-4o-mini":
        quality = "very good"
    else:
        quality = "good"

    temp_desc = (
        "creative"
        if temperature > 0.7
        else "balanced" if temperature > 0.3 else "focused"
    )

    return f"[{model}|temp={temperature}|{temp_desc}|{quality}] Answer to: {question}"


# Test function for Parameter mode
@optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],
    },
    injection_mode="parameter",
    config_param="config",
    objectives=["accuracy"],
)
def parameter_qa_agent(question: str, config: TraigentConfig) -> str:
    """Q&A agent using parameter injection - config passed explicitly."""
    # Get Tuned Variables from config parameter
    model = config.get("model", "gpt-3.5-turbo")
    temperature = config.get("temperature", 0.7)

    # Simulate different model behaviors
    if model == "gpt-4o":
        quality = "excellent"
    elif model == "gpt-4o-mini":
        quality = "very good"
    else:
        quality = "good"

    temp_desc = (
        "creative"
        if temperature > 0.7
        else "balanced" if temperature > 0.3 else "focused"
    )

    return f"[{model}|temp={temperature}|{temp_desc}|{quality}] Answer to: {question}"


def create_mock_optimization_result():
    """Create a mock optimization result for testing."""
    best_config = {"model": "gpt-4o", "temperature": 0.1}

    trials = [
        TrialResult(
            trial_id="trial_1",
            config={"model": "gpt-3.5-turbo", "temperature": 0.5},
            metrics={"accuracy": 0.75},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
            metadata={},
        ),
        TrialResult(
            trial_id="trial_2",
            config={"model": "gpt-4o-mini", "temperature": 0.9},
            metrics={"accuracy": 0.82},
            status=TrialStatus.COMPLETED,
            duration=1.2,
            timestamp=datetime.now(),
            metadata={},
        ),
        TrialResult(
            trial_id="trial_3",
            config=best_config,
            metrics={"accuracy": 0.95},
            status=TrialStatus.COMPLETED,
            duration=1.5,
            timestamp=datetime.now(),
            metadata={},
        ),
    ]

    return OptimizationResult(
        trials=trials,
        best_config=best_config,
        best_score=0.95,
        optimization_id="test_opt",
        duration=5.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="test",
        timestamp=datetime.now(),
        metadata={},
    )


def test_seamless_mode():
    """Test apply_best_config with Seamless injection mode."""
    print("\n=== Testing Seamless Mode ===")

    # Before optimization - uses default values
    print("\n1. Before optimization (default values):")
    result = seamless_qa_agent("What is AI?")
    print(f"   Result: {result}")

    # Simulate optimization results
    mock_results = create_mock_optimization_result()
    seamless_qa_agent._optimization_results = mock_results

    # Apply best config
    print("\n2. Applying best config...")
    success = seamless_qa_agent.apply_best_config()
    print(f"   Apply success: {success}")
    print(f"   Best config applied: {seamless_qa_agent._current_config}")

    # After applying best config - should use optimized values
    print("\n3. After apply_best_config (optimized values):")
    result = seamless_qa_agent("What is AI?")
    print(f"   Result: {result}")

    # Verify the result contains optimized parameters
    assert "gpt-4o" in result, "Seamless mode should use optimized model"
    assert "temp=0.1" in result, "Seamless mode should use optimized temperature"
    assert "focused" in result, "Low temperature should be 'focused'"
    assert "excellent" in result, "gpt-4o should have 'excellent' quality"

    print("\n✅ Seamless mode test passed!")


def test_parameter_mode():
    """Test apply_best_config with Parameter injection mode."""
    print("\n=== Testing Parameter Mode ===")

    # Before optimization - uses default values
    print("\n1. Before optimization (default values):")
    result = parameter_qa_agent("What is ML?")
    print(f"   Result: {result}")

    # Simulate optimization results
    mock_results = create_mock_optimization_result()
    parameter_qa_agent._optimization_results = mock_results

    # Apply best config
    print("\n2. Applying best config...")
    success = parameter_qa_agent.apply_best_config()
    print(f"   Apply success: {success}")
    print(f"   Best config applied: {parameter_qa_agent._current_config}")

    # After applying best config - should use optimized values
    print("\n3. After apply_best_config (optimized values):")
    result = parameter_qa_agent("What is ML?")
    print(f"   Result: {result}")

    # Verify the result contains optimized parameters
    assert "gpt-4o" in result, "Parameter mode should use optimized model"
    assert "temp=0.1" in result, "Parameter mode should use optimized temperature"
    assert "focused" in result, "Low temperature should be 'focused'"
    assert "excellent" in result, "gpt-4o should have 'excellent' quality"

    print("\n✅ Parameter mode test passed!")


# Test function for Context mode
@optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],
    },
    injection_mode="context",
    objectives=["accuracy"],
)
def context_qa_agent(question: str) -> str:
    """Q&A agent using context injection - config passed via context variables."""
    # Get Tuned Variables from context
    from traigent.config.context import get_config

    config = get_config()

    if config:
        model = config.get("model", "gpt-3.5-turbo")
        temperature = config.get("temperature", 0.7)
    else:
        model = "gpt-3.5-turbo"
        temperature = 0.7

    # Simulate different model behaviors
    if model == "gpt-4o":
        quality = "excellent"
    elif model == "gpt-4o-mini":
        quality = "very good"
    else:
        quality = "good"

    temp_desc = (
        "creative"
        if temperature > 0.7
        else "balanced" if temperature > 0.3 else "focused"
    )

    return f"[{model}|temp={temperature}|{temp_desc}|{quality}] Answer to: {question}"


def test_context_mode():
    """Test apply_best_config with Context injection mode."""
    print("\n=== Testing Context Mode ===")

    # Before optimization - uses default values
    print("\n1. Before optimization (default values):")
    result = context_qa_agent("What is NLP?")
    print(f"   Result: {result}")

    # Simulate optimization results
    mock_results = create_mock_optimization_result()
    context_qa_agent._optimization_results = mock_results

    # Apply best config
    print("\n2. Applying best config...")
    success = context_qa_agent.apply_best_config()
    print(f"   Apply success: {success}")
    print(f"   Best config applied: {context_qa_agent._current_config}")

    # After applying best config - should use optimized values
    print("\n3. After apply_best_config (optimized values):")
    result = context_qa_agent("What is NLP?")
    print(f"   Result: {result}")

    # Verify the result contains optimized parameters
    assert "gpt-4o" in result, "Context mode should use optimized model"
    assert "temp=0.1" in result, "Context mode should use optimized temperature"
    assert "focused" in result, "Low temperature should be 'focused'"
    assert "excellent" in result, "gpt-4o should have 'excellent' quality"

    print("\n✅ Context mode test passed!")


def main():
    """Run all tests."""
    print("🚀 Testing apply_best_config with different injection modes")
    print("=" * 60)

    try:
        # Test Seamless mode
        test_seamless_mode()

        # Test Parameter mode
        test_parameter_mode()

        # Test Context mode
        test_context_mode()

        print("\n" + "=" * 60)
        print(
            "🎉 All tests passed! apply_best_config works correctly with all 3 injection modes."
        )
        print("\nKey findings:")
        print(
            "1. Seamless mode: Traigent intercepts variable assignments in the function"
        )
        print("2. Parameter mode: Traigent passes config as explicit parameter")
        print(
            "3. Context mode: Traigent uses Python's contextvars for config injection"
        )
        print("4. All modes correctly apply the best config after optimization")
        print(
            "5. The _setup_function_wrapper method properly handles all injection strategies"
        )

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
