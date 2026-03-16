#!/usr/bin/env python3
"""
Debug script to understand how config application works in OptimizedFunction.
"""

import sys
from datetime import datetime

sys.path.insert(0, ".")

try:
    from traigent.api.types import (
        OptimizationResult,
        OptimizationStatus,
        TrialResult,
        TrialStatus,
    )
    from traigent.core.optimized_function import OptimizedFunction

    def test_function(
        text: str, model: str = "default", temperature: float = 0.5
    ) -> str:
        return f"MODEL:{model} TEMP:{temperature} TEXT:{text.upper()}"

    print("🔍 Debugging OptimizedFunction config application...")

    # Create optimized function with parameter injection
    opt_func = OptimizedFunction(
        func=test_function,
        config_space={
            "model": ["gpt-4o-mini", "GPT-4o"],
            "temperature": [0.1, 0.5, 0.9],
        },
        objectives=["accuracy"],
        injection_mode="parameter",  # Use parameter injection instead of context
    )

    print(f"Initial config: {opt_func._current_config}")

    # Test function with initial config
    result1 = opt_func("hello")
    print(f"Initial execution: {result1}")

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
        optimization_id="debug",
        duration=5.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="random",
        timestamp=datetime.now(),
        metadata={},
    )

    # Apply configuration
    print(f"\nApplying config: {optimization_result.best_config}")
    success = opt_func.apply_best_config(optimization_result)
    print(f"Apply result: {success}")
    print(f"New config: {opt_func._current_config}")

    # Test function with new config
    result2 = opt_func("hello")
    print(f"After apply execution: {result2}")

    # Test direct function call to see expected output
    direct_result = test_function("hello", model="GPT-4o", temperature=0.1)
    print(f"Direct call: {direct_result}")

    # Check if the function wrapper is working
    print(f"\nOptimized function type: {type(opt_func.func)}")
    print(f"Original function type: {type(test_function)}")
    print(f"Wrapped function type: {type(opt_func._wrapped_func)}")
    print(f"Provider type: {type(opt_func._provider)}")

    # Test wrapped function directly
    wrapped_result = opt_func._wrapped_func("hello")
    print(f"Wrapped function direct call: {wrapped_result}")

    # Check injection mode
    print(f"Injection mode: {opt_func.injection_mode}")
    print(f"Current config: {opt_func._current_config}")

except Exception as e:
    import traceback

    print(f"Error: {e}")
    traceback.print_exc()
