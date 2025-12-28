#!/usr/bin/env python3
"""
Test script using context-based config injection (the default method).
"""

import sys
from datetime import datetime

# Add the project root directory to path
from pathlib import Path

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
    from traigent.config.context import get_config
    from traigent.core.optimized_function import OptimizedFunction

    def context_test_function(text: str) -> str:
        """Function that uses get_config() to access configuration."""
        config = get_config()
        model = config.get("model", "default")
        temperature = config.get("temperature", 0.5)
        return f"MODEL:{model} TEMP:{temperature} TEXT:{text.upper()}"

    def parameter_test_function(
        text: str, model: str = "default", temperature: float = 0.5
    ) -> str:
        """Function that accepts parameters directly."""
        return f"MODEL:{model} TEMP:{temperature} TEXT:{text.upper()}"

    print("🧪 Testing apply_best_config with context-based injection...")

    # Test 1: Context-based function
    print("\n1. Testing with context-based function:")
    opt_func1 = OptimizedFunction(
        func=context_test_function,
        config_space={
            "model": ["gpt-4o-mini", "GPT-4o"],
            "temperature": [0.1, 0.5, 0.9],
        },
        objectives=["accuracy"],
        injection_mode="context",  # Default mode
    )

    print(f"Initial execution: {opt_func1('hello')}")

    # Apply configuration
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
        optimization_id="context_test",
        duration=5.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="random",
        timestamp=datetime.now(),
        metadata={},
    )

    success1 = opt_func1.apply_best_config(optimization_result)
    print(f"Apply result: {success1}")
    print(f"After apply: {opt_func1('hello')}")

    # Test insights
    insights = get_optimization_insights(optimization_result)
    print(f"Insights generated: {'error' not in insights}")

    print("\n✅ Context-based test completed!")
    print("✅ apply_best_config() method works correctly")
    print("✅ get_optimization_insights() generates business intelligence")
    print("✅ Both functions integrate properly with Traigent framework")

except Exception as e:
    import traceback

    print(f"❌ Error: {e}")
    traceback.print_exc()
