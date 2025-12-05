#!/usr/bin/env python3
"""Debug evaluation to see what's happening with costs."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

os.environ["TRAIGENT_REAL_MODE"] = "true"

# Create a mock output similar to what LangChain returns
mock_output = {
    "input_tokens": 9,
    "output_tokens": 2,
    "total_tokens": 11,
    # This is what we typically get from the function
    "result": "positive",
}

# Import the evaluator components
from traigent.evaluators.metrics_tracker import extract_llm_metrics

print("Testing metrics extraction with mock LLM output")
print(f"Mock output: {mock_output}")

# This simulates what happens during evaluation
model = "claude-3-haiku-20240307"
metrics = extract_llm_metrics(
    response=mock_output,
    model_name=model,
    original_prompt="Test prompt",
    response_text="positive",
)

print("\nExtracted metrics:")
print(f"  Input tokens: {metrics.tokens.input_tokens}")
print(f"  Output tokens: {metrics.tokens.output_tokens}")
print(f"  Input cost: ${metrics.cost.input_cost:.8f}")
print(f"  Output cost: ${metrics.cost.output_cost:.8f}")
print(f"  Total cost: ${metrics.cost.total_cost:.8f}")

# Now test what happens when we evaluate a function that returns this
print("\n" + "=" * 60)
print("Simulating actual evaluation flow:")

from traigent.core.types import ExampleResult
from traigent.evaluators.local import LocalEvaluator

# Create a simple evaluator
evaluator = LocalEvaluator(detailed=True)

# Simulate what the evaluator does
outputs = ["positive"]
errors = [None]

# This is like what happens in _evaluate_batch
example_results = [
    ExampleResult(
        input_data={"text": "test"},
        output="positive",
        expected_output="positive",
        success=True,
        execution_time=0.5,
        metrics={},
    )
]

# The evaluator should extract metrics from the output
# In real mode, the output might be a dict with token info
# Let's simulate that
outputs_with_tokens = [mock_output]

print(f"Output with tokens: {outputs_with_tokens[0]}")

# Extract metrics like the evaluator does
extracted = extract_llm_metrics(
    response=outputs_with_tokens[0],
    model_name=model,
    original_prompt="Test",
    response_text="positive",
)

print("\nFinal extracted metrics:")
print(
    f"  Tokens: input={extracted.tokens.input_tokens}, output={extracted.tokens.output_tokens}"
)
print(
    f"  Costs: input=${extracted.cost.input_cost:.8f}, output=${extracted.cost.output_cost:.8f}"
)
print(f"  Total cost: ${extracted.cost.total_cost:.8f}")

# This is what should be in the final results
if extracted.cost.total_cost > 0:
    example_results[0].metrics["input_cost"] = extracted.cost.input_cost
    example_results[0].metrics["output_cost"] = extracted.cost.output_cost
    example_results[0].metrics["total_cost"] = extracted.cost.total_cost

print(f"\nFinal example result metrics: {example_results[0].metrics}")
