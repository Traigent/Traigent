#!/usr/bin/env python3
"""
Product & Technical Agent - Code Generation

This agent generates Python functions based on specifications and optimizes
for correctness (test pass rate) and code quality.

Usage:
    export TRAIGENT_MOCK_MODE=true
    python use-cases/product-technical/agent/code_agent.py
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import evaluator from sibling directory
import importlib.util

import traigent
from traigent.api.decorators import EvaluationOptions, ExecutionOptions

_evaluator_path = Path(__file__).parent.parent / "eval" / "evaluator.py"
_spec = importlib.util.spec_from_file_location("code_evaluator", _evaluator_path)
_evaluator_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluator_module)
CodeEvaluator = _evaluator_module.CodeEvaluator

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "coding_tasks.jsonl"


CODE_GENERATION_PROMPT = """You are an expert Python programmer. Generate a function based on the following specification.

Task: {task}
Function Name: {function_name}
Signature: {signature}

Requirements:
- Write clean, efficient Python code
- Handle edge cases appropriately
- Follow Python best practices
- Style: {coding_style}

{approach_instruction}

Return ONLY the function code, no explanations or markdown:"""


def is_mock_mode() -> bool:
    """Check if mock mode is enabled via environment variable."""
    return os.environ.get("TRAIGENT_MOCK_MODE", "").lower() in ("true", "1", "yes")


def get_approach_instruction(approach: str) -> str:
    """Get instruction based on coding approach."""
    if approach == "test_first":
        return """Think about test cases first:
- What are the normal cases?
- What are the edge cases?
- What should happen with invalid input?
Then implement a solution that handles all these cases."""
    else:  # direct
        return "Implement a straightforward solution for the task."


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.2, 0.4],
        "coding_style": ["concise", "verbose", "documented"],
        "approach": ["direct", "test_first"],
    },
    objectives=["test_pass_rate", "code_quality", "efficiency", "cost"],
    evaluation=EvaluationOptions(
        eval_dataset=str(DATASET_PATH),
        # CodeEvaluator has scoring_function interface: (prediction, expected, input_data) -> dict
        scoring_function=CodeEvaluator(),
    ),
    execution=ExecutionOptions(execution_mode="edge_analytics"),
)
def code_generation_agent(
    task: str,
    function_name: str,
    signature: str,
) -> dict[str, Any]:
    """
    Generate Python code based on a specification.

    Args:
        task: Description of what the function should do
        function_name: Name of the function to generate
        signature: Function signature including types

    Returns:
        Dictionary with 'code', 'function_name', and metadata
    """
    # Get current configuration
    config = traigent.get_config()

    # Extract tuned variables with defaults
    model = config.get("model", "gpt-4o-mini")
    temperature = config.get("temperature", 0.2)
    coding_style = config.get("coding_style", "concise")
    approach = config.get("approach", "direct")

    # Get approach instruction
    approach_instruction = get_approach_instruction(approach)

    # Build the prompt
    prompt = CODE_GENERATION_PROMPT.format(
        task=task,
        function_name=function_name,
        signature=signature,
        coding_style=coding_style,
        approach_instruction=approach_instruction,
    )

    if is_mock_mode():
        return generate_mock_code(task, function_name, signature)

    # Use LangChain for LLM call
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required when TRAIGENT_MOCK_MODE is disabled. "
            "Install it with: pip install langchain-openai"
        ) from exc

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
    )
    response = llm.invoke(prompt)
    code = extract_code(response.content)

    return {
        "code": code,
        "function_name": function_name,
        "model": model,
        "temperature": temperature,
        "style": coding_style,
        "approach": approach,
    }


def extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
    # Remove markdown code blocks if present
    if "```python" in response:
        start = response.find("```python") + 9
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()

    if "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()

    # Return as-is if no code blocks
    return response.strip()


def generate_mock_code(
    task: str,
    function_name: str,
    signature: str,
) -> dict[str, Any]:
    """Generate mock code for testing without LLM."""
    # Simple mock implementations for common tasks
    mock_implementations = {
        "is_prime": """def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True""",
        "factorial": """def factorial(n: int) -> int:
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result""",
        "reverse_string": """def reverse_string(s: str) -> str:
    return s[::-1]""",
        "fibonacci": """def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b""",
    }

    # Return mock implementation if available, otherwise generic
    code = mock_implementations.get(
        function_name,
        f'''{signature}:
    """Mock implementation for {task}"""
    pass  # TODO: Implement''',
    )

    return {
        "code": code,
        "function_name": function_name,
        "model": "mock",
        "temperature": 0.0,
        "style": "mock",
        "approach": "mock",
    }


async def run_optimization():
    """Run the code generation agent optimization."""
    print("=" * 60)
    print("Product & Technical Agent - Traigent Optimization")
    print("=" * 60)

    # Check if mock mode is enabled
    mock_mode = is_mock_mode()
    print(f"\nMock Mode: {'Enabled' if mock_mode else 'Disabled'}")

    if not mock_mode:
        print("\nWARNING: Running without mock mode will incur API costs!")
        print("Set TRAIGENT_MOCK_MODE=true for testing.\n")

    print("\nStarting optimization...")
    print("Configuration Space:")
    print("  - Models: gpt-3.5-turbo, gpt-4o-mini, gpt-4o")
    print("  - Temperature: 0.0, 0.2, 0.4")
    print("  - Coding Style: concise, verbose, documented")
    print("  - Approach: direct, test_first")
    print("\nObjectives: test_pass_rate, code_quality, efficiency, cost")
    print("-" * 60)

    # Run optimization
    results = await code_generation_agent.optimize(
        algorithm="random",
        max_trials=20,
    )

    # Display results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print("\nBest Configuration:")
    for key, value in results.best_config.items():
        print(f"  {key}: {value}")
    print(f"\nBest Score: {results.best_score:.4f}")

    # Apply best config
    code_generation_agent.apply_best_config(results)
    print("\nBest configuration applied!")

    # Test with sample task
    print("\n" + "-" * 60)
    print("Testing optimized agent with sample task...")
    print("-" * 60)

    result = code_generation_agent(
        task="Write a function that checks if a number is prime",
        function_name="is_prime",
        signature="def is_prime(n: int) -> bool",
    )

    print(f"\nGenerated Code:\n{result['code']}")
    print("\nMetadata:")
    print(f"  Model: {result['model']}")
    print(f"  Style: {result['style']}")
    print(f"  Approach: {result['approach']}")

    return results


def main():
    """Main entry point."""
    asyncio.run(run_optimization())


if __name__ == "__main__":
    main()
