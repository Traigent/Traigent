#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false, reportUndefinedVariable=false, reportArgumentType=false
# mypy: ignore-errors
# flake8: noqa
# ruff: noqa
# pylint: disable=all

"""Example 6: Custom Evaluator - Define your own success metrics."""

import asyncio
import re

from _shared import add_repo_root_to_sys_path, dataset_path, ensure_dataset, init_mock_mode

add_repo_root_to_sys_path(__file__)

CODE_GEN_DATASET = dataset_path(__file__, "code_gen.jsonl")
ensure_dataset(
    CODE_GEN_DATASET,
    [
        {
            "input": {"task": "sum a list of numbers"},
            "expected_output": "def calculate_sum",
        },
        {
            "input": {"task": "sort an array"},
            "expected_output": "def sort_array",
        },
        {
            "input": {"task": "reverse a string"},
            "expected_output": "def reverse_string",
        },
    ],
)

import traigent

MOCK = init_mock_mode()


def custom_code_evaluator(output: str, expected: str) -> float:
    """
    Custom evaluator for code generation tasks.

    Evaluates based on:
    - Syntax correctness
    - Presence of required keywords
    - Code structure
    """
    score = 0.0

    # Check if output contains expected function/class names
    if "def " in expected and "def " in output:
        score += 0.3

    # Check for required imports
    if "import" in expected:
        expected_imports = re.findall(r"import \w+", expected)
        for imp in expected_imports:
            if imp in output:
                score += 0.2

    # Check for basic structure
    if output.strip() and "error" not in output.lower():
        score += 0.3

    # Check for documentation
    if '"""' in output or "#" in output:
        score += 0.2

    return min(score, 1.0)  # Cap at 1.0


@traigent.optimize(
    eval_dataset=str(CODE_GEN_DATASET),
    objectives=["accuracy", "cost"],
    custom_evaluator=custom_code_evaluator,  # Use our custom evaluator!
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.2, 0.5],
        "style": ["verbose", "concise", "documented"],
        "language": ["python", "javascript"],
    },
    execution_mode="edge_analytics",
)
def generate_code(task: str) -> str:
    """Generate code with custom evaluation."""

    config = traigent.get_config()
    style = config.get("style", "concise")
    language = config.get("language", "python")

    print(f"  Generating {style} {language} code...")

    if MOCK:
        # Generate mock code based on configuration
        if language == "python":
            if style == "verbose":
                return '''def calculate_sum(numbers):
    """
    Calculate the sum of a list of numbers.

    Args:
        numbers: List of numbers to sum

    Returns:
        The sum of all numbers
    """
    total = 0
    for number in numbers:
        total = total + number
    return total'''
            elif style == "documented":
                return '''def calculate_sum(numbers):
    """Calculate sum of numbers."""
    # Initialize sum
    total = 0
    # Add each number
    for n in numbers:
        total += n
    return total'''
            else:  # concise
                return "def calculate_sum(nums): return sum(nums)"
        else:  # javascript
            return (
                "function calculateSum(nums) { return nums.reduce((a,b) => a+b, 0); }"
            )

    # Real implementation
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=config.get("model", "gpt-3.5-turbo"),
        temperature=config.get("temperature", 0.2),
    )

    # Build prompt based on style
    style_instructions = {
        "verbose": "Include detailed comments and documentation",
        "concise": "Write compact, efficient code",
        "documented": "Add docstrings and inline comments",
    }

    prompt = f"""Generate {language} code for: {task}

Style: {style_instructions.get(style, '')}

Code:"""

    response = llm.invoke(prompt)
    return response.content


async def main():
    print("🎯 TraiGent Example 6: Custom Evaluator")
    print("=" * 50)
    print("🎯 Define your own success metrics\n")

    print("Custom Code Evaluator checks for:")
    print("  • Function/class definitions (30%)")
    print("  • Required imports (20%)")
    print("  • Basic structure (30%)")
    print("  • Documentation (20%)\n")

    print("This is more nuanced than simple string matching!\n")

    # Run optimization
    print("🔍 Testing different code generation styles...\n")
    results = await generate_code.optimize(
        algorithm="grid", max_trials=24 if not MOCK else 8  # Test all combinations
    )

    # Display results
    print("\n" + "=" * 50)
    print("🏆 BEST CODE GENERATION CONFIG:")
    print("=" * 50)

    best = results.best_config
    print(f"Model: {best.get('model')}")
    print(f"Temperature: {best.get('temperature')}")
    print(f"Style: {best.get('style')}")
    print(f"Language: {best.get('language')}")

    print(f"\n📊 Custom Evaluator Score: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"Cost: ${results.best_metrics.get('cost', 0):.6f}")

    print("\n💡 Benefits of custom evaluation:")
    print("  • Domain-specific quality metrics")
    print("  • Nuanced scoring beyond exact matches")
    print("  • Weighted importance of different aspects")
    print("  • Business logic validation")

    print("\n🎯 Create evaluators that match YOUR success criteria!")


if __name__ == "__main__":
    asyncio.run(main())
