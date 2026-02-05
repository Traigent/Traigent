#!/usr/bin/env python3
"""Math Q&A example for CI/CD integration - exact match arithmetic evaluation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


try:
    import traigent
except ImportError:  # pragma: no cover - support IDE execution paths
    import importlib

    module_path = Path(__file__).resolve()
    for depth in (2, 3):
        try:
            sys.path.append(str(module_path.parents[depth]))
        except IndexError:
            continue
    traigent = importlib.import_module("traigent")

# Dataset file path
DATASET_FILE = os.path.join(os.path.dirname(__file__), "math_qa.jsonl")


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.1, 0.3],
        "prompt_style": ["direct", "step-by-step", "chain-of-thought"],
        "max_tokens": [50, 100, 150],
    },
    eval_dataset=DATASET_FILE,
    objectives=["accuracy", "cost", "response_time"],
    execution_mode="edge_analytics",
    max_trials=10,
)
def solve_arithmetic(expression: str) -> str:
    """Solve arithmetic expressions and return the numeric result as a string.

    Args:
        expression: Arithmetic expression like "2 + 3 * 4"

    Returns:
        String representation of the numeric answer
    """
    # Get configuration - works both during optimization and normal calls
    try:
        current = traigent.get_trial_config()
        config = current if isinstance(current, dict) else {}
    except traigent.utils.exceptions.OptimizationStateError:
        # Not in an optimization trial - use applied config or defaults
        config = getattr(solve_arithmetic, "current_config", {}) or {}

    # Map prompt styles to system prompts
    prompt_styles = {
        "direct": "Calculate the following expression and return only the numeric answer:",
        "step-by-step": "Solve this step by step, then give the final numeric answer:",
        "chain-of-thought": "Think through this problem carefully. Show your work, then provide the final answer as a number:",
    }

    prompt_template = prompt_styles.get(config.get("prompt_style", "direct"))

    # In mock mode, return calculated answer
    if os.environ.get("TRAIGENT_MOCK_LLM") == "true":
        try:
            # Safe evaluation for simple arithmetic
            import ast
            import operator as op

            # Supported operators
            ops = {
                ast.Add: op.add,
                ast.Sub: op.sub,
                ast.Mult: op.mul,
                ast.Div: op.truediv,
                ast.Mod: op.mod,
                ast.Pow: op.pow,
                ast.USub: op.neg,
            }

            def eval_expr(node):
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.BinOp):
                    return ops[type(node.op)](
                        eval_expr(node.left), eval_expr(node.right)
                    )
                elif isinstance(node, ast.UnaryOp):
                    return ops[type(node.op)](eval_expr(node.operand))
                else:
                    raise ValueError(f"Unsupported expression: {node}")

            result = eval_expr(ast.parse(expression, mode="eval").body)
            # Return as integer if whole number, else float
            if isinstance(result, float) and result.is_integer():
                return str(int(result))
            return str(result)
        except Exception:
            # Fallback for mock mode
            return "42"

    # Production mode would use LLM here
    from langchain.schema import HumanMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=config.get("model", "gpt-3.5-turbo"),
        temperature=config.get("temperature", 0.0),
        max_tokens=config.get("max_tokens", 50),
    )

    full_prompt = f"{prompt_template}\n\n{expression}"
    response = llm.invoke([HumanMessage(content=full_prompt)])

    # Extract numeric answer from response
    answer = str(response.content).strip()
    # Try to extract just the number if there's extra text
    import re

    match = re.search(r"-?\d+\.?\d*", answer)
    if match:
        return match.group()
    return answer


if __name__ == "__main__":
    # Test the function
    print("Testing solve_arithmetic function...")
    test_expr = "2 + 3 * 4"
    result = solve_arithmetic(test_expr)
    print(f"{test_expr} = {result}")
