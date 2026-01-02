#!/usr/bin/env python3
"""Example 6: Custom Evaluator with LLM-as-Judge - AI-powered code quality assessment.

This example demonstrates using an LLM as a judge to evaluate code generation quality.
The judge evaluates correctness, style, and documentation using a detailed rubric.

Usage:
    export OPENAI_API_KEY="your-key"
    python 06_custom_evaluator.py
"""

import asyncio
import json

from langchain_openai import ChatOpenAI

import traigent

# LLM Judge for code evaluation - uses a smaller, cheaper model
_judge_llm = None


def get_judge_llm() -> ChatOpenAI:
    """Get or create the judge LLM (singleton for efficiency)."""
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _judge_llm


LLM_JUDGE_PROMPT = """You are an expert code reviewer evaluating Python code quality.

## Task Description
The code was generated for this task: {task}

## Generated Code
```python
{code}
```

## Evaluation Rubric
Score the code on these criteria (0.0 to 1.0 each):

1. **Correctness (40%)**: Does the code correctly solve the task?
   - 1.0: Fully correct, handles edge cases
   - 0.7: Mostly correct, minor issues
   - 0.4: Partially correct, has bugs
   - 0.0: Incorrect or doesn't compile

2. **Code Quality (30%)**: Is the code well-structured and Pythonic?
   - 1.0: Clean, efficient, follows best practices
   - 0.7: Good structure, minor improvements possible
   - 0.4: Works but poorly structured
   - 0.0: Messy, hard to understand

3. **Documentation (30%)**: Is the code well-documented?
   - 1.0: Has docstring, clear comments, type hints
   - 0.7: Has docstring or good comments
   - 0.4: Minimal documentation
   - 0.0: No documentation

## Response Format
Respond with ONLY a JSON object (no markdown, no explanation):
{{"correctness": <score>, "quality": <score>, "documentation": <score>, "reasoning": "<brief explanation>"}}
"""


def llm_code_evaluator(output: str, expected: str, **kwargs) -> float:
    """LLM-as-Judge evaluator for code generation quality.

    Uses GPT-4o-mini to evaluate code on correctness, quality, and documentation.
    Falls back to heuristic evaluation if LLM fails.
    """
    # Get the task from kwargs (passed by Traigent)
    task = kwargs.get("input_data", {}).get("task", "unknown task")

    try:
        judge = get_judge_llm()
        prompt = LLM_JUDGE_PROMPT.format(task=task, code=output)
        response = judge.invoke(prompt)

        # Parse JSON response
        result = json.loads(str(response.content).strip())

        # Calculate weighted score
        correctness = float(result.get("correctness", 0))
        quality = float(result.get("quality", 0))
        documentation = float(result.get("documentation", 0))

        weighted_score = (correctness * 0.4) + (quality * 0.3) + (documentation * 0.3)
        return min(max(weighted_score, 0.0), 1.0)

    except Exception:
        # Fallback to simple heuristic if LLM evaluation fails
        score = 0.0
        if "def " in output or "class " in output:
            score += 0.4
        if output.strip() and "error" not in output.lower():
            score += 0.3
        if '"""' in output or "# " in output:
            score += 0.3
        return min(score, 1.0)


@traigent.optimize(
    eval_dataset="./code_gen.jsonl",
    objectives=["accuracy", "cost"],
    custom_evaluator=llm_code_evaluator,
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.2, 0.5],
        "style": ["verbose", "concise", "documented"],
    },
    execution_mode="edge_analytics",
)
def generate_code(task: str) -> str:
    """Generate code with configurable style."""
    config = traigent.get_config()
    style = config.get("style", "concise")

    style_instructions = {
        "verbose": "Include detailed comments explaining each step",
        "concise": "Write minimal, efficient code without comments",
        "documented": "Add comprehensive docstrings and type hints",
    }

    llm = ChatOpenAI(
        model=config.get("model", "gpt-3.5-turbo"),
        temperature=config.get("temperature", 0.2),
    )

    prompt = f"""Write Python code for: {task}

Requirements:
- {style_instructions[style]}
- Include proper error handling where appropriate
- Use descriptive variable names

Return ONLY the Python code, no explanations."""

    response = llm.invoke(prompt)
    return str(response.content)


async def main() -> None:
    print("Traigent Example 6: LLM-as-Judge Custom Evaluator")
    print("=" * 55)
    print("Using GPT-4o-mini as a judge to evaluate code quality.")
    print("Scoring: Correctness (40%), Quality (30%), Docs (30%).\n")

    results = await generate_code.optimize(algorithm="random", max_trials=10, random_seed=42)

    print("\nBest Code Generation Config:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print(f"  Style: {results.best_config.get('style')}")

    print(f"\nLLM Judge Score: {results.best_metrics.get('accuracy', 0):.2%}")


if __name__ == "__main__":
    asyncio.run(main())
