#!/usr/bin/env python3
"""Example 6: Custom Evaluator with LLM-as-Judge - AI-powered code quality assessment.

This example demonstrates using an LLM as a judge to evaluate code generation quality.
The judge evaluates correctness, code quality, and documentation using a detailed rubric.

Usage (run in a terminal from repo root, works without activating venv):
    export OPENAI_API_KEY="your-key"  # pragma: allowlist secret
    .venv/bin/python walkthrough/real/06_custom_evaluator.py

If OPENAI_API_KEY is missing, this script exits with an error and suggests
running the mock walkthrough instead.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import maybe_run_mock_example

maybe_run_mock_example(__file__)

from langchain_openai import ChatOpenAI
from utils.helpers import (
    configure_logging,
    print_cost_estimate,
    print_estimated_time,
    print_optimization_config,
    print_results_table,
    sanitize_traigent_api_key,
)

import traigent

sanitize_traigent_api_key()
configure_logging()

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

traigent.initialize(execution_mode="edge_analytics")

# Dataset path relative to this file
DATASETS = Path(__file__).parent.parent / "datasets"
OBJECTIVES = ["accuracy", "cost"]
# Valid model names: https://models.litellm.ai/
CONFIG_SPACE = {
    "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
    "temperature": [0.0, 0.2, 0.5],
    "style": ["verbose", "concise", "documented"],
}

# LLM Judge for code evaluation - uses a smaller, cheaper model
_judge_llm = None
_DATASET_WARNING_FILTER_ADDED = False


def get_judge_llm() -> ChatOpenAI:
    """Get or create the judge LLM (singleton for efficiency)."""
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _judge_llm


def _suppress_code_gen_warning() -> None:
    global _DATASET_WARNING_FILTER_ADDED
    if _DATASET_WARNING_FILTER_ADDED:
        return
    base_logger = logging.getLogger("traigent.evaluators.base")

    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage()
            if "code_gen.jsonl" in message and "has no expected outputs" in message:
                return False
            return True

    base_logger.addFilter(_Filter())
    _DATASET_WARNING_FILTER_ADDED = True


LLM_JUDGE_PROMPT = """You are an expert code reviewer evaluating Python code quality.

## Task Description
The code was generated for this task: {task}

## Generated Code
```python
{code}
```

## Evaluation Rubric
Score each criterion from 0.0 to 1.0 and apply weights:

1. **Correctness (40%)**: Does the code solve the task correctly?
   - 1.0: Correct output for typical + edge cases
   - 0.7: Mostly correct, minor mistakes or missed edge cases
   - 0.4: Partially correct, important bugs present
   - 0.0: Incorrect or does not run

2. **Code Quality (30%)**: Is the code clear, structured, and Pythonic?
   - 1.0: Clean structure, readable naming, idiomatic Python
   - 0.7: Generally good, small improvements possible
   - 0.4: Works but messy or hard to follow
   - 0.0: Unstructured or very hard to read

3. **Documentation (30%)**: Is the code explained for a junior reader?
   - 1.0: Docstring + helpful comments (and optional type hints)
   - 0.7: Docstring or helpful comments present
   - 0.4: Minimal or unclear documentation
   - 0.0: No documentation

## Response Format
Respond with ONLY a JSON object (no markdown, no extra text):
{{"correctness": <score>, "quality": <score>, "documentation": <score>, "reasoning": "<1-2 sentences>"}}
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
    eval_dataset=str(DATASETS / "code_gen.jsonl"),
    objectives=OBJECTIVES,
    scoring_function=llm_code_evaluator,
    configuration_space=CONFIG_SPACE,
    injection_mode="context",  # default injection mode, added explicitly for clarity
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

    try:
        response = llm.invoke(prompt)
        return str(response.content)
    except Exception as exc:
        print(f"LLM call failed: {type(exc).__name__}: {exc}")
        return f"Error: {type(exc).__name__}: {exc}"


async def main() -> None:
    print("Traigent Example 6: LLM-as-Judge Custom Evaluator")
    print("=" * 55)
    print("Using GPT-4o-mini as a judge to evaluate code quality.")
    print("Scoring: Correctness (40%), Quality (30%), Documentation (30%).")
    print_optimization_config(OBJECTIVES, CONFIG_SPACE)
    print_cost_estimate(
        models=CONFIG_SPACE["model"],
        dataset_size=20,
        task_type="code_generation",
        num_trials=10,
    )

    _suppress_code_gen_warning()

    print_estimated_time("06_custom_evaluator.py")
    results = await generate_code.optimize(
        algorithm="random",
        max_trials=10,
        show_progress=True,
        random_seed=42,
    )

    print_results_table(results, CONFIG_SPACE, OBJECTIVES, is_mock=False)

    print("\nBest Configuration Found:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print(f"  Style: {results.best_config.get('style')}")

    print(f"\nLLM Judge Score: {results.best_metrics.get('accuracy', 0):.2%}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
