#!/usr/bin/env python3
"""Example 2: Zero Code Change - Seamless mode intercepts hardcoded values.

Usage (run in a terminal from repo root, works without activating venv):
    export OPENAI_API_KEY="your-key"  # pragma: allowlist secret
    .venv/bin/python walkthrough/real/02_zero_code_change.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from utils.helpers import (
    configure_logging,
    print_cost_estimate,
    print_estimated_time,
    print_optimization_config,
    print_results_table,
    require_openai_key,
    sanitize_traigent_api_key,
)
from utils.scoring import token_match_score

import traigent

require_openai_key("02_zero_code_change.py")
sanitize_traigent_api_key()
configure_logging()

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

traigent.initialize(execution_mode="edge_analytics")

# Dataset path relative to this file
DATASETS = Path(__file__).parent.parent / "datasets"
OBJECTIVES = ["accuracy", "cost"]
CONFIG_SPACE = {
    "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
    "temperature": [0.1, 0.5, 0.9],
}


@traigent.optimize(
    eval_dataset=str(DATASETS / "simple_questions.jsonl"),
    objectives=OBJECTIVES,
    scoring_function=token_match_score,
    configuration_space=CONFIG_SPACE,
    injection_mode="seamless",
    execution_mode="edge_analytics",
)
def answer_question(question: str) -> str:
    """Your existing code - Traigent overrides the hardcoded values below."""
    # These hardcoded values will be overridden by Traigent!
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    try:
        response = llm.invoke(f"Answer: {question}")
        return str(response.content)
    except Exception as exc:
        print(f"LLM call failed: {type(exc).__name__}: {exc}")
        return f"Error: {type(exc).__name__}: {exc}"


async def main() -> None:
    print("Traigent Example 2: Zero Code Change")
    print("=" * 50)
    print("Seamless mode overrides hardcoded LLM parameters.")
    print_optimization_config(OBJECTIVES, CONFIG_SPACE)
    print_cost_estimate(
        models=CONFIG_SPACE["model"],
        dataset_size=20,
        task_type="simple_qa",
        num_trials=10,
    )

    print_estimated_time("02_zero_code_change.py")
    results = await answer_question.optimize(
        algorithm="random",
        max_trials=10,
        show_progress=True,
        random_seed=42,
    )

    print_results_table(results, CONFIG_SPACE, OBJECTIVES, is_mock=False)

    print("\nBest Configuration Found:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"  Cost: ${results.best_metrics.get('cost', 0):.6f}")
    print("\nYour original code stayed exactly the same!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
