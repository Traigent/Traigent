#!/usr/bin/env python3
"""Example 3: Parameter Mode - Explicit configuration control.

Usage (run in a terminal from repo root, works without activating venv):
    export OPENAI_API_KEY="your-key"  # pragma: allowlist secret
    .venv/bin/python walkthrough/real/03_parameter_mode.py

If OPENAI_API_KEY is missing, this script exits with an error and suggests
running the mock walkthrough instead.
"""

import asyncio
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
from utils.scoring import token_match_score

import traigent
from traigent import TraigentConfig

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
    "temperature": [0.0, 0.5, 1.0],
    "max_tokens": [50, 150, 300],
    "use_system_prompt": [True, False],
}


@traigent.optimize(
    eval_dataset=str(DATASETS / "simple_questions.jsonl"),
    objectives=OBJECTIVES,
    injection_mode="parameter",
    scoring_function=token_match_score,
    configuration_space=CONFIG_SPACE,
    execution_mode="edge_analytics",
)
def answer_with_control(question: str, config: TraigentConfig) -> str:
    """Function with explicit configuration parameter."""
    model = config.get("model", "gpt-3.5-turbo")
    temperature = config.get("temperature", 0.5)
    max_tokens = config.get("max_tokens", 150)
    use_system = config.get("use_system_prompt", True)

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if use_system:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]
    else:
        messages = [{"role": "user", "content": question}]

    try:
        response = llm.invoke(messages)
        return str(response.content)
    except Exception as exc:
        print(f"LLM call failed: {type(exc).__name__}: {exc}")
        return f"Error: {type(exc).__name__}: {exc}"


async def main() -> None:
    print("Traigent Example 3: Parameter Mode")
    print("=" * 50)
    print("Full control with explicit configuration parameter.")
    print_optimization_config(OBJECTIVES, CONFIG_SPACE)
    print_cost_estimate(
        models=CONFIG_SPACE["model"],
        dataset_size=20,
        task_type="simple_qa",
        num_trials=10,
    )

    print_estimated_time("03_parameter_mode.py")
    results = await answer_with_control.optimize(
        algorithm="random",
        max_trials=10,
        show_progress=True,
        random_seed=42,
    )

    print_results_table(results, CONFIG_SPACE, OBJECTIVES, is_mock=False)

    print("\nBest Configuration Found:")
    for key, value in results.best_config.items():
        if key == "use_system_prompt":
            label = "system_prompt_enabled"
            value = "yes" if value else "no"
        else:
            label = key
        print(f"  {label}: {value}")

    print(f"\nAccuracy: {results.best_metrics.get('accuracy', 0):.2%}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
