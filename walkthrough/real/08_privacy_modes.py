#!/usr/bin/env python3
"""Example 8: Privacy Modes - Local-only privacy-first execution (current).

Usage (run in a terminal from repo root, works without activating venv):
    export OPENAI_API_KEY="your-key"  # pragma: allowlist secret
    .venv/bin/python walkthrough/real/08_privacy_modes.py

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
)
from utils.scoring import token_match_score

import traigent
from traigent import TraigentConfig

configure_logging()

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")
os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")

traigent.initialize(
    config=TraigentConfig(
        execution_mode="edge_analytics",
        minimal_logging=True,
        enable_usage_analytics=False,
    )
)

# Dataset path relative to this file
DATASETS = Path(__file__).parent.parent / "datasets"
RESULTS_DIR = os.getenv("TRAIGENT_RESULTS_FOLDER", "./local_results")
OBJECTIVES = ["accuracy"]
# Valid model names: https://models.litellm.ai/
CONFIG_SPACE = {
    "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
    "temperature": [0.1, 0.5],
}


@traigent.optimize(
    eval_dataset=str(DATASETS / "simple_questions.jsonl"),
    objectives=OBJECTIVES,
    scoring_function=token_match_score,
    configuration_space=CONFIG_SPACE,
    injection_mode="context",  # default injection mode, added explicitly for clarity
    execution_mode="edge_analytics",
    local_storage_path=RESULTS_DIR,
)
def local_mode(question: str) -> str:
    """Local mode - all data stays on your machine and backend calls are disabled."""
    config = traigent.get_config()
    llm = ChatOpenAI(
        model=config.get("model"),
        temperature=config.get("temperature"),
    )
    try:
        response = llm.invoke(question)
        return str(response.content)
    except Exception as exc:
        print(f"LLM call failed: {type(exc).__name__}: {exc}")
        return f"Error: {type(exc).__name__}: {exc}"


async def main() -> None:
    print("Traigent Example 8: Privacy Modes (local-only for now)")
    print("=" * 50)
    print_optimization_config(OBJECTIVES, CONFIG_SPACE)
    print_cost_estimate(
        models=CONFIG_SPACE["model"],
        dataset_size=20,
        task_type="simple_qa",
        num_trials=4,
    )

    print("\nLOCAL - All data stays on your machine")
    print_estimated_time("08_privacy_modes.py")
    results = await local_mode.optimize(
        algorithm="grid",
        max_trials=4,
        show_progress=True,
        random_seed=42,
    )

    print_results_table(results, CONFIG_SPACE, OBJECTIVES, is_mock=False)

    print("\nBest Configuration Found:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print("\nLocal Storage:")
    print(f"  Results stored in: {RESULTS_DIR}")
    print("  Look inside: sessions/ and experiments/ for saved runs")
    print("\nThis walkthrough focuses on privacy-first local execution.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
