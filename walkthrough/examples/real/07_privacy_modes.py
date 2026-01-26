#!/usr/bin/env python3
"""Example 7: Privacy Modes - Local-only privacy-first execution (current).

Usage (run in a terminal from repo root, works without activating venv):
    export OPENAI_API_KEY="your-key"
    .venv/bin/python walkthrough/examples/real/07_privacy_modes.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI

import traigent

from utils.helpers import configure_logging, print_estimated_time, require_openai_key
from utils.scoring import token_match_score

require_openai_key("07_privacy_modes.py")
configure_logging()

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

# Dataset path relative to this file
DATASETS = Path(__file__).parent.parent / "datasets"
RESULTS_DIR = os.getenv("TRAIGENT_RESULTS_FOLDER", "./local_results")


@traigent.optimize(
    eval_dataset=str(DATASETS / "simple_questions.jsonl"),
    objectives=["accuracy"],
    scoring_function=token_match_score,
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.1, 0.5],
    },
    injection_mode="context",  # default injection mode, added explicitly for clarity
    execution_mode="edge_analytics",
    local_storage_path=RESULTS_DIR,
)
def local_mode(question: str) -> str:
    """Local mode - all data stays on your machine."""
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
    print("Traigent Example 7: Privacy Modes (local-only for now)")
    print("=" * 50)

    print("\nLOCAL - All data stays on your machine")
    print_estimated_time("07_privacy_modes.py")
    results = await local_mode.optimize(
        algorithm="grid",
        max_trials=4,
        show_progress=True,
        random_seed=42,
    )

    print("\nBest Configuration Found:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print("\nLocal Storage:")
    print(f"  Results stored in: {RESULTS_DIR}")
    print("  Look inside: sessions/ and experiments/ for saved runs")
    print("\nThis walkthrough focuses on privacy-first local execution.")


if __name__ == "__main__":
    asyncio.run(main())
