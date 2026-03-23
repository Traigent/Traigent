"""Find the best model and temperature for your task — in one decorator.

No API keys needed — mock mode is on by default and simulates LLM
responses so you can see the optimization flow instantly.
To use real LLMs: TRAIGENT_MOCK_LLM=false python hello_world.py
"""
import asyncio
import os
import sys
from pathlib import Path

# Mock mode: simulates LLM calls so you don't need API keys.
os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")
# Provide a dummy key so ChatOpenAI can be constructed in mock mode.
os.environ.setdefault("OPENAI_API_KEY", "mock-key-for-demos")
# Offline mode: no Traigent backend connection needed.
os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")

sys.path.append(str(Path(__file__).parent / "walkthrough"))

import traigent
from langchain_openai import ChatOpenAI
from utils.helpers import print_results_table

DATASET = str(Path(__file__).parent / "examples" / "datasets" / "quickstart" / "qa_samples.jsonl")
OBJECTIVES = ["accuracy"]
CONFIG_SPACE = {
    "model": ["gpt-4o-mini", "gpt-4o"],
    "temperature": [0.0, 0.7, 1.0],
}


@traigent.optimize(
    configuration_space=CONFIG_SPACE,
    objectives=OBJECTIVES,
    eval_dataset=DATASET,
    execution_mode="edge_analytics",
)
def answer(question: str) -> str:
    """Call an LLM with the current trial's config."""
    cfg = traigent.get_config()
    llm = ChatOpenAI(model=cfg["model"], temperature=cfg["temperature"])
    return llm.invoke(question).content


if __name__ == "__main__":
    result = asyncio.run(answer.optimize(max_trials=6, algorithm="grid"))
    is_mock = os.environ.get("TRAIGENT_MOCK_LLM", "").lower() in ("1", "true", "yes")
    print_results_table(result, CONFIG_SPACE, OBJECTIVES, is_mock=is_mock)
