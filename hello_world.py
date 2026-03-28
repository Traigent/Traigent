"""Find the best model and temperature for your task — in one decorator.

No API keys needed — runs in mock mode and simulates LLM responses
so you can see the optimization flow instantly.
For a real run with actual LLM calls, see walkthrough/real/01_tuning_qa.py.
"""
import asyncio
import os
from pathlib import Path

# hello_world.py always runs in mock mode — no API keys needed.
# Hard-set so stale shell env vars don't silently break the quickstart.
os.environ["TRAIGENT_MOCK_LLM"] = "true"
os.environ.setdefault("OPENAI_API_KEY", "mock-key-for-demos")
os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")

from langchain_openai import ChatOpenAI

import traigent

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
