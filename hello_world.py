"""Find the best model and temperature for your task — in one decorator.

No API keys needed — runs in mock mode and simulates LLM responses
so you can see the optimization flow instantly.
For a real run with actual LLM calls, see walkthrough/real/01_tuning_qa.py.
"""
import asyncio
import os
from pathlib import Path

# Default to mock mode so the quickstart works without API keys.
# Override with: TRAIGENT_MOCK_LLM=false python hello_world.py
os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")
if os.environ.get("TRAIGENT_MOCK_LLM", "").lower() in ("1", "true", "yes"):
    os.environ.setdefault("OPENAI_API_KEY", "mock-key-for-demos")
    os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")

import litellm

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
    response = litellm.completion(
        model=cfg["model"],
        temperature=cfg["temperature"],
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    result = asyncio.run(answer.optimize(max_trials=6, algorithm="grid"))
