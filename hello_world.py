"""Find the best model and temperature for your task — in one decorator."""
import asyncio
import os
import sys
from pathlib import Path

# Hello world runs locally — no backend connection needed.
os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")

sys.path.append(str(Path(__file__).parent / "walkthrough"))

import traigent
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
    """Your LLM call goes here. Mock mode simulates it."""
    # Replace with your LLM call, e.g.:
    #   from langchain_openai import ChatOpenAI
    #   config = traigent.get_config()
    #   return ChatOpenAI(model=config["model"]).invoke(question).content
    return f"Answer to: {question}"


if __name__ == "__main__":
    result = asyncio.run(answer.optimize(max_trials=6, algorithm="grid"))
    print_results_table(result, CONFIG_SPACE, OBJECTIVES, is_mock=True)
