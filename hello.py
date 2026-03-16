"""Find the best model and temperature for your task — in one decorator."""
import asyncio
from pathlib import Path

import traigent

DATASET = str(Path(__file__).parent / "examples" / "datasets" / "quickstart" / "qa_samples.jsonl")


@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.7, 1.0],
    },
    objectives=["accuracy"],
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


result = asyncio.run(answer.optimize(max_trials=6, algorithm="grid"))
print(f"Best config: {result.best_config}")
print(f"Best score:  {result.best_score:.2%}")
