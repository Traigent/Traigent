#!/usr/bin/env python3
"""Traigent Playground — try the SDK in under a minute.

Usage (mock mode, no API keys needed):
    TRAIGENT_MOCK_LLM=true python playground.py

Usage (real mode, requires OpenAI key):
    OPENAI_API_KEY=sk-... python playground.py
"""

import asyncio
from pathlib import Path

import traigent
from langchain_openai import ChatOpenAI
from traigent.api.decorators import EvaluationOptions, ExecutionOptions

DATASET = str(Path(__file__).resolve().parent / "examples" / "datasets" / "quickstart" / "qa_samples.jsonl")


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],
    },
    objectives=["accuracy", "cost"],
    evaluation=EvaluationOptions(eval_dataset=DATASET),
    execution=ExecutionOptions(execution_mode="edge_analytics"),
)
def simple_qa_agent(question: str) -> str:
    """Simple Q&A agent — Traigent automatically tests different models & temperatures."""

    # Your normal code. Traigent intercepts ChatOpenAI and swaps in the trial's config.
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    response = llm.invoke(f"Question: {question}\nAnswer:")
    return str(response.content)


async def main():
    result = await simple_qa_agent.optimize(algorithm="grid", max_trials=6)

    print(f"\nBest config: {result.best_config}")
    print(f"Best score:  {result.best_score}")

    df = result.to_aggregated_dataframe(primary_objective="accuracy")
    cols = [c for c in ["model", "temperature", "accuracy", "cost"] if c in df.columns]
    if cols:
        print(f"\n{df[cols].to_string(index=False)}")


if __name__ == "__main__":
    asyncio.run(main())
