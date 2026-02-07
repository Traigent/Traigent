#!/usr/bin/env python3
"""Example: Tuned Variables with Factory Methods and Constraints - Real LangChain Version

Demonstrates factory methods and constraints with actual LLM calls.

Requirements:
- Set OPENAI_API_KEY environment variable
- pip install langchain langchain-openai

Run with: python 01_tuned_variables.py
"""

import asyncio
import os
from pathlib import Path

import traigent
from traigent.api.constraints import implies
from traigent.api.parameter_ranges import Choices, IntRange, Range

# Compute dataset path relative to this script
SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = str((SCRIPT_DIR / ".." / "simple_questions.jsonl").resolve())

# Check for API key before initialization
if not os.getenv("OPENAI_API_KEY"):
    print("Error: Set OPENAI_API_KEY environment variable to run this example")
    print("Example: export OPENAI_API_KEY='your-key-here'")  # pragma: allowlist secret
    exit(1)

try:
    from langchain.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
except ImportError:
    print("Error: LangChain not installed")
    print("Install with: pip install langchain langchain-openai")
    exit(1)

# Initialize Traigent for real execution
traigent.initialize(execution_mode="edge_analytics")

# Factory-created parameter ranges
temperature = Range.temperature(creative=True)  # [0.7, 1.5]
max_tokens = IntRange.max_tokens(task="short")  # [50, 256]
model = Choices.model(provider="openai", tier="fast")  # [gpt-4o-mini]

# Constraints
constraints = [
    # If temperature is high (creative), allow more tokens
    implies(temperature.gte(1.0), max_tokens.gte(128)),
]


@traigent.optimize(
    temperature=temperature,
    max_tokens=max_tokens,
    model=model,
    constraints=constraints,
    objectives=["accuracy", "latency"],
    eval_dataset=DATASET_PATH,
    execution_mode="edge_analytics",
)
async def qa_agent(question: str) -> str:
    """Real QA agent using LangChain."""
    config = traigent.get_config()

    llm = ChatOpenAI(
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Answer concisely."),
            ("user", "{question}"),
        ]
    )

    chain = prompt | llm
    response = await chain.ainvoke({"question": question})
    return response.content


async def main() -> None:
    print("Traigent Example: Tuned Variables with LangChain")
    print("=" * 60)

    print("\nParameter Ranges:")
    print(f"  temperature: {temperature.low} - {temperature.high}")
    print(f"  max_tokens: {max_tokens.low} - {max_tokens.high}")
    print(f"  model: {model.values}")

    print("\nConstraints:")
    print("  temperature >= 1.0 -> max_tokens >= 128")

    print("\nRunning optimization (this will make real API calls)...")
    results = await qa_agent.optimize(algorithm="random", max_trials=6, random_seed=42)

    print("\nBest Configuration:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print(f"  Max Tokens: {results.best_config.get('max_tokens')}")

    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"  Latency: {results.best_metrics.get('latency', 0):.3f}s")
    print(f"  Cost: ${results.best_metrics.get('cost', 0):.6f}")


if __name__ == "__main__":
    asyncio.run(main())
