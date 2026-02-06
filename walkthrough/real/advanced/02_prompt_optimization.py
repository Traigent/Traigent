#!/usr/bin/env python3
"""Example: Prompt Optimization - Real LangChain Version

Demonstrates system prompt tuning with actual LLM calls.

Requirements:
- Set OPENAI_API_KEY environment variable
- pip install langchain langchain-openai

Run with: python 02_prompt_optimization.py
"""

import asyncio
import os
from pathlib import Path

import traigent
from traigent.api.parameter_ranges import Choices, Range

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

# Initialize Traigent
traigent.initialize(execution_mode="edge_analytics")

# Define system prompt variants to optimize
SYSTEM_PROMPTS = [
    "You are a helpful assistant. Answer concisely.",
    "You are an expert. Provide accurate, detailed answers.",
    "Think step by step before answering. Be thorough.",
    "You are friendly and explain things simply.",
]

# Create Choices for system prompts
system_prompt = Choices(SYSTEM_PROMPTS, name="system_prompt")

# Other parameters to tune alongside prompts
temperature = Range.temperature()  # [0.0, 1.0]
model = Choices.model(provider="openai", tier="fast")


@traigent.optimize(
    system_prompt=system_prompt,
    temperature=temperature,
    model=model,
    objectives=["accuracy", "latency"],
    eval_dataset=DATASET_PATH,
    execution_mode="edge_analytics",
)
async def qa_agent(question: str) -> str:
    """Real QA agent with tunable system prompt."""
    config = traigent.get_config()

    llm = ChatOpenAI(
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=256,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", config["system_prompt"]),
            ("user", "{question}"),
        ]
    )

    chain = prompt | llm
    response = await chain.ainvoke({"question": question})
    return response.content


async def main() -> None:
    print("Traigent Example: Prompt Optimization with LangChain")
    print("=" * 60)

    print("\nSystem Prompt Variants:")
    for i, prompt in enumerate(SYSTEM_PROMPTS, 1):
        print(f"  {i}. {prompt}")

    print("\nOther Parameters:")
    print(f"  temperature: {temperature.low} - {temperature.high}")
    print(f"  model: {model.values}")

    print("\nRunning optimization (this will make real API calls)...")
    results = await qa_agent.optimize(algorithm="random", max_trials=8, random_seed=42)

    print("\nBest Configuration:")
    best_prompt = results.best_config.get("system_prompt", "N/A")
    print(f"  System Prompt: {best_prompt}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print(f"  Model: {results.best_config.get('model')}")

    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"  Latency: {results.best_metrics.get('latency', 0):.3f}s")
    print(f"  Cost: ${results.best_metrics.get('cost', 0):.6f}")

    # Analyze which prompts performed best
    print("\nPrompt Performance Summary:")
    prompt_scores: dict[str, list[float]] = {}
    for trial in results.trials:
        prompt = trial.config.get("system_prompt", "unknown")
        if prompt not in prompt_scores:
            prompt_scores[prompt] = []
        prompt_scores[prompt].append(trial.metrics.get("accuracy", 0))

    for prompt, scores in sorted(
        prompt_scores.items(),
        key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0,
        reverse=True,
    ):
        avg_score = sum(scores) / len(scores) if scores else 0
        prompt_preview = prompt[:40] + "..." if len(prompt) > 40 else prompt
        print(f"  {prompt_preview}: avg accuracy = {avg_score:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
