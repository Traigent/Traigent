#!/usr/bin/env python3
"""OpenRouter Tiered Model Comparison - Traigent Integrated.

Tests model tiers with multiple config dimensions.

Usage:
    source .env.local
    python walkthrough/hotpotQA/run_openrouter_traigent.py
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

# Real mode, online Traigent
os.environ["TRAIGENT_MOCK_LLM"] = "false"
os.environ["TRAIGENT_OFFLINE_MODE"] = "false"
os.environ["TRAIGENT_COST_APPROVED"] = "true"

ROOT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("TRAIGENT_DATASET_ROOT", str(ROOT_DIR))

import traigent

# Models across 3 tiers
MODELS = [
    "anthropic/claude-3.5-sonnet",        # Tier 1 Premium
    "openai/gpt-4o",                       # Tier 1 Premium
    "anthropic/claude-3.5-haiku",         # Tier 2 Mid
    "openai/gpt-4o-mini",                 # Tier 2 Mid
    "meta-llama/llama-3.1-70b-instruct",  # Tier 3 Budget
    "qwen/qwen-2.5-72b-instruct",         # Tier 3 Budget
]

DATASET_PATH = ROOT_DIR / "walkthrough" / "examples" / "mock" / "simple_questions.jsonl"


def get_tier(model: str) -> str:
    if "sonnet" in model or model.endswith("gpt-4o"):
        return "Premium"
    if "haiku" in model or "mini" in model:
        return "Mid"
    return "Budget"


def invoke_openrouter(
    model: str, prompt: str, temperature: float, max_tokens: int
) -> str:
    """Call OpenRouter API with retry on 500 errors."""
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Retry logic for 500 errors
    for attempt in range(3):
        try:
            start = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers={
                    "HTTP-Referer": "https://github.com/Traigent/Traigent",
                    "X-Title": "Traigent Model Tier Demo",
                },
            )
            latency = (time.perf_counter() - start) * 1000

            content = response.choices[0].message.content or ""
            short = model.split("/")[-1]
            tier = get_tier(model)
            print(f"  [{tier}] {short}: {latency:.0f}ms, T={temperature}")

            return content.strip()
        except Exception as e:
            if "500" in str(e) and attempt < 2:
                print(f"  [RETRY] {model.split('/')[-1]}: 500 error, retrying...")
                time.sleep(1)
                continue
            raise

    return ""


def score_answer(prediction: str, expected: str) -> float:
    """Score answer - check containment and word overlap."""
    pred = prediction.lower().strip()
    exp = expected.lower().strip()

    # Empty check
    if not exp or not pred:
        return 0.0

    # Exact containment (expected in prediction)
    if exp in pred:
        return 1.0

    # Word overlap score
    pred_words = set(pred.split())
    exp_words = set(exp.split())
    if not exp_words:
        return 0.0

    overlap = pred_words & exp_words
    return len(overlap) / len(exp_words)


def custom_scorer(example: Any, output: str) -> float:
    """Extract expected output and score."""
    # Handle different expected_output formats
    expected = example.expected_output
    if isinstance(expected, dict):
        expected = expected.get("output", expected.get("answer", str(expected)))
    expected = str(expected) if expected else ""

    score = score_answer(output, expected)
    return score


@traigent.optimize(
    eval_dataset=str(DATASET_PATH),
    objectives=["accuracy"],
    configuration_space={
        # 6 models
        "model": MODELS,
        # 3 temperatures
        "temperature": [0.1, 0.4, 0.7],
        # 2 token limits
        "max_tokens": [50, 150],
        # 2 prompt styles
        "prompt_style": ["concise", "detailed"],
    },
    scoring_function=custom_scorer,
    execution_mode="edge_analytics",
)
def qa_openrouter(question: str) -> str:
    """QA agent using OpenRouter with configurable parameters."""
    config: dict[str, Any] = traigent.get_config()

    model = config.get("model", "openai/gpt-4o-mini")
    temp = float(config.get("temperature", 0.3))
    max_tok = int(config.get("max_tokens", 100))
    style = config.get("prompt_style", "concise")

    # Build prompt based on style
    if style == "detailed":
        prompt = (
            f"Please answer the following question thoroughly but accurately.\n\n"
            f"Question: {question}\n\n"
            f"Provide your answer:"
        )
    else:
        prompt = f"Answer concisely:\n\nQ: {question}\n\nA:"

    return invoke_openrouter(model, prompt, temp, max_tok)


async def main() -> None:
    print("=" * 60)
    print("OpenRouter Multi-Config Optimization")
    print("=" * 60)

    print(f"\nConfig space:")
    print(f"  Models: {len(MODELS)} (Premium/Mid/Budget)")
    print(f"  Temperatures: [0.1, 0.4, 0.7]")
    print(f"  Max tokens: [50, 150]")
    print(f"  Prompt styles: [concise, detailed]")
    print(f"  Total configs: {6 * 3 * 2 * 2} = 72 possible")

    print(f"\nDataset: {DATASET_PATH.name}")
    print("-" * 60)

    # Max 20 trials, 4 parallel
    result = await qa_openrouter.optimize(
        algorithm="random",
        max_trials=20,
        parallel_config={"trial_concurrency": 4},
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nBest config:")
    for k, v in result.best_config.items():
        if k == "model":
            v = v.split("/")[-1]
        print(f"  {k}: {v}")
    print(f"\nBest accuracy: {result.best_metrics.get('accuracy', 0):.1%}")


if __name__ == "__main__":
    asyncio.run(main())
