#!/usr/bin/env python3
"""OpenRouter Tiered Model Comparison - Direct Testing of All Models.

Tests 3 model tiers to show when model quality matters:
- Tier 1 (Premium): claude-3.5-sonnet, gpt-4o
- Tier 2 (Mid): claude-3.5-haiku, gpt-4o-mini
- Tier 3 (Budget): llama-3.1-70b, qwen-2.5-72b

Usage:
    source .env.local  # loads OPENROUTER_API_KEY
    python walkthrough/hotpotQA/run_openrouter_tiers.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Suppress warnings
os.environ["TRAIGENT_COST_APPROVED"] = "true"

ROOT_DIR = Path(__file__).resolve().parents[2]

# Model tiers for OpenRouter
MODEL_TIERS = {
    "Tier1_Premium": [
        ("anthropic/claude-3.5-sonnet", "~$3/1M"),
        ("openai/gpt-4o", "~$5/1M"),
    ],
    "Tier2_Mid": [
        ("anthropic/claude-3.5-haiku", "~$0.25/1M"),
        ("openai/gpt-4o-mini", "~$0.15/1M"),
    ],
    "Tier3_Budget": [
        ("meta-llama/llama-3.1-70b-instruct", "~$0.50/1M"),
        ("qwen/qwen-2.5-72b-instruct", "~$0.35/1M"),
    ],
}

DATASET_PATH = ROOT_DIR / "walkthrough" / "examples" / "mock" / "simple_questions.jsonl"


def load_dataset() -> list[dict]:
    """Load evaluation dataset."""
    examples = []
    with open(DATASET_PATH) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def invoke_openrouter(model: str, prompt: str) -> tuple[str, float, int]:
    """Call OpenRouter API. Returns (response, latency_ms, total_tokens)."""
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY (source .env.local)")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=100,
        extra_headers={
            "HTTP-Referer": "https://github.com/Traigent/Traigent",
            "X-Title": "Traigent Model Tier Demo",
        },
    )
    latency_ms = (time.perf_counter() - start) * 1000

    content = response.choices[0].message.content or ""
    total_tokens = (
        (response.usage.prompt_tokens + response.usage.completion_tokens)
        if response.usage
        else 0
    )

    return content.strip(), latency_ms, total_tokens


def score_answer(prediction: str, expected: str) -> float:
    """Score - 1.0 if expected in prediction, else word overlap."""
    pred = prediction.lower().strip()
    exp = expected.lower().strip()

    if exp in pred:
        return 1.0

    pred_words = set(pred.split())
    exp_words = set(exp.split())
    if not exp_words:
        return 0.0
    overlap = pred_words & exp_words
    return len(overlap) / len(exp_words)


def test_model(model: str, examples: list[dict]) -> dict:
    """Test a single model on all examples."""
    scores = []
    latencies = []
    tokens = []

    for ex in examples:
        question = ex["input"]["question"]
        expected = ex["output"]
        prompt = f"Answer concisely:\n\nQ: {question}\n\nA:"

        try:
            answer, latency_ms, tok = invoke_openrouter(model, prompt)
            score = score_answer(answer, expected)
            scores.append(score)
            latencies.append(latency_ms)
            tokens.append(tok)
        except Exception as e:
            print(f"    Error: {e}")
            scores.append(0.0)
            latencies.append(0.0)
            tokens.append(0)

    return {
        "accuracy": sum(scores) / len(scores) if scores else 0,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "avg_tokens": sum(tokens) / len(tokens) if tokens else 0,
        "total_calls": len(scores),
    }


def main() -> None:
    print("=" * 70)
    print("OpenRouter Tiered Model Comparison")
    print("=" * 70)

    examples = load_dataset()
    print(f"\nDataset: {len(examples)} questions from {DATASET_PATH.name}")
    print("Temperature: 0.2, Max tokens: 100 (fixed for fair comparison)")
    print("-" * 70)

    results: dict[str, dict] = {}

    for tier_name, models in MODEL_TIERS.items():
        print(f"\n{tier_name}:")
        print("-" * 40)

        for model, price in models:
            short_name = model.split("/")[-1]
            print(f"  Testing {short_name}...", end=" ", flush=True)

            result = test_model(model, examples)
            results[model] = {**result, "tier": tier_name, "price": price}

            acc = result["accuracy"]
            lat = result["avg_latency_ms"]
            print(f"✓ {acc:.1%} accuracy, {lat:.0f}ms avg latency")

    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<30} {'Tier':<15} {'Accuracy':>10} {'Latency':>10} {'Price':>12}")
    print("-" * 77)

    sorted_results = sorted(results.items(), key=lambda x: -x[1]["accuracy"])
    for model, data in sorted_results:
        short = model.split("/")[-1]
        tier = data["tier"].replace("_", " ")
        acc = f"{data['accuracy']:.1%}"
        lat = f"{data['avg_latency_ms']:.0f}ms"
        price = data["price"]
        print(f"{short:<30} {tier:<15} {acc:>10} {lat:>10} {price:>12}")

    # Best per tier
    print("\n" + "-" * 70)
    print("BEST PER TIER:")
    for tier_name in MODEL_TIERS:
        tier_models = [(m, d) for m, d in results.items() if d["tier"] == tier_name]
        if tier_models:
            best = max(tier_models, key=lambda x: x[1]["accuracy"])
            short = best[0].split("/")[-1]
            print(f"  {tier_name}: {short} ({best[1]['accuracy']:.1%})")

    # Overall best
    best_overall = max(results.items(), key=lambda x: x[1]["accuracy"])
    print(f"\nOVERALL BEST: {best_overall[0].split('/')[-1]} ({best_overall[1]['accuracy']:.1%})")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)

    tier1_acc = max(d["accuracy"] for m, d in results.items() if "Tier1" in d["tier"])
    tier3_acc = max(d["accuracy"] for m, d in results.items() if "Tier3" in d["tier"])
    gap = tier1_acc - tier3_acc

    if gap < 0.05:
        print("Budget models perform nearly as well as premium models for simple QA!")
        print(f"Gap: only {gap:.1%} difference - consider cost savings.")
    elif gap < 0.15:
        print(f"Moderate gap ({gap:.1%}) between tiers - task complexity matters.")
    else:
        print(f"Significant gap ({gap:.1%}) - premium models worth it for this task.")


if __name__ == "__main__":
    main()
