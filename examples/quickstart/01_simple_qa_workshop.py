#!/usr/bin/env python
"""
Workshop Demo: Simple Q&A Optimization

Python equivalent of the JS workshop demo (traigent-js 01_simple_qa.mjs).
Deterministic grid search -- 9 trials, mock responses, no API keys needed.

Run:
    python examples/quickstart/01_simple_qa_workshop.py
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

# This demo is fully self-contained: deterministic answers, no LLM calls.
# Mock mode suppresses litellm pricing warnings for abstract model names.
os.environ["TRAIGENT_MOCK_LLM"] = "true"
os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")
os.environ.setdefault(
    "TRAIGENT_RESULTS_FOLDER",
    str(Path(__file__).parent / ".traigent_results"),
)

try:
    import traigent
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import traigent

from traigent.api.decorators import EvaluationOptions

# ---------------------------------------------------------------------------
# Inline dataset (mirrors traigent-js/examples/datasets/simple_questions.jsonl)
# ---------------------------------------------------------------------------
DATASET = [
    {"input": {"question": "What is 2+2?"}, "output": "4"},
    {"input": {"question": "What is the capital of France?"}, "output": "Paris"},
    {"input": {"question": "What color is the sky?"}, "output": "blue"},
    {"input": {"question": "How many days are in a week?"}, "output": "7"},
    {"input": {"question": "What is the largest planet?"}, "output": "Jupiter"},
    {"input": {"question": "What year did World War II end?"}, "output": "1945"},
    {"input": {"question": "What is H2O commonly called?"}, "output": "water"},
    {"input": {"question": "How many continents are there?"}, "output": "7"},
    {"input": {"question": "What is the speed of light in km/s?"}, "output": "299792"},
    {"input": {"question": "Who wrote Romeo and Juliet?"}, "output": "Shakespeare"},
    {
        "input": {"question": "What causes ocean tides?"},
        "output": "The gravitational pull of the moon and sun",
    },
    {
        "input": {"question": "Why is the sky blue?"},
        "output": "Rayleigh scattering of sunlight by atmosphere",
    },
    {
        "input": {"question": "How does photosynthesis work?"},
        "output": "Plants convert light energy to chemical energy using chlorophyll",
    },
    {
        "input": {"question": "What is machine learning?"},
        "output": "A method where computers learn patterns from data",
    },
    {
        "input": {"question": "Why do we have seasons?"},
        "output": "Earth's axial tilt causes varying sun angles throughout the year",
    },
    {
        "input": {"question": "Explain opportunity cost with an example"},
        "output": "The value of the next best alternative foregone when making a choice",
    },
    {
        "input": {"question": "What is the difference between weather and climate?"},
        "output": "Weather is short-term atmospheric conditions; climate is long-term patterns",
    },
    {
        "input": {"question": "How does a vaccine work?"},
        "output": "It trains the immune system to recognize and fight pathogens",
    },
    {
        "input": {"question": "What is cognitive bias?"},
        "output": "Systematic patterns of deviation from rational judgment",
    },
    {
        "input": {"question": "Explain the concept of compound interest"},
        "output": "Interest calculated on both principal and accumulated interest",
    },
]


# ---------------------------------------------------------------------------
# Scoring helpers (mirrors JS exactMatchScore + cost metric)
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text)


def exact_match(output: str, expected: str) -> float:
    """1.0 if normalised output contains the normalised expected answer."""
    # Inside a trial, traigent.with_usage wraps the answer in a {"text": ...}
    # envelope; unwrap it before scoring.
    if isinstance(output, dict):
        output = output.get("text") or ""
    return 1.0 if _normalize(expected) in _normalize(output) else 0.0


def simulated_cost(question: str) -> float:
    """Simulated per-call cost in USD: 0.0012 for 'capital' questions, else 0.0005.

    Kept small so a 9-trial x 20-example run stays far below the default cost
    limit; these are synthetic numbers, not real spend.
    """
    return 0.0012 if "capital" in question else 0.0005


# ---------------------------------------------------------------------------
# Optimised agent -- same deterministic logic as the JS demo
# ---------------------------------------------------------------------------
@traigent.optimize(
    configuration_space={
        "model": ["cheap", "balanced", "accurate"],
        "temperature": [0, 0.2, 0.4],
    },
    objectives=["accuracy", "cost"],
    evaluation=EvaluationOptions(
        eval_dataset=DATASET,
        scoring_function=exact_match,
    ),
    offline=True,
)
def answer_question(question: str) -> str:
    """Deterministic mock agent -- no LLM calls.

    The simulated cost is reported through ``traigent.with_usage`` (the same
    supported mechanism the real-model quickstarts use), so strict cost
    accounting sees a cost for every call.
    """
    cfg = traigent.get_config()
    model = cfg["model"]

    if model == "accurate":
        answer = "Paris" if "capital" in question else "4"
    elif model == "balanced" and "2+2" in question:
        answer = "4"
    else:
        answer = "unknown"

    return traigent.with_usage(answer, total_cost=simulated_cost(question))


# ---------------------------------------------------------------------------
# Run + summary (mirrors JS printSummary)
# ---------------------------------------------------------------------------
async def main() -> None:
    print("=" * 60)
    print("  TRAIGENT WORKSHOP DEMO — Simple Q&A Optimization")
    print("  Mock mode: no real LLM calls, deterministic results")
    print("=" * 60)
    print()

    result = await answer_question.optimize(algorithm="grid", max_trials=9)

    trials = result.trials if hasattr(result, "trials") and result.trials else []
    best_metrics = {}
    if trials and result.best_config:
        best_trial = next(
            (t for t in trials if t.config == result.best_config), trials[0]
        )
        best_metrics = {
            k: v for k, v in best_trial.metrics.items() if k in ("accuracy", "cost")
        }

    print(
        json.dumps(
            {
                "example": "quickstart/01_simple_qa_workshop",
                "bestConfig": result.best_config,
                "bestMetrics": best_metrics,
                "stopReason": result.stop_reason,
                "trialCount": len(trials),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
