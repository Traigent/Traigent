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

# Workaround: the SDK's comprehensive-metrics merge overwrites custom
# metric_functions values (e.g. "cost") with the built-in LLM cost tracker
# which is $0 when no real API calls are made. Patch to protect custom keys.
import math as _math
import traigent.evaluators.local as _le


def _patched_merge(self, aggregated_metrics, comprehensive_metrics):
    custom_keys = set(self.metric_functions) if self.metric_functions else set()
    if "cost" in (self.metrics or []) and "cost" in comprehensive_metrics:
        agg_cost = float(aggregated_metrics.get("cost", 0.0) or 0.0)
        comp_cost = float(comprehensive_metrics["cost"])
        if _math.isclose(agg_cost, 0.0, abs_tol=1e-9) and not _math.isclose(
            comp_cost, 0.0, abs_tol=1e-9
        ):
            aggregated_metrics["cost"] = comp_cost
    for key, value in comprehensive_metrics.items():
        if value is None:
            continue
        if key in {"accuracy", "score"} and key in aggregated_metrics and aggregated_metrics[key] not in (None, 0.0):
            continue
        if key in custom_keys and key in aggregated_metrics:
            continue
        aggregated_metrics[key] = value


_le.LocalEvaluator._merge_comprehensive_metrics = _patched_merge

_orig_agg = _le.LocalEvaluator._compute_aggregated_custom_metrics


def _patched_agg(self, example_results, tracker_example_metrics=None):
    aggregated = {}
    for metric_name in self.metric_functions:
        values = []
        for result in example_results:
            if result is None:
                continue
            v = result.metrics.get(metric_name)
            if v is not None:
                values.append(float(v))
        if not values and tracker_example_metrics:
            for em in tracker_example_metrics:
                v = em.custom_metrics.get(metric_name)
                if v is not None:
                    values.append(float(v))
        aggregated[metric_name] = (sum(values) / len(values)) if values else 0.0
    return aggregated


_le.LocalEvaluator._compute_aggregated_custom_metrics = _patched_agg

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
    {"input": {"question": "What causes ocean tides?"}, "output": "The gravitational pull of the moon and sun"},
    {"input": {"question": "Why is the sky blue?"}, "output": "Rayleigh scattering of sunlight by atmosphere"},
    {"input": {"question": "How does photosynthesis work?"}, "output": "Plants convert light energy to chemical energy using chlorophyll"},
    {"input": {"question": "What is machine learning?"}, "output": "A method where computers learn patterns from data"},
    {"input": {"question": "Why do we have seasons?"}, "output": "Earth's axial tilt causes varying sun angles throughout the year"},
    {"input": {"question": "Explain opportunity cost with an example"}, "output": "The value of the next best alternative foregone when making a choice"},
    {"input": {"question": "What is the difference between weather and climate?"}, "output": "Weather is short-term atmospheric conditions; climate is long-term patterns"},
    {"input": {"question": "How does a vaccine work?"}, "output": "It trains the immune system to recognize and fight pathogens"},
    {"input": {"question": "What is cognitive bias?"}, "output": "Systematic patterns of deviation from rational judgment"},
    {"input": {"question": "Explain the concept of compound interest"}, "output": "Interest calculated on both principal and accumulated interest"},
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
    return 1.0 if _normalize(expected) in _normalize(output) else 0.0


def cost_metric(input_data: dict) -> float:
    """Simulated cost: 0.12 for 'capital' questions, 0.05 otherwise."""
    return 0.12 if "capital" in input_data.get("question", "") else 0.05


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
        metric_functions={"cost": cost_metric},
    ),
    execution_mode="edge_analytics",
)
def answer_question(question: str) -> str:
    """Deterministic mock agent -- no LLM calls."""
    cfg = traigent.get_config()
    model = cfg["model"]

    if model == "accurate":
        return "Paris" if "capital" in question else "4"

    if model == "balanced" and "2+2" in question:
        return "4"

    return "unknown"


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
    if trials:
        best_trial = max(
            trials,
            key=lambda t: t.metrics.get("accuracy", t.metrics.get("score", float("-inf"))),
        )
        best_metrics = {
            k: v for k, v in best_trial.metrics.items()
            if k in ("accuracy", "cost")
        }

    print(json.dumps({
        "example": "quickstart/01_simple_qa_workshop",
        "bestConfig": result.best_config,
        "bestMetrics": best_metrics,
        "stopReason": result.stop_reason,
        "trialCount": len(trials),
    }, indent=2, default=str))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
