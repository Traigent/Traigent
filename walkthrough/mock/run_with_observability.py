#!/usr/bin/env python3
"""Run mock examples and emit observability traces to the v1beta surface.

This wrapper runs each mock example, captures the optimization results,
and sends structured observability traces so the frontend Observability
page populates with real trial data.

Usage:
    TRAIGENT_BACKEND_URL=http://localhost:5001 \
    TRAIGENT_API_KEY=<key> \
    python walkthrough/mock/run_with_observability.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
WALKTHROUGH_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WALKTHROUGH_ROOT))

from utils.mock_answers import (  # noqa: E402
    ANSWERS,
    DEFAULT_MOCK_MODEL,
    configure_mock_notice,
    get_mock_accuracy,
    get_mock_cost,
    normalize_text,
    set_mock_model,
)

os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")
os.environ.setdefault("TRAIGENT_DATASET_ROOT", str(WALKTHROUGH_ROOT))

import traigent  # noqa: E402
from traigent import TraigentConfig  # noqa: E402

BACKEND_URL = os.environ.get("TRAIGENT_BACKEND_URL", "http://localhost:5001")
API_KEY = os.environ.get("TRAIGENT_API_KEY", "")

DATASETS = WALKTHROUGH_ROOT / "datasets"

# ---------------------------------------------------------------------------
# Observability trace emitter
# ---------------------------------------------------------------------------

MOCK_COST_MAP = {
    "gpt-3.5-turbo": {"input_cpm": 0.50, "output_cpm": 1.50, "latency_ms": 280},
    "gpt-4o-mini": {"input_cpm": 0.15, "output_cpm": 0.60, "latency_ms": 250},
    "gpt-4o": {"input_cpm": 2.50, "output_cpm": 10.00, "latency_ms": 450},
    "gpt-4.1-nano": {"input_cpm": 0.10, "output_cpm": 0.40, "latency_ms": 200},
    "claude-3-5-sonnet-20241022": {"input_cpm": 3.00, "output_cpm": 15.00, "latency_ms": 850},
    "claude-3-5-haiku-20241022": {"input_cpm": 0.80, "output_cpm": 4.00, "latency_ms": 300},
}


def _model_stats(model: str) -> dict:
    return MOCK_COST_MAP.get(model, {"input_cpm": 1.0, "output_cpm": 3.0, "latency_ms": 300})


async def emit_observability_traces(
    example_name: str,
    results,
    config_space: dict,
) -> int:
    """Send v1beta observability traces for each trial in the result."""

    if not API_KEY:
        print("  [obs] No API key — skipping observability emission")
        return 0

    session_id = f"obs-session-{example_name.replace('.py', '')}-{uuid.uuid4().hex[:8]}"
    base_time = datetime.now(UTC) - timedelta(minutes=5)

    traces = []
    for i, trial in enumerate(results.all_trials):
        model = trial.config.get("model", "gpt-4o-mini")
        stats = _model_stats(model)
        temperature = trial.config.get("temperature", 0.5)

        input_tokens = 150 + int(temperature * 50)
        output_tokens = 40 + int(temperature * 20)
        cost_usd = (input_tokens * stats["input_cpm"] + output_tokens * stats["output_cpm"]) / 1_000_000
        latency_ms = stats["latency_ms"] + int(temperature * 100)

        trial_start = base_time + timedelta(seconds=i * 3)
        trial_end = trial_start + timedelta(milliseconds=latency_ms)

        accuracy = trial.metrics.get("accuracy", 0)

        trace_id = f"trace-{example_name.replace('.py','')}-trial-{i+1}-{uuid.uuid4().hex[:6]}"

        observations = [
            {
                "id": f"{trace_id}-eval",
                "type": "span",
                "name": "evaluate_dataset",
                "status": "completed",
                "started_at": trial_start.isoformat(),
                "ended_at": (trial_start + timedelta(milliseconds=50)).isoformat(),
                "latency_ms": 50,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "metadata": {"dataset_size": 20, "scoring": "contains_match"},
            },
            {
                "id": f"{trace_id}-gen",
                "type": "generation",
                "name": f"{model} inference",
                "status": "completed",
                "parent_observation_id": f"{trace_id}-eval",
                "model_name": model,
                "started_at": (trial_start + timedelta(milliseconds=50)).isoformat(),
                "ended_at": trial_end.isoformat(),
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "input_data": {"question": "Sample question", "temperature": temperature},
                "output_data": {"answer": "Mock answer", "accuracy": accuracy},
                "metadata": {
                    "temperature": temperature,
                    "model": model,
                    "trial_index": i + 1,
                    "accuracy": accuracy,
                },
            },
        ]

        traces.append({
            "id": trace_id,
            "name": f"{example_name} — trial {i+1} ({model}, t={temperature})",
            "status": "completed",
            "session_id": session_id,
            "tags": ["mock", "walkthrough", example_name.replace(".py", "")],
            "metadata": {
                "example": example_name,
                "trial_index": i + 1,
                "config": trial.config,
                "accuracy": accuracy,
                "model": model,
                "temperature": temperature,
            },
            "started_at": trial_start.isoformat(),
            "ended_at": trial_end.isoformat(),
            "session": {
                "id": session_id,
                "tags": ["walkthrough", "mock"],
                "metadata": {"example": example_name, "total_trials": len(results.all_trials)},
                "started_at": base_time.isoformat(),
                "ended_at": (base_time + timedelta(minutes=1)).isoformat(),
            },
            "observations": observations,
        })

    # Send in a single batch
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{BACKEND_URL}/api/v1beta/observability/ingest",
            json={"traces": traces},
            headers={
                "X-API-Key": API_KEY,
                "Content-Type": "application/json",
            },
        )

        if resp.status_code in (200, 201):
            data = resp.json().get("data", {})
            print(f"  [obs] Ingested {data.get('accepted', 0)} traces into observability")
            return data.get("accepted", 0)
        else:
            print(f"  [obs] Failed to ingest: {resp.status_code} — {resp.text[:200]}")
            return 0


# ---------------------------------------------------------------------------
# Example definitions (reusing mock example logic)
# ---------------------------------------------------------------------------

OBJECTIVES = ["accuracy", "cost"]
CONFIG_SPACE_01 = {
    "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4.1-nano"],
    "temperature": [0.1, 0.7],
}

MOCK_MODE_CONFIG = {"base_accuracy": 0.80, "variance": 0.0, "random_seed": 42}


def results_match_score(output: str, expected: str, config: dict | None = None, **_) -> float:
    if os.getenv("TRAIGENT_MOCK_LLM", "").lower() in ("1", "true", "yes"):
        model = config.get("model", DEFAULT_MOCK_MODEL) if config else DEFAULT_MOCK_MODEL
        return get_mock_accuracy(model, "simple_qa")
    if output is None or expected is None:
        return 0.0
    return 1.0 if expected.strip().lower() in str(output).lower() else 0.0


traigent.initialize(
    config=TraigentConfig(execution_mode="edge_analytics", minimal_logging=True)
)


@traigent.optimize(
    eval_dataset=str(DATASETS / "simple_questions.jsonl"),
    objectives=OBJECTIVES,
    scoring_function=results_match_score,
    configuration_space=CONFIG_SPACE_01,
    injection_mode="context",
    execution_mode="edge_analytics",
    mock_mode_config=MOCK_MODE_CONFIG,
)
def answer_question(question: str) -> str:
    config = traigent.get_config()
    set_mock_model(config.get("model", DEFAULT_MOCK_MODEL))
    return ANSWERS.get(normalize_text(question), "I don't know")


# ---------------------------------------------------------------------------
# Multi-provider config (example 07 style)
# ---------------------------------------------------------------------------

CONFIG_SPACE_07 = {
    "model": ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
    "temperature": [0.1, 0.5],
}


@traigent.optimize(
    eval_dataset=str(DATASETS / "simple_questions.jsonl"),
    objectives=["accuracy", "cost", "latency"],
    scoring_function=results_match_score,
    configuration_space=CONFIG_SPACE_07,
    injection_mode="context",
    execution_mode="edge_analytics",
    mock_mode_config=MOCK_MODE_CONFIG,
)
def multi_provider_qa(question: str) -> str:
    config = traigent.get_config()
    set_mock_model(config.get("model", DEFAULT_MOCK_MODEL))
    return ANSWERS.get(normalize_text(question), "I don't know")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 60)
    print("Running mock examples with observability trace emission")
    print(f"Backend: {BACKEND_URL}")
    print(f"API Key: {'set' if API_KEY else 'NOT SET'}")
    print("=" * 60)

    total_traces = 0

    # Example 1: Basic QA tuning
    print("\n--- Example 1: Basic QA Tuning ---")
    configure_mock_notice("01_tuning_qa.py")
    results = await answer_question.optimize(algorithm="grid", max_trials=8, random_seed=42)
    print(f"  Best: {results.best_config.get('model')} @ t={results.best_config.get('temperature')}")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    total_traces += await emit_observability_traces("01_tuning_qa.py", results, CONFIG_SPACE_01)

    # Example 7: Multi-provider
    print("\n--- Example 7: Multi-Provider ---")
    results = await multi_provider_qa.optimize(algorithm="grid", max_trials=8, random_seed=42)
    print(f"  Best: {results.best_config.get('model')} @ t={results.best_config.get('temperature')}")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    total_traces += await emit_observability_traces("07_multi_provider.py", results, CONFIG_SPACE_07)

    print(f"\n{'=' * 60}")
    print(f"Total observability traces emitted: {total_traces}")
    print(f"Check the Observability page at {BACKEND_URL.replace('5001', '3000')}/observability")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130)
