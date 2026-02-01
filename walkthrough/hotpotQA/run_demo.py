#!/usr/bin/env python3
"""HotpotQA Multi-Hop QA Optimization Demo using Traigent.

This demo shows how Traigent optimizes a multi-hop QA agent on HotpotQA data.
See how model, temperature, retrieval depth, and prompt style affect quality/cost/latency.

Usage:
    python run_demo.py         # Mock mode (default)
    python run_demo.py --real  # Real mode (requires OPENAI_API_KEY)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import warnings
from functools import reduce
from pathlib import Path
from typing import Any
from collections.abc import Mapping

# Suppress library warnings that clutter demo output.
# tiktoken: "gpt-4o may update over time" (using aliases is more user-friendly for demos)
# anthropic: "token counting API is in beta" (beta warnings not relevant for end users)
warnings.filterwarnings("ignore", message=".*may update over time.*")
warnings.filterwarnings("ignore", message=".*token counting.*beta.*")
warnings.filterwarnings("ignore", message=".*Token counting.*")
warnings.filterwarnings("ignore", category=UserWarning, module="anthropic")

# Suppress tokencost logger warnings about Anthropic beta API
import logging
logging.getLogger("tokencost").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HotpotQA Multi-Hop QA Optimization Demo (mock mode by default)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Run with real LLM API calls (requires OPENAI_API_KEY)",
    )
    return parser.parse_args()


# Parse args early to set environment before imports
_args = parse_args()
_use_mock = not _args.real

# Set environment based on CLI flag
os.environ["TRAIGENT_MOCK_LLM"] = "true" if _use_mock else "false"
ROOT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("TRAIGENT_DATASET_ROOT", str(ROOT_DIR))

import sys  # noqa: E402

# Ensure ROOT_DIR is in path for paper_experiments module
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import traigent  # noqa: E402
    from traigent.utils.callbacks import ProgressBarCallback  # noqa: E402
except ImportError:  # pragma: no cover - support direct script execution
    import importlib

    traigent = importlib.import_module("traigent")
    callbacks_module = importlib.import_module("traigent.utils.callbacks")
    ProgressBarCallback = callbacks_module.ProgressBarCallback

from paper_experiments.case_study_rag.dataset import (  # noqa: E402
    dataset_path,
    load_case_study_dataset,
)
from paper_experiments.case_study_rag.metrics import (  # noqa: E402
    build_hotpot_metric_functions,
)
from paper_experiments.case_study_rag.simulator import (  # noqa: E402
    generate_case_study_answer,
)

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

# ============================================================================
# Configuration
# ============================================================================

DATASET = str(dataset_path())
USE_MOCK = _use_mock  # Set by CLI args (--mock or --real)

# Objectives: what we're optimizing for
OBJECTIVES = ["quality", "latency_p95_ms", "cost_usd_per_1k"]

# Configuration space: what combinations Traigent will explore
CONFIG_SPACE = {
    "model": ["gpt-4o", "gpt-4o-mini", "haiku-3.5"],
    "temperature": [0.1, 0.3, 0.7],
    "retriever_k": [3, 5, 8],  # Number of context passages to retrieve
    "prompt_style": ["vanilla", "cot"],  # Direct answer vs chain-of-thought
    "retrieval_reranker": ["none", "mono_t5"],
    # Note: 384+ tokens needed for CoT reasoning; prompt requests concise final answer
    "max_output_tokens": [384, 512],
}

# Optimization settings (adjust these for longer/more thorough runs)
MAX_TRIALS = 12  # Total configurations to try
TIMEOUT_SECONDS = 300  # 5 min; use `timeout=` in optimize() (not timeout_seconds)
PARALLEL_TRIALS = 4  # Parallel works in both mock and real mode (LangChain auto-tracks tokens)


# ============================================================================
# Output Formatting Helpers
# ============================================================================

def print_header() -> None:
    """Print demo header with title and mode indicator."""
    mode = "MOCK" if USE_MOCK else "REAL"
    print()
    print("=" * 60)
    print("  Traigent HotpotQA Multi-Hop QA Optimization Demo")
    print("=" * 60)
    print(f"  Mode: {mode}")
    if USE_MOCK:
        print("  (No API keys required - using simulated responses)")
    else:
        print("  (Using real LLM API calls - costs will apply)")
    print()


def print_optimization_config() -> None:
    """Print configuration summary before optimization starts."""
    # Calculate total combinations
    param_counts = [len(values) for values in CONFIG_SPACE.values()]
    total_combinations = reduce(lambda x, y: x * y, param_counts, 1)

    # Build breakdown string
    breakdown_parts = [f"{len(v)} {k}" for k, v in CONFIG_SPACE.items()]
    breakdown_str = " x ".join(breakdown_parts)

    print("Optimization Configuration:")
    print(f"  Objectives: {', '.join(OBJECTIVES)}")
    print("    - quality: Answer correctness (exact match + F1 score)")
    print("    - latency_p95_ms: 95th percentile response time (95% of calls within this)")
    print("    - cost_usd_per_1k: Cost per 1000 tokens")
    print(f"  Total combinations: {total_combinations} ({breakdown_str})")
    print(f"  Max trials: {MAX_TRIALS}")
    print(f"  Timeout: {TIMEOUT_SECONDS}s (edit TIMEOUT_SECONDS in run_demo.py for longer runs)")
    print(f"  Parallel trials: {PARALLEL_TRIALS}")
    print()
    print("Configuration Space:")
    for param, values in CONFIG_SPACE.items():
        values_str = ", ".join(repr(v) for v in values)
        print(f"  - {param}: [{values_str}]")
    print()


def print_cost_warning() -> None:
    """Print cost warning for real mode."""
    if USE_MOCK:
        return

    dataset = load_case_study_dataset()
    num_examples = len(dataset.examples)
    total_calls = num_examples * MAX_TRIALS

    print("Real Mode Info:")
    print(f"  Dataset: {num_examples} examples × {MAX_TRIALS} trials = {total_calls} LLM calls")
    print("  Costs vary by model - gpt-4o-mini is cheapest, gpt-4o most expensive")
    print("  Check your provider dashboard for actual usage after the run.")
    print()


def print_results_table(trials: list, best_config: dict) -> None:
    """Print a formatted results table showing all trials."""
    if not trials:
        print("\nNo trials to display.")
        return

    print()
    print("=" * 80)
    mode_label = "MOCK" if USE_MOCK else "REAL"
    print(f"  Trial Results ({mode_label} - {len(trials)} trials)")
    print("=" * 80)

    # Table header
    print(f"{'#':>3} | {'Model':<12} | {'Temp':>4} | {'K':>2} | {'Style':<7} | "
          f"{'Quality':>8} | {'Latency':>8} | {'Cost':>10}")
    print("-" * 80)

    # Find best values for highlighting
    best_quality = max(getattr(t, "metrics", {}).get("quality", 0) for t in trials)
    best_latency = min(getattr(t, "metrics", {}).get("latency_p95_ms", float("inf")) for t in trials)
    best_cost = min(getattr(t, "metrics", {}).get("cost_usd_per_1k", float("inf")) for t in trials)

    for i, trial in enumerate(trials, 1):
        config = getattr(trial, "config", {})
        metrics = getattr(trial, "metrics", {})

        model = config.get("model", "?")[:12]
        temp = config.get("temperature", 0)
        k = config.get("retriever_k", 0)
        style = config.get("prompt_style", "?")[:7]

        quality = metrics.get("quality", 0)
        latency = metrics.get("latency_p95_ms", 0)
        cost = metrics.get("cost_usd_per_1k", 0)

        # Mark best values with asterisk
        q_mark = "*" if quality == best_quality else " "
        l_mark = "*" if latency == best_latency else " "
        c_mark = "*" if cost == best_cost else " "

        # Mark overall best config
        is_best = config == best_config
        row_mark = ">" if is_best else " "

        print(f"{row_mark}{i:>2} | {model:<12} | {temp:>4.1f} | {k:>2} | {style:<7} | "
              f"{quality:>7.1%}{q_mark} | {latency:>6.0f}ms{l_mark} | ${cost:>8.5f}{c_mark}")

    print("-" * 80)
    print("Legend: * = Best in column, > = Overall best configuration")
    print()


def print_best_config(best_config: dict, best_metrics: dict) -> None:
    """Print the best configuration found."""
    print("Best Configuration Found:")
    print("-" * 40)
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print()

    # Only show the metrics we're optimizing for (not internal system metrics)
    print("Best Metrics:")
    print("-" * 40)
    for key in OBJECTIVES:
        if key in best_metrics:
            value = best_metrics[key]
            if "quality" in key:
                print(f"  {key}: {value:.1%}")
            elif "cost" in key:
                print(f"  {key}: ${value:.5f}")
            elif "latency" in key:
                print(f"  {key}: {value:.0f}ms")
            else:
                print(f"  {key}: {value:.4f}")
    print()


def print_sample_answer(question: str, answer: str, expected: str) -> None:
    """Print a sample question and answer for verification."""
    print("Sample Question & Answer:")
    print("-" * 40)
    print(f"  Q: {question}")
    print()
    print("  Model Answer:")
    # Indent the answer for readability
    for line in answer.split("\n"):
        print(f"    {line}")
    print()
    print(f"  Expected: {expected}")
    print()


# ============================================================================
# LLM Invocation (Real Mode) - Using LangChain for automatic token tracking
# ============================================================================

def _validate_real_credentials(model: str) -> None:
    """Validate that required API keys are set for real mode."""
    lowered = model.lower()
    if "gpt" in lowered and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not set. Required for GPT models in real mode.\n"
            "Set it with: export OPENAI_API_KEY='your-key'\n"
            "Or run in mock mode: python run_demo.py (no --real flag)"
        )
    if any(
        keyword in lowered for keyword in ("claude", "haiku", "sonnet")
    ) and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Required for Claude/Haiku models in real mode.\n"
            "Set it with: export ANTHROPIC_API_KEY='your-key'\n"
            "Or run in mock mode: python run_demo.py (no --real flag)"
        )


def _create_llm_client(model: str, temperature: float, max_tokens: int) -> Any:
    """Create LangChain client for the model.

    Using LangChain allows Traigent to automatically track token usage and cost
    without needing manual thread-local storage workarounds.
    """
    lowered = model.lower()

    if "gpt" in lowered:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "langchain-openai not installed. Install with: pip install langchain-openai"
            ) from exc
        return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)

    elif any(k in lowered for k in ("claude", "haiku", "sonnet")):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "langchain-anthropic not installed. Install with: pip install langchain-anthropic"
            ) from exc
        # Resolve user-friendly names to actual model IDs
        if "haiku" in lowered:
            resolved_model = "claude-3-haiku-20240307"
        elif "sonnet" in lowered:
            resolved_model = "claude-3-5-sonnet-20241022"
        else:
            resolved_model = model
        return ChatAnthropic(model=resolved_model, temperature=temperature, max_tokens=max_tokens)

    else:
        raise ValueError(f"Unknown model: {model}")


def generate_real_answer(
    question: str,
    context: list[str],
    config: Mapping[str, Any] | dict[str, Any],
) -> str:
    """Generate answer using LangChain (Traigent auto-tracks tokens)."""
    model = str(config.get("model", "gpt-4o-mini"))
    temperature = float(config.get("temperature", 0.3))
    retriever_k = max(1, int(config.get("retriever_k", 4)))
    prompt_style = str(config.get("prompt_style", "vanilla"))
    reranker = str(config.get("retrieval_reranker", config.get("reranker", "none")))
    max_tokens = int(config.get("max_output_tokens", 384))

    _validate_real_credentials(model)

    # Select top-k context passages
    selected_context = context[:retriever_k] if context else []
    context_block = (
        "\n\n".join(selected_context)
        if selected_context
        else "No retrieved context provided."
    )

    # Build prompt based on style
    # Key insight: HotpotQA scoring (EM + F1) penalizes verbose answers heavily.
    # We need to be very explicit about the answer format to get good scores.
    if prompt_style == "cot":
        reasoning_instruction = (
            "Think step by step and explain how each retrieved passage contributes "
            "to answering the question."
        )
    else:
        reasoning_instruction = "Use evidence from the retrieved passages to answer."

    prompt = (
        f"Question: {question}\n\n"
        f"Retrieved context (top-{retriever_k}, reranker={reranker}):\n{context_block}\n\n"
        f"{reasoning_instruction}\n\n"
        "IMPORTANT: End your response with 'Answer:' followed by ONLY the final answer "
        "(a few words maximum - e.g., 'Answer: Yes' or 'Answer: Paris'). "
        "Do not repeat the question or add explanations after 'Answer:'."
    )

    # Use LangChain client - Traigent automatically captures token usage
    try:
        llm = _create_llm_client(model, temperature, max_tokens)
        response = llm.invoke(prompt)
        return str(response.content).strip()
    except Exception as e:
        # Return error indicator instead of crashing the entire optimization
        error_type = type(e).__name__
        error_msg = str(e)[:100]
        return f"[Error calling {model}: {error_type} - {error_msg}]"


# ============================================================================
# Traigent-Optimized Agent
# ============================================================================

@traigent.optimize(
    eval_dataset=DATASET,
    objectives=OBJECTIVES,
    configuration_space=CONFIG_SPACE,
    metric_functions=build_hotpot_metric_functions(mock_mode=USE_MOCK),
    mock_mode_config={"enabled": USE_MOCK, "override_evaluator": False},
)
def hotpot_agent(question: str, context: list[str] | None = None) -> str:
    """Multi-hop QA agent optimized by Traigent.

    This function answers questions that require combining evidence from
    multiple passages (multi-hop reasoning). Traigent optimizes the
    configuration (model, temperature, retrieval depth, prompt style)
    to maximize quality while minimizing cost and latency.

    Args:
        question: The question to answer
        context: List of context passages (from HotpotQA dataset)

    Returns:
        The model's answer to the question
    """
    context = context or []
    config: dict[str, Any] = traigent.get_config()

    if USE_MOCK:
        return generate_case_study_answer(question, config)
    return generate_real_answer(question, context, config)


# ============================================================================
# Main Entry Point
# ============================================================================

async def main() -> None:
    """Run the HotpotQA optimization demo."""
    # Print header and configuration
    print_header()
    print_optimization_config()
    print_cost_warning()

    print("Starting optimization...")
    if not USE_MOCK:
        print("(Real mode - making actual LLM API calls)")
    print()

    # Use progress callback in real mode for visibility during slower API calls
    callbacks = [ProgressBarCallback()] if not USE_MOCK else []

    # Run optimization
    result = await hotpot_agent.optimize(
        algorithm="optuna_nsga2",
        max_trials=MAX_TRIALS,
        timeout=TIMEOUT_SECONDS,
        parallel_config={"trial_concurrency": PARALLEL_TRIALS},
        callbacks=callbacks,
    )

    # Display results
    trials = getattr(result, "trials", [])
    print_results_table(trials, result.best_config)
    print_best_config(result.best_config, result.best_metrics)

    # Show sample answer with best config
    dataset = load_case_study_dataset()
    example = dataset.examples[0]
    hotpot_agent.set_config(result.best_config)

    sample_answer = hotpot_agent(
        question=str(example.input_data.get("question", "")),
        context=list(example.input_data.get("context", [])),
    )

    print_sample_answer(
        question=str(example.input_data.get("question", "")),
        answer=sample_answer,
        expected=str(example.expected_output),
    )

    print("=" * 60)
    print("  Demo Complete!")
    print("=" * 60)
    print()
    if USE_MOCK:
        print("This was a MOCK run. To use real LLM APIs:")
        print("  1. Set your API key: export OPENAI_API_KEY='your-key'")
        print("  2. Run: python run_demo.py --real")
    else:
        print("This was a REAL run using live LLM APIs.")
        print("Check your provider dashboard for actual usage and costs.")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
