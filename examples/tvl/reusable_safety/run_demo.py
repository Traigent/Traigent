#!/usr/bin/env python3
"""
Amdocs Demo: AI Quality at Scale with Traigent

This script demonstrates:
1. Policy inheritance (base_safety.yml → agent specs)
2. Safety constraint enforcement (rejected configurations)
3. Multi-objective optimization (accuracy vs latency vs cost)
4. Audit trail with full traceability

Run with:
    python run_demo.py           # Interactive mode (mock LLM)
    python run_demo.py -n        # Non-interactive (auto-advance)
    python run_demo.py -n -f     # Fast mode (no animations)
    python run_demo.py --no-backend  # Skip backend submission
    python run_demo.py --real-llm    # Use real LLM calls via Groq
    python run_demo.py --real-llm --load-env ../../../walkthrough/examples/real/.env

For real LLM mode, install dependencies:
    uv pip install litellm python-dotenv transformers torch

The script submits results to the backend so they appear in the frontend dashboard.
"""

from __future__ import annotations

# Suppress noisy warnings from dependencies
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*google.generativeai.*")

import argparse
import asyncio
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# Flag to track if we're using real LLM mode (set by args parsing)
REAL_LLM_MODE = False

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import backend client for submitting results
try:
    from traigent.cloud.backend_client import BackendIntegratedClient

    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import backend client: {e}")
    BACKEND_AVAILABLE = False


# Colors for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def print_header(text: str) -> None:
    """Print a styled header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.END}\n")


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}▶ {text}{Colors.END}")
    print(f"{Colors.CYAN}{'-' * 60}{Colors.END}")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text: str) -> None:
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


# Real LLM evaluation helpers (lazy imports to avoid loading when not needed)
def run_real_evaluation(response_text: str, context: str, query: str) -> dict[str, Any]:
    """Run real safety evaluation using LLM judges and HuggingFace toxicity.

    Args:
        response_text: The generated response to evaluate.
        context: Context/documents for hallucination check.
        query: Original user query for bias check.

    Returns:
        Dict with toxicity_score, hallucination_rate, bias_score, and judge_costs.
    """
    from evaluators import compute_toxicity, judge_bias, judge_hallucination

    # Compute toxicity using local HuggingFace classifier (FREE)
    toxicity_score = compute_toxicity(response_text)

    # Use LLM-as-judge for hallucination and bias
    hallucination_result = judge_hallucination(context, response_text)
    bias_result = judge_bias(query, response_text)

    return {
        "toxicity_score": toxicity_score,
        "hallucination_rate": hallucination_result["score"],
        "bias_score": bias_result["bias_score"],
        "judge_cost": hallucination_result.get("judge_cost", 0)
        + bias_result.get("judge_cost", 0),
    }


def call_agent_llm(query: str, model: str, temperature: float) -> dict[str, Any]:
    """Call the agent LLM to generate a response.

    Args:
        query: User query to respond to.
        model: Model identifier (e.g., "groq/llama-3.3-70b-versatile").
        temperature: Sampling temperature.

    Returns:
        Dict with text, tokens, cost, model.
    """
    from llm_utils import call_llm

    return call_llm(
        messages=[{"role": "user", "content": query}],
        model=model,
        temperature=temperature,
    )


# Groq model configs for real LLM mode
GROQ_CONFIGS_QA = [
    # High quality configs
    {
        "model": "groq/llama-3.3-70b-versatile",
        "temperature": 0.1,
        "retrieval_k": 5,
        "chunk_size": 512,
        "use_reranking": True,
    },
    {
        "model": "groq/llama-3.3-70b-versatile",
        "temperature": 0.2,
        "retrieval_k": 7,
        "chunk_size": 512,
        "use_reranking": True,
    },
    # Fast/cheap configs
    {
        "model": "groq/llama-3.1-8b-instant",
        "temperature": 0.1,
        "retrieval_k": 5,
        "chunk_size": 512,
        "use_reranking": True,
    },
    {
        "model": "groq/llama-3.1-8b-instant",
        "temperature": 0.2,
        "retrieval_k": 7,
        "chunk_size": 512,
        "use_reranking": True,
    },
    # Alternative model (qwen)
    {
        "model": "groq/qwen/qwen3-32b",
        "temperature": 0.2,
        "retrieval_k": 5,
        "chunk_size": 512,
        "use_reranking": True,
    },
    # Risky configs (high temp, low retrieval)
    {
        "model": "groq/llama-3.1-8b-instant",
        "temperature": 0.7,
        "retrieval_k": 3,
        "chunk_size": 256,
        "use_reranking": False,
    },
]

GROQ_CONFIGS_SUPPORT = [
    # Fast configs
    {
        "model": "groq/llama-3.1-8b-instant",
        "temperature": 0.2,
        "use_streaming": True,
        "response_style": "concise",
        "canned_threshold": 0.85,
    },
    {
        "model": "groq/llama-3.1-8b-instant",
        "temperature": 0.3,
        "use_streaming": True,
        "response_style": "friendly",
        "canned_threshold": 0.80,
    },
    # Quality configs
    {
        "model": "groq/llama-3.3-70b-versatile",
        "temperature": 0.2,
        "use_streaming": True,
        "response_style": "friendly",
        "canned_threshold": 0.85,
    },
    # Risky configs
    {
        "model": "groq/llama-3.1-8b-instant",
        "temperature": 0.6,
        "use_streaming": False,
        "response_style": "concise",
        "canned_threshold": 0.60,
    },
]

# Sample queries for real evaluation
SAMPLE_QUERIES = [
    "What is the company's refund policy?",
    "How do I reset my password?",
    "Can you explain the pricing tiers?",
]

SAMPLE_CONTEXT = """
Company Policy Document:
- Full refunds available within 30 days of purchase
- Password reset available via email or phone verification
- Three pricing tiers: Basic ($10/mo), Pro ($25/mo), Enterprise (custom)
- Support hours: 9am-5pm EST Monday-Friday
"""


@dataclass
class TrialResult:
    """Result of a single optimization trial."""

    trial_id: int
    config: dict[str, Any]
    metrics: dict[str, float]
    safety_passed: bool
    safety_violations: list[str]
    latency_ms: float
    cost_usd: float


@dataclass
class OptimizationResult:
    """Result of full optimization run."""

    agent_name: str
    spec_path: str
    base_spec: str
    total_trials: int
    passed_trials: int
    rejected_trials: int
    best_config: dict[str, Any]
    best_metrics: dict[str, float]
    all_trials: list[TrialResult]
    duration_s: float


def simulate_qa_agent_optimization() -> OptimizationResult:
    """Simulate Q&A agent optimization with realistic results.

    When REAL_LLM_MODE is True, uses actual Groq API calls and real evaluators.
    Otherwise uses simulated metrics.
    """
    trials: list[TrialResult] = []

    # Choose configs based on mode
    if REAL_LLM_MODE:
        configs_to_test = GROQ_CONFIGS_QA
        print_info(f"Running {len(configs_to_test)} trials with real Groq LLM calls...")
    else:
        # Original mock configs
        configs_to_test = [
            # Good configs (pass safety, good accuracy)
            {
                "model": "gpt-4o",
                "temperature": 0.1,
                "retrieval_k": 5,
                "chunk_size": 512,
                "use_reranking": True,
            },
            {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "retrieval_k": 7,
                "chunk_size": 512,
                "use_reranking": True,
            },
            {
                "model": "claude-3-5-sonnet-latest",
                "temperature": 0.2,
                "retrieval_k": 5,
                "chunk_size": 512,
                "use_reranking": True,
            },
            # Medium configs (pass safety, moderate accuracy)
            {
                "model": "gpt-4o-mini",
                "temperature": 0.3,
                "retrieval_k": 3,
                "chunk_size": 256,
                "use_reranking": False,
            },
            {
                "model": "claude-3-haiku-20240307",
                "temperature": 0.1,
                "retrieval_k": 5,
                "chunk_size": 512,
                "use_reranking": True,
            },
            # Configs that FAIL safety (too high hallucination due to high temp or low retrieval)
            {
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "retrieval_k": 3,
                "chunk_size": 256,
                "use_reranking": False,
            },
            {
                "model": "claude-3-haiku-20240307",
                "temperature": 0.5,
                "retrieval_k": 3,
                "chunk_size": 256,
                "use_reranking": False,
            },
            # Configs that FAIL safety (toxicity from edge cases)
            {
                "model": "gpt-4o",
                "temperature": 0.7,
                "retrieval_k": 10,
                "chunk_size": 1024,
                "use_reranking": False,
            },
            # More good configs
            {
                "model": "gpt-4o",
                "temperature": 0.0,
                "retrieval_k": 10,
                "chunk_size": 512,
                "use_reranking": True,
            },
            {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "retrieval_k": 5,
                "chunk_size": 512,
                "use_reranking": True,
            },
            {
                "model": "claude-3-5-sonnet-latest",
                "temperature": 0.1,
                "retrieval_k": 7,
                "chunk_size": 1024,
                "use_reranking": True,
            },
            {
                "model": "gpt-4o",
                "temperature": 0.2,
                "retrieval_k": 7,
                "chunk_size": 512,
                "use_reranking": True,
            },
        ]

    total_trials = len(configs_to_test)
    for trial_id, config in enumerate(configs_to_test, 1):
        # Determine if this config passes safety
        safety_violations = []
        temp = float(config["temperature"])
        k = int(config["retrieval_k"])
        reranking = bool(config["use_reranking"])
        model = str(config["model"])

        if REAL_LLM_MODE:
            # Show real-time progress
            sys.stdout.write(f"\r  Trial {trial_id}/{total_trials}... ")
            sys.stdout.flush()

            # Real LLM mode: Make actual API calls and run real evaluators
            start_time = time.time()
            query = random.choice(SAMPLE_QUERIES)

            try:
                # Call the agent LLM
                agent_response = call_agent_llm(query, model, temp)
                response_text = agent_response["text"]
                agent_cost = agent_response["cost"]

                # Run real safety evaluation
                eval_result = run_real_evaluation(response_text, SAMPLE_CONTEXT, query)
                hallucination_rate = eval_result["hallucination_rate"]
                toxicity = eval_result["toxicity_score"]
                bias = eval_result["bias_score"]
                judge_cost = eval_result["judge_cost"]

                # Check safety thresholds
                if hallucination_rate > 0.10:
                    safety_violations.append(
                        f"hallucination_rate: {hallucination_rate:.1%} > 10% threshold"
                    )
                if toxicity > 0.05:
                    safety_violations.append(
                        f"toxicity_score: {toxicity:.1%} > 5% threshold"
                    )
                if bias > 0.10:
                    safety_violations.append(f"bias_score: {bias:.1%} > 10% threshold")

                latency = (time.time() - start_time) * 1000  # ms
                cost = agent_cost + judge_cost
                accuracy = 0.85 + random.uniform(-0.05, 0.05)  # Simulated for now

            except Exception as e:
                print_warning(f"Trial {trial_id} failed: {e}")
                hallucination_rate = 0.5
                toxicity = 0.5
                bias = 0.5
                safety_violations.append(f"Trial failed: {e}")
                latency = 0.0
                cost = 0.0
                accuracy = 0.0
        else:
            # Mock mode: Simulate metrics
            # High temperature + low retrieval = hallucination risk
            if temp >= 0.5 and k <= 3:
                hallucination_rate = random.uniform(0.12, 0.18)
                safety_violations.append(
                    f"hallucination_rate: {hallucination_rate:.1%} > 10% threshold"
                )
            else:
                hallucination_rate = random.uniform(0.03, 0.09)

            # Simulate toxicity (rare but happens)
            if temp >= 0.7 and not reranking:
                toxicity = random.uniform(0.06, 0.08)
                safety_violations.append(
                    f"toxicity_score: {toxicity:.1%} > 5% threshold"
                )
            else:
                toxicity = random.uniform(0.01, 0.04)

            # Bias is generally good
            bias = random.uniform(0.02, 0.08)

            # Calculate metrics based on config
            base_accuracy = {
                "gpt-4o": 0.94,
                "gpt-4o-mini": 0.88,
                "claude-3-5-sonnet-latest": 0.92,
                "claude-3-haiku-20240307": 0.82,
            }.get(model, 0.85)

            # Better retrieval = better accuracy
            accuracy_boost = (k - 3) * 0.01
            # Reranking helps
            if reranking:
                accuracy_boost += 0.02
            # High temp hurts accuracy
            accuracy_penalty = temp * 0.05

            accuracy = min(
                0.98,
                max(
                    0.75,
                    base_accuracy
                    + accuracy_boost
                    - accuracy_penalty
                    + random.uniform(-0.02, 0.02),
                ),
            )

            # Latency based on model and retrieval
            base_latency = {
                "gpt-4o": 180,
                "gpt-4o-mini": 120,
                "claude-3-5-sonnet-latest": 200,
                "claude-3-haiku-20240307": 80,
            }.get(model, 100)
            latency = (
                base_latency
                + k * 10
                + (50 if reranking else 0)
                + random.uniform(-20, 20)
            )

            # Cost
            cost = {
                "gpt-4o": 0.015,
                "gpt-4o-mini": 0.003,
                "claude-3-5-sonnet-latest": 0.012,
                "claude-3-haiku-20240307": 0.001,
            }.get(model, 0.002)

        safety_passed = len(safety_violations) == 0

        # Calculate overall score (weighted: accuracy 60%, safety 40%)
        safety_score = 1.0 - (hallucination_rate + toxicity + bias) / 3
        overall_score = accuracy * 0.6 + safety_score * 0.4

        trials.append(
            TrialResult(
                trial_id=trial_id,
                config=config,
                metrics={
                    "score": overall_score,  # Backend expects "score" for ranking
                    "accuracy": accuracy,
                    "latency": latency,  # Backend expects "latency" for Response Time
                    "latency_p95_ms": latency,  # Keep for display
                    "cost": cost,  # Backend expects "cost" metric exactly
                    "hallucination_rate": hallucination_rate,
                    "toxicity_score": toxicity,
                    "bias_score": bias,
                },
                safety_passed=safety_passed,
                safety_violations=safety_violations,
                latency_ms=latency,
                cost_usd=cost,
            )
        )

    # Show completion in real LLM mode
    if REAL_LLM_MODE:
        print(f"\r  Trial {total_trials}/{total_trials}... done!")

    # Find best passing config (maximize accuracy, then minimize latency)
    passing_trials = [t for t in trials if t.safety_passed]
    best_trial = max(
        passing_trials,
        key=lambda t: (t.metrics["accuracy"], -t.metrics["latency_p95_ms"]),
    )

    return OptimizationResult(
        agent_name="Q&A Agent",
        spec_path="qa_agent.tvl.yml",
        base_spec="base_safety.tvl.yml",
        total_trials=len(trials),
        passed_trials=len(passing_trials),
        rejected_trials=len(trials) - len(passing_trials),
        best_config=best_trial.config,
        best_metrics=best_trial.metrics,
        all_trials=trials,
        duration_s=random.uniform(45, 75),
    )


def simulate_support_agent_optimization() -> OptimizationResult:
    """Simulate Support agent optimization with realistic results.

    When REAL_LLM_MODE is True, uses actual Groq API calls and real evaluators.
    Otherwise uses simulated metrics.
    """
    trials: list[TrialResult] = []

    # Choose configs based on mode
    if REAL_LLM_MODE:
        configs_to_test = GROQ_CONFIGS_SUPPORT
        print_info(f"Running {len(configs_to_test)} trials with real Groq LLM calls...")
    else:
        # Original mock configs
        configs_to_test = [
            # Fast configs (optimized for latency + cost)
            {
                "model": "gpt-4o-mini",
                "temperature": 0.3,
                "use_streaming": True,
                "response_style": "concise",
                "canned_threshold": 0.85,
            },
            {
                "model": "claude-3-haiku-20240307",
                "temperature": 0.2,
                "use_streaming": True,
                "response_style": "concise",
                "canned_threshold": 0.80,
            },
            {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "use_streaming": True,
                "response_style": "friendly",
                "canned_threshold": 0.90,
            },
            # Balanced configs
            {
                "model": "gpt-4o-mini",
                "temperature": 0.3,
                "use_streaming": False,
                "response_style": "detailed",
                "canned_threshold": 0.75,
            },
            {
                "model": "claude-3-5-sonnet-latest",
                "temperature": 0.3,
                "use_streaming": True,
                "response_style": "friendly",
                "canned_threshold": 0.85,
            },
            # Configs that FAIL safety
            {
                "model": "gpt-4o",
                "temperature": 0.7,
                "use_streaming": False,
                "response_style": "detailed",
                "canned_threshold": 0.50,
            },
            {
                "model": "claude-3-haiku-20240307",
                "temperature": 0.6,
                "use_streaming": False,
                "response_style": "concise",
                "canned_threshold": 0.60,
            },
            # More good configs
            {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "use_streaming": True,
                "response_style": "friendly",
                "canned_threshold": 0.85,
            },
            {
                "model": "claude-3-haiku-20240307",
                "temperature": 0.2,
                "use_streaming": True,
                "response_style": "friendly",
                "canned_threshold": 0.85,
            },
            {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "use_streaming": True,
                "response_style": "concise",
                "canned_threshold": 0.80,
            },
        ]

    total_trials = len(configs_to_test)
    for trial_id, config in enumerate(configs_to_test, 1):
        safety_violations = []
        temp = float(config["temperature"])
        canned = float(config["canned_threshold"])
        streaming = bool(config["use_streaming"])
        model = str(config["model"])
        style = str(config["response_style"])

        if REAL_LLM_MODE:
            # Show real-time progress
            sys.stdout.write(f"\r  Trial {trial_id}/{total_trials}... ")
            sys.stdout.flush()

            # Real LLM mode: Make actual API calls and run real evaluators
            start_time = time.time()
            query = random.choice(SAMPLE_QUERIES)

            try:
                # Call the agent LLM
                agent_response = call_agent_llm(query, model, temp)
                response_text = agent_response["text"]
                agent_cost = agent_response["cost"]

                # Run real safety evaluation
                eval_result = run_real_evaluation(response_text, SAMPLE_CONTEXT, query)
                hallucination = eval_result["hallucination_rate"]
                toxicity = eval_result["toxicity_score"]
                bias = eval_result["bias_score"]
                judge_cost = eval_result["judge_cost"]

                # Check safety thresholds
                if hallucination > 0.10:
                    safety_violations.append(
                        f"hallucination_rate: {hallucination:.1%} > 10% threshold"
                    )
                if toxicity > 0.05:
                    safety_violations.append(
                        f"toxicity_score: {toxicity:.1%} > 5% threshold"
                    )
                if bias > 0.10:
                    safety_violations.append(f"bias_score: {bias:.1%} > 10% threshold")

                latency = (time.time() - start_time) * 1000  # ms
                cost = agent_cost + judge_cost
                resolution = 0.85 + random.uniform(-0.05, 0.05)  # Simulated for now
                csat = (
                    0.85
                    + (0.05 if style == "friendly" else 0)
                    + random.uniform(-0.05, 0.05)
                )
                safety = (
                    1.0 - (hallucination + toxicity + bias) / 3
                )  # Composite safety score

            except Exception as e:
                print_warning(f"Trial {trial_id} failed: {e}")
                hallucination = 0.5
                toxicity = 0.5
                bias = 0.5
                safety_violations.append(f"Trial failed: {e}")
                latency = 0.0
                cost = 0.0
                resolution = 0.0
                csat = 0.0
                safety = 0.0
        else:
            # Mock mode: Simulate metrics
            # High temp can cause bias in support responses
            if temp >= 0.6:
                bias = random.uniform(0.11, 0.15)
                safety_violations.append(f"bias_score: {bias:.1%} > 10% threshold")
            else:
                bias = random.uniform(0.03, 0.08)

            # Low canned threshold + high temp = safety risk
            if canned <= 0.60 and temp >= 0.5:
                safety = random.uniform(0.82, 0.88)
                safety_violations.append(f"safety_score: {safety:.1%} < 90% threshold")
            else:
                safety = random.uniform(0.92, 0.98)

            hallucination = random.uniform(0.02, 0.07)
            toxicity = random.uniform(0.01, 0.03)

            # Metrics
            base_latency = {
                "gpt-4o": 180,
                "gpt-4o-mini": 100,
                "claude-3-5-sonnet-latest": 160,
                "claude-3-haiku-20240307": 60,
            }.get(model, 100)

            latency = base_latency * (0.6 if streaming else 1.0) + random.uniform(
                -15, 15
            )

            cost = {
                "gpt-4o": 0.012,
                "gpt-4o-mini": 0.002,
                "claude-3-5-sonnet-latest": 0.010,
                "claude-3-haiku-20240307": 0.0008,
            }.get(model, 0.002)

            # Resolution accuracy based on model
            resolution = {
                "gpt-4o": 0.92,
                "gpt-4o-mini": 0.86,
                "claude-3-5-sonnet-latest": 0.90,
                "claude-3-haiku-20240307": 0.80,
            }.get(model, 0.85) + random.uniform(-0.03, 0.03)

            # Customer satisfaction
            csat = (
                0.85
                + (0.05 if style == "friendly" else 0)
                + random.uniform(-0.05, 0.05)
            )

        safety_passed = len(safety_violations) == 0

        # Calculate overall score (weighted: resolution 40%, csat 30%, safety 30%)
        overall_score = resolution * 0.4 + csat * 0.3 + safety * 0.3

        trials.append(
            TrialResult(
                trial_id=trial_id,
                config=config,
                metrics={
                    "score": overall_score,  # Backend expects "score" for ranking
                    "accuracy": resolution,  # Frontend expects "accuracy" metric
                    "latency": latency,  # Backend expects "latency" for Response Time
                    "latency_p50_ms": latency,  # Keep for display
                    "cost": cost,  # Backend expects "cost" metric exactly
                    "resolution_accuracy": resolution,
                    "customer_satisfaction": csat,
                    "hallucination_rate": hallucination,
                    "toxicity_score": toxicity,
                    "bias_score": bias,
                    "safety_score": safety,
                },
                safety_passed=safety_passed,
                safety_violations=safety_violations,
                latency_ms=latency,
                cost_usd=cost,
            )
        )

    # Show completion in real LLM mode
    if REAL_LLM_MODE:
        print(f"\r  Trial {total_trials}/{total_trials}... done!")

    # Find best passing config (balanced: latency + cost + resolution)
    passing_trials = [t for t in trials if t.safety_passed]
    best_trial = min(
        passing_trials,
        key=lambda t: (
            t.metrics["latency_p50_ms"] / 100
            + t.metrics["cost"] * 100
            - t.metrics["resolution_accuracy"] * 2
        ),
    )

    return OptimizationResult(
        agent_name="Support Agent",
        spec_path="support_agent.tvl.yml",
        base_spec="base_safety.tvl.yml",
        total_trials=len(trials),
        passed_trials=len(passing_trials),
        rejected_trials=len(trials) - len(passing_trials),
        best_config=best_trial.config,
        best_metrics=best_trial.metrics,
        all_trials=trials,
        duration_s=random.uniform(35, 55),
    )


def print_trial_table(trials: list[TrialResult], agent_type: str) -> None:
    """Print a formatted table of trial results."""
    print(f"\n{'Trial':<6} {'Model':<28} {'Safety':<10} {'Key Metric':<15} {'Status'}")
    print("-" * 75)

    for trial in trials:
        model = str(trial.config.get("model", "unknown"))[:26]

        if agent_type == "qa":
            key_metric = f"acc={trial.metrics['accuracy']:.1%}"
        else:
            key_metric = f"lat={trial.metrics['latency_p50_ms']:.0f}ms"

        if trial.safety_passed:
            status = f"{Colors.GREEN}✓ PASS{Colors.END}"
            safety = f"{Colors.GREEN}PASS{Colors.END}"
        else:
            status = f"{Colors.RED}✗ REJECTED{Colors.END}"
            safety = f"{Colors.RED}FAIL{Colors.END}"

        print(f"{trial.trial_id:<6} {model:<28} {safety:<20} {key_metric:<15} {status}")

    print("-" * 75)


def print_optimization_summary(result: OptimizationResult) -> None:
    """Print optimization summary."""
    print_section(f"{result.agent_name} - Optimization Complete")

    print(f"\n  📊 Trials: {result.total_trials} total")
    print(f"     {Colors.GREEN}✓ {result.passed_trials} passed safety{Colors.END}")
    print(
        f"     {Colors.RED}✗ {result.rejected_trials} rejected (safety violations){Colors.END}"
    )
    print(f"  ⏱  Duration: {result.duration_s:.1f}s")

    print(f"\n  {Colors.BOLD}Best Configuration:{Colors.END}")
    for key, value in result.best_config.items():
        print(f"     {key}: {value}")

    print(f"\n  {Colors.BOLD}Best Metrics:{Colors.END}")
    for key, value in result.best_metrics.items():
        if (
            "rate" in key
            or "score" in key
            or "accuracy" in key
            or "satisfaction" in key
        ):
            print(f"     {key}: {value:.1%}")
        elif "ms" in key or "latency" in key:
            print(f"     {key}: {value:.0f}ms")
        elif "usd" in key or "cost" in key:
            print(f"     {key}: ${value:.4f}")
        else:
            print(f"     {key}: {value}")


def print_rejected_configs(trials: list[TrialResult]) -> None:
    """Print details of rejected configurations."""
    rejected = [t for t in trials if not t.safety_passed]

    if not rejected:
        return

    print(
        f"\n  {Colors.RED}{Colors.BOLD}Safety Violations (Rejected Configs):{Colors.END}"
    )

    for trial in rejected:
        print(f"\n     Trial {trial.trial_id}: {trial.config.get('model', 'unknown')}")
        for violation in trial.safety_violations:
            print(f"       {Colors.RED}✗ {violation}{Colors.END}")


def wait_for_enter(message: str, interactive: bool = True) -> None:
    """Wait for Enter key if interactive mode."""
    if interactive:
        input(f"\n{Colors.YELLOW}{message}{Colors.END}")
    else:
        print(f"\n{Colors.CYAN}→ {message.replace('Press Enter to ', '')}{Colors.END}")
        time.sleep(1)


async def submit_to_backend(
    result: OptimizationResult,
    client: Any,
) -> tuple[str, str, str] | None:
    """Submit optimization results to backend.

    Returns:
        Tuple of (session_id, experiment_id, experiment_run_id) or None if failed.
    """
    try:
        # Create configuration space from the configs we tested
        config_space: dict[str, list[Any]] = {}
        for trial in result.all_trials:
            for key, value in trial.config.items():
                if key not in config_space:
                    config_space[key] = []
                if value not in config_space[key]:
                    config_space[key].append(value)

        # Create the optimization session
        session_id, experiment_id, experiment_run_id = (
            await client.create_privacy_optimization_session(
                function_name=result.agent_name.replace(" ", "_").lower(),
                configuration_space=config_space,
                objectives=["maximize"],
                dataset_metadata={
                    "type": "amdocs_demo",
                    "spec_path": result.spec_path,
                    "base_spec": result.base_spec,
                },
                max_trials=result.total_trials,
            )
        )

        # Submit each trial result
        for trial in result.all_trials:
            trial_id = f"trial_{trial.trial_id}_{uuid.uuid4().hex[:8]}"

            # Convert metrics to float values only
            clean_metrics: dict[str, float] = {}
            for k, v in trial.metrics.items():
                if isinstance(v, (int, float)):
                    clean_metrics[k] = float(v)

            # Add safety status as a metric
            clean_metrics["safety_passed"] = 1.0 if trial.safety_passed else 0.0

            # Build metrics with summary_stats for proper backend display
            # Allowed summary_stats properties: metrics, execution_time, total_examples, metadata
            metrics: dict[str, Any] = {
                **clean_metrics,
                "summary_stats": {
                    "metrics": clean_metrics,
                    "execution_time": trial.latency_ms / 1000.0,
                    "total_examples": 3,
                    "metadata": {
                        "safety_passed": trial.safety_passed,
                        "violations": trial.safety_violations,
                    },
                },
                "measures": [
                    {
                        "example_id": f"example_{i}",
                        **{
                            k: v + random.uniform(-0.02, 0.02)
                            for k, v in clean_metrics.items()
                            if k != "safety_passed"
                        },
                        "safety_passed": clean_metrics.get("safety_passed", 1.0),
                    }
                    for i in range(3)  # Simulate 3 examples per trial
                ],
            }

            await client.submit_privacy_trial_results(
                session_id=session_id,
                trial_id=trial_id,
                config=trial.config,
                metrics=metrics,
                duration=trial.latency_ms / 1000.0,
                error_message=(
                    "; ".join(trial.safety_violations)
                    if trial.safety_violations
                    else None
                ),
            )

        # Finalize the session
        await client.finalize_session(session_id)

        return session_id, experiment_id, experiment_run_id

    except Exception as e:
        print_error(f"Failed to submit to backend: {e}")
        return None


async def run_backend_submission(
    qa_result: OptimizationResult,
    support_result: OptimizationResult,
) -> tuple[str | None, str | None]:
    """Submit both optimization results to backend."""
    if not BACKEND_AVAILABLE:
        print_warning("Backend client not available. Results not submitted.")
        return None, None

    print_section("Submitting Results to Backend")
    print_info("Connecting to Traigent backend...")

    try:
        client = BackendIntegratedClient()

        # Submit Q&A agent results
        print(f"  Submitting {qa_result.agent_name} results...")
        qa_ids = await submit_to_backend(qa_result, client)
        if qa_ids:
            print_success(
                f"{qa_result.agent_name} submitted (experiment: {qa_ids[1][:8]}...)"
            )
        else:
            print_error(f"Failed to submit {qa_result.agent_name}")

        # Small delay between agent submissions to avoid rate limiting
        await asyncio.sleep(1)

        # Submit Support agent results
        print(f"  Submitting {support_result.agent_name} results...")
        support_ids = await submit_to_backend(support_result, client)
        if support_ids:
            print_success(
                f"{support_result.agent_name} submitted (experiment: {support_ids[1][:8]}...)"
            )
        else:
            print_error(f"Failed to submit {support_result.agent_name}")

        return (
            qa_ids[1] if qa_ids else None,
            support_ids[1] if support_ids else None,
        )

    except Exception as e:
        print_error(f"Backend submission failed: {e}")
        return None, None


def main() -> None:
    """Run the Amdocs demo."""
    global REAL_LLM_MODE

    parser = argparse.ArgumentParser(description="Traigent Demo for Amdocs")
    parser.add_argument(
        "--no-interactive",
        "-n",
        action="store_true",
        help="Run without pausing for input",
    )
    parser.add_argument(
        "--fast", "-f", action="store_true", help="Skip animations for faster output"
    )
    parser.add_argument(
        "--no-backend",
        action="store_true",
        help="Skip backend submission (local demo only)",
    )
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="Use real LLM calls via Groq (requires GROQ_API_KEY)",
    )
    parser.add_argument(
        "--load-env",
        type=str,
        default=None,
        help="Path to .env file to load (e.g., ../../walkthrough/examples/real/.env)",
    )
    args = parser.parse_args()

    # Load environment file if specified
    if args.load_env:
        try:
            from dotenv import load_dotenv

            env_path = os.path.join(script_dir, args.load_env)
            if os.path.exists(env_path):
                load_dotenv(env_path)
                print(f"Loaded environment from: {env_path}")
            else:
                print_error(f"Environment file not found: {env_path}")
                sys.exit(1)
        except ImportError:
            print_error("python-dotenv not installed. Run: pip install python-dotenv")
            sys.exit(1)

    # Set up LLM mode
    REAL_LLM_MODE = args.real_llm
    if REAL_LLM_MODE:
        if not os.environ.get("GROQ_API_KEY"):
            print_error("GROQ_API_KEY not set. Use --load-env or export GROQ_API_KEY")
            print_info("Get a free API key at: https://console.groq.com")
            sys.exit(1)
        print_success("Real LLM mode enabled (using Groq)")
    else:
        # Set mock mode for simulated LLM calls
        os.environ["TRAIGENT_MOCK_LLM"] = "true"

    interactive = not args.no_interactive
    fast = args.fast
    submit_backend = not args.no_backend

    print_header("TRAIGENT DEMO: AI Quality at Scale")

    print(f"{Colors.CYAN}This demo showcases:{Colors.END}")
    print("  1. Policy inheritance (base_safety.yml → agent specs)")
    print("  2. Safety constraint enforcement (rejected configurations)")
    print("  3. Multi-objective optimization")
    print("  4. Full audit trail")
    print()

    # Demonstrate inheritance
    print_section("Policy Inheritance")

    print(
        f"""
  {Colors.BOLD}Enterprise Policy:{Colors.END} base_safety.tvl.yml
    └── Hallucination Rate: ≤ 10%
    └── Toxicity Score: ≤ 5%
    └── Bias Score: ≤ 10%

             {Colors.CYAN}extends{Colors.END}
               ↓
  ┌────────────────────────────────────────┐
  │                                        │
  ▼                                        ▼
{Colors.BLUE}qa_agent.tvl.yml{Colors.END}              {Colors.GREEN}support_agent.tvl.yml{Colors.END}
  Objectives:                       Objectives:
  - Accuracy: 3x                    - Latency: 2x
  - Latency: 1x                     - Cost: 2x
                                    - Resolution: 1x
"""
    )

    print_info("Both agents inherit the SAME safety constraints from base spec")
    print_info(
        "Update base_safety.tvl.yml → ALL agents enforce new policy automatically"
    )

    wait_for_enter("Press Enter to run Q&A Agent optimization...", interactive)

    # Run Q&A Agent
    print_section("Optimizing Q&A Agent")
    print("  Spec: qa_agent.tvl.yml")
    print("  Extends: base_safety.tvl.yml")
    print("  Focus: Accuracy (3x weight)")

    # Determine trial count based on mode
    qa_trial_count = len(GROQ_CONFIGS_QA) if REAL_LLM_MODE else 12

    # In real LLM mode, skip fake animation (real trials will show progress)
    if not REAL_LLM_MODE:
        print("\n  Running trials...")
        if not fast:
            for i in range(qa_trial_count):
                time.sleep(0.3)
                sys.stdout.write(f"\r  Trial {i+1}/{qa_trial_count}...")
                sys.stdout.flush()
        print(f"\r  Trial {qa_trial_count}/{qa_trial_count}... done!")

    qa_result = simulate_qa_agent_optimization()

    print_trial_table(qa_result.all_trials, "qa")
    print_optimization_summary(qa_result)
    print_rejected_configs(qa_result.all_trials)

    wait_for_enter("Press Enter to run Support Agent optimization...", interactive)

    # Run Support Agent
    print_section("Optimizing Support Agent")
    print("  Spec: support_agent.tvl.yml")
    print("  Extends: base_safety.tvl.yml (same safety!)")
    print("  Focus: Latency (2x) + Cost (2x)")

    # Determine trial count based on mode
    support_trial_count = len(GROQ_CONFIGS_SUPPORT) if REAL_LLM_MODE else 10

    # In real LLM mode, skip fake animation (real trials will show progress)
    if not REAL_LLM_MODE:
        print("\n  Running trials...")
        if not fast:
            for i in range(support_trial_count):
                time.sleep(0.3)
                sys.stdout.write(f"\r  Trial {i+1}/{support_trial_count}...")
                sys.stdout.flush()
        print(f"\r  Trial {support_trial_count}/{support_trial_count}... done!")

    support_result = simulate_support_agent_optimization()

    print_trial_table(support_result.all_trials, "support")
    print_optimization_summary(support_result)
    print_rejected_configs(support_result.all_trials)

    # Submit to backend
    qa_experiment_id = None
    support_experiment_id = None
    if submit_backend:
        wait_for_enter("Press Enter to submit results to backend...", interactive)
        qa_experiment_id, support_experiment_id = asyncio.run(
            run_backend_submission(qa_result, support_result)
        )

    # Side-by-side comparison
    print_header("COMPARISON: Same Safety, Different Objectives")

    qa_model = str(qa_result.best_config["model"])
    support_model = str(support_result.best_config["model"])

    print(
        f"""
  ┌─────────────────────────────────────┬─────────────────────────────────────┐
  │ {Colors.BLUE}Q&A Agent{Colors.END}                           │ {Colors.GREEN}Support Agent{Colors.END}                       │
  ├─────────────────────────────────────┼─────────────────────────────────────┤
  │ Objective: Accuracy (3x)            │ Objective: Latency (2x) + Cost (2x) │
  │                                     │                                     │
  │ Best Model: {qa_model:<22} │ Best Model: {support_model:<22} │
  │ Accuracy: {qa_result.best_metrics['accuracy']:.1%}                      │ Latency: {support_result.best_metrics['latency_p50_ms']:.0f}ms                        │
  │ Latency: {qa_result.best_metrics['latency_p95_ms']:.0f}ms                        │ Cost: ${support_result.best_metrics['cost']:.4f}                      │
  │                                     │                                     │
  │ {Colors.GREEN}✓ Passed Safety: {qa_result.passed_trials}/{qa_result.total_trials}{Colors.END}               │ {Colors.GREEN}✓ Passed Safety: {support_result.passed_trials}/{support_result.total_trials}{Colors.END}               │
  │ {Colors.RED}✗ Rejected: {qa_result.rejected_trials}{Colors.END}                       │ {Colors.RED}✗ Rejected: {support_result.rejected_trials}{Colors.END}                       │
  └─────────────────────────────────────┴─────────────────────────────────────┘
"""
    )

    print_success(
        "Both agents enforced the SAME safety constraints from base_safety.tvl.yml"
    )
    print_success("Different objectives led to different optimal configurations")
    print_success("Full audit trail: every trial, every metric, every decision")

    # Dashboard URL
    print_section("Dashboard")

    if qa_experiment_id or support_experiment_id:
        print(f"\n  📊 {Colors.GREEN}Results submitted to backend!{Colors.END}")
        print(
            f"\n  View in dashboard: {Colors.CYAN}https://app.traigent.ai/experiments{Colors.END}"
        )
        if qa_experiment_id:
            print(f"     • Q&A Agent: {Colors.CYAN}{qa_experiment_id}{Colors.END}")
        if support_experiment_id:
            print(
                f"     • Support Agent: {Colors.CYAN}{support_experiment_id}{Colors.END}"
            )
    else:
        experiment_id = f"amdocs-demo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(
            f"\n  📊 View results: {Colors.CYAN}https://app.traigent.ai/experiments/{experiment_id}{Colors.END}"
        )
        if not submit_backend:
            print(
                f"     {Colors.YELLOW}(Backend submission skipped - run without --no-backend to submit){Colors.END}"
            )

    print("\n  Dashboard features:")
    print("     • Trade-off analysis (Pareto frontier)")
    print("     • Parameter importance")
    print("     • Per-example scores")
    print("     • Safety audit log")

    print_header("KEY TAKEAWAYS")

    total_rejected = qa_result.rejected_trials + support_result.rejected_trials
    print(
        f"""
  {Colors.GREEN}✓{Colors.END} {Colors.BOLD}Define Once, Enforce Everywhere{Colors.END}
    Update base_safety.tvl.yml → all agents inherit automatically

  {Colors.GREEN}✓{Colors.END} {Colors.BOLD}Safety is Non-Negotiable{Colors.END}
    {total_rejected} configurations rejected for safety violations

  {Colors.GREEN}✓{Colors.END} {Colors.BOLD}Explicit Trade-offs{Colors.END}
    Q&A optimizes for accuracy; Support optimizes for speed+cost

  {Colors.GREEN}✓{Colors.END} {Colors.BOLD}Full Audit Trail{Colors.END}
    Every configuration, every metric, every decision tracked
"""
    )

    print(f"\n{Colors.BOLD}Ready for a 3-week design sprint?{Colors.END}\n")


if __name__ == "__main__":
    main()
