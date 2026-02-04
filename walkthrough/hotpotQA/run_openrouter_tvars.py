#!/usr/bin/env python3
"""OpenRouter Optimization with Tuned Variable Presets and Conditional Constraints.

Demonstrates Traigent's domain-specific parameter presets (TVars) for LLM
optimization, plus conditional parameter activation using implies() constraints.

This script showcases:
1. Pre-configured parameter ranges that encode domain knowledge:
   - Range.temperature() - pre-tuned temperature ranges
   - Range.top_p() - nucleus sampling parameter
   - Range.frequency_penalty() - repetition control
   - Range.presence_penalty() - topic diversity
   - IntRange.max_tokens() - output length by task type
   - Choices.prompting_strategy() - common prompting approaches

2. Conditional "Review and Improve" step:
   - use_reviewer: Toggle to enable/disable review step
   - reviewer_model, reviewer_temperature: Only active when use_reviewer=True
   - implies() constraint ensures valid config when review is enabled
   - Demonstrates multi-agent grouping with agent= parameter

Usage:
    source .env.local  # loads OPENROUTER_API_KEY
    python walkthrough/hotpotQA/run_openrouter_tvars.py
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Real mode, online Traigent (not mock, not offline)
os.environ["TRAIGENT_MOCK_LLM"] = "false"
os.environ["TRAIGENT_OFFLINE_MODE"] = "false"
os.environ["TRAIGENT_COST_APPROVED"] = "true"

ROOT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("TRAIGENT_DATASET_ROOT", str(ROOT_DIR))

import logging

# Enable verbose logging to see pruning events
logging.basicConfig(level=logging.INFO)
logging.getLogger("traigent.optimizers.pruners").setLevel(logging.DEBUG)

import traigent
from traigent.api.constraints import implies
from traigent.api.parameter_ranges import Choices, IntRange, Range
from traigent.optimizers import CeilingPruner

# =============================================================================
# OpenRouter Models - Budget-friendly selection (~$0.001/trial)
# =============================================================================
# Using affordable paid models for reliability. Free tier has strict rate limits
# and provider routing issues. These models cost ~$0.15/1M input tokens.
# Total cost for 25 trials × 8 examples ≈ $0.02

OPENROUTER_MODELS = [
    # Budget-friendly models (~$0.15-0.25/1M tokens)
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
]

# Reviewer models - same budget tier
REVIEWER_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
]

DATASET_PATH = ROOT_DIR / "walkthrough" / "examples" / "mock" / "simple_questions.jsonl"


# OpenRouter pricing per 1M tokens (approximate, for cost estimation)
OPENROUTER_PRICING = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "anthropic/claude-3.5-haiku": {"input": 0.25, "output": 1.25},
}


@dataclass
class LLMResult:
    """Result from an LLM call with usage metrics."""

    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str

    @property
    def total_cost(self) -> float:
        """Estimate total cost based on model pricing."""
        pricing = OPENROUTER_PRICING.get(self.model, {"input": 0.15, "output": 0.60})
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


def invoke_openrouter(
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
) -> LLMResult:
    """Call OpenRouter API with full parameter control and rate limit handling.

    Returns LLMResult with content and usage metrics for cost tracking.
    """
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY (source .env.local)")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Retry logic with exponential backoff for rate limits and transient errors
    max_retries = 5
    for attempt in range(max_retries):
        try:
            start = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                extra_headers={
                    "HTTP-Referer": "https://github.com/Traigent/Traigent",
                    "X-Title": "Traigent TVar Presets Demo",
                },
            )
            latency_ms = (time.perf_counter() - start) * 1000

            content = response.choices[0].message.content or ""
            short = model.split("/")[-1]
            output_len = len(content)

            # Extract token usage from response
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0

            print(
                f"  {short}: T={temperature:.2f} P={top_p:.2f} "
                f"freq={frequency_penalty:.1f} pres={presence_penalty:.1f} "
                f"({latency_ms:.0f}ms, {output_len} chars, {input_tokens}+{output_tokens} tok)"
            )

            return LLMResult(
                content=content.strip(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                model=model,
            )
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate limit" in err_str.lower()
            is_transient = "500" in err_str or "502" in err_str or "503" in err_str

            if (is_rate_limit or is_transient) and attempt < max_retries - 1:
                # Exponential backoff: 4s, 8s, 16s, 32s for rate limits
                wait_time = 4 * (2 ** attempt) if is_rate_limit else 2 * (attempt + 1)
                short = model.split("/")[-1]
                reason = "rate limit" if is_rate_limit else "transient error"
                print(f"  [RETRY {attempt+1}/{max_retries}] {short}: {reason}, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            raise

    return LLMResult(content="", input_tokens=0, output_tokens=0, latency_ms=0.0, model=model)


def score_answer(prediction: str, expected: str) -> float:
    """Score answer with LENGTH PENALTY to demonstrate CeilingPruner.

    Scoring rules:
    1. Base score: 1.0 if expected answer is contained in prediction
    2. Length penalty: Penalize verbose outputs (>100 chars get reduced score)
       - This causes verbose_analysis strategy to score poorly
       - Enables demonstration of trial-level pruning

    The length penalty simulates real-world scenarios where:
    - Concise answers are preferred for downstream processing
    - Verbose outputs increase latency and cost
    - Users want direct answers, not essays
    """
    pred = prediction.lower().strip()
    exp = expected.lower().strip()

    if not exp or not pred:
        return 0.0

    # Base score: containment check
    base_score = 0.0
    if exp in pred:
        base_score = 1.0
    else:
        # Word overlap fallback
        pred_words = set(pred.split())
        exp_words = set(exp.split())
        if exp_words:
            overlap = pred_words & exp_words
            base_score = len(overlap) / len(exp_words)

    # LENGTH PENALTY: Penalize verbose outputs
    # - Outputs ≤100 chars: no penalty
    # - Outputs >100 chars: penalty scales with length
    # - At 500+ chars: score reduced by 80%
    pred_len = len(prediction)
    if pred_len <= 100:
        length_penalty = 1.0
    elif pred_len <= 500:
        # Linear penalty from 1.0 to 0.2 as length goes from 100 to 500
        length_penalty = 1.0 - 0.8 * (pred_len - 100) / 400
    else:
        # Severe penalty for very long outputs
        length_penalty = 0.2

    final_score = base_score * length_penalty
    return final_score


def custom_scorer(example: Any, output: str | dict) -> float:
    """Extract expected output and score.

    Handles both plain string outputs and with_usage() wrapped outputs.
    """
    # Handle with_usage() wrapped output (dict with __traigent_meta__)
    if isinstance(output, dict):
        output = output.get("text", str(output))

    expected = example.expected_output
    if isinstance(expected, dict):
        expected = expected.get("output", expected.get("answer", str(expected)))
    expected = str(expected) if expected else ""
    return score_answer(str(output), expected)


# =============================================================================
# Prompt Templates for Different Strategies
# =============================================================================

PROMPT_TEMPLATES = {
    "direct": "Answer the following question directly and concisely:\n\nQ: {question}\n\nA:",
    "chain_of_thought": (
        "Answer the following question. Think step by step before giving "
        "your final answer.\n\nQuestion: {question}\n\nLet me think through this:"
    ),
    "react": (
        "Answer the question using the ReAct format.\n\n"
        "Question: {question}\n\n"
        "Thought: Let me analyze this question.\n"
        "Action: Reason about the answer.\n"
        "Observation:"
    ),
    "self_consistency": (
        "Provide a clear and confident answer to this question. "
        "If you're uncertain, give your best answer anyway.\n\n"
        "Q: {question}\n\nAnswer:"
    ),
    # Deliberately verbose strategy - produces long outputs that won't match short expected answers
    # This is useful for demonstrating trial-level pruning (CeilingPruner)
    "verbose_analysis": (
        "Provide an extremely detailed, multi-paragraph academic analysis of this question. "
        "Include historical context, philosophical implications, multiple perspectives, "
        "and related tangents. Do NOT provide a short answer. "
        "Your response should be at least 500 words.\n\n"
        "Question to analyze in depth: {question}\n\n"
        "Comprehensive Analysis:"
    ),
}

# Review prompt template - used when use_reviewer=True
REVIEW_PROMPT = """Review and improve this answer if needed.

Question: {question}

Initial Answer: {initial_answer}

If the answer is correct and complete, return it as-is.
If it can be improved, provide a better answer.

Final Answer:"""


# =============================================================================
# Define parameters for conditional constraint
# =============================================================================

# Toggle for review step - disabled by default to speed up trials for demo
# Set to True for multi-agent demonstration
use_reviewer = Choices([False], default=False, name="use_reviewer")  # Disabled for faster trials

# Reviewer agent parameters - only relevant when use_reviewer=True
reviewer_model = Choices(
    REVIEWER_MODELS,
    default="openai/gpt-4o-mini",
    name="reviewer_model",
    agent="reviewer",  # Groups in UI under "reviewer" agent
)
reviewer_temperature = Range(
    0.0, 0.3,  # Lower range for reviewer - we want consistent critique
    default=0.1,
    name="reviewer_temperature",
    agent="reviewer",
)


@traigent.optimize(
    eval_dataset=str(DATASET_PATH),
    objectives=["accuracy"],
    configuration_space={
        # Model selection - budget-friendly models (default to gpt-4o-mini)
        "model": Choices(OPENROUTER_MODELS, default="openai/gpt-4o-mini", name="model"),
        # Temperature preset - conservative for factual QA
        "temperature": Range.temperature(conservative=True),
        # Top-p preset - standard nucleus sampling range
        "top_p": Range.top_p(),
        # Frequency penalty preset - control repetition
        "frequency_penalty": Range.frequency_penalty(),
        # Presence penalty preset - topic diversity
        "presence_penalty": Range.presence_penalty(),
        # Max tokens preset - short task (QA answers)
        "max_tokens": IntRange.max_tokens(task="short"),
        # Prompting strategy - only 2 options to force verbose_analysis to be sampled
        # verbose_analysis is deliberately bad to demonstrate CeilingPruner
        "prompting_strategy": Choices(
            ["direct", "verbose_analysis"],
            default="direct",
            name="prompting_strategy",
        ),
        # =====================================================================
        # Conditional Review Step - parameters only active when use_reviewer=True
        # =====================================================================
        "use_reviewer": use_reviewer,
        "reviewer_model": reviewer_model,
        "reviewer_temperature": reviewer_temperature,
    },
    # Constraint: reviewer params must be valid when review is enabled
    # This demonstrates conditional parameter activation
    constraints=[
        implies(
            use_reviewer.equals(True),
            reviewer_model.is_in(REVIEWER_MODELS),
            description="Reviewer model must be from REVIEWER_MODELS when review is enabled",
        ),
    ],
    scoring_function=custom_scorer,
    execution_mode="edge_analytics",
)
def qa_with_tvars(question: str) -> str:
    """QA agent using Traigent's tuned variable presets.

    This function demonstrates:
    - Domain presets that simplify configuration
    - Conditional reviewer step (only runs when use_reviewer=True)
    - Multi-agent parameter grouping with the `agent=` attribute
    - Usage tracking via traigent.with_usage() for accurate cost metrics
    """
    config: dict[str, Any] = traigent.get_config()

    # Extract generator parameters
    model = config.get("model", "openai/gpt-4o-mini")
    temperature = float(config.get("temperature", 0.2))
    top_p = float(config.get("top_p", 0.9))
    frequency_penalty = float(config.get("frequency_penalty", 0.0))
    presence_penalty = float(config.get("presence_penalty", 0.0))
    max_tokens = int(config.get("max_tokens", 128))
    strategy = config.get("prompting_strategy", "direct")

    # Track total usage across LLM calls
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    total_latency_ms = 0.0

    # Build prompt based on strategy
    template = PROMPT_TEMPLATES.get(strategy, PROMPT_TEMPLATES["direct"])
    prompt = template.format(question=question)

    # Step 1: Generate initial answer
    result = invoke_openrouter(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    total_input_tokens += result.input_tokens
    total_output_tokens += result.output_tokens
    total_cost += result.total_cost
    total_latency_ms += result.latency_ms
    final_answer = result.content

    # Step 2: Conditional review step - only runs when use_reviewer=True
    # This demonstrates how constraints ensure reviewer params are only
    # configured when the review step is actually enabled
    if config.get("use_reviewer", False):
        rev_model = config.get("reviewer_model", "openai/gpt-4o-mini")
        rev_temp = float(config.get("reviewer_temperature", 0.1))

        review_prompt = REVIEW_PROMPT.format(
            question=question, initial_answer=final_answer
        )

        print(f"  [REVIEW] {rev_model.split('/')[-1]} reviewing...")
        review_result = invoke_openrouter(
            model=rev_model,
            prompt=review_prompt,
            temperature=rev_temp,
            max_tokens=max_tokens,
            top_p=0.9,  # Fixed for reviewer
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        total_input_tokens += review_result.input_tokens
        total_output_tokens += review_result.output_tokens
        total_cost += review_result.total_cost
        total_latency_ms += review_result.latency_ms
        final_answer = review_result.content

    # Wrap output with usage metadata for Traigent to track
    # This ensures cost/token/latency metrics appear in the dashboard for all trials
    return traigent.with_usage(
        text=final_answer,
        total_cost=total_cost,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        response_time_ms=total_latency_ms,
    )


async def main() -> None:
    print("=" * 70)
    print("OpenRouter Optimization with Tuned Variable Presets")
    print("=" * 70)

    print("\nTuned Variable Presets Used:")
    print("-" * 70)
    print("  Range.temperature(conservative=True)  → [0.0, 0.5] default=0.2")
    print("  Range.top_p()                         → [0.1, 1.0] default=0.9")
    print("  Range.frequency_penalty()             → [0.0, 2.0] default=0.0")
    print("  Range.presence_penalty()              → [0.0, 2.0] default=0.0")
    print("  IntRange.max_tokens(task='short')     → [50, 256] step=64 default=128")
    print("  Choices(prompting_strategy)           → [direct, verbose_analysis*]")
    print("    * verbose_analysis produces long outputs penalized by scorer")
    print("\nScorer: LENGTH-PENALIZED containment")
    print("-" * 70)
    print("  ≤100 chars: full score")
    print("  100-500 chars: linear penalty (1.0 → 0.2)")
    print("  >500 chars: 80% penalty (score × 0.2)")
    print("  → verbose_analysis (~1400 chars) will score ~20% even if correct!")

    print("\nConditional Reviewer Agent (demonstrates implies() constraint):")
    print("-" * 70)
    print("  use_reviewer: Choices([True, False])  → enables/disables review step")
    print("  reviewer_model: agent='reviewer'      → only active when use_reviewer=True")
    print("  reviewer_temperature: agent='reviewer'→ [0.0, 0.3] for consistent critique")
    print("  Constraint: implies(use_reviewer=True, reviewer_model ∈ REVIEWER_MODELS)")

    print("\nEarly Stopping Configuration:")
    print("-" * 70)
    print("  max_examples=5       → 5 samples per configuration (faster trials)")
    print("  trial_concurrency=1  → Sequential trials for reliability")
    print("  timeout=300s         → 5 minute timeout")
    print("  plateau_window=6     → Stop optimization if best unchanged for 6 trials")
    print("  plateau_epsilon=0.05 → 5% min improvement to continue")
    print("  CeilingPruner:       → Prunes inferior trials mid-execution")
    print("    min_completed=1    → Start pruning after 1 completed trial")
    print("    warmup_steps=2     → Start checking after 2 examples")
    print("    epsilon=0.35       → AGGRESSIVE: Prune if ceiling < best - 35%")
    print("\n  Example: If best=80%, prune trials with 0% after 3 examples")

    print("\nModels (budget-friendly, ~$0.02 total for 25 trials):")
    print("-" * 70)
    print("  Generator models:")
    for m in OPENROUTER_MODELS:
        print(f"    - {m}")
    print("  Reviewer models:")
    for m in REVIEWER_MODELS:
        print(f"    - {m}")

    # Calculate config space size
    # 3 models × continuous temps × continuous top_p × continuous freq × continuous pres × 4 tokens × 4 strategies × 2 reviewer toggle
    # For discrete estimation: 3 × 5 × 9 × 20 × 20 × 4 × 4 × 2 = large space
    # Traigent samples intelligently from this space
    print(f"\nDataset: {DATASET_PATH.name}")
    print("-" * 70)

    # Custom pruner: prune clearly inferior configs
    # BALANCED pruning settings - gives trials a fair chance while still pruning hopeless ones:
    # - min_completed_trials=2: Need 2 baselines before pruning (more stable reference)
    # - warmup_steps=3: Evaluate 3 examples before making pruning decisions
    # - epsilon=0.2: Prune if ceiling < best - 20% (less aggressive)
    #
    # With best=100% and epsilon=0.2, prune threshold = 80%
    # After 3 examples: ceiling = (correct + 2)/5
    # - 0 correct: ceiling = 40% < 80% → PRUNED (clearly hopeless)
    # - 1 correct: ceiling = 60% < 80% → PRUNED (unlikely to catch up)
    # - 2 correct: ceiling = 80% = 80% → CONTINUE (still has a chance)
    #
    # This is less aggressive than before but still prunes obviously bad configs
    early_pruner = CeilingPruner(
        min_completed_trials=2,  # Wait for 2 completed trials before pruning
        warmup_steps=3,          # Evaluate 3 examples before checking
        epsilon=0.2,             # BALANCED: Prune if ceiling < best - 20%
    )

    # Run optimization with Optuna and early stopping
    # - max_examples: 8 samples per trial
    # - plateau_window/epsilon: Stop optimization if best score stabilizes
    # - CeilingPruner: Prunes inferior trials mid-execution
    #
    # Goal: Demonstrate both types of early stopping:
    # 1. Trial-level: CeilingPruner prunes individual trials mid-execution
    # 2. Optimization-level: plateau detection stops before max_trials
    result = await qa_with_tvars.optimize(
        algorithm="optuna",
        max_trials=15,            # Reduced for faster demo
        max_examples=5,           # 5 samples per config (faster trials)
        parallel_config={"trial_concurrency": 1},  # Sequential to avoid timeout
        timeout=300,              # 5 minute timeout
        # Early stopping: stop if best accuracy plateaus for 6 consecutive trials
        plateau_window=6,
        plateau_epsilon=0.05,     # 5% min improvement to continue
        # Custom pruner for progressive early stopping of inferior configs
        pruner=early_pruner,
    )

    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    print("\nBest Configuration:")
    for k, v in result.best_config.items():
        if k == "model":
            v = v.split("/")[-1]
        elif isinstance(v, float):
            v = f"{v:.3f}"
        print(f"  {k}: {v}")

    print(f"\nBest Accuracy: {result.best_metrics.get('accuracy', 0):.1%}")

    # Show stop reason if early stopped
    stop_reason = getattr(result, "stop_reason", None)
    if stop_reason and stop_reason != "max_trials":
        print(f"\n⚡ Early stopped: {stop_reason}")
        print(f"   Completed {len(result.trials)} of 25 max trials")

    # Show preset benefits
    print("\n" + "-" * 70)
    print("PRESET BENEFITS")
    print("-" * 70)
    print("✓ Domain knowledge encoded - no need to guess parameter ranges")
    print("✓ Conservative temperature keeps factual QA grounded")
    print("✓ Prompting strategies compared systematically")
    print("✓ Penalty parameters explore repetition/diversity tradeoffs")
    print("✓ Conditional reviewer: params only active when use_reviewer=True")
    print("✓ implies() constraint ensures valid config when review enabled")
    print("✓ Early stopping: plateau detection + inferior trial pruning")


if __name__ == "__main__":
    asyncio.run(main())
