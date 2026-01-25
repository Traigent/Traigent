#!/usr/bin/env python3
"""Long Context Agent - Validates GPT-4.1 long context improvements.

This agent tests long context comprehension based on OpenAI's GPT-4.1 claims:
- OpenAI-MRCR (2 needle, 128k): GPT-4.1 (57.2%) vs GPT-4o (31.9%)
- Graphwalks BFS (<128k): GPT-4.1 (61.7%) vs GPT-4o (41.7%)
- Context window: 1M tokens for GPT-4.1 vs 128k for GPT-4o

Test Categories:
- Single needle retrieval: Find specific info in large context
- Multi-needle disambiguation: Find and distinguish multiple similar items (MRCR-style)
- Multi-hop reasoning: Reason across multiple positions in context (Graphwalks-style)

Usage:
    # Mock mode (recommended for testing)
    export TRAIGENT_MOCK_LLM=true
    python use-cases/gpt-4.1-study/agent/long_context_agent.py

    # Real mode with OpenAI API
    export OPENAI_API_KEY=sk-...
    python use-cases/gpt-4.1-study/agent/long_context_agent.py --max-trials 20
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# =============================================================================
# Constants
# =============================================================================

DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"

MODEL_GPT_4_1 = "gpt-4.1"
MODEL_GPT_4_1_MINI = "gpt-4.1-mini"
MODEL_GPT_4_1_NANO = "gpt-4.1-nano"
MODEL_GPT_4O = "gpt-4o"
MODEL_GPT_4O_MINI = "gpt-4o-mini"

DEFAULT_MODEL = MODEL_GPT_4O

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

import traigent  # noqa: E402
from traigent.api.decorators import EvaluationOptions, ExecutionOptions  # noqa: E402

# Import evaluator
_evaluator_path = Path(__file__).parent.parent / "eval" / "long_context_evaluator.py"
_spec = importlib.util.spec_from_file_location(
    "long_context_evaluator", _evaluator_path
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load evaluator module from {_evaluator_path}")
_evaluator_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluator_module)

if TYPE_CHECKING:
    from types import ModuleType

    _evaluator_module: ModuleType

LongContextEvaluator = _evaluator_module.LongContextEvaluator
retrieval_accuracy = _evaluator_module.retrieval_accuracy
multi_hop_accuracy = _evaluator_module.multi_hop_accuracy

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "long_context_dataset.jsonl"


# =============================================================================
# Configuration Space
# =============================================================================

CONFIGURATION_SPACE = {
    "model": [
        MODEL_GPT_4_1,
        MODEL_GPT_4_1_MINI,
        MODEL_GPT_4_1_NANO,
        MODEL_GPT_4O,
        MODEL_GPT_4O_MINI,
    ],
    "context_strategy": ["full_context", "chunked"],
}


# =============================================================================
# Task Categories
# =============================================================================

TASK_CATEGORIES = {
    "single_needle": "Find a single specific piece of information",
    "multi_needle": "Find and disambiguate multiple similar items (MRCR-style)",
    "multi_hop": "Reason across multiple positions in context (Graphwalks-style)",
}


# =============================================================================
# Mock Mode & API
# =============================================================================


def is_mock_mode() -> bool:
    """Check if running in mock mode."""
    return os.environ.get("TRAIGENT_MOCK_LLM", "").lower() in ("true", "1", "yes")


MODEL_COSTS = {
    MODEL_GPT_4_1: {"input": 2.00, "output": 8.00},
    MODEL_GPT_4_1_MINI: {"input": 0.40, "output": 1.60},
    MODEL_GPT_4_1_NANO: {"input": 0.10, "output": 0.40},
    MODEL_GPT_4O: {"input": 2.50, "output": 10.00},
    MODEL_GPT_4O_MINI: {"input": 0.15, "output": 0.60},
}

DEFAULT_MODEL_PRICING = {"input": 2.50, "output": 10.00}

# Model profiles for long context (based on blog claims)
MODEL_PROFILES = {
    MODEL_GPT_4_1: {
        "single_needle": 0.95,  # Near-perfect needle retrieval
        "multi_needle": 0.57,  # Based on MRCR 2-needle 128k claim
        "multi_hop": 0.62,  # Based on Graphwalks BFS claim
    },
    MODEL_GPT_4_1_MINI: {
        "single_needle": 0.90,
        "multi_needle": 0.47,  # Based on MRCR claim
        "multi_hop": 0.62,  # Same as GPT-4.1 per blog
    },
    MODEL_GPT_4_1_NANO: {
        "single_needle": 0.80,
        "multi_needle": 0.37,  # Based on MRCR claim
        "multi_hop": 0.25,
    },
    MODEL_GPT_4O: {
        "single_needle": 0.85,
        "multi_needle": 0.32,  # Based on MRCR claim
        "multi_hop": 0.42,  # Based on Graphwalks claim
    },
    MODEL_GPT_4O_MINI: {
        "single_needle": 0.75,
        "multi_needle": 0.25,  # Based on MRCR claim
        "multi_hop": 0.29,
    },
}

DEFAULT_PROFILE = {
    "single_needle": 0.80,
    "multi_needle": 0.35,
    "multi_hop": 0.40,
}


def _get_deterministic_seed(task_id: str, model: str) -> int:
    """Generate deterministic seed for mock reproducibility."""
    combined = f"{task_id}:{model}"
    return int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)


def _get_api_config() -> tuple[str | None, str]:
    """Get API key and base URL from environment."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    api_base = os.environ.get("LLM_API_BASE", DEFAULT_OPENAI_API_BASE)
    return api_key, api_base


def call_llm(prompt: str, model: str, temperature: float = 0.0) -> str:
    """Call LLM via OpenAI-compatible API."""
    api_key, api_base = _get_api_config()

    if not api_key:
        print("Warning: No API key found.")
        return ""

    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 1000,
    }

    try:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=180) as response:  # noqa: S310
            result = json.loads(response.read().decode("utf-8"))
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except (HTTPError, URLError, Exception) as e:
        print(f"LLM call error: {e}")
        return ""


# =============================================================================
# Mock Generation
# =============================================================================


def generate_mock_output(
    task_id: str,
    category: str,
    context: str,
    question: str,
    expected_answer: str,
    needle_positions: list[int] | None,
    model: str,
) -> dict[str, Any]:
    """Generate mock output based on model profiles."""
    seed = _get_deterministic_seed(task_id, model)
    rng = random.Random(seed)  # noqa: S311

    profile = MODEL_PROFILES.get(model, DEFAULT_PROFILE)

    # Determine success probability based on category
    category_key = category.replace("-", "_").lower()
    success_prob = profile.get(category_key, 0.5)

    success = rng.random() < success_prob

    if success:
        answer = expected_answer
        retrieval_correct = True
        reasoning_correct = category != "multi_hop" or rng.random() < 0.9
    else:
        # Generate plausible but incorrect answer
        if category == "multi_needle":
            # Return wrong instance
            answer = f"Instance at position {rng.randint(1, 10)}"
        elif category == "multi_hop":
            answer = "Unable to determine the connection"
        else:
            answer = "The requested information was not found"
        retrieval_correct = False
        reasoning_correct = False

    return {
        "answer": answer,
        "retrieval_correct": retrieval_correct,
        "reasoning_correct": reasoning_correct,
        "context_length": len(context),
        "needle_count": len(needle_positions) if needle_positions else 1,
    }


def estimate_mock_cost(context: str, model: str) -> float:
    """Estimate mock cost for the task."""
    input_tokens = len(context) // 4 + 100
    output_tokens = 200

    pricing = MODEL_COSTS.get(model, DEFAULT_MODEL_PRICING)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


# =============================================================================
# Metric Functions
# =============================================================================


def cost_metric(
    output: dict[str, Any] | None,
    expected: dict[str, Any],
    **kwargs: Any,
) -> float:
    """Calculate cost for the long context task."""
    input_data = kwargs.get("input_data", {})
    config = kwargs.get("config", {})

    context = input_data.get("context", "")
    model = config.get("model", DEFAULT_MODEL)

    return estimate_mock_cost(context, model)


# =============================================================================
# Main Agent
# =============================================================================


@traigent.optimize(
    configuration_space=CONFIGURATION_SPACE,
    objectives=["retrieval_accuracy", "multi_hop_accuracy", "cost"],
    metric_functions={
        "retrieval_accuracy": retrieval_accuracy,
        "multi_hop_accuracy": multi_hop_accuracy,
        "cost": cost_metric,
    },
    evaluation=EvaluationOptions(eval_dataset=str(DATASET_PATH)),
    execution=ExecutionOptions(execution_mode="edge_analytics"),
)
def long_context_agent(
    task_id: str,
    category: str,
    context: str,
    question: str,
    expected_answer: str,
    needle_positions: list[int] | None = None,
) -> dict[str, Any]:
    """Test long context comprehension capabilities.

    This agent validates GPT-4.1's claimed improvements in:
    - Needle-in-haystack retrieval
    - Multi-needle disambiguation (MRCR-style)
    - Multi-hop reasoning across context (Graphwalks-style)

    Args:
        task_id: Unique identifier for the task
        category: One of single_needle, multi_needle, multi_hop
        context: The large context to search/reason over
        question: The question to answer
        expected_answer: The correct answer for evaluation
        needle_positions: Positions of relevant info in context (for multi-needle)

    Returns:
        Dict with answer and accuracy metrics
    """
    config = traigent.get_config()
    model = config.get("model", DEFAULT_MODEL)
    context_strategy = config.get("context_strategy", "full_context")

    # Mock mode
    if is_mock_mode():
        return generate_mock_output(
            task_id=task_id,
            category=category,
            context=context,
            question=question,
            expected_answer=expected_answer,
            needle_positions=needle_positions,
            model=model,
        )

    # Build prompt
    if context_strategy == "chunked":
        # For very long contexts, could implement chunking strategy
        # For now, use full context
        prompt = f"""Context:
{context}

Question: {question}

Answer the question based ONLY on the information in the context above.
Be precise and specific. If the context contains multiple similar items,
make sure to identify the correct one based on the question."""
    else:
        prompt = f"""Context:
{context}

Question: {question}

Answer:"""

    # Call LLM
    response = call_llm(prompt, model)

    # Check if answer is correct
    answer_lower = response.lower().strip()
    expected_lower = expected_answer.lower().strip()

    retrieval_correct = expected_lower in answer_lower or answer_lower == expected_lower

    # For multi-hop, also check reasoning chain
    reasoning_correct = True
    if category == "multi_hop" and needle_positions:
        # Check if response indicates understanding of connection
        reasoning_correct = retrieval_correct and len(response) > len(expected_answer)

    return {
        "answer": response,
        "retrieval_correct": retrieval_correct,
        "reasoning_correct": reasoning_correct,
        "context_length": len(context),
        "needle_count": len(needle_positions) if needle_positions else 1,
    }


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Run the long context agent for testing."""
    parser = argparse.ArgumentParser(description="GPT-4.1 Long Context Agent")
    parser.add_argument(
        "--max-trials", type=int, default=10, help="Max optimization trials"
    )
    parser.add_argument("--demo", action="store_true", help="Run demo task")
    args = parser.parse_args()

    if args.demo:
        print("Running demo task...")
        # Simple needle-in-haystack demo
        context = "A " * 1000 + "The secret code is ALPHA-7. " + "B " * 1000
        result = long_context_agent(
            task_id="demo_001",
            category="single_needle",
            context=context,
            question="What is the secret code?",
            expected_answer="ALPHA-7",
        )
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print(f"Running optimization with {args.max_trials} trials...")


if __name__ == "__main__":
    main()
