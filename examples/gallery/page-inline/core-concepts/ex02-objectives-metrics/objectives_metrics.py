#!/usr/bin/env python3
"""Example: Objectives & Metrics - Defining What to Optimize For."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# --- Setup for running from repo without installation ---
# Set TRAIGENT_SDK_PATH to override when running from outside the repo tree.
_sdk_override = os.environ.get("TRAIGENT_SDK_PATH")
if _sdk_override:
    if _sdk_override not in sys.path:
        sys.path.insert(0, _sdk_override)
else:
    _module_path = Path(__file__).resolve()
    for _depth in range(1, 7):
        try:
            _repo_root = _module_path.parents[_depth]
            if (_repo_root / "traigent").is_dir() and (_repo_root / "examples").is_dir():
                if str(_repo_root) not in sys.path:
                    sys.path.insert(0, str(_repo_root))
                break
        except IndexError:
            continue
from examples.utils.langchain_compat import ChatOpenAI, HumanMessage

try:
    import traigent
except ImportError:  # pragma: no cover - support IDE execution paths
    import importlib

    _sdk = os.environ.get("TRAIGENT_SDK_PATH")
    if _sdk:
        sys.path.insert(0, _sdk)
    else:
        module_path = Path(__file__).resolve()
        for depth in (2, 3):
            try:
                sys.path.append(str(module_path.parents[depth]))
            except IndexError:
                continue
    traigent = importlib.import_module("traigent")

from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema  # noqa: E402

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


# Custom metric function
# Create dataset file
DATASET_FILE = os.path.join(os.path.dirname(__file__), "support_tickets.jsonl")


def create_sample_dataset() -> str:
    """Create sample support tickets dataset."""
    dataset = [
        {
            "input": {"query": "How do I reset my password?"},
            "expected_output": {"contains": ["password", "reset", "steps"]},
        },
        {
            "input": {"query": "My order hasn't arrived yet"},
            "expected_output": {"contains": ["order", "tracking", "delivery"]},
        },
        {
            "input": {"query": "How to cancel subscription?"},
            "expected_output": {"contains": ["cancel", "subscription", "process"]},
        },
        {
            "input": {"query": "Refund request for damaged product"},
            "expected_output": {"contains": ["refund", "damaged", "return"]},
        },
        {
            "input": {"query": "Technical issue with app login"},
            "expected_output": {"contains": ["login", "technical", "troubleshoot"]},
        },
    ]

    # Write to JSONL file
    with open(DATASET_FILE, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    return DATASET_FILE


# Create the dataset file
create_sample_dataset()

BALANCED_SUPPORT_OBJECTIVES = ObjectiveSchema.from_objectives(
    [
        ObjectiveDefinition("cost", orientation="minimize", weight=0.3),
        ObjectiveDefinition("quality", orientation="maximize", weight=0.5),
        ObjectiveDefinition("response_time", orientation="minimize", weight=0.2),
    ]
)


def response_quality_score(output: str, expected: dict[str, Any]) -> float:
    """Custom metric to evaluate response quality."""
    score = 0.0

    # Check if key terms are present
    if "contains" in expected:
        for term in expected["contains"]:
            if term.lower() in output.lower():
                score += 0.25

    # Check response length (penalize too short or too long)
    word_count = len(output.split())
    if 50 <= word_count <= 200:
        score += 0.2
    elif 20 <= word_count < 50:
        score += 0.1

    # Check for structure (sentences, punctuation)
    if "." in output and len(output.split(".")) > 1:
        score += 0.1

    # Check for polite/professional tone
    polite_terms = ["please", "thank you", "would", "could", "happy to help"]
    if any(term in output.lower() for term in polite_terms):
        score += 0.1

    return min(score, 1.0)  # Cap at 1.0


def _max_cost_per_call(
    config: dict[str, Any], metrics: dict[str, Any] | None = None
) -> bool:
    cost = (metrics or {}).get("cost")
    if cost is None:
        return True
    try:
        return float(cost) <= 0.05
    except (TypeError, ValueError):
        return True


def _min_quality_score(
    config: dict[str, Any], metrics: dict[str, Any] | None = None
) -> bool:
    score = (metrics or {}).get("response_quality")
    if score is None:
        return True
    try:
        return float(score) >= 0.7
    except (TypeError, ValueError):
        return True


# Example 1: Single objective optimization
@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.1, 0.5, 0.9],
    },
    eval_dataset=DATASET_FILE,
    objectives=["cost"],  # Minimize cost only
    execution_mode="edge_analytics",
    max_trials=10,
)
def cost_optimized_bot(query: str) -> str:
    """Bot optimized purely for cost efficiency."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    response = llm.invoke([HumanMessage(content=query)])
    return str(getattr(response, "content", response))


# Example 2: Multi-objective optimization
@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.3, 0.5, 0.7],
        "max_tokens": [100, 200, 300],
    },
    eval_dataset=DATASET_FILE,
    objectives=BALANCED_SUPPORT_OBJECTIVES,
    execution_mode="edge_analytics",
    max_trials=10,
)
def balanced_support_bot(query: str) -> str:
    """Bot balancing cost, quality, and response time."""
    current = traigent.get_config()
    config = current if isinstance(current, dict) else {}

    llm = ChatOpenAI(
        model=config.get("model", "gpt-3.5-turbo"),
        temperature=config.get("temperature", 0.5),
        max_tokens=config.get("max_tokens", 200),
    )

    prompt = f"As a helpful support agent, answer: {query}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return str(getattr(response, "content", response))


# Example 3: Custom metrics with constraints
@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.3, 0.5],
        "response_format": ["brief", "standard", "detailed"],
    },
    eval_dataset=DATASET_FILE,
    objectives=["cost", "response_quality"],
    metric_functions={"response_quality": response_quality_score},
    constraints=[
        _max_cost_per_call,
        _min_quality_score,
    ],
    execution_mode="edge_analytics",
    max_trials=10,
)
def quality_constrained_bot(query: str) -> str:
    """Bot with quality constraints and custom metrics."""
    current = traigent.get_config()
    config = current if isinstance(current, dict) else {}

    format_instructions = {
        "brief": "Answer in 1-2 sentences.",
        "standard": "Provide a clear, moderate answer.",
        "detailed": "Give a comprehensive response with details.",
    }

    llm = ChatOpenAI(
        model=config.get("model", "gpt-3.5-turbo"),
        temperature=config.get("temperature", 0.3),
    )

    format_style = str(config.get("response_format", "standard"))
    instructions = format_instructions.get(
        format_style, format_instructions["standard"]
    )
    prompt = f"{instructions}\n\nQuery: {query}"

    response = llm.invoke([HumanMessage(content=prompt)])
    return str(getattr(response, "content", response))


def demonstrate_objective_types() -> None:
    """Show different types of objectives and metrics."""

    print("Traigent Objective Types:")
    print("-" * 40)

    # Built-in objectives
    print("\n1. Built-in Objectives:")
    built_in = ["cost", "latency", "accuracy", "quality", "response_time"]
    for obj in built_in:
        print(f"   - {obj}")

    # Custom metrics
    print("\n2. Custom Metrics:")
    print("   - response_quality_score (custom function)")
    print("   - customer_satisfaction (0-1 scale)")
    print("   - task_completion_rate (percentage)")

    # Multi-objective optimization
    print("\n3. Multi-Objective Optimization:")
    print("   - Balance multiple objectives with weights")
    print("   - Example: 30% cost, 50% quality, 20% speed")
    print("   - Traigent finds Pareto-optimal solutions")

    # Constraints
    print("\n4. Constraints:")
    print("   - max_cost_per_call: $0.05")
    print("   - min_quality_score: 0.7")
    print("   - max_response_time: 2 seconds")


def analyze_tradeoffs() -> None:
    """Analyze objective tradeoffs."""

    print("\n" + "=" * 50)
    print("Objective Tradeoffs Analysis")
    print("=" * 50)

    tradeoffs = {
        "Cost vs Quality": {
            "cheap": {"model": "gpt-3.5-turbo", "cost": "$0.002", "quality": "70%"},
            "balanced": {"model": "gpt-4o-mini", "cost": "$0.01", "quality": "85%"},
            "premium": {"model": "gpt-4o", "cost": "$0.03", "quality": "95%"},
        },
        "Speed vs Accuracy": {
            "fast": {"max_tokens": 100, "time": "1s", "accuracy": "75%"},
            "balanced": {"max_tokens": 200, "time": "2s", "accuracy": "85%"},
            "thorough": {"max_tokens": 500, "time": "4s", "accuracy": "92%"},
        },
    }

    for tradeoff_name, options in tradeoffs.items():
        print(f"\n{tradeoff_name}:")
        for option_name, metrics in options.items():
            print(f"  {option_name:10} - {metrics}")


if __name__ == "__main__":
    try:
        print("=" * 60)
        print("Traigent Core Concepts: Objectives & Metrics")
        print("=" * 60)

        # Demonstrate objective types
        demonstrate_objective_types()

        # Analyze tradeoffs
        analyze_tradeoffs()

        print("\n" + "=" * 50)
        print("Optimization Examples")
        print("=" * 50)

        print("\n1. Cost-optimized bot (single objective)")
        print("2. Balanced support bot (multi-objective)")
        print("3. Quality-constrained bot (custom metrics + constraints)")

        print("\nThese examples show how Traigent optimizes for different goals.")
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
