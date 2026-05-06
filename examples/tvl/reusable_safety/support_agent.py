"""Customer Support Agent Example with Inherited Safety Constraints.

This example demonstrates how to create a support agent that inherits
safety constraints from a base specification. The agent balances
latency, cost, and accuracy for high-volume customer interactions.

Key Features:
- Inherits 3 safety constraints from base_safety.tvl.yml
- Balanced objectives: latency (2x), cost (2x), accuracy (1x)
- Optimized for high-volume customer support
- Task-specific support dataset (50 examples)

Usage:
    # With mock LLM for testing
    TRAIGENT_MOCK_LLM=true python support_agent.py

    # With real LLM
    python support_agent.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Import Traigent SDK
from traigent.api import optimize
from traigent.api.safety import (
    bias_score,
    hallucination_rate,
    safety_score,
    toxicity_score,
)

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent

# Path to the TVL spec (demonstrates inheritance)
TVL_SPEC_PATH = SCRIPT_DIR / "support_agent.tvl.yml"


# Canned responses for common issues
CANNED_RESPONSES = {
    "password_reset": "I can help you reset your password. Please click the 'Forgot Password' link on the login page, and you'll receive a reset email within a few minutes.",
    "tracking_update": "I've checked your order status. Your package is currently in transit and should arrive within the estimated delivery window.",
    "general": "I understand your concern. Let me help you with that.",
}


def get_customer_context(customer_id: str) -> dict[str, Any]:
    """Retrieve customer context for personalized support.

    Args:
        customer_id: The customer's identifier.

    Returns:
        Customer context including history and preferences.
    """
    # Mock customer context - in production, query CRM
    return {
        "customer_id": customer_id,
        "tier": "pro",
        "tenure_months": 12,
        "last_contact": "3 days ago",
        "sentiment": "neutral",
    }


@optimize(
    spec=str(TVL_SPEC_PATH),
    objectives=[
        "latency_p50",
        "cost_per_request",
        "resolution_accuracy",
        "customer_satisfaction",
    ],
    # Safety constraints inherited from base_safety.tvl.yml:
    # - hallucination_rate <= 10%
    # - toxicity_score <= 5%
    # - bias_score <= 10%
    # Plus additional safety constraint for this agent:
    safety_constraints=[
        hallucination_rate().below(0.1),
        toxicity_score().below(0.05),
        bias_score().below(0.1),
        safety_score().above(0.9),  # High safety requirement for customer-facing
    ],
)
def support_agent(
    customer_query: str,
    customer_id: str = "anonymous",
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 512,
    use_streaming: bool = True,
    response_style: str = "friendly",
    include_escalation_option: bool = True,
    canned_response_threshold: float = 0.85,
) -> dict[str, Any]:
    """Handle customer support queries efficiently.

    This agent provides fast, cost-effective support responses while
    maintaining quality and safety standards inherited from the base spec.

    Args:
        customer_query: The customer's question or issue.
        customer_id: Customer identifier for context.
        model: LLM model to use.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        use_streaming: Enable streaming for faster perceived latency.
        response_style: Response tone (concise/detailed/friendly).
        include_escalation_option: Include human escalation option.
        canned_response_threshold: Confidence for using canned responses.

    Returns:
        Dict containing the response and metadata.
    """
    # Step 1: Get customer context
    context = get_customer_context(customer_id)

    # Step 2: Check for canned response match
    query_lower = customer_query.lower()
    canned_key = None
    if "password" in query_lower:
        canned_key = "password_reset"
    elif "order" in query_lower and ("track" in query_lower or "where" in query_lower):
        canned_key = "tracking_update"

    # Step 3: Generate response
    if os.environ.get("TRAIGENT_MOCK_LLM"):
        if canned_key:
            response = CANNED_RESPONSES[canned_key]
        else:
            response = f"Thank you for reaching out! {CANNED_RESPONSES['general']} [Mock response for: {customer_query[:50]}...]"
        latency_ms = 80.0 if use_streaming else 150.0
        cost_usd = 0.002 if model == "gpt-4o-mini" else 0.008
    else:
        # In production: call LLM API with appropriate model
        response = "Thank you for reaching out. I'm here to help with your concern..."
        latency_ms = 100.0 if use_streaming else 200.0
        cost_usd = 0.003

    # Step 4: Add escalation option if enabled
    if include_escalation_option:
        response += "\n\nIf you'd like to speak with a human agent, just let me know."

    return {
        "response": response,
        "customer_id": customer_id,
        "query": customer_query,
        "model": model,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "used_canned_response": canned_key is not None,
        "customer_context": context,
        "metadata": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "use_streaming": use_streaming,
            "response_style": response_style,
            "include_escalation": include_escalation_option,
        },
    }


def main() -> None:
    """Run the support agent example."""
    print("=" * 60)
    print("Customer Support Agent with Inherited Safety Constraints")
    print("=" * 60)
    print()

    # Show TVL spec path
    print(f"TVL Spec: {TVL_SPEC_PATH}")
    print()

    # Demonstrate inheritance by loading the spec
    from traigent.tvl.spec_loader import load_tvl_spec

    spec = load_tvl_spec(spec_path=TVL_SPEC_PATH)
    print("Loaded TVL Spec:")
    print(f"  - Module: {spec.tvl_header.module if spec.tvl_header else 'N/A'}")
    print(f"  - Config space params: {list(spec.configuration_space.keys())}")
    print(f"  - Objectives: {[o.name for o in spec.objective_schema.objectives]}")
    print(f"  - Constraints: {len(spec.constraints)} total (includes inherited)")
    print()

    # Example support queries
    queries = [
        ("I can't log into my account", "cust_001"),
        ("Where is my order #12345?", "cust_002"),
        ("I want to cancel my subscription", "cust_003"),
    ]

    print("Running Support Agent:")
    print("-" * 40)

    for query, customer_id in queries:
        print(f"\nCustomer {customer_id}: support query received")
        result = support_agent(query, customer_id)
        print("Agent: response generated")
        print(
            f"   Latency: {result['latency_ms']:.1f}ms | Cost: ${result['cost_usd']:.4f}"
        )
        print(
            f"   Canned response: {'Yes' if result['used_canned_response'] else 'No'}"
        )

    print()
    print("=" * 60)
    print("Safety constraints from base_safety.tvl.yml are enforced!")
    print("Both agents share the same safety requirements.")
    print("=" * 60)


if __name__ == "__main__":
    main()
