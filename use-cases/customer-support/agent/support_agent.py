#!/usr/bin/env python3
"""
Customer Support Agent - ShopEasy Support Bot

This agent handles customer support inquiries for ShopEasy, an e-commerce platform.
It optimizes for resolution accuracy, tone quality, and escalation decisions.

Usage:
    export TRAIGENT_MOCK_MODE=true
    python use-cases/customer-support/agent/support_agent.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import traigent
from traigent.api.decorators import EvaluationOptions, ExecutionOptions

# Import evaluator
from use_cases.customer_support.eval.evaluator import SupportEvaluator


SUPPORT_SYSTEM_PROMPT = """You are a customer support agent for ShopEasy, a popular e-commerce platform.
Your role is to help customers with their inquiries while maintaining a {tone} tone.

Guidelines:
- Be {empathy_level} empathetic to customer concerns
- Provide clear, actionable solutions
- Follow company policies on refunds, returns, and exchanges
- Escalate complex issues when appropriate (threshold: {escalation_threshold})

Company Policies:
- Returns accepted within 30 days with original receipt
- Refunds processed within 5-7 business days
- VIP customers (gold/platinum tier) get priority support
- Order modifications possible until shipping begins
- Damaged items: immediate replacement or refund

Escalation Criteria (when to escalate):
- Customer explicitly requests supervisor/manager
- Legal threats or serious complaints
- Safety or security issues
- Requests beyond agent authority (large refunds > $500, policy exceptions)
- Repeated failed resolution attempts
- Customer satisfaction score very low"""

SUPPORT_USER_PROMPT = """Customer Context:
- Customer Tier: {customer_tier}
- Customer Sentiment: {sentiment}
- Order Status: {order_status}
- Previous Interactions: {previous_interactions}

Customer Query:
{query}

Provide a helpful response. If escalation is needed, indicate it clearly.

Response Format:
1. Greeting and acknowledgment
2. Solution or next steps
3. Closing with any follow-up needed

Respond naturally to the customer:"""


def build_escalation_decision_prompt(
    query: str,
    customer_context: dict[str, Any],
    response: str,
) -> str:
    """Build prompt to determine if escalation is needed."""
    return f"""Based on the following support interaction, determine if escalation to a supervisor is needed.

Customer Query: {query}
Customer Tier: {customer_context.get('customer_tier', 'standard')}
Customer Sentiment: {customer_context.get('sentiment', 'neutral')}

Agent Response: {response}

Escalation should happen if:
1. Issue cannot be resolved at current level
2. Customer explicitly requests supervisor
3. Legal/safety concerns mentioned
4. High-value customer with unresolved complaint
5. Policy exception needed

Should this be escalated? Respond with only "YES" or "NO":"""


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.3, 0.5, 0.7],
        "tone": ["professional", "friendly", "empathetic"],
        "empathy_level": ["moderate", "high", "very_high"],
        "escalation_threshold": ["conservative", "moderate", "aggressive"],
    },
    objectives=["resolution_accuracy", "tone_quality", "escalation_accuracy", "cost"],
    evaluation=EvaluationOptions(
        eval_dataset="use-cases/customer-support/datasets/support_tickets.jsonl",
        custom_evaluator=SupportEvaluator(),
    ),
    execution=ExecutionOptions(execution_mode="edge_analytics"),
)
def customer_support_agent(
    query: str,
    customer_context: dict[str, Any],
) -> dict[str, Any]:
    """
    Handle a customer support inquiry.

    Args:
        query: The customer's question or issue
        customer_context: Context about the customer (tier, sentiment, order status, etc.)

    Returns:
        Dictionary with 'response', 'should_escalate', 'resolution_type', and metadata
    """
    # Get current configuration
    config = traigent.get_config()

    # Extract tuned variables with defaults
    model = config.get("model", "gpt-4o-mini")
    temperature = config.get("temperature", 0.5)
    tone = config.get("tone", "professional")
    empathy_level = config.get("empathy_level", "high")
    escalation_threshold = config.get("escalation_threshold", "moderate")

    # Build the prompt
    system_prompt = SUPPORT_SYSTEM_PROMPT.format(
        tone=tone,
        empathy_level=empathy_level,
        escalation_threshold=escalation_threshold,
    )

    user_prompt = SUPPORT_USER_PROMPT.format(
        customer_tier=customer_context.get("customer_tier", "standard"),
        sentiment=customer_context.get("sentiment", "neutral"),
        order_status=customer_context.get("order_status", "unknown"),
        previous_interactions=customer_context.get("previous_interactions", 0),
        query=query,
    )

    # Use LangChain for LLM call
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
        )

        # Generate response
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = llm.invoke(messages)
        response_text = response.content

        # Determine escalation
        escalation_prompt = build_escalation_decision_prompt(
            query, customer_context, response_text
        )
        escalation_response = llm.invoke(escalation_prompt)
        should_escalate = "YES" in escalation_response.content.upper()

        # Determine resolution type
        resolution_type = determine_resolution_type(response_text, should_escalate)

        return {
            "response": response_text,
            "should_escalate": should_escalate,
            "resolution_type": resolution_type,
            "model": model,
            "temperature": temperature,
            "tone": tone,
            "empathy_level": empathy_level,
        }

    except ImportError:
        # Fallback for mock mode without LangChain
        return generate_mock_response(query, customer_context, tone, empathy_level)


def determine_resolution_type(response: str, should_escalate: bool) -> str:
    """Determine the type of resolution based on response content."""
    response_lower = response.lower()

    if should_escalate:
        return "escalated"

    if any(word in response_lower for word in ["refund", "credited", "money back"]):
        return "refund"
    if any(word in response_lower for word in ["replacement", "new item", "exchange"]):
        return "replacement"
    if any(word in response_lower for word in ["track", "tracking", "shipment"]):
        return "information"
    if any(word in response_lower for word in ["update", "change", "modify"]):
        return "order_modification"
    if any(word in response_lower for word in ["cancel", "cancelled"]):
        return "cancellation"

    return "resolved"


def generate_mock_response(
    query: str,
    customer_context: dict[str, Any],
    tone: str,
    empathy_level: str,
) -> dict[str, Any]:
    """Generate a mock response for testing without LLM."""
    query_lower = query.lower()
    customer_tier = customer_context.get("customer_tier", "standard")
    sentiment = customer_context.get("sentiment", "neutral")

    # Determine if escalation is needed based on keywords
    escalation_keywords = ["supervisor", "manager", "lawyer", "sue", "unacceptable"]
    should_escalate = any(kw in query_lower for kw in escalation_keywords)

    # Also escalate for very negative sentiment on high-value customers
    if sentiment == "very_negative" and customer_tier in ["gold", "platinum"]:
        should_escalate = True

    # Generate appropriate response based on query type
    if "refund" in query_lower or "money back" in query_lower:
        response = f"""Thank you for reaching out to ShopEasy support.

I understand you're inquiring about a refund, and I'm here to help. As a {customer_tier} customer, we want to ensure your complete satisfaction.

I've reviewed your account and can process your refund request. The amount will be credited to your original payment method within 5-7 business days.

Is there anything else I can assist you with today?"""
        resolution_type = "refund"

    elif "track" in query_lower or "where is" in query_lower:
        response = f"""Hello and thank you for contacting ShopEasy!

I can help you track your order. Based on the latest shipping update, your package is currently in transit and should arrive within 2-3 business days.

You can also track your order in real-time using the tracking link sent to your email.

Please let me know if you need any additional assistance!"""
        resolution_type = "information"

    elif "cancel" in query_lower:
        response = f"""Thank you for reaching out to ShopEasy support.

I understand you'd like to cancel your order. I've checked your order status, and I'm happy to help process this cancellation for you.

Your order has been cancelled and you won't be charged. If any payment was already processed, it will be refunded within 3-5 business days.

Is there anything else I can help you with?"""
        resolution_type = "cancellation"

    elif "damaged" in query_lower or "broken" in query_lower:
        response = f"""I'm so sorry to hear that your item arrived damaged. That's definitely not the experience we want you to have with ShopEasy.

As a {customer_tier} customer, I'd like to make this right immediately. I can offer you either:
1. A full replacement shipped with express delivery at no extra cost
2. A complete refund to your original payment method

Please let me know which option you'd prefer, and I'll process it right away."""
        resolution_type = "replacement"

    elif should_escalate:
        response = f"""Thank you for your patience, and I sincerely apologize for the frustration you've experienced.

I understand this situation requires additional attention. I'm going to escalate this to our senior support team who will be able to provide a more comprehensive resolution.

A supervisor will contact you within 24 hours. As a {customer_tier} customer, your case will be prioritized.

Is there anything immediate I can assist with while we arrange this?"""
        resolution_type = "escalated"

    else:
        response = f"""Thank you for contacting ShopEasy support!

I'm happy to help you with your inquiry. I've reviewed your account and order history to better assist you.

Based on your question, here's what I can tell you: Our team is committed to ensuring your satisfaction. If you have any specific concerns, please don't hesitate to provide more details.

Is there anything else I can help you with today?"""
        resolution_type = "resolved"

    return {
        "response": response,
        "should_escalate": should_escalate,
        "resolution_type": resolution_type,
        "model": "mock",
        "temperature": 0.5,
        "tone": tone,
        "empathy_level": empathy_level,
    }


async def run_optimization():
    """Run the customer support agent optimization."""
    print("=" * 60)
    print("Customer Support Agent - TraiGent Optimization")
    print("=" * 60)

    # Check if mock mode is enabled
    mock_mode = os.environ.get("TRAIGENT_MOCK_MODE", "false").lower() == "true"
    print(f"\nMock Mode: {'Enabled' if mock_mode else 'Disabled'}")

    if not mock_mode:
        print("\nWARNING: Running without mock mode will incur API costs!")
        print("Set TRAIGENT_MOCK_MODE=true for testing.\n")

    print("\nStarting optimization...")
    print("Configuration Space:")
    print("  - Models: gpt-3.5-turbo, gpt-4o-mini, gpt-4o")
    print("  - Temperature: 0.3, 0.5, 0.7")
    print("  - Tone: professional, friendly, empathetic")
    print("  - Empathy Level: moderate, high, very_high")
    print("  - Escalation Threshold: conservative, moderate, aggressive")
    print("\nObjectives: resolution_accuracy, tone_quality, escalation_accuracy, cost")
    print("-" * 60)

    # Run optimization
    results = await customer_support_agent.optimize(
        algorithm="random",
        max_trials=20,
    )

    # Display results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print("\nBest Configuration:")
    for key, value in results.best_config.items():
        print(f"  {key}: {value}")
    print(f"\nBest Score: {results.best_score:.4f}")

    # Apply best config
    customer_support_agent.apply_best_config(results)
    print("\nBest configuration applied!")

    # Test with sample query
    print("\n" + "-" * 60)
    print("Testing optimized agent with sample query...")
    print("-" * 60)

    result = customer_support_agent(
        query="I received a damaged laptop and I want a refund immediately. This is unacceptable!",
        customer_context={
            "customer_tier": "gold",
            "sentiment": "negative",
            "order_status": "delivered",
            "previous_interactions": 2,
        },
    )

    print(f"\nResponse:\n{result['response']}")
    print(f"\nMetadata:")
    print(f"  Should Escalate: {result['should_escalate']}")
    print(f"  Resolution Type: {result['resolution_type']}")
    print(f"  Model: {result['model']}")
    print(f"  Tone: {result['tone']}")

    return results


def main():
    """Main entry point."""
    asyncio.run(run_optimization())


if __name__ == "__main__":
    main()
