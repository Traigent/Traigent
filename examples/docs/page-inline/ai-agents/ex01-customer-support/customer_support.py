#!/usr/bin/env python3
"""Customer Support Agent - Intent Classification with Multi-Objective Optimization."""
import asyncio
import json
import os
import sys
from pathlib import Path

# --- Setup for running from repo without installation ---
# Add repo root to path so we can import examples.utils and traigent
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

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


try:
    import traigent
except ImportError:  # pragma: no cover - support IDE execution paths
    import importlib

    traigent = importlib.import_module("traigent")

# Create dataset file path
DATASET_FILE = os.path.join(os.path.dirname(__file__), "support_tickets.jsonl")


def create_support_dataset():
    """Create evaluation dataset for customer support intent classification."""
    dataset = [
        {
            "input": {"message": "I was charged twice for my subscription"},
            "output": "billing",
        },
        {"input": {"message": "How do I reset my password?"}, "output": "account"},
        {
            "input": {"message": "The app keeps crashing on my phone"},
            "output": "technical",
        },
        {"input": {"message": "What are your business hours?"}, "output": "general"},
        {
            "input": {"message": "I want to cancel my subscription"},
            "output": "cancellation",
        },
        {
            "input": {"message": "My invoice shows the wrong amount"},
            "output": "billing",
        },
        {"input": {"message": "Can't log into my account"}, "output": "account"},
        {
            "input": {"message": "Feature X is not working properly"},
            "output": "technical",
        },
        {"input": {"message": "How do I contact support?"}, "output": "general"},
        {
            "input": {"message": "Please cancel my account immediately"},
            "output": "cancellation",
        },
    ]

    # Write to JSONL file
    with open(DATASET_FILE, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    return DATASET_FILE


# Create the dataset
create_support_dataset()


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.1, 0.3, 0.5],
        "prompt_style": ["direct", "structured", "few_shot"],
        "max_tokens": [50, 100, 150],
        "response_format": ["label_only", "with_confidence", "with_explanation"],
    },
    eval_dataset=DATASET_FILE,
    objectives=["accuracy", "cost", "response_time"],
    execution_mode="edge_analytics",
    max_trials=10,
)
def classify_support_intent(message: str) -> str:
    """Classify customer support message into predefined intent categories.

    Args:
        message: Customer support message to classify

    Returns:
        Intent category: billing, account, technical, general, or cancellation
    """
    # Get configuration - works both during optimization and normal calls
    # During optimization: get_trial_config() returns trial params
    # Outside optimization: use the function's current_config (applied best config or defaults)
    try:
        current = traigent.get_trial_config()
        config = current if isinstance(current, dict) else {}
    except traigent.utils.exceptions.OptimizationStateError:
        # Not in an optimization trial - use applied config or defaults
        config = getattr(classify_support_intent, "current_config", {}) or {}

    # Build prompt based on style
    prompt_styles = {
        "direct": f"""Classify this support message into one of these categories:
billing, account, technical, general, cancellation.

Message: {message}

Category:""",
        "structured": f"""Task: Customer Support Intent Classification
Categories:
- billing: Payment, charges, invoices
- account: Login, password, profile
- technical: Bugs, errors, features not working
- general: Information, business hours, contact
- cancellation: Cancel subscription or account

Message: {message}

Category:""",
        "few_shot": f"""Classify support messages into categories.

Examples:
"I was charged incorrectly" -> billing
"Can't access my profile" -> account
"App crashes on startup" -> technical

Message: {message}

Category:""",
    }

    prompt_key = str(config.get("prompt_style", "direct"))
    prompt = prompt_styles.get(prompt_key, prompt_styles["direct"])

    # In mock mode, return deterministic result based on keywords
    if os.environ.get("TRAIGENT_MOCK_LLM") == "true":
        message_lower = message.lower()
        if any(word in message_lower for word in ["charge", "bill", "invoice", "pay"]):
            intent = "billing"
        elif any(
            word in message_lower
            for word in ["password", "login", "account", "profile"]
        ):
            intent = "account"
        elif any(
            word in message_lower for word in ["crash", "bug", "error", "not working"]
        ):
            intent = "technical"
        elif any(word in message_lower for word in ["cancel", "unsubscribe"]):
            intent = "cancellation"
        else:
            intent = "general"

        # Format response based on config
        response_format = config.get("response_format", "label_only")
        if response_format == "with_confidence":
            return f"{intent} (confidence: 0.95)"
        elif response_format == "with_explanation":
            return f"{intent} - Classified based on keyword matching"
        else:
            return intent

    # Production mode would use actual LLM
    llm = ChatOpenAI(
        model=config.get("model", "gpt-3.5-turbo"),
        temperature=config.get("temperature", 0.1),
        max_tokens=config.get("max_tokens", 50),
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    result = str(getattr(response, "content", response)).strip().lower()

    # Extract just the category label
    for category in ["billing", "account", "technical", "general", "cancellation"]:
        if category in result:
            return category

    return "general"  # Default fallback


def demonstrate_customer_support():
    """Demonstrate the customer support agent with various configurations."""

    print("=" * 60)
    print("Customer Support Agent - Intent Classification")
    print("=" * 60)

    # Test messages
    test_messages = [
        "I've been charged twice this month",
        "How do I change my email address?",
        "The mobile app won't open",
        "What time do you close?",
        "I want to cancel my subscription",
    ]

    # Run with default config
    print("\n1. Testing with default configuration:")
    for msg in test_messages[:2]:
        intent = classify_support_intent(msg)
        print(f"   Message: '{msg}'")
        print(f"   Intent:  {intent}")

    # Optimize and test
    print("\n2. Running optimization...")
    results = asyncio.run(classify_support_intent.optimize())

    if results:
        print("\n3. Best configuration found:")
        print(f"   Model: {results.best_config.get('model')}")
        print(f"   Temperature: {results.best_config.get('temperature')}")
        print(f"   Prompt Style: {results.best_config.get('prompt_style')}")
        print(f"   Response Format: {results.best_config.get('response_format')}")
        print(f"   Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")

        # Test with optimized config
        print("\n4. Testing with optimized configuration:")
        for msg in test_messages:
            intent = classify_support_intent(msg)
            print(f"   '{msg[:40]}...' → {intent}")


if __name__ == "__main__":
    demonstrate_customer_support()
