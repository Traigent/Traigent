#!/usr/bin/env python3
"""Example: Applying Saved Configurations - Save and Load Optimal Configs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


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

# Step 1: Run optimization and save the best configuration (MCQ format)


@traigent.optimize(
    configuration_space={
        # Keep grid size small so total trials stay reasonable (<=10)
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.1, 0.5],
        "max_tokens": [100, 200],
    },
    eval_dataset=os.path.join(os.path.dirname(__file__), "customer_queries.jsonl"),
    objectives=["accuracy"],
    execution_mode="edge_analytics",
    max_trials=10,
)
def customer_support_agent(question: str, choices: list[str]) -> int:
    """Select the correct answer index for a customer support MCQ.

    Returns a 0-based index corresponding to the best choice.
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, max_tokens=120)

    labeled_choices = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(choices))
    prompt = f"""You are a helpful customer support assistant.
Choose the single best answer to the question below from the options A-D.
Respond with only the letter A, B, C, or D.

Question: {question}

Options:
{labeled_choices}
"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    text = (getattr(resp, "content", "") or "").strip().upper()

    # Parse letter (A-D) or digit (1-4) to 0-based index
    idx = 0
    if text:
        if text[0] in {"A", "B", "C", "D"}:
            idx = ord(text[0]) - ord("A")
        elif text[0].isdigit():
            d = int(text[0])
            if 1 <= d <= len(choices):
                idx = d - 1
        else:
            # Try to match by choice text snippet
            for i, opt in enumerate(choices):
                if opt.lower() in text.lower():
                    idx = i
                    break
    return max(0, min(idx, len(choices) - 1))


# Step 2: Example of using the optimized function after optimization
def use_optimized_support():
    """Example showing how to use the function after optimization."""

    # Run optimization (in production, this would be done separately)
    print("Running optimization...")
    results = customer_support_agent.optimize()

    # Save the best configuration to a file
    best_config_file = "optimal_support_config.json"
    with open(best_config_file, "w") as f:
        json.dump(results.best_config, f, indent=2)
    print(f"Saved best config to {best_config_file}")

    # The function now automatically uses the best config
    q = "How can a user reset their account password?"
    ch = [
        "Email support to request a reset",
        "Use the Reset Password link in account settings",
        "Create a new account",
        "Send a message in the chat",
    ]
    pred = customer_support_agent(q, ch)
    print(f"Predicted choice index: {pred}  -> {chr(65+pred)}. {ch[pred]}")

    return results


# Step 3: Load and apply saved configuration in a new session
def load_and_apply_saved_config():
    """Load previously saved configuration and apply it."""

    config_file = "optimal_support_config.json"

    # Check if config file exists
    if os.path.exists(config_file):
        # Load the saved configuration
        with open(config_file) as f:
            saved_config = json.load(f)

        print(f"Loaded config: {saved_config}")

        # Create a new function instance with the saved config
        @traigent.optimize(
            configuration_space={
                "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
                "temperature": [0.1, 0.3, 0.5, 0.7],
                "max_tokens": [100, 200, 300],
            },
            # No eval_dataset needed since we're just applying config
            objectives=["accuracy"],
            execution_mode="edge_analytics",
        )
        def production_support_agent(question: str, choices: list[str]) -> int:
            """Production MCQ selector using saved config."""
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",  # Overridden by saved config
                temperature=0.3,  # Overridden by saved config
                max_tokens=120,  # Overridden by saved config
            )
            labeled = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(choices))
            prompt = f"""Choose the best answer (A-D) for the question below.
Respond with only the letter A, B, C, or D.

Question: {question}

Options:
{labeled}
"""
            resp = llm.invoke([HumanMessage(content=prompt)])
            text = (getattr(resp, "content", "") or "").strip().upper()
            idx = 0
            if text:
                if text[0] in {"A", "B", "C", "D"}:
                    idx = ord(text[0]) - ord("A")
                elif text[0].isdigit():
                    d = int(text[0])
                    if 1 <= d <= len(choices):
                        idx = d - 1
                else:
                    for i, opt in enumerate(choices):
                        if opt.lower() in text.lower():
                            idx = i
                            break
            return max(0, min(idx, len(choices) - 1))

        # Apply the saved configuration
        production_support_agent.set_config(saved_config)

        # Now the function uses the saved optimal parameters
        tests = [
            (
                "How do I track my order?",
                [
                    "Tracking is not available",
                    "Call the courier",
                    "Use the tracking link in the shipping confirmation email",
                    "Wait 14 days",
                ],
            ),
            (
                "How to update a billing address?",
                [
                    "It cannot be updated",
                    "Ask the bank to change it",
                    "Update it in Billing Settings",
                    "Send a paper form",
                ],
            ),
        ]

        print("\nTesting with saved configuration:")
        for q, ch in tests:
            pred = production_support_agent(q, ch)
            print(f"Q: {q}")
            print(f"Chosen: {chr(65+pred)}. {ch[pred]}\n")
    else:
        print(f"Config file {config_file} not found. Run optimization first.")


if __name__ == "__main__":
    try:
        # Example usage
        print("=" * 60)
        print("Traigent: Apply Saved Configuration Example")
        print("=" * 60)

        # You can either:
        # 1. Run optimization and save config
        # results = use_optimized_support()

        # 2. Or load and use a previously saved config
        load_and_apply_saved_config()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
