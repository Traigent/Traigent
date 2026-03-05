#!/usr/bin/env python3
"""
Production Deployment - Loading and applying optimized configurations.

This example demonstrates the production deployment workflow:
1. Run optimization to find the best configuration
2. Save the best configuration to a JSON file
3. Load the saved configuration in a "production" context
4. Run the function with the saved (frozen) configuration
"""

import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


def _prepare_mock_paths(base: Path) -> None:
    """Use a writable local directory for mock runs."""
    results_dir = base / ".traigent_local"
    os.environ.setdefault("HOME", str(base))
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRAIGENT_RESULTS_FOLDER", str(results_dir))


# --- Setup ---
MOCK = str(os.getenv("TRAIGENT_MOCK_LLM", "")).lower() in {"1", "true", "yes", "y"}
BASE = Path(__file__).parent
if MOCK:
    _prepare_mock_paths(BASE)

# --- Import Traigent ---
try:
    import traigent
except ImportError:
    import importlib

    module_path = Path(__file__).resolve()
    for depth in (2, 3):
        try:
            sys.path.append(str(module_path.parents[depth]))
        except IndexError:
            continue
    traigent = importlib.import_module("traigent")

# --- Configuration ---
DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "production-deployment"
DATASET_PATH = DATA_ROOT / "evaluation_set.jsonl"
if not DATASET_PATH.is_file():
    raise FileNotFoundError(f"Evaluation dataset not found at {DATASET_PATH}")
DATASET = str(DATASET_PATH)
CONFIG_DIR = BASE / ".traigent_local"
if MOCK:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
SAVED_CONFIG_PATH = CONFIG_DIR / "best_config.json"
SAVED_CONFIG_PATH = CONFIG_DIR / "best_config.json"

if MOCK:
    try:
        traigent.initialize(execution_mode="edge_analytics")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Traigent mock initialization skipped: {exc}")


# --- Mock Implementation ---


def _mock_answer(query: str) -> str:
    """Deterministic mock answers for the evaluation set."""
    q = (query or "").lower()
    mapping = [
        (
            ["reset", "password"],
            "Go to Settings > Security > Reset Password and follow the prompts.",
        ),
        (["business hours", "hours"], "We are open Monday to Friday, 9 AM to 5 PM."),
        (
            ["cancel", "subscription"],
            "Go to Account > Subscription > Cancel and confirm.",
        ),
        (
            ["payment", "methods"],
            "We accept Visa, Mastercard, PayPal, and bank transfers.",
        ),
        (["contact", "support"], "Email support@example.com or call 1-800-555-0123."),
        (["order history"], "Go to Account > Orders to view your purchase history."),
        (
            ["billing", "address"],
            "Go to Account > Billing > Edit Address and save changes.",
        ),
        (
            ["refund", "policy"],
            "Full refund within 30 days of purchase, no questions asked.",
        ),
        (
            ["two-factor", "2fa"],
            "Go to Settings > Security > Enable 2FA and scan the QR code.",
        ),
        (["username"], "Go to Settings > Profile > Edit Username and save."),
        (
            ["export", "data"],
            "Go to Settings > Privacy > Export Data to download your information.",
        ),
        (["browser"], "We support Chrome, Firefox, Safari, and Edge."),
    ]
    for keywords, answer in mapping:
        if any(kw in q for kw in keywords):
            return answer
    return "Please contact support for assistance."


# --- Optimized Function ---


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["accuracy"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o", "claude-3-haiku-20240307"],
        "temperature": [0.0, 0.3, 0.7],
        "max_tokens": [256, 512, 1024],
    },
    injection_mode="seamless",
    execution_mode="edge_analytics",
)
def answer_query(query: str) -> str:
    """Answer a customer support query using the current configuration."""
    config = traigent.get_config()
    model = str(config.get("model", "gpt-4o-mini"))
    temperature = float(config.get("temperature", 0.0))
    max_tokens = int(config.get("max_tokens", 256))

    print(f"  Config: model={model}, temp={temperature}, max_tokens={max_tokens}")

    if MOCK:
        return _mock_answer(query)

    # Real implementation would call the LLM here
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
    response = llm.invoke([HumanMessage(content=f"Answer this support query: {query}")])
    return str(response.content)


# --- Production Helpers ---


def save_best_config(result, path: Path) -> dict:
    """Save the best configuration from an optimization result."""
    best_config = result.best_config
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"Saved best config to {path}")
    return best_config


def load_config(path: Path) -> dict:
    """Load a saved configuration from disk."""
    with open(path) as f:
        config = json.load(f)
    print(f"Loaded config from {path}: {config}")
    return config


def run_with_config(config: dict, query: str) -> str:
    """Run the optimized function with a specific (frozen) configuration."""
    # In production, you apply the saved config directly
    print(f"  [Production] Using saved config: {config}")
    if MOCK:
        return _mock_answer(query)
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI(
        model=config.get("model", "gpt-4o-mini"),
        temperature=config.get("temperature", 0.0),
        max_tokens=config.get("max_tokens", 256),
    )
    response = llm.invoke([HumanMessage(content=f"Answer this support query: {query}")])
    return str(response.content)


# --- Main ---


async def main():
    """Run the full production deployment workflow."""
    print("=" * 60)
    print("Production Deployment Example")
    print("=" * 60)

    # Phase 1: Optimize
    print("\n--- Phase 1: Run Optimization ---")
    result = await answer_query.optimize(max_trials=5)
    print(f"\nBest Score: {result.best_score}")
    print(f"Best Config: {result.best_config}")

    # Phase 2: Save
    print("\n--- Phase 2: Save Best Config ---")
    save_best_config(result, SAVED_CONFIG_PATH)

    # Phase 3: Load in "production"
    print("\n--- Phase 3: Load Config in Production ---")
    production_config = load_config(SAVED_CONFIG_PATH)

    # Phase 4: Run with saved config
    print("\n--- Phase 4: Run with Saved Config ---")
    test_queries = [
        "How do I reset my password?",
        "What payment methods do you accept?",
        "How do I export my data?",
    ]
    for query in test_queries:
        print(f"\nQuery: {query}")
        answer = run_with_config(production_config, query)
        print(f"Answer: {answer}")

    # Cleanup
    if SAVED_CONFIG_PATH.exists():
        SAVED_CONFIG_PATH.unlink()

    print("\n" + "=" * 60)
    print("Production deployment workflow complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
