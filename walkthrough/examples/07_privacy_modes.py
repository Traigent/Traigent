#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false, reportUndefinedVariable=false, reportArgumentType=false
# mypy: ignore-errors
# flake8: noqa
# ruff: noqa
# pylint: disable=all

"""Example 7: Privacy Modes - Local, Cloud, and Hybrid execution."""

import asyncio
import os

from _shared import add_repo_root_to_sys_path, dataset_path, ensure_dataset, init_mock_mode

add_repo_root_to_sys_path(__file__)

SIMPLE_PRIVACY_DATASET = dataset_path(__file__, "simple_questions.jsonl")
ensure_dataset(
    SIMPLE_PRIVACY_DATASET,
    [
        {
            "input": {"text": "What is AI?"},
            "expected_output": "Artificial Intelligence",
        },
        {
            "input": {"text": "Explain privacy modes"},
            "expected_output": "local, cloud, and hybrid",
        },
    ],
)

import traigent

MOCK = init_mock_mode()

# Example 1: Local Mode - Complete Privacy
@traigent.optimize(
    eval_dataset=str(SIMPLE_PRIVACY_DATASET),
    objectives=["accuracy"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.1, 0.5]
    },
    execution_mode="edge_analytics",  # Everything stays on your machine!
    local_storage_path="./local_results",
    minimal_logging=True
)
def local_mode_example(text: str) -> str:
    """Complete privacy - no data leaves your machine."""
    if MOCK:
        return f"Local response for: {text[:20]}..."

    from langchain_openai import ChatOpenAI
    config = traigent.get_current_config()
    llm = ChatOpenAI(model=config.get("model"), temperature=config.get("temperature"))
    return llm.invoke(text).content

# Example 2: Cloud Mode - Advanced Optimization
@traigent.optimize(
    eval_dataset=str(SIMPLE_PRIVACY_DATASET),
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3, 0.6, 0.9]
    },
    execution_mode="cloud",  # Use TraiGent cloud for smart optimization
)
def cloud_mode_example(text: str) -> str:
    """Cloud mode with Bayesian optimization."""
    if MOCK:
        return f"Cloud-optimized response: {text[:20]}..."

    from langchain_openai import ChatOpenAI
    config = traigent.get_current_config()
    llm = ChatOpenAI(model=config.get("model"), temperature=config.get("temperature"))
    return llm.invoke(text).content

# Example 3: Hybrid Mode - Best of Both
@traigent.optimize(
    eval_dataset=str(SIMPLE_PRIVACY_DATASET),
    objectives=["accuracy", "cost", "latency"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.1, 0.5],
        "max_tokens": [100, 200]
    },
    execution_mode="hybrid",  # Local execution, cloud intelligence
    privacy_enabled=True  # Never send actual data to cloud
)
def hybrid_mode_example(text: str) -> str:
    """Hybrid - local execution with cloud optimization guidance."""
    if MOCK:
        return f"Hybrid response: {text[:20]}..."

    from langchain_openai import ChatOpenAI
    config = traigent.get_current_config()
    llm = ChatOpenAI(
        model=config.get("model"),
        temperature=config.get("temperature"),
        model_kwargs={"max_tokens": config.get("max_tokens")}
    )
    return llm.invoke(text).content


MODE_CONFIGS = {
    "local": {
        "func": local_mode_example,
        "emoji": "🏠",
        "label": "LOCAL mode",
        "opt_kwargs": {"algorithm": "grid", "max_trials": 4 if not MOCK else 2},
    },
    "cloud": {
        "func": cloud_mode_example,
        "emoji": "☁️",
        "label": "CLOUD mode",
        "opt_kwargs": {
            "algorithm": "bayesian" if not MOCK else "random",
            "max_trials": 12 if not MOCK else 4,
        },
    },
    "hybrid": {
        "func": hybrid_mode_example,
        "emoji": "🔀",
        "label": "HYBRID mode",
        "opt_kwargs": {"algorithm": "random", "max_trials": 6 if not MOCK else 3},
    },
}


async def main():
    print("🎯 TraiGent Example 7: Privacy & Execution Modes")
    print("="*50)
    print("🔐 Choose your privacy level\n")

    print("🏠 LOCAL MODE - Complete Privacy:")
    print("  • All data stays on your machine")
    print("  • No external API calls for optimization")
    print("  • Results stored locally")
    print("  • Perfect for sensitive data\n")

    print("☁️ CLOUD MODE - Advanced Algorithms:")
    print("  • Bayesian optimization")
    print("  • Smart parameter exploration")
    print("  • Faster convergence")
    print("  • Team collaboration features\n")

    print("🔀 HYBRID MODE - Best of Both:")
    print("  • Data stays local")
    print("  • Only metadata to cloud")
    print("  • Cloud optimization strategies")
    print("  • Privacy + Performance\n")

    requested_mode = os.getenv("TRAIGENT_PRIVACY_MODE", "local").lower()
    if requested_mode == "all":
        modes_to_run = list(MODE_CONFIGS.keys())
    elif requested_mode in MODE_CONFIGS:
        modes_to_run = [requested_mode]
    else:
        print(f"Unknown mode '{requested_mode}', defaulting to LOCAL.")
        modes_to_run = ["local"]

    for mode_key in modes_to_run:
        cfg = MODE_CONFIGS[mode_key]
        print(f"\n{cfg['emoji']} Running {cfg['label']} optimization...\n")
        if mode_key == "cloud" and MOCK:
            print("  (Simulated cloud optimization in mock mode)")
        results = await cfg["func"].optimize(**cfg["opt_kwargs"])
        print(f"✅ {cfg['label']} optimization complete!")
        print(f"   Best Config: {results.best_config}")
        print(f"   Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")

    print("\n💡 Choose the mode that fits your needs:")
    print("  • Sensitive data? Use LOCAL")
    print("  • Complex optimization? Use CLOUD")
    print("  • Want both? Use HYBRID")

if __name__ == "__main__":
    asyncio.run(main())
