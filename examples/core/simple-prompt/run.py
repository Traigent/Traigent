#!/usr/bin/env python3
"""
Simple Prompt Optimization - The "Hello World" of TraiGent.

This example demonstrates the most basic usage of TraiGent:
1. Decorating a function with @traigent.optimize
2. Defining a configuration space (model, temperature)
3. Running the optimization loop
"""

import asyncio
import os
import sys
from pathlib import Path


def _prepare_mock_paths(base: Path) -> None:
    """
    Use a writable local directory for mock runs to avoid permission issues.
    """
    results_dir = base / ".traigent_local"
    os.environ.setdefault("HOME", str(base))
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRAIGENT_RESULTS_FOLDER", str(results_dir))


# --- Setup for local development/testing ---
MOCK = str(os.getenv("TRAIGENT_MOCK_MODE", "")).lower() in {"1", "true", "yes", "y"}
BASE = Path(__file__).parent
if MOCK:
    _prepare_mock_paths(BASE)

# --- Import TraiGent ---
try:
    import traigent
except ImportError:
    # Fallback for running directly from the repo without installation
    import importlib

    module_path = Path(__file__).resolve()
    for depth in (2, 3):
        try:
            sys.path.append(str(module_path.parents[depth]))
        except IndexError:
            continue
    traigent = importlib.import_module("traigent")

# --- Configuration ---
DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "simple-prompt"
DATASET_PATH = DATA_ROOT / "evaluation_set.jsonl"
if not DATASET_PATH.is_file():
    raise FileNotFoundError(f"Evaluation dataset not found at {DATASET_PATH}")
DATASET = str(DATASET_PATH)

if MOCK:
    # Initialize in edge_analytics mode for local execution
    try:
        traigent.initialize(execution_mode="edge_analytics")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Traigent mock initialization skipped: {exc}")


# --- The Optimized Function ---


@traigent.optimize(
    # 1. The dataset to evaluate against
    eval_dataset=DATASET,
    # 2. The metric(s) to optimize (default is 'accuracy' if not specified)
    objectives=["accuracy"],
    # 3. The parameters to tune
    configuration_space={
        "model": ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"],
        "temperature": [0.0, 0.7],
        "prompt_style": ["concise", "detailed"],
    },
    # 4. How parameters are injected (seamless = auto-injected into traigent.get_config())
    injection_mode="seamless",
    # 5. Execution mode (edge_analytics = local execution + analytics)
    execution_mode="edge_analytics",
)
def summarize_text(text: str) -> str:
    """Summarize the input text based on the current configuration.

    Args:
        text: The text to summarize.

    Returns:
        A summary of the input text.
    """
    # Get the current configuration chosen by the optimizer
    config = traigent.get_config()

    model = str(config.get("model", "claude-3-haiku-20240307"))
    temperature = float(config.get("temperature", 0.0))
    style = str(config.get("prompt_style", "concise"))

    print(f"Running with: model={model}, temp={temperature}, style={style}")

    # --- Mock Implementation (No API Key needed) ---
    if MOCK:
        # Simulate different behaviors based on config
        if style == "concise":
            return f"Summary of: {text[:20]}..."
        return f"Here is a detailed summary of the text: {text}"

    # --- Real Implementation (Requires API Key) ---
    # In a real app, you would call your LLM here using the config
    # e.g., client.chat.completions.create(model=model, temperature=temperature, ...)

    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("Please set ANTHROPIC_API_KEY or use TRAIGENT_MOCK_MODE=true")

    prompt = f"Please summarize the following text. Style: {style}.\n\nText: {text}"

    llm = ChatAnthropic(model=model, temperature=temperature)
    response = llm.invoke([HumanMessage(content=prompt)])
    return str(response.content)


if __name__ == "__main__":
    print("Starting Simple Prompt Optimization...")

    async def main():
        try:
            # Run the optimization
            # max_trials determines how many configurations to test
            result = await summarize_text.optimize(max_trials=5)

            print("\nOptimization Complete!")
            print(f"Best Score: {result.best_score}")
            print(f"Best Configuration: {result.best_config}")

            # Show a summary table
            df = result.to_aggregated_dataframe()
            print("\nResults Summary:")
            print(
                df[["model", "temperature", "prompt_style", "accuracy", "cost"]].to_string(
                    index=False
                )
            )
        except Exception as e:
            import traceback
            print(f"\n❌ EXAMPLE FAILED WITH ERROR: {e}")
            traceback.print_exc()
            raise

    try:
        asyncio.run(main())
    except Exception as e:
        import traceback
        print(f"\n❌ ASYNCIO.RUN FAILED WITH ERROR: {e}")
        traceback.print_exc()
        raise
