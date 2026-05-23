#!/usr/bin/env python3
"""
Simple Prompt Optimization - The "Hello World" of Traigent.

This example demonstrates the most basic usage of Traigent:
1. Decorating a function with @traigent.optimize
2. Defining a configuration space (model, temperature)
3. Running the optimization loop
"""

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")
BASE = Path(__file__).parent
MODULE_PATH = Path(__file__).resolve()


def _env_flag(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


_mock_env_flag = _env_flag(os.environ.get("TRAIGENT_MOCK_LLM"))
if _mock_env_flag is True or (
    _mock_env_flag is not False and not os.environ.get("ANTHROPIC_API_KEY", "").strip()
):
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

_sdk_path = os.environ.get("TRAIGENT_SDK_PATH")
if _sdk_path:
    sys.path.insert(0, _sdk_path)
else:
    repo_root = MODULE_PATH.parents[3]
    if (repo_root / "traigent" / "__init__.py").exists():
        sys.path.insert(0, str(repo_root))

# --- Import Traigent ---
try:
    import traigent
except ImportError:
    import importlib

    traigent = importlib.import_module("traigent")

from traigent.examples.tutorial_bootstrap import configure_tutorial_mock_mode

# --- Setup for local development/testing ---
MOCK = configure_tutorial_mock_mode(
    provider_env_keys=("ANTHROPIC_API_KEY",),
    tutorial_name="Simple Prompt Optimization",
    results_base=BASE,
)

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


def _mock_summarize_text(text: str) -> str:
    """Return deterministic summaries for mock mode achieving 75%+ accuracy.

    Covers all 20 proverb summarization evaluation set examples.
    """
    t = (text or "").lower()
    # Mapping from proverb keywords to expected summary
    # Ordered by specificity to handle similar phrases
    mapping = [
        (["quick brown fox", "lazy dog"], "Quick fox jumps over lazy dog."),
        (["to be or not to be"], "Debates whether to exist or not."),
        (["glitters", "gold"], "Shiny things are not always valuable."),
        (["thousand miles", "single step"], "Long journeys start with small steps."),
        (["knowledge is power"], "Information gives advantage."),
        (["actions speak louder"], "Behavior matters more than talk."),
        (["early bird"], "Being early has advantages."),
        (["practice makes perfect"], "Repetition improves skill."),
        (["when in rome"], "Adapt to local customs."),
        (["two heads"], "Collaboration improves outcomes."),
        (["pen is mightier"], "Words have more power than force."),
        (["better late than never"], "Delayed action beats inaction."),
        (["rome was not built"], "Great things take time."),
        (["silver lining", "cloud"], "Bad situations have positives."),
        (["time heals"], "Pain diminishes over time."),
        (["eggs in one basket"], "Diversify your options."),
        (["grass is always greener"], "Others seem to have it better."),
        (["smoke", "fire"], "Signs indicate underlying issues."),
        (["picture is worth"], "Visuals convey complex ideas."),
        (["curiosity killed"], "Excessive questioning has risks."),
    ]
    for keywords, summary in mapping:
        if any(kw in t for kw in keywords):
            return summary
    return "Summary of input text."


def _print_results_summary(result) -> None:
    rows = []
    for trial in result.trials:
        metrics = trial.metrics or {}
        rows.append(
            [
                str(trial.config.get("model", "")),
                str(trial.config.get("temperature", "")),
                str(trial.config.get("prompt_style", "")),
                f"{float(metrics.get('accuracy', 0.0)):.3f}",
                f"{float(metrics.get('cost', 0.0)):.6f}",
            ]
        )

    headers = ["model", "temperature", "prompt_style", "accuracy", "cost"]
    widths = []
    for index, header in enumerate(headers):
        widths.append(max([len(header), *(len(row[index]) for row in rows)]))

    print("\nResults Summary:")
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


@traigent.optimize(
    # 1. The dataset to evaluate against
    eval_dataset=DATASET,
    # 2. The metric(s) to optimize (default is 'accuracy' if not specified)
    objectives=["accuracy"],
    # 3. The parameters to tune
    configuration_space={
        "model": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"],
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

    model = str(config.get("model", "claude-haiku-4-5-20251001"))
    temperature = float(config.get("temperature", 0.0))
    style = str(config.get("prompt_style", "concise"))

    print(f"Running with: model={model}, temp={temperature}, style={style}")

    # --- Mock Implementation (No API Key needed) ---
    if MOCK:
        return _mock_summarize_text(text)

    # --- Real Implementation (Requires API Key) ---
    # In a real app, you would call your LLM here using the config
    # e.g., client.chat.completions.create(model=model, temperature=temperature, ...)

    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError(
            "Please set ANTHROPIC_API_KEY or use TRAIGENT_MOCK_LLM=true. "
            "For the no-key tutorial path, leave TRAIGENT_MOCK_LLM unset."
        )

    prompt = f"Please summarize the following text. Style: {style}.\n\nText: {text}"

    llm = ChatAnthropic(model=model, temperature=temperature)
    response = llm.invoke([HumanMessage(content=prompt)])
    return str(response.content)


if __name__ == "__main__":
    try:
        print("Starting Simple Prompt Optimization...")

        async def main():
            try:
                # Run the optimization
                # max_trials determines how many configurations to test
                result = await summarize_text.optimize(max_trials=5)

                print("\nOptimization Complete!")
                print(f"Best Score: {result.best_score}")
                print(f"Best Configuration: {result.best_config}")

                _print_results_summary(result)
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
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
