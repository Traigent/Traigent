#!/usr/bin/env python3
"""
Error Handling — Graceful failure modes and fallback patterns.

This example demonstrates 5 error scenarios:
1. Invalid configuration space (bad types)
2. Sample-budget exceeded (budget_limit / budget_metric)
3. Optimization timeout (stop_reason == "timeout")
4. Preflight validation (environment checks)
5. Graceful fallback (try/except with default config)
"""

import asyncio
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
DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "error-handling"
DATASET_PATH = DATA_ROOT / "evaluation_set.jsonl"
if not DATASET_PATH.is_file():
    raise FileNotFoundError(f"Evaluation dataset not found at {DATASET_PATH}")
DATASET = str(DATASET_PATH)

if MOCK:
    try:
        traigent.initialize(execution_mode="edge_analytics")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Traigent mock initialization skipped: {exc}")


# --- Mock Implementation ---


def _mock_answer(text: str) -> str:
    """Deterministic mock answers."""
    t = (text or "").lower()
    mapping = [
        (["machine learning"], "A subset of AI that learns from data."),
        (["cloud computing"], "On-demand delivery of IT resources via the internet."),
        (
            ["neural network"],
            "A computing system inspired by biological neural networks.",
        ),
        (
            ["devops"],
            "A set of practices combining software development and IT operations.",
        ),
        (
            ["containerization"],
            "Packaging software with its dependencies into isolated containers.",
        ),
        (
            ["microservices"],
            "An architecture where applications are collections of small services.",
        ),
        (
            ["ci/cd"],
            "Continuous integration and continuous delivery for software releases.",
        ),
        (["api"], "An interface that allows software applications to communicate."),
        (["version control"], "A system for tracking changes to files over time."),
        (
            ["encryption"],
            "Converting data into a coded format to prevent unauthorized access.",
        ),
    ]
    for keywords, answer in mapping:
        if any(kw in t for kw in keywords):
            return answer
    return "An important concept in technology."


# --- Working optimized function (used by scenarios 2, 3, 5) ---


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["accuracy"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.7],
    },
    injection_mode="seamless",
    execution_mode="edge_analytics",
)
def explain_concept(text: str) -> str:
    """Explain a technical concept."""
    config = traigent.get_config()
    if MOCK:
        return _mock_answer(text)
    return f"Explanation using {config.get('model')}: {text}"


# --- Scenario Functions ---


def print_scenario(number: int, title: str):
    """Print a scenario header."""
    print(f"\n{'=' * 60}")
    print(f"Scenario {number}: {title}")
    print("=" * 60)


async def demo_invalid_config():
    """Scenario 1: Invalid configuration space — bad types raise ValueError."""
    print_scenario(1, "Invalid Configuration Space")

    try:

        @traigent.optimize(
            eval_dataset=DATASET,
            objectives=["accuracy"],
            configuration_space={
                "model": "not-a-list",  # Must be a list — this triggers ValueError
            },
            injection_mode="seamless",
            execution_mode="edge_analytics",
        )
        def bad_function(text: str) -> str:
            return text

        # If decoration didn't raise, try running
        await bad_function.optimize(max_trials=2)
        print("  Unexpected: no error raised")
    except (ValueError, TypeError) as e:
        print(f"  Caught expected error: {type(e).__name__}: {e}")
        print("  Resolution: Fix configuration_space to use lists for each parameter.")
    except Exception as e:
        print(f"  Caught error: {type(e).__name__}: {e}")
        print("  Resolution: Validate configuration_space types before decorating.")

    print("  Status: HANDLED")


async def demo_budget_exceeded():
    """Scenario 2: Sample-budget exceeded via budget_limit/budget_metric."""
    print_scenario(2, "Sample-Budget Exceeded")

    try:
        # Use budget_limit with budget_metric="examples_attempted" to cap samples.
        # The orchestrator maps "budget" → "cost_limit" internally (orchestrator.py:2034).
        result = await explain_concept.optimize(
            max_trials=20,
            budget_limit=2,
            budget_metric="examples_attempted",
        )
        print(f"  Optimization stopped: stop_reason={result.stop_reason}")
        print(f"  Trials completed: {len(result.trials)}")
        # Budget stop is mapped to "cost_limit" by the orchestrator.
        if result.stop_reason in ("budget", "cost_limit"):
            print("  Budget/cost limit reached — optimization capped as expected.")
        else:
            print(f"  Stopped for other reason: {result.stop_reason}")
        print("  Resolution: Increase budget_limit or reduce dataset size.")
    except Exception as e:
        print(f"  Caught error: {type(e).__name__}: {e}")
        print("  Resolution: Check budget_metric matches a metric your trials produce.")

    print("  Status: HANDLED")


async def demo_timeout():
    """Scenario 3: Optimization timeout — result.stop_reason == 'timeout'."""
    print_scenario(3, "Optimization Timeout")

    try:
        # Set a very short timeout (1 second) so the optimization times out
        result = await explain_concept.optimize(
            max_trials=100,
            timeout=1,
        )
        print(f"  Optimization completed: stop_reason={result.stop_reason}")
        print(f"  Trials completed: {len(result.trials)}")
        if result.stop_reason == "timeout":
            print("  Timeout reached as expected — optimization stopped gracefully.")
        else:
            print(f"  Stopped for other reason: {result.stop_reason}")
        print("  Resolution: Increase timeout or reduce max_trials/dataset size.")
    except Exception as e:
        print(f"  Caught error: {type(e).__name__}: {e}")
        print("  Resolution: Check timeout parameter and system resources.")

    print("  Status: HANDLED")


async def demo_preflight_validation():
    """Scenario 4: Preflight validation — check environment before running."""
    print_scenario(4, "Preflight Validation")

    # Check for required environment variables
    required_vars = {
        "OPENAI_API_KEY": "Required for OpenAI models",
        "ANTHROPIC_API_KEY": "Required for Anthropic models",
    }

    missing = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            print(f"  {var}: SET")
        else:
            print(f"  {var}: MISSING — {description}")
            missing.append(var)

    if missing and not MOCK:
        print(
            f"\n  Would fail in production: {len(missing)} required variable(s) missing."
        )
        print("  Resolution: Set environment variables or use TRAIGENT_MOCK_LLM=true.")
    elif missing and MOCK:
        print(f"\n  {len(missing)} variable(s) missing (OK in mock mode).")
        print("  In production, these would cause AuthenticationError.")
        print("  Pattern: Always validate environment before calling optimize().")
    else:
        print("\n  All environment variables are set.")

    print("  Status: HANDLED")


async def demo_graceful_fallback():
    """Scenario 5: Graceful fallback — use default config on any failure."""
    print_scenario(5, "Graceful Fallback to Default Config")

    default_config = {"model": "gpt-4o-mini", "temperature": 0.0}

    try:
        result = await explain_concept.optimize(max_trials=3)
        config = result.best_config
        print(f"  Optimization succeeded: best_config={config}")
    except Exception as e:
        print(f"  Optimization failed: {type(e).__name__}: {e}")
        config = default_config
        print(f"  Falling back to default config: {config}")

    # Use whichever config we got
    print(f"\n  Running with config: {config}")
    if MOCK:
        answer = _mock_answer("What is machine learning?")
    else:
        answer = f"Answer using {config}"
    print(f"  Result: {answer}")
    print("  Pattern: Always have a fallback config for production resilience.")

    print("  Status: HANDLED")


# --- Main ---


async def run_all_demos():
    """Run all error handling scenarios."""
    print("=" * 60)
    print("Error Handling Example — 5 Scenarios")
    print("=" * 60)

    await demo_invalid_config()
    await demo_budget_exceeded()
    await demo_timeout()
    await demo_preflight_validation()
    await demo_graceful_fallback()

    print("\n" + "=" * 60)
    print("All 5 error scenarios handled successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(run_all_demos())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
