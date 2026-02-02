#!/usr/bin/env python3
"""Example 1: Basic QA Tuning - Model and temperature optimization.

Usage (run in a terminal from repo root, works without activating venv):
    export OPENAI_API_KEY="your-key"
    .venv/bin/python walkthrough/examples/real/01_tuning_qa.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI

import traigent
from traigent import TraigentConfig

from utils.helpers import (
    configure_logging,
    print_cost_estimate,
    print_estimated_time,
    print_optimization_config,
    print_results_table,
    require_openai_key,
    sanitize_traigent_api_key,
    setup_example_logger,
)
from utils.scoring import STOPWORDS, token_match_score, token_matches, tokenize

require_openai_key("01_tuning_qa.py")
sanitize_traigent_api_key()
configure_logging()

logger = setup_example_logger("01_tuning_qa")

# Avoid per-run cost estimate prompts in walkthrough examples.
os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

# Toggle minimal logging via TRAIGENT_MINIMAL_LOGGING=1
MINIMAL_LOGGING = os.getenv("TRAIGENT_MINIMAL_LOGGING", "1").lower() in (
    "1",
    "true",
    "yes",
)
REQUIRE_CONFIRM = os.getenv("TRAIGENT_REQUIRE_CONFIRM", "1").lower() in (
    "1",
    "true",
    "yes",
)
traigent.initialize(
    config=TraigentConfig(
        execution_mode="edge_analytics",
        minimal_logging=MINIMAL_LOGGING,
    )
)

# Dataset path relative to this file
DATASETS = Path(__file__).parent.parent / "datasets"
DEBUG_EVAL = os.getenv("TRAIGENT_DEBUG_EVAL", "").lower() in ("1", "true", "yes")
DEBUG_EVAL_PATH = os.getenv("TRAIGENT_DEBUG_EVAL_PATH")
OBJECTIVES = ["accuracy", "cost"]
CONFIG_SPACE = {
    "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
    "temperature": [0.1, 0.7],
}

# Required fraction of expected tokens that must match (used for debug logging)
_REQUIRED_FRACTION = 0.8


def _resolve_debug_path() -> Path | None:
    if DEBUG_EVAL_PATH:
        return Path(DEBUG_EVAL_PATH)
    return None


def results_match_score(output: str, expected: str, **kwargs) -> float:
    """Return 1.0 when >=80% of expected tokens appear in output (case-insensitive).

    Wraps the shared token_match_score with debug logging support.
    """
    if not DEBUG_EVAL:
        return token_match_score(output, expected, **kwargs)

    # Debug mode: compute match details for logging
    if output is None or expected is None:
        return 0.0
    output_text = str(output)
    expected_text = str(expected).strip()
    if not expected_text:
        return 0.0
    output_tokens = {t for t in tokenize(output_text) if t not in STOPWORDS}
    expected_tokens = [t for t in tokenize(expected_text) if t not in STOPWORDS]
    if not expected_tokens:
        return 0.0
    missing = [
        token for token in expected_tokens if not token_matches(token, output_tokens)
    ]
    match_ratio = (len(expected_tokens) - len(missing)) / len(expected_tokens)
    match = match_ratio >= _REQUIRED_FRACTION
    if missing:
        debug_path = _resolve_debug_path()
        if debug_path is not None:
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            with debug_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"[eval-miss] expected={expected_text!r} output={output_text!r} missing={missing!r} match_ratio={match_ratio:.2f}\n"
                )
        logger.info(
            "eval-miss expected=%r output=%r missing=%r match_ratio=%.2f",
            expected_text,
            output_text,
            missing,
            match_ratio,
        )
    return 1.0 if match else 0.0


@traigent.optimize(
    eval_dataset=str(DATASETS / "simple_questions.jsonl"),
    objectives=OBJECTIVES,
    scoring_function=results_match_score,
    configuration_space=CONFIG_SPACE,
    injection_mode="context",  # default injection mode, added explicitly for clarity
    execution_mode="edge_analytics",
)
def answer_question(question: str) -> str:
    """Simple Q&A function using OpenAI."""
    config = traigent.get_config()
    llm = ChatOpenAI(
        model=config["model"],
        temperature=config["temperature"],
    )
    try:
        response = llm.invoke(

                "Answer with the final answer only. Do not ask questions. "
                "Keep it concise and use the question's terminology. "
                "If the answer is numeric, use digits only (no commas), no units, "
                "and follow the units implied by the question. Do not round. "
                f"{question}"

        )
        return str(response.content)
    except Exception as exc:
        logger.warning("LLM call failed: %s: %s", type(exc).__name__, exc)
        return f"Error: {type(exc).__name__}: {exc}"


async def main() -> None:
    logger.info("Traigent Example 1: Simple Optimization")
    logger.info("=" * 50)
    print_optimization_config(OBJECTIVES, CONFIG_SPACE)
    print_cost_estimate(
        models=CONFIG_SPACE["model"],
        dataset_size=20,
        task_type="simple_qa",
        num_trials=8,
    )

    if REQUIRE_CONFIRM and sys.stdin.isatty():
        response = input("Run this example? [y/N] ").strip().lower()
        if response not in ("y", "yes"):
            logger.info("Cancelled.")
            return

    if DEBUG_EVAL:
        debug_path = _resolve_debug_path()
        if debug_path is not None:
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text("", encoding="utf-8")

    print_estimated_time("01_tuning_qa.py")
    results = await answer_question.optimize(
        algorithm="grid",
        max_trials=8,
        timeout=160,
        show_progress=True,
        random_seed=42,
    )

    print_results_table(results, CONFIG_SPACE, OBJECTIVES, is_mock=False)

    logger.info("Best Configuration Found:")
    logger.info("  Model: %s", results.best_config.get("model"))
    logger.info("  Temperature: %s", results.best_config.get("temperature"))
    logger.info("Performance:")
    logger.info("  Accuracy: %.2f%%", results.best_metrics.get("accuracy", 0) * 100)
    logger.info("  Cost: $%.6f", results.best_metrics.get("cost", 0))


if __name__ == "__main__":
    asyncio.run(main())
