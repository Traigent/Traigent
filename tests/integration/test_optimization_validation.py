#!/usr/bin/env python3
"""
Tests for the Traigent optimization-validation discovery surface.

Previously this module was a script: it defined decorated functions but no
``test_*`` callables AND was wrapped in a module-level ``pytest.skip`` that
fired whenever the (untracked) ``data/`` directory was missing — so pytest
silently treated the whole file as skipped, hiding any regression in the
``@traigent.optimize`` decorator's registration / discovery path.

The fixtures used to live in an external ``data/`` directory; they are now
written inline (3 small JSONL files in a per-session temp dir) so the tests
exercise the decorator + discovery flow on every run.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Enable mock mode for testing
os.environ["TRAIGENT_MOCK_LLM"] = "true"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


# Build minimal inline datasets in a session-scoped temp dir so the decorators
# below can reference real, existing files without depending on an untracked
# repo-level ``data/`` directory.
_DATA_DIR = Path(tempfile.mkdtemp(prefix="traigent_test_opt_validation_"))
_TEST_DATASET = str(_DATA_DIR / "test_dataset.jsonl")
_QA_DATASET = str(_DATA_DIR / "qa_dataset.jsonl")
_SIMPLE_DATASET = str(_DATA_DIR / "simple_dataset.jsonl")

_write_jsonl(
    Path(_TEST_DATASET),
    [
        {"input": "This is good", "expected_output": "positive"},
        {"input": "This is bad", "expected_output": "negative"},
        {"input": "This is okay", "expected_output": "neutral"},
    ],
)
_write_jsonl(
    Path(_QA_DATASET),
    [
        {"input": "What is 2+2?", "expected_output": "4"},
        {"input": "Capital of France?", "expected_output": "Paris"},
    ],
)
_write_jsonl(
    Path(_SIMPLE_DATASET),
    [
        {"input": "alpha", "expected_output": "Processed alpha using fast method"},
    ],
)

import traigent


@traigent.optimize(
    eval_dataset=_TEST_DATASET,
    objectives=["accuracy", "cost"],
    configuration_space={
        "temperature": [0.0, 0.5, 1.0],
        "model": ["gpt-3.5-turbo", "gpt-4"],
    },
)
def analyze_sentiment(
    text: str, temperature: float = 0.7, model: str = "gpt-3.5-turbo"
) -> dict[str, Any]:
    """Analyze sentiment with configurable parameters."""
    confidence = 0.8 if temperature < 0.5 else 0.7
    if model == "gpt-4":
        confidence += 0.1

    sentiment = (
        "positive"
        if "good" in text.lower()
        else "negative"
        if "bad" in text.lower()
        else "neutral"
    )

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "model_used": model,
        "temperature_used": temperature,
    }


@traigent.optimize(
    eval_dataset=_QA_DATASET,
    objectives=["accuracy"],
    configuration_space={
        "max_tokens": [50, 100, 200],
        "temperature": [0.0, 0.3, 0.7],
    },
)
def answer_question(
    question: str, max_tokens: int = 100, temperature: float = 0.0
) -> str:
    """Answer questions with configurable parameters."""
    return f"Answer for {question!r} (tokens={max_tokens}, temp={temperature})"


def unoptimized_function(text: str) -> str:
    """This function is not decorated and should not be discovered."""
    return f"Unoptimized: {text}"


@traigent.optimize(
    eval_dataset=_SIMPLE_DATASET,
    objectives=["speed"],
    configuration_space={"method": ["fast", "slow"]},
)
def process_data(data: str, method: str) -> str:
    """Process data with no default parameters."""
    return f"Processed {data} using {method} method"


# --------------------------------------------------------------------------- #
# Real tests — every assertion below was previously masked by the module skip.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "func, expected_name",
    [
        (analyze_sentiment, "analyze_sentiment"),
        (answer_question, "answer_question"),
        (process_data, "process_data"),
    ],
)
def test_decorated_function_remains_callable(func, expected_name):
    """The @traigent.optimize decorator must preserve callability and __name__."""
    assert callable(func), f"{expected_name} is not callable after decoration"
    # Decorator should preserve the underlying function's name.
    assert func.__name__ == expected_name, (
        f"Decorator clobbered __name__: got {func.__name__!r}, "
        f"expected {expected_name!r}"
    )


def test_undecorated_function_is_plain_callable():
    """Undecorated functions must remain plain Python functions."""
    assert callable(unoptimized_function)
    assert unoptimized_function("x") == "Unoptimized: x"
    # Plain functions do not carry traigent decorator state.
    assert not hasattr(unoptimized_function, "_traigent_optimized")


def test_analyze_sentiment_executes_under_mock_mode():
    """Decorated function must still execute with default params under mock mode."""
    result = analyze_sentiment("This is a good day")
    assert isinstance(result, dict)
    assert result["sentiment"] == "positive"
    assert "confidence" in result
    assert result["model_used"] == "gpt-3.5-turbo"


def test_process_data_executes_without_defaults():
    """Decorated function with no default param values must still execute."""
    result = process_data("payload", "fast")
    assert result == "Processed payload using fast method"


def test_decorated_functions_register_configuration_space():
    """The decorator must expose / honor the declared configuration_space.

    The exact attribute name has shifted over releases; we accept any of the
    documented surfaces. Failure here indicates the decorator stopped attaching
    its configuration to the wrapped function — a real regression.
    """
    candidates = (
        "_traigent_config",
        "_traigent_decorator_config",
        "__traigent_optimize__",
        "_optimization_config",
        "configuration_space",
    )
    found = any(hasattr(analyze_sentiment, attr) for attr in candidates)
    assert found, (
        "Decorated function exposes none of the expected configuration "
        f"attributes ({candidates}); decorator may have stopped attaching state"
    )
