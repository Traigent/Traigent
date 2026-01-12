"""Evaluator for GPT-4.1 Coding Study.

Metrics:
- coding_accuracy: Whether the generated code completes the task
- diff_compliance: Whether diff format was followed correctly
"""

from __future__ import annotations

import difflib
from typing import Any


class CodingEvaluator:
    """Evaluator for coding tasks."""

    def __call__(
        self,
        output: dict[str, Any] | None,
        expected: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, float]:
        """Evaluate coding task output.

        Args:
            output: Agent output dict with generated_code, task_completed, etc.
            expected: Expected output with expected_code

        Returns:
            Dict with accuracy and diff_compliance scores
        """
        if output is None:
            return {"accuracy": 0.0, "diff_compliance": 0.0}

        accuracy = coding_accuracy(output, expected, **kwargs)
        compliance = diff_compliance(output, expected, **kwargs)

        return {
            "accuracy": accuracy,
            "diff_compliance": compliance,
        }


def coding_accuracy(
    output: dict[str, Any] | None,
    expected: dict[str, Any],
    **kwargs: Any,
) -> float:
    """Calculate task completion accuracy.

    Compares generated code against expected code using:
    - Exact match check
    - Similarity ratio for partial credit
    - Task completion flag

    Args:
        output: Agent output with generated_code
        expected: Expected output with expected_code

    Returns:
        Accuracy score between 0.0 and 1.0
    """
    if output is None:
        return 0.0

    # Check task_completed flag first
    if not output.get("task_completed", False):
        return 0.0

    generated = output.get("generated_code", "")
    expected_code = expected.get("expected_code", "")

    if not generated or not expected_code:
        return 0.0

    # Normalize code for comparison
    generated_normalized = _normalize_code(generated)
    expected_normalized = _normalize_code(expected_code)

    # Exact match
    if generated_normalized == expected_normalized:
        return 1.0

    # Calculate similarity ratio
    similarity = difflib.SequenceMatcher(
        None, generated_normalized, expected_normalized
    ).ratio()

    # Apply threshold - need at least 70% similarity for partial credit
    if similarity < 0.7:
        return 0.0

    return similarity


def diff_compliance(
    output: dict[str, Any] | None,
    expected: dict[str, Any],
    **kwargs: Any,
) -> float:
    """Calculate diff format compliance.

    Checks if the agent followed the diff format correctly when requested.

    Args:
        output: Agent output with diff_compliant flag
        expected: Expected output (not used for this metric)

    Returns:
        1.0 if compliant or not applicable, 0.0 otherwise
    """
    if output is None:
        return 0.0

    # If diff format wasn't requested, return 1.0
    output_format = output.get("output_format", "whole_file")
    if output_format != "diff":
        return 1.0

    # Check diff_compliant flag
    return 1.0 if output.get("diff_compliant", False) else 0.0


def extraneous_edit_rate(
    output: dict[str, Any] | None,
    expected: dict[str, Any],
    **kwargs: Any,
) -> float:
    """Calculate extraneous edit rate.

    Measures how often the model makes unnecessary edits.
    Lower is better (this is a "cost" metric).

    Args:
        output: Agent output with has_extraneous_edits flag
        expected: Expected output (not used)

    Returns:
        1.0 if extraneous edits present, 0.0 otherwise
    """
    if output is None:
        return 0.0

    return 1.0 if output.get("has_extraneous_edits", False) else 0.0


def _normalize_code(code: str) -> str:
    """Normalize code for comparison.

    - Strip whitespace
    - Remove comments
    - Normalize indentation
    """
    lines = []
    for line in code.strip().split("\n"):
        # Remove inline comments (but keep strings with #)
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return "\n".join(lines)
