"""Evaluator for GPT-4.1 Long Context Study.

Metrics:
- retrieval_accuracy: Whether correct information was retrieved
- multi_hop_accuracy: Whether multi-hop reasoning was correct
"""

from __future__ import annotations

from typing import Any


class LongContextEvaluator:
    """Evaluator for long context tasks."""

    def __call__(
        self,
        output: dict[str, Any] | None,
        expected: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, float]:
        """Evaluate long context output.

        Args:
            output: Agent output with answer and accuracy flags
            expected: Expected output with expected_answer

        Returns:
            Dict with retrieval_accuracy and multi_hop_accuracy scores
        """
        if output is None:
            return {"retrieval_accuracy": 0.0, "multi_hop_accuracy": 0.0}

        retrieval = retrieval_accuracy(output, expected, **kwargs)
        multi_hop = multi_hop_accuracy(output, expected, **kwargs)

        return {
            "retrieval_accuracy": retrieval,
            "multi_hop_accuracy": multi_hop,
        }


def retrieval_accuracy(
    output: dict[str, Any] | None,
    expected: dict[str, Any],
    **kwargs: Any,
) -> float:
    """Calculate retrieval accuracy.

    Checks if the correct information was retrieved from the context.

    Args:
        output: Agent output with answer and retrieval_correct flag
        expected: Expected output with expected_answer

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    if output is None:
        return 0.0

    # Check retrieval_correct flag first
    if "retrieval_correct" in output:
        return 1.0 if output["retrieval_correct"] else 0.0

    # Fall back to answer comparison
    answer = output.get("answer", "").lower().strip()
    expected_answer = expected.get("expected_answer", "").lower().strip()

    if not answer or not expected_answer:
        return 0.0

    # Check for exact or substring match
    if expected_answer in answer or answer == expected_answer:
        return 1.0

    return 0.0


def multi_hop_accuracy(
    output: dict[str, Any] | None,
    expected: dict[str, Any],
    **kwargs: Any,
) -> float:
    """Calculate multi-hop reasoning accuracy.

    Checks if the model correctly reasoned across multiple
    positions in the context (Graphwalks-style).

    Args:
        output: Agent output with reasoning_correct flag
        expected: Expected output

    Returns:
        1.0 if reasoning correct, 0.0 otherwise
    """
    if output is None:
        return 0.0

    # Check reasoning_correct flag
    if "reasoning_correct" in output:
        return 1.0 if output["reasoning_correct"] else 0.0

    # For non-multi-hop tasks, return retrieval accuracy
    return retrieval_accuracy(output, expected, **kwargs)


def needle_position_accuracy(
    output: dict[str, Any] | None,
    expected: dict[str, Any],
    **kwargs: Any,
) -> float:
    """Calculate accuracy for specific needle position retrieval.

    For MRCR-style tasks where the model must identify
    a specific instance among multiple similar items.

    Args:
        output: Agent output with answer
        expected: Expected output with expected_answer and needle_position

    Returns:
        1.0 if correct needle identified, 0.0 otherwise
    """
    if output is None:
        return 0.0

    # This metric combines retrieval with disambiguation
    answer = output.get("answer", "").lower().strip()
    expected_answer = expected.get("expected_answer", "").lower().strip()

    if not answer or not expected_answer:
        return 0.0

    # Must match exactly for disambiguation tasks
    if expected_answer in answer:
        # Check if no other needle answers are present
        input_data = kwargs.get("input_data", {})
        other_needles = input_data.get("distractor_answers", [])

        for distractor in other_needles:
            if distractor.lower() in answer:
                return 0.5  # Partial credit if answer contains distractors

        return 1.0

    return 0.0


def context_utilization_score(
    output: dict[str, Any] | None,
    expected: dict[str, Any],
    **kwargs: Any,
) -> float:
    """Calculate how well the model utilized the context.

    For tasks where the answer quality depends on
    comprehensively using the provided context.

    Args:
        output: Agent output
        expected: Expected output

    Returns:
        Score between 0.0 and 1.0
    """
    if output is None:
        return 0.0

    # Base score on retrieval accuracy
    base_score = retrieval_accuracy(output, expected, **kwargs)

    # Bonus for longer, more comprehensive answers (when appropriate)
    answer = output.get("answer", "")
    expected_answer = expected.get("expected_answer", "")

    if base_score > 0 and len(answer) > len(expected_answer) * 1.5:
        # Model provided additional context - slight bonus
        return min(1.0, base_score + 0.1)

    return base_score
