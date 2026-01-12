"""Evaluator for GPT-4.1 Instruction Following Study.

Metrics:
- format_compliance: Whether output format matches requirements
- instruction_adherence: Whether all instructions were followed
"""

from __future__ import annotations

from typing import Any


class InstructionFollowingEvaluator:
    """Evaluator for instruction following tasks."""

    def __call__(
        self,
        output: dict[str, Any] | None,
        expected: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, float]:
        """Evaluate instruction following output.

        Args:
            output: Agent output with compliance flags
            expected: Expected output

        Returns:
            Dict with format_compliance and instruction_adherence scores
        """
        if output is None:
            return {"format_compliance": 0.0, "instruction_adherence": 0.0}

        fmt_compliance = format_compliance(output, expected, **kwargs)
        adherence = instruction_adherence(output, expected, **kwargs)

        return {
            "format_compliance": fmt_compliance,
            "instruction_adherence": adherence,
        }


def format_compliance(
    output: dict[str, Any] | None,
    expected: dict[str, Any],
    **kwargs: Any,
) -> float:
    """Calculate format compliance score.

    Checks if the output format matches the expected format
    (JSON, XML, YAML, Markdown).

    Args:
        output: Agent output with format_compliant flag
        expected: Expected output with expected_format

    Returns:
        1.0 if compliant, 0.0 otherwise
    """
    if output is None:
        return 0.0

    return 1.0 if output.get("format_compliant", False) else 0.0


def instruction_adherence(
    output: dict[str, Any] | None,
    expected: dict[str, Any],
    **kwargs: Any,
) -> float:
    """Calculate overall instruction adherence score.

    Combines multiple compliance checks:
    - Negative instruction compliance (didn't include forbidden terms)
    - Content requirements (included required terms)
    - Order compliance (followed ordered instructions)

    Args:
        output: Agent output with compliance flags
        expected: Expected output

    Returns:
        Score between 0.0 and 1.0 based on compliance
    """
    if output is None:
        return 0.0

    # Check overall compliance flag first
    if output.get("overall_compliance", False):
        return 1.0

    # Calculate weighted score from individual compliance flags
    scores = []
    weights = []

    # Negative instructions (high weight - these are often "hard" per OpenAI)
    if "negative_instruction_followed" in output:
        scores.append(1.0 if output["negative_instruction_followed"] else 0.0)
        weights.append(0.35)

    # Content requirements
    if "content_requirements_met" in output:
        scores.append(1.0 if output["content_requirements_met"] else 0.0)
        weights.append(0.30)

    # Order compliance
    if "order_followed" in output:
        scores.append(1.0 if output["order_followed"] else 0.0)
        weights.append(0.20)

    # Format compliance (also contributes to adherence)
    if "format_compliant" in output:
        scores.append(1.0 if output["format_compliant"] else 0.0)
        weights.append(0.15)

    if not scores:
        return 0.0

    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0

    weighted_score = sum(s * w for s, w in zip(scores, weights, strict=True)) / total_weight
    return weighted_score


def negative_instruction_compliance(
    output: dict[str, Any] | None,
    expected: dict[str, Any],
    **kwargs: Any,
) -> float:
    """Check compliance with negative instructions.

    Negative instructions are "don't do X" type instructions.

    Args:
        output: Agent output with negative_instruction_followed flag
        expected: Expected output

    Returns:
        1.0 if compliant, 0.0 otherwise
    """
    if output is None:
        return 0.0

    return 1.0 if output.get("negative_instruction_followed", True) else 0.0


def content_requirement_compliance(
    output: dict[str, Any] | None,
    expected: dict[str, Any],
    **kwargs: Any,
) -> float:
    """Check compliance with content requirements.

    Content requirements are "must include X" type instructions.

    Args:
        output: Agent output with content_requirements_met flag
        expected: Expected output

    Returns:
        1.0 if compliant, 0.0 otherwise
    """
    if output is None:
        return 0.0

    return 1.0 if output.get("content_requirements_met", True) else 0.0
