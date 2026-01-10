#!/usr/bin/env python3
"""Evaluator for Code Review Agent.

Measures detection accuracy using precision, recall, and F1 score
by comparing predicted issue types against ground truth labels.

The evaluator is resilient to various input formats:
- Parsed list of dicts from the agent
- Raw JSON string from LLM response
- Empty responses (parse failures)
"""

import json
import re
from typing import Any


# Valid issue types for validation
VALID_ISSUE_TYPES = frozenset([
    "MISSING_DOCS",
    "IMPLICIT_ASSUMPTION",
    "SIDE_EFFECT",
    "COMPLEXITY",
    "BROAD_EXCEPTION",
    "PRINCIPLE_VIOLATION",
    "TODO_KNOWN_ISSUE",
    "TYPE_HANDLING",
    "API_DESIGN",
    "THREADING_ISSUE",
])


class CodeReviewEvaluator:
    """Evaluator for code review agent issue detection.

    Compares predicted issues against ground truth using type-level matching.
    This approach is robust since issue descriptions may vary in wording
    while the issue type (category) should match exactly.

    Metrics returned:
    - detection_precision: Of issues predicted, how many are correct types?
    - detection_recall: Of actual issues, how many types were detected?
    - detection_f1: Harmonic mean of precision and recall
    - overall: Same as detection_f1 (for Traigent compatibility)
    """

    def __call__(
        self,
        prediction: list[dict[str, str]] | str | None,
        expected: dict[str, Any] | None,
        input_data: dict[str, Any],
    ) -> dict[str, float]:
        """Evaluate predicted issues against ground truth.

        Args:
            prediction: List of {"issue_type": "...", "description": "..."}
                or raw JSON string from LLM, or None on failure
            expected: {"issues": [{"issue_type": "...", "description": "..."}]}
            input_data: {"function_code": "...", "function_name": "...", "source_file": "..."}

        Returns:
            Dictionary with precision, recall, f1 scores
        """
        # Parse prediction to list of issue dicts
        issues = self._parse_prediction(prediction)

        # Extract issue types
        predicted_types = self._extract_valid_types(issues)
        expected_issues = expected.get("issues", []) if expected else []
        expected_types = self._extract_valid_types(expected_issues)

        # Calculate metrics
        return self._calculate_metrics(predicted_types, expected_types)

    def _parse_prediction(
        self, prediction: list[dict[str, str]] | str | None
    ) -> list[dict[str, str]]:
        """Parse prediction to list of issue dicts.

        Handles:
        - Already parsed list of dicts
        - Raw JSON string from LLM
        - None or empty values
        """
        if prediction is None:
            return []

        if isinstance(prediction, list):
            return prediction

        if isinstance(prediction, str):
            prediction = prediction.strip()
            if not prediction:
                return []

            # Try direct JSON parse
            try:
                parsed = json.loads(prediction)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

            # Try to extract JSON array from response
            # (handles cases where LLM wraps JSON in markdown or adds explanation)
            match = re.search(r"\[.*\]", prediction, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass

            return []  # Parse failed

        return []

    def _extract_valid_types(
        self, issues: list[dict[str, str]]
    ) -> set[str]:
        """Extract valid issue types from a list of issue dicts."""
        types = set()
        for issue in issues:
            if isinstance(issue, dict):
                issue_type = issue.get("issue_type", "")
                if issue_type in VALID_ISSUE_TYPES:
                    types.add(issue_type)
        return types

    def _calculate_metrics(
        self,
        predicted_types: set[str],
        expected_types: set[str],
    ) -> dict[str, float]:
        """Calculate precision, recall, and F1 from type sets."""
        # Handle edge cases
        if not predicted_types and not expected_types:
            # Both empty - perfect match (no issues to find, none found)
            return {
                "detection_precision": 1.0,
                "detection_recall": 1.0,
                "detection_f1": 1.0,
                "overall": 1.0,
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
            }

        # Calculate set operations
        true_positives = predicted_types & expected_types
        false_positives = predicted_types - expected_types
        false_negatives = expected_types - predicted_types

        tp = len(true_positives)
        fp = len(false_positives)
        fn = len(false_negatives)

        # Calculate metrics with zero-division protection
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "detection_precision": precision,
            "detection_recall": recall,
            "detection_f1": f1,
            "overall": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }


# Standalone metric function for use with metric_functions parameter
def detection_f1_metric(
    output: list[dict[str, str]] | str | None,
    expected: dict[str, Any],
    **kwargs: Any,
) -> float:
    """Calculate F1 score for issue type detection.

    This function is designed for use with Traigent's metric_functions parameter
    to properly wire the detection_f1 objective.

    Args:
        output: Agent output - list of issues or raw string
        expected: Expected output with "issues" key
        **kwargs: Additional context (input_data, etc.)

    Returns:
        F1 score as float between 0.0 and 1.0
    """
    evaluator = CodeReviewEvaluator()
    input_data = kwargs.get("input_data", {})
    result = evaluator(output, expected, input_data)
    return result["detection_f1"]


def demo_evaluator() -> None:
    """Demonstrate the evaluator with sample inputs."""
    evaluator = CodeReviewEvaluator()

    print("=" * 60)
    print("Code Review Evaluator Demo")
    print("=" * 60)

    # Test case 1: Perfect match
    print("\n[Test 1] Perfect match - 2 issues correctly identified")
    prediction = [
        {"issue_type": "MISSING_DOCS", "description": "No docstring"},
        {"issue_type": "BROAD_EXCEPTION", "description": "except Exception"},
    ]
    expected = {
        "issues": [
            {"issue_type": "MISSING_DOCS", "description": "Function lacks docstring"},
            {"issue_type": "BROAD_EXCEPTION", "description": "Too broad exception"},
        ]
    }
    result = evaluator(prediction, expected, {})
    print(f"  Precision: {result['detection_precision']:.2f}")
    print(f"  Recall: {result['detection_recall']:.2f}")
    print(f"  F1: {result['detection_f1']:.2f}")

    # Test case 2: Partial match with false positive
    print("\n[Test 2] Partial match - 1 correct, 1 false positive")
    prediction = [
        {"issue_type": "MISSING_DOCS", "description": "No docstring"},
        {"issue_type": "THREADING_ISSUE", "description": "Race condition"},
    ]
    expected = {
        "issues": [
            {"issue_type": "MISSING_DOCS", "description": "Missing docstring"},
        ]
    }
    result = evaluator(prediction, expected, {})
    print(f"  Precision: {result['detection_precision']:.2f} (1/2 correct)")
    print(f"  Recall: {result['detection_recall']:.2f} (found all expected)")
    print(f"  F1: {result['detection_f1']:.2f}")

    # Test case 3: Missed issues (false negatives)
    print("\n[Test 3] Missed issues - only found 1 of 3")
    prediction = [
        {"issue_type": "COMPLEXITY", "description": "Deep nesting"},
    ]
    expected = {
        "issues": [
            {"issue_type": "COMPLEXITY", "description": "Deeply nested"},
            {"issue_type": "MISSING_DOCS", "description": "No docstring"},
            {"issue_type": "SIDE_EFFECT", "description": "Global mutation"},
        ]
    }
    result = evaluator(prediction, expected, {})
    print(f"  Precision: {result['detection_precision']:.2f} (1/1 correct)")
    print(f"  Recall: {result['detection_recall']:.2f} (1/3 found)")
    print(f"  F1: {result['detection_f1']:.2f}")

    # Test case 4: Raw JSON string input
    print("\n[Test 4] Raw JSON string from LLM")
    prediction_str = '[{"issue_type": "IMPLICIT_ASSUMPTION", "description": "No null check"}]'
    expected = {"issues": [{"issue_type": "IMPLICIT_ASSUMPTION", "description": "Missing validation"}]}
    result = evaluator(prediction_str, expected, {})
    print(f"  Parsed successfully: F1 = {result['detection_f1']:.2f}")

    # Test case 5: No issues (clean code)
    print("\n[Test 5] Clean code - no issues expected or found")
    prediction = []
    expected = {"issues": []}
    result = evaluator(prediction, expected, {})
    print(f"  F1: {result['detection_f1']:.2f} (perfect for clean code)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_evaluator()
