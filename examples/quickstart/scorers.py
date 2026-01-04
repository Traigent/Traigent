"""Shared scoring functions for TraiGent quickstart examples."""

import re


def custom_accuracy_scorer(output: str, expected: str, llm_metrics: dict = None) -> float:
    """Custom scoring function that checks if expected answer is in output.

    For single-letter answers (A-J), looks for patterns like "A.", "A)", "(A)",
    "Answer: A", "answer is A" to avoid false positives from letters in words.

    For other answers, uses case-insensitive containment check.

    Args:
        output: The LLM's response
        expected: The expected answer from the dataset
        llm_metrics: Additional metrics from the LLM call (tokens, latency, etc.)

    Returns:
        Score between 0 and 1
    """
    if not output or not expected:
        return 0.0

    expected = expected.strip()
    output_lower = output.lower()
    expected_lower = expected.lower()

    # For single-letter answers (multiple choice), use strict matching
    if len(expected) == 1 and expected.upper() in "ABCDEFGHIJ":
        letter = expected.upper()
        # Look for patterns that clearly indicate the answer choice
        patterns = [
            rf"\b{letter}\.",  # "A."
            rf"\b{letter}\)",  # "A)"
            rf"\({letter}\)",  # "(A)"
            rf"answer[:\s]+{letter}\b",  # "Answer: A" or "answer A"
            rf"answer is[:\s]+{letter}\b",  # "answer is A"
            rf"correct[:\s]+{letter}\b",  # "correct: A"
            rf"^{letter}\b",  # Starts with "A" as a word
            rf"\b{letter}\s*$",  # Ends with "A" as a word
        ]
        for pattern in patterns:
            if re.search(pattern, output, re.IGNORECASE):
                return 1.0
        return 0.0

    # For yes/no/true/false, use word boundary matching
    if expected_lower in ["yes", "no", "true", "false"]:
        # Match as whole word, not substring
        pattern = rf"\b{expected_lower}\b"
        return 1.0 if re.search(pattern, output_lower) else 0.0

    # For other answers, use simple containment
    return 1.0 if expected_lower in output_lower else 0.0
