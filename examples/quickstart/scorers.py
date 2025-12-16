"""Shared scoring functions for TraiGent quickstart examples."""


def custom_accuracy_scorer(output: str, expected: str, llm_metrics: dict = None) -> float:
    """Custom scoring function that checks if expected answer is in output.

    Args:
        output: The LLM's response
        expected: The expected answer from the dataset
        llm_metrics: Additional metrics from the LLM call (tokens, latency, etc.)

    Returns:
        Score between 0 and 1
    """
    if not output or not expected:
        return 0.0
    # Case-insensitive containment check
    return 1.0 if expected.lower() in output.lower() else 0.0
