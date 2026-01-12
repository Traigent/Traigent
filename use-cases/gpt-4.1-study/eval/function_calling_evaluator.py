"""Evaluator for GPT-4.1 Function Calling Study.

Metrics:
- tool_selection_accuracy: Whether correct tools were selected
- parameter_accuracy: Whether parameters were correctly specified
"""

from __future__ import annotations

from typing import Any


class FunctionCallingEvaluator:
    """Evaluator for function calling tasks."""

    def __call__(
        self,
        output: dict[str, Any] | None,
        expected: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, float]:
        """Evaluate function calling output.

        Args:
            output: Agent output with tool_calls and accuracy flags
            expected: Expected output with expected_tools and expected_parameters

        Returns:
            Dict with tool_selection_accuracy and parameter_accuracy scores
        """
        if output is None:
            return {"tool_selection_accuracy": 0.0, "parameter_accuracy": 0.0}

        tool_acc = tool_selection_accuracy(output, expected, **kwargs)
        param_acc = parameter_accuracy(output, expected, **kwargs)

        return {
            "tool_selection_accuracy": tool_acc,
            "parameter_accuracy": param_acc,
        }


def tool_selection_accuracy(
    output: dict[str, Any] | None,
    expected: dict[str, Any],
    **kwargs: Any,
) -> float:
    """Calculate tool selection accuracy.

    Checks if the correct tools were selected for the task.

    Args:
        output: Agent output with tool_calls and tool_selection_correct flag
        expected: Expected output with expected_tools

    Returns:
        1.0 if correct tools selected, 0.0 otherwise
    """
    if output is None:
        return 0.0

    # Check tool_selection_correct flag first
    if "tool_selection_correct" in output:
        return 1.0 if output["tool_selection_correct"] else 0.0

    # Fall back to comparing tool lists
    selected_tools = set(output.get("selected_tools", []))
    expected_tools = set(expected.get("expected_tools", []))

    if not expected_tools:
        return 1.0 if not selected_tools else 0.0

    # Calculate Jaccard similarity
    intersection = selected_tools & expected_tools
    union = selected_tools | expected_tools

    if not union:
        return 1.0

    return len(intersection) / len(union)


def parameter_accuracy(
    output: dict[str, Any] | None,
    expected: dict[str, Any],
    **kwargs: Any,
) -> float:
    """Calculate parameter accuracy.

    Checks if tool parameters were correctly specified.

    Args:
        output: Agent output with tool_calls and parameters_correct flag
        expected: Expected output with expected_parameters

    Returns:
        Score between 0.0 and 1.0 based on parameter correctness
    """
    if output is None:
        return 0.0

    # Check parameters_correct flag first
    if "parameters_correct" in output:
        return 1.0 if output["parameters_correct"] else 0.0

    # Fall back to comparing parameters
    tool_calls = output.get("tool_calls", [])
    expected_params = expected.get("expected_parameters", {})

    if not expected_params:
        return 1.0  # No expected parameters to check

    if not tool_calls:
        return 0.0

    total_params = 0
    correct_params = 0

    for tc in tool_calls:
        tool_name = tc.get("name", "")
        actual_args = tc.get("arguments", {})
        expected_args = expected_params.get(tool_name, {})

        for key, expected_value in expected_args.items():
            total_params += 1
            actual_value = actual_args.get(key)

            if _values_match(actual_value, expected_value):
                correct_params += 1

    if total_params == 0:
        return 1.0

    return correct_params / total_params


def _values_match(actual: Any, expected: Any) -> bool:
    """Check if two values match (with some flexibility).

    Args:
        actual: The actual parameter value
        expected: The expected parameter value

    Returns:
        True if values match
    """
    if actual == expected:
        return True

    # String comparison (case-insensitive)
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.lower().strip() == expected.lower().strip()

    # List comparison (order-independent for some cases)
    if isinstance(actual, list) and isinstance(expected, list):
        return set(actual) == set(expected)

    # Dict comparison
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual == expected

    return False


def tool_call_count_accuracy(
    output: dict[str, Any] | None,
    expected: dict[str, Any],
    **kwargs: Any,
) -> float:
    """Check if the correct number of tool calls were made.

    For multi-tool tasks, the model should make the right
    number of tool calls.

    Args:
        output: Agent output with tool_calls
        expected: Expected output with expected_tool_count

    Returns:
        1.0 if count matches, scaled score otherwise
    """
    if output is None:
        return 0.0

    actual_count = len(output.get("tool_calls", []))
    expected_count = expected.get(
        "expected_tool_count", len(expected.get("expected_tools", []))
    )

    if expected_count == 0:
        return 1.0 if actual_count == 0 else 0.0

    if actual_count == expected_count:
        return 1.0

    # Partial credit based on how close the count is
    diff = abs(actual_count - expected_count)
    return max(0.0, 1.0 - (diff / expected_count))


def multi_tool_orchestration_score(
    output: dict[str, Any] | None,
    expected: dict[str, Any],
    **kwargs: Any,
) -> float:
    """Calculate score for multi-tool orchestration.

    Checks if multiple tools were called correctly and
    in the right order (if order matters).

    Args:
        output: Agent output with tool_calls
        expected: Expected output with expected_tools and order_matters flag

    Returns:
        Score between 0.0 and 1.0
    """
    if output is None:
        return 0.0

    tool_calls = output.get("tool_calls", [])
    expected_tools = expected.get("expected_tools", [])

    if not expected_tools:
        return 1.0 if not tool_calls else 0.0

    selected_tools = [tc.get("name", "") for tc in tool_calls]

    # Check if all expected tools are present
    if set(selected_tools) != set(expected_tools):
        # Partial credit for having some correct tools
        correct = len(set(selected_tools) & set(expected_tools))
        return correct / len(expected_tools)

    # Check order if it matters
    order_matters = expected.get("order_matters", False)
    if order_matters:
        # Tools must be in correct order
        if selected_tools == expected_tools:
            return 1.0
        else:
            return 0.8  # Correct tools but wrong order

    return 1.0
