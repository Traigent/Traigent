"""
Function calling evaluation system.

This module provides tools and evaluation functions for testing function calling
reliability, including tool selection accuracy and parameter validation.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

from function_config import FunctionConfig, validate_function_call


@dataclass
class FunctionCallResult:
    """Result of a function call attempt."""

    success: bool
    function_name: str
    arguments: dict[str, Any]
    execution_time_ms: float
    error_message: str | None = None
    retry_count: int = 0


@dataclass
class ToolExecutionResult:
    """Complete tool execution result with metrics."""

    tool_selection_correct: bool
    parameters_valid: bool
    execution_successful: bool
    latency_ms: float
    retry_count: int
    error_details: dict[str, Any] | None = None


def create_mock_tools() -> list[dict[str, Any]]:
    """Create a set of mock tools for testing."""

    return [
        {
            "name": "calculator",
            "description": "Performs basic mathematical calculations",
            "category": "math",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The mathematical operation to perform",
                    },
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["operation", "a", "b"],
            },
            "examples": [
                {"operation": "add", "a": 5, "b": 3},
                {"operation": "multiply", "a": 10, "b": 2.5},
            ],
        },
        {
            "name": "string_formatter",
            "description": "Formats and manipulates strings",
            "category": "text",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to format"},
                    "format_type": {
                        "type": "string",
                        "enum": ["uppercase", "lowercase", "title_case", "reverse"],
                        "description": "Type of formatting to apply",
                    },
                    "prefix": {
                        "type": "string",
                        "description": "Optional prefix to add",
                        "default": "",
                    },
                },
                "required": ["text", "format_type"],
            },
            "examples": [
                {"text": "hello world", "format_type": "title_case"},
                {"text": "test", "format_type": "uppercase", "prefix": ">> "},
            ],
        },
        {
            "name": "date_calculator",
            "description": "Calculates dates and time differences",
            "category": "datetime",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "format": "date",
                        "description": "Start date in YYYY-MM-DD format",
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["add_days", "subtract_days", "days_between"],
                        "description": "Date operation to perform",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days (for add/subtract operations)",
                    },
                    "end_date": {
                        "type": "string",
                        "format": "date",
                        "description": "End date (for days_between operation)",
                    },
                },
                "required": ["start_date", "operation"],
            },
            "examples": [
                {"start_date": "2024-01-01", "operation": "add_days", "days": 30}
            ],
        },
        {
            "name": "data_filter",
            "description": "Filters and queries data arrays",
            "category": "data",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Array of objects to filter",
                    },
                    "field": {
                        "type": "string",
                        "description": "Field name to filter by",
                    },
                    "operator": {
                        "type": "string",
                        "enum": ["equals", "greater_than", "less_than", "contains"],
                        "description": "Comparison operator",
                    },
                    "value": {
                        "type": ["string", "number", "boolean"],
                        "description": "Value to compare against",
                    },
                },
                "required": ["data", "field", "operator", "value"],
            },
            "examples": [
                {
                    "data": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}],
                    "field": "age",
                    "operator": "greater_than",
                    "value": 27,
                }
            ],
        },
        {
            "name": "weather_api",
            "description": "Gets weather information for a location",
            "category": "api",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or coordinates",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit", "kelvin"],
                        "description": "Temperature units",
                        "default": "celsius",
                    },
                    "include_forecast": {
                        "type": "boolean",
                        "description": "Include 5-day forecast",
                        "default": False,
                    },
                },
                "required": ["location"],
            },
            "examples": [
                {"location": "London", "units": "celsius"},
                {"location": "New York", "include_forecast": True},
            ],
        },
    ]


def simulate_llm_function_call(
    task_description: str, available_tools: list[dict[str, Any]], config: FunctionConfig
) -> dict[str, Any]:
    """Simulate LLM function calling behavior."""

    # This simulates what an LLM would return for function calling
    # In a real implementation, this would call the actual LLM

    start_time = time.time()

    # Simulate processing time
    processing_delay = random.uniform(0.05, 0.2)
    time.sleep(processing_delay)

    # Simple logic to select appropriate tool based on task
    selected_tool = None

    if "calculate" in task_description.lower() or "math" in task_description.lower():
        selected_tool = next(
            (t for t in available_tools if t["name"] == "calculator"), None
        )
    elif "format" in task_description.lower() or "text" in task_description.lower():
        selected_tool = next(
            (t for t in available_tools if t["name"] == "string_formatter"), None
        )
    elif "date" in task_description.lower() or "time" in task_description.lower():
        selected_tool = next(
            (t for t in available_tools if t["name"] == "date_calculator"), None
        )
    elif "filter" in task_description.lower() or "data" in task_description.lower():
        selected_tool = next(
            (t for t in available_tools if t["name"] == "data_filter"), None
        )
    elif "weather" in task_description.lower():
        selected_tool = next(
            (t for t in available_tools if t["name"] == "weather_api"), None
        )
    else:
        # Default to first tool or random selection
        selected_tool = available_tools[0] if available_tools else None

    if not selected_tool:
        return {
            "error": "No suitable tool found",
            "execution_time_ms": (time.time() - start_time) * 1000,
        }

    # Generate parameters based on task and tool schema
    arguments = generate_mock_arguments(task_description, selected_tool, config)

    return {
        "function_call": {"name": selected_tool["name"], "arguments": arguments},
        "execution_time_ms": (time.time() - start_time) * 1000,
    }


def _generate_calculator_args(description: str) -> dict[str, Any]:
    lowered = description.lower()
    if "add" in lowered or "plus" in lowered:
        return {"operation": "add", "a": 10, "b": 5}
    if "multiply" in lowered:
        return {"operation": "multiply", "a": 8, "b": 3}
    return {"operation": "add", "a": 15, "b": 7}


def _string_formatter_args(description: str) -> dict[str, Any]:
    lowered = description.lower()
    format_type = "uppercase" if "upper" in lowered else "title_case"
    return {"text": "example text", "format_type": format_type}


def _date_calculator_args() -> dict[str, Any]:
    return {"start_date": "2024-01-01", "operation": "add_days", "days": 30}


def _data_filter_args() -> dict[str, Any]:
    return {
        "data": [{"name": "Alice", "score": 85}, {"name": "Bob", "score": 92}],
        "field": "score",
        "operator": "greater_than",
        "value": 90,
    }


def _weather_api_args() -> dict[str, Any]:
    return {"location": "London", "units": "celsius"}


def _inject_parameter_errors(
    arguments: dict[str, Any],
    parameters: dict[str, Any],
    required: list[str],
    error_rate: float,
) -> dict[str, Any]:
    if random.random() >= error_rate:
        return arguments
    mutated = arguments.copy()
    if required and random.random() < 0.5:
        param_to_remove = random.choice(required)
        mutated.pop(param_to_remove, None)
    else:
        param_name = random.choice(list(parameters.keys()))
        mutated[param_name] = "invalid_value"
    return mutated


def generate_mock_arguments(
    task_description: str, tool: dict[str, Any], config: FunctionConfig
) -> dict[str, Any]:
    """Generate mock arguments for a tool based on task description."""

    tool_name = tool["name"]
    parameters = tool["parameters"]["properties"]
    required = tool["parameters"].get("required", [])

    arguments = {}

    generators = {
        "calculator": lambda: _generate_calculator_args(task_description),
        "string_formatter": lambda: _string_formatter_args(task_description),
        "date_calculator": _date_calculator_args,
        "data_filter": _data_filter_args,
        "weather_api": _weather_api_args,
    }
    generator = generators.get(tool_name, lambda: {})
    arguments = generator()

    if config.temperature > 0.1:
        error_rate = min(0.3, config.temperature)
        arguments = _inject_parameter_errors(
            arguments, parameters, required, error_rate
        )

    return arguments


def execute_function_call(
    function_call: dict[str, Any], tool_schema: dict[str, Any], config: FunctionConfig
) -> FunctionCallResult:
    """Execute a function call and return results."""

    start_time = time.time()

    function_name = function_call.get("name", "")
    arguments = function_call.get("arguments", {})

    # Validate the function call
    validation_result = validate_function_call(
        {"name": function_name, "arguments": arguments}, tool_schema
    )

    if not validation_result["valid"]:
        return FunctionCallResult(
            success=False,
            function_name=function_name,
            arguments=arguments,
            execution_time_ms=(time.time() - start_time) * 1000,
            error_message="; ".join(validation_result["errors"]),
        )

    # Simulate actual execution
    execution_delay = random.uniform(0.01, 0.1)
    time.sleep(execution_delay)

    # Simulate occasional execution failures
    failure_rate = 0.05  # 5% failure rate
    if random.random() < failure_rate:
        return FunctionCallResult(
            success=False,
            function_name=function_name,
            arguments=arguments,
            execution_time_ms=(time.time() - start_time) * 1000,
            error_message="Simulated execution error",
        )

    return FunctionCallResult(
        success=True,
        function_name=function_name,
        arguments=arguments,
        execution_time_ms=(time.time() - start_time) * 1000,
    )


def evaluate_tool_selection(
    predicted_tool: str, expected_tool: str, available_tools: list[str]
) -> float:
    """Evaluate tool selection accuracy."""

    if predicted_tool == expected_tool:
        return 1.0

    # Partial credit for selecting a tool from the same category
    # This would require more sophisticated logic in a real implementation
    return 0.0


def evaluate_function_calling_task(
    task: dict[str, Any], config: FunctionConfig
) -> ToolExecutionResult:
    """Evaluate a complete function calling task."""

    task_description = task["description"]
    expected_tool = task["expected_tool"]
    available_tools = task["available_tools"]

    start_time = time.time()

    # Get the expected tool schema
    expected_tool_schema = next(
        (tool for tool in available_tools if tool["name"] == expected_tool), None
    )

    if not expected_tool_schema:
        return ToolExecutionResult(
            tool_selection_correct=False,
            parameters_valid=False,
            execution_successful=False,
            latency_ms=(time.time() - start_time) * 1000,
            retry_count=0,
            error_details={"error": "Expected tool not found"},
        )

    retry_count = 0
    last_error = None

    for _attempt in range(config.max_retries + 1):
        # Simulate LLM function call
        llm_response = simulate_llm_function_call(
            task_description, available_tools, config
        )

        if "error" in llm_response:
            last_error = llm_response["error"]
            retry_count += 1
            continue

        function_call = llm_response["function_call"]

        # Evaluate tool selection
        tool_selection_correct = function_call["name"] == expected_tool

        # Execute function call
        execution_result = execute_function_call(
            function_call, expected_tool_schema, config
        )

        # Check if successful
        if execution_result.success:
            return ToolExecutionResult(
                tool_selection_correct=tool_selection_correct,
                parameters_valid=True,
                execution_successful=True,
                latency_ms=(time.time() - start_time) * 1000,
                retry_count=retry_count,
            )
        else:
            last_error = execution_result.error_message
            retry_count += 1

    # All retries failed
    return ToolExecutionResult(
        tool_selection_correct=False,
        parameters_valid=False,
        execution_successful=False,
        latency_ms=(time.time() - start_time) * 1000,
        retry_count=retry_count,
        error_details={"error": last_error},
    )


def calculate_reliability_metrics(
    results: list[ToolExecutionResult],
) -> dict[str, float]:
    """Calculate reliability metrics from evaluation results."""

    if not results:
        return {
            "tool_selection_accuracy": 0.0,
            "parameter_validity_rate": 0.0,
            "execution_success_rate": 0.0,
            "unnecessary_retry_rate": 1.0,
            "avg_latency_ms": 1000.0,
        }

    # Calculate metrics
    tool_selection_accuracy = sum(r.tool_selection_correct for r in results) / len(
        results
    )
    parameter_validity_rate = sum(r.parameters_valid for r in results) / len(results)
    execution_success_rate = sum(r.execution_successful for r in results) / len(results)
    avg_retry_count = sum(r.retry_count for r in results) / len(results)
    unnecessary_retry_rate = min(1.0, avg_retry_count / 3.0)  # Normalize to 0-1
    avg_latency_ms = sum(r.latency_ms for r in results) / len(results)

    # Calculate overall reliability score
    reliability_score = (
        tool_selection_accuracy * 0.4
        + parameter_validity_rate * 0.3
        + execution_success_rate * 0.3
    )

    return {
        "tool_selection_accuracy": tool_selection_accuracy,
        "parameter_validity_rate": parameter_validity_rate,
        "execution_success_rate": execution_success_rate,
        "unnecessary_retry_rate": unnecessary_retry_rate,
        "avg_latency_ms": avg_latency_ms,
        "reliability_score": reliability_score,
        # Additional metrics
        "total_tasks": len(results),
        "successful_tasks": sum(r.execution_successful for r in results),
        "avg_retry_count": avg_retry_count,
    }
