"""
Dataset generation for function calling reliability testing.

This module creates diverse function calling scenarios across different tool categories
and complexity levels to test tool selection and parameter validation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any

from evaluator import create_mock_tools


class TaskComplexity(Enum):
    """Task complexity levels."""

    SIMPLE = "simple"  # Single tool, obvious mapping
    MODERATE = "moderate"  # Tool selection required
    COMPLEX = "complex"  # Sequential tools or conditional logic
    AMBIGUOUS = "ambiguous"  # Multiple valid approaches


class TaskCategory(Enum):
    """Task categories matching tool types."""

    MATHEMATICAL = "mathematical"
    TEXT_PROCESSING = "text_processing"
    DATE_TIME = "date_time"
    DATA_MANIPULATION = "data_manipulation"
    API_CALLS = "api_calls"


@dataclass
class FunctionCallingTask:
    """A task requiring function calling."""

    id: str
    description: str
    expected_tool: str
    expected_arguments: dict[str, Any]
    task_type: TaskCategory
    complexity: TaskComplexity
    available_tools: list[dict[str, Any]]
    success_criteria: dict[str, Any]
    edge_case: bool = False


def generate_mathematical_tasks() -> list[FunctionCallingTask]:
    """Generate mathematical calculation tasks."""

    tasks = [
        # Simple calculations
        FunctionCallingTask(
            id="math_001",
            description="Calculate the sum of 45 and 23",
            expected_tool="calculator",
            expected_arguments={"operation": "add", "a": 45, "b": 23},
            task_type=TaskCategory.MATHEMATICAL,
            complexity=TaskComplexity.SIMPLE,
            available_tools=create_mock_tools(),
            success_criteria={"exact_match": True},
        ),
        FunctionCallingTask(
            id="math_002",
            description="What is 12 multiplied by 3.5?",
            expected_tool="calculator",
            expected_arguments={"operation": "multiply", "a": 12, "b": 3.5},
            task_type=TaskCategory.MATHEMATICAL,
            complexity=TaskComplexity.SIMPLE,
            available_tools=create_mock_tools(),
            success_criteria={"exact_match": True},
        ),
        # More complex mathematical tasks
        FunctionCallingTask(
            id="math_003",
            description="I need to divide my budget of 500 dollars among 8 people equally",
            expected_tool="calculator",
            expected_arguments={"operation": "divide", "a": 500, "b": 8},
            task_type=TaskCategory.MATHEMATICAL,
            complexity=TaskComplexity.MODERATE,
            available_tools=create_mock_tools(),
            success_criteria={"operation_correct": True},
        ),
        # Edge case - zero division
        FunctionCallingTask(
            id="math_004",
            description="Calculate 10 divided by 0",
            expected_tool="calculator",
            expected_arguments={"operation": "divide", "a": 10, "b": 0},
            task_type=TaskCategory.MATHEMATICAL,
            complexity=TaskComplexity.SIMPLE,
            available_tools=create_mock_tools(),
            success_criteria={"handles_error": True},
            edge_case=True,
        ),
    ]

    return tasks


def generate_text_processing_tasks() -> list[FunctionCallingTask]:
    """Generate text processing tasks."""

    tasks = [
        # Simple formatting
        FunctionCallingTask(
            id="text_001",
            description="Convert 'hello world' to title case",
            expected_tool="string_formatter",
            expected_arguments={"text": "hello world", "format_type": "title_case"},
            task_type=TaskCategory.TEXT_PROCESSING,
            complexity=TaskComplexity.SIMPLE,
            available_tools=create_mock_tools(),
            success_criteria={"exact_match": True},
        ),
        FunctionCallingTask(
            id="text_002",
            description="Make this text uppercase: programming is fun",
            expected_tool="string_formatter",
            expected_arguments={
                "text": "programming is fun",
                "format_type": "uppercase",
            },
            task_type=TaskCategory.TEXT_PROCESSING,
            complexity=TaskComplexity.SIMPLE,
            available_tools=create_mock_tools(),
            success_criteria={"operation_correct": True},
        ),
        # With prefix
        FunctionCallingTask(
            id="text_003",
            description="Format 'important message' as uppercase and add '>>> ' as prefix",
            expected_tool="string_formatter",
            expected_arguments={
                "text": "important message",
                "format_type": "uppercase",
                "prefix": ">>> ",
            },
            task_type=TaskCategory.TEXT_PROCESSING,
            complexity=TaskComplexity.MODERATE,
            available_tools=create_mock_tools(),
            success_criteria={"all_parameters": True},
        ),
        # Ambiguous task
        FunctionCallingTask(
            id="text_004",
            description="Process this text: 'Hello World'",
            expected_tool="string_formatter",
            expected_arguments={"text": "Hello World", "format_type": "lowercase"},
            task_type=TaskCategory.TEXT_PROCESSING,
            complexity=TaskComplexity.AMBIGUOUS,
            available_tools=create_mock_tools(),
            success_criteria={"tool_selected": True},
        ),
    ]

    return tasks


def generate_datetime_tasks() -> list[FunctionCallingTask]:
    """Generate date and time calculation tasks."""

    tasks = [
        # Simple date addition
        FunctionCallingTask(
            id="date_001",
            description="Add 30 days to January 15, 2024",
            expected_tool="date_calculator",
            expected_arguments={
                "start_date": "2024-01-15",
                "operation": "add_days",
                "days": 30,
            },
            task_type=TaskCategory.DATE_TIME,
            complexity=TaskComplexity.SIMPLE,
            available_tools=create_mock_tools(),
            success_criteria={"exact_match": True},
        ),
        # Date difference
        FunctionCallingTask(
            id="date_002",
            description="How many days between March 1, 2024 and March 15, 2024?",
            expected_tool="date_calculator",
            expected_arguments={
                "start_date": "2024-03-01",
                "operation": "days_between",
                "end_date": "2024-03-15",
            },
            task_type=TaskCategory.DATE_TIME,
            complexity=TaskComplexity.MODERATE,
            available_tools=create_mock_tools(),
            success_criteria={"operation_correct": True},
        ),
        # Vacation planning
        FunctionCallingTask(
            id="date_003",
            description="I'm planning a 2-week vacation starting December 20, 2024. When will it end?",
            expected_tool="date_calculator",
            expected_arguments={
                "start_date": "2024-12-20",
                "operation": "add_days",
                "days": 14,
            },
            task_type=TaskCategory.DATE_TIME,
            complexity=TaskComplexity.MODERATE,
            available_tools=create_mock_tools(),
            success_criteria={"reasoning_correct": True},
        ),
    ]

    return tasks


def generate_data_manipulation_tasks() -> list[FunctionCallingTask]:
    """Generate data filtering and manipulation tasks."""

    sample_data = [
        {"name": "Alice", "age": 30, "department": "Engineering", "salary": 85000},
        {"name": "Bob", "age": 25, "department": "Marketing", "salary": 60000},
        {"name": "Charlie", "age": 35, "department": "Engineering", "salary": 95000},
        {"name": "Diana", "age": 28, "department": "Sales", "salary": 70000},
    ]

    tasks = [
        # Simple filtering
        FunctionCallingTask(
            id="data_001",
            description="Find all employees older than 30",
            expected_tool="data_filter",
            expected_arguments={
                "data": sample_data,
                "field": "age",
                "operator": "greater_than",
                "value": 30,
            },
            task_type=TaskCategory.DATA_MANIPULATION,
            complexity=TaskComplexity.SIMPLE,
            available_tools=create_mock_tools(),
            success_criteria={"exact_match": True},
        ),
        # Department filtering
        FunctionCallingTask(
            id="data_002",
            description="Show me all Engineering department employees",
            expected_tool="data_filter",
            expected_arguments={
                "data": sample_data,
                "field": "department",
                "operator": "equals",
                "value": "Engineering",
            },
            task_type=TaskCategory.DATA_MANIPULATION,
            complexity=TaskComplexity.MODERATE,
            available_tools=create_mock_tools(),
            success_criteria={"field_and_value_correct": True},
        ),
        # Complex filtering
        FunctionCallingTask(
            id="data_003",
            description="Filter employees earning more than 80000",
            expected_tool="data_filter",
            expected_arguments={
                "data": sample_data,
                "field": "salary",
                "operator": "greater_than",
                "value": 80000,
            },
            task_type=TaskCategory.DATA_MANIPULATION,
            complexity=TaskComplexity.MODERATE,
            available_tools=create_mock_tools(),
            success_criteria={"numeric_comparison": True},
        ),
    ]

    return tasks


def generate_api_call_tasks() -> list[FunctionCallingTask]:
    """Generate API calling tasks."""

    tasks = [
        # Simple weather query
        FunctionCallingTask(
            id="api_001",
            description="Get the weather for London",
            expected_tool="weather_api",
            expected_arguments={"location": "London"},
            task_type=TaskCategory.API_CALLS,
            complexity=TaskComplexity.SIMPLE,
            available_tools=create_mock_tools(),
            success_criteria={"location_correct": True},
        ),
        # Weather with units
        FunctionCallingTask(
            id="api_002",
            description="Check the weather in Tokyo with temperature in Fahrenheit",
            expected_tool="weather_api",
            expected_arguments={"location": "Tokyo", "units": "fahrenheit"},
            task_type=TaskCategory.API_CALLS,
            complexity=TaskComplexity.MODERATE,
            available_tools=create_mock_tools(),
            success_criteria={"location_and_units": True},
        ),
        # Weather with forecast
        FunctionCallingTask(
            id="api_003",
            description="Get weather for New York including the 5-day forecast",
            expected_tool="weather_api",
            expected_arguments={"location": "New York", "include_forecast": True},
            task_type=TaskCategory.API_CALLS,
            complexity=TaskComplexity.MODERATE,
            available_tools=create_mock_tools(),
            success_criteria={"forecast_included": True},
        ),
    ]

    return tasks


def generate_edge_case_tasks() -> list[FunctionCallingTask]:
    """Generate edge case and error handling tasks."""

    tasks = [
        # Missing required parameter
        FunctionCallingTask(
            id="edge_001",
            description="Calculate something with numbers",  # Vague request
            expected_tool="calculator",
            expected_arguments={"operation": "add", "a": 0, "b": 0},  # Default values
            task_type=TaskCategory.MATHEMATICAL,
            complexity=TaskComplexity.AMBIGUOUS,
            available_tools=create_mock_tools(),
            success_criteria={"handles_ambiguity": True},
            edge_case=True,
        ),
        # Invalid parameter type
        FunctionCallingTask(
            id="edge_002",
            description="Format the number 123 as uppercase",
            expected_tool="string_formatter",
            expected_arguments={"text": "123", "format_type": "uppercase"},
            task_type=TaskCategory.TEXT_PROCESSING,
            complexity=TaskComplexity.SIMPLE,
            available_tools=create_mock_tools(),
            success_criteria={"handles_type_conversion": True},
            edge_case=True,
        ),
        # No matching tool
        FunctionCallingTask(
            id="edge_003",
            description="Send an email to john@example.com",
            expected_tool="email_sender",  # Tool doesn't exist
            expected_arguments={"to": "john@example.com", "subject": "Test"},
            task_type=TaskCategory.API_CALLS,
            complexity=TaskComplexity.SIMPLE,
            available_tools=create_mock_tools(),
            success_criteria={"handles_missing_tool": True},
            edge_case=True,
        ),
    ]

    return tasks


def create_function_calling_dataset(
    total_tasks: int = 100, include_edge_cases: bool = True
) -> list[FunctionCallingTask]:
    """Create a comprehensive function calling dataset."""

    # Generate all task types
    all_tasks = []
    all_tasks.extend(generate_mathematical_tasks())
    all_tasks.extend(generate_text_processing_tasks())
    all_tasks.extend(generate_datetime_tasks())
    all_tasks.extend(generate_data_manipulation_tasks())
    all_tasks.extend(generate_api_call_tasks())

    if include_edge_cases:
        all_tasks.extend(generate_edge_case_tasks())

    # Expand dataset to reach desired size
    while len(all_tasks) < total_tasks:
        # Create variations of existing tasks
        base_task = random.choice(
            all_tasks[: len(all_tasks) // 2]
        )  # Avoid infinite recursion

        # Create a variation
        variation = FunctionCallingTask(
            id=f"{base_task.id}_var_{len(all_tasks)}",
            description=f"{base_task.description} (variation)",
            expected_tool=base_task.expected_tool,
            expected_arguments=base_task.expected_arguments.copy(),
            task_type=base_task.task_type,
            complexity=base_task.complexity,
            available_tools=base_task.available_tools,
            success_criteria=base_task.success_criteria,
            edge_case=base_task.edge_case,
        )

        all_tasks.append(variation)

    return all_tasks[:total_tasks]


def get_baseline_performance() -> dict[str, float]:
    """Get expected baseline performance metrics."""

    return {
        "simple_baseline": {
            "tool_selection_accuracy": 0.65,
            "parameter_validity_rate": 0.70,
            "execution_success_rate": 0.60,
            "avg_latency_ms": 150,
            "reliability_score": 0.63,
        },
        "standard_practice": {
            "tool_selection_accuracy": 0.80,
            "parameter_validity_rate": 0.85,
            "execution_success_rate": 0.78,
            "avg_latency_ms": 180,
            "reliability_score": 0.81,
        },
        "robust_approach": {
            "tool_selection_accuracy": 0.88,
            "parameter_validity_rate": 0.92,
            "execution_success_rate": 0.85,
            "avg_latency_ms": 220,
            "reliability_score": 0.88,
        },
    }


def analyze_task_distribution(tasks: list[FunctionCallingTask]) -> dict[str, Any]:
    """Analyze the distribution of tasks in the dataset."""

    # Count by category
    category_counts = {}
    for task in tasks:
        category = task.task_type.value
        category_counts[category] = category_counts.get(category, 0) + 1

    # Count by complexity
    complexity_counts = {}
    for task in tasks:
        complexity = task.complexity.value
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1

    # Count edge cases
    edge_case_count = sum(1 for task in tasks if task.edge_case)

    return {
        "total_tasks": len(tasks),
        "category_distribution": category_counts,
        "complexity_distribution": complexity_counts,
        "edge_cases": edge_case_count,
        "edge_case_percentage": (edge_case_count / len(tasks)) * 100 if tasks else 0,
    }
