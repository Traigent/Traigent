#!/usr/bin/env python3
"""Function Calling Agent - Validates GPT-4.1 function calling capabilities.

This agent tests tool use reliability based on OpenAI's GPT-4.1 claims:
- ComplexFuncBench: GPT-4.1 (65.5%) vs GPT-4o (66.5%)
- Taubench airline: GPT-4.1 (49.4%) vs GPT-4o (42.8%)
- Taubench retail: GPT-4.1 (68.0%) vs GPT-4o (60.3%)

Test Categories:
- Single tool selection: Choose correct tool for a given task
- Multi-tool orchestration: Coordinate multiple tools in sequence
- Complex parameter schemas: Handle nested/complex parameter structures

Usage:
    # Mock mode (recommended for testing)
    export TRAIGENT_MOCK_LLM=true
    python use-cases/gpt-4.1-study/agent/function_calling_agent.py

    # Real mode with OpenAI API
    export OPENAI_API_KEY=sk-...
    python use-cases/gpt-4.1-study/agent/function_calling_agent.py --max-trials 25
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# =============================================================================
# Constants
# =============================================================================

DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"

MODEL_GPT_4_1 = "gpt-4.1"
MODEL_GPT_4_1_MINI = "gpt-4.1-mini"
MODEL_GPT_4_1_NANO = "gpt-4.1-nano"
MODEL_GPT_4O = "gpt-4o"
MODEL_GPT_4O_MINI = "gpt-4o-mini"

DEFAULT_MODEL = MODEL_GPT_4O

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

import traigent  # noqa: E402
from traigent.api.decorators import EvaluationOptions, ExecutionOptions  # noqa: E402

# Import evaluator
_evaluator_path = (
    Path(__file__).parent.parent / "eval" / "function_calling_evaluator.py"
)
_spec = importlib.util.spec_from_file_location(
    "function_calling_evaluator", _evaluator_path
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load evaluator module from {_evaluator_path}")
_evaluator_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluator_module)

if TYPE_CHECKING:
    from types import ModuleType

    _evaluator_module: ModuleType

FunctionCallingEvaluator = _evaluator_module.FunctionCallingEvaluator
tool_selection_accuracy = _evaluator_module.tool_selection_accuracy
parameter_accuracy = _evaluator_module.parameter_accuracy

DATASET_PATH = (
    Path(__file__).parent.parent / "datasets" / "function_calling_dataset.jsonl"
)


# =============================================================================
# Configuration Space
# =============================================================================

CONFIGURATION_SPACE = {
    "model": [
        MODEL_GPT_4_1,
        MODEL_GPT_4_1_MINI,
        MODEL_GPT_4_1_NANO,
        MODEL_GPT_4O,
        MODEL_GPT_4O_MINI,
    ],
    "parallel_tool_calls": [True, False],
}


# =============================================================================
# Sample Tool Definitions
# =============================================================================

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search for records in a database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "table": {"type": "string", "description": "Table to search"},
                    "limit": {"type": "integer", "description": "Max results"},
                },
                "required": ["query", "table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                    "cc": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "CC recipients",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": "Create a new task in the task management system",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Task title"},
                    "description": {
                        "type": "string",
                        "description": "Task description",
                    },
                    "assignee": {"type": "string", "description": "Assigned user"},
                    "due_date": {
                        "type": "string",
                        "description": "Due date (ISO format)",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Task priority",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Task tags",
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_metrics",
            "description": "Calculate business metrics from data",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_type": {
                        "type": "string",
                        "enum": ["revenue", "churn", "growth", "retention"],
                        "description": "Type of metric",
                    },
                    "date_range": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "string"},
                            "end": {"type": "string"},
                        },
                        "description": "Date range for calculation",
                    },
                    "filters": {
                        "type": "object",
                        "additionalProperties": True,
                        "description": "Additional filters",
                    },
                },
                "required": ["metric_type"],
            },
        },
    },
]


# =============================================================================
# Mock Mode & API
# =============================================================================


def is_mock_mode() -> bool:
    """Check if running in mock mode."""
    return os.environ.get("TRAIGENT_MOCK_LLM", "").lower() in ("true", "1", "yes")


MODEL_COSTS = {
    MODEL_GPT_4_1: {"input": 2.00, "output": 8.00},
    MODEL_GPT_4_1_MINI: {"input": 0.40, "output": 1.60},
    MODEL_GPT_4_1_NANO: {"input": 0.10, "output": 0.40},
    MODEL_GPT_4O: {"input": 2.50, "output": 10.00},
    MODEL_GPT_4O_MINI: {"input": 0.15, "output": 0.60},
}

DEFAULT_MODEL_PRICING = {"input": 2.50, "output": 10.00}

# Model profiles for function calling (based on blog claims)
# Note: GPT-4.1 and GPT-4o have similar function calling performance per blog
MODEL_PROFILES = {
    MODEL_GPT_4_1: {
        "tool_selection": 0.88,  # Based on Taubench results
        "parameter_accuracy": 0.85,
        "multi_tool": 0.80,
        "complex_params": 0.75,
    },
    MODEL_GPT_4_1_MINI: {
        "tool_selection": 0.78,
        "parameter_accuracy": 0.75,
        "multi_tool": 0.65,
        "complex_params": 0.60,
    },
    MODEL_GPT_4_1_NANO: {
        "tool_selection": 0.55,
        "parameter_accuracy": 0.50,
        "multi_tool": 0.40,
        "complex_params": 0.35,
    },
    MODEL_GPT_4O: {
        "tool_selection": 0.85,  # Similar to GPT-4.1 per blog
        "parameter_accuracy": 0.82,
        "multi_tool": 0.75,
        "complex_params": 0.70,
    },
    MODEL_GPT_4O_MINI: {
        "tool_selection": 0.70,
        "parameter_accuracy": 0.65,
        "multi_tool": 0.55,
        "complex_params": 0.50,
    },
}

DEFAULT_PROFILE = {
    "tool_selection": 0.75,
    "parameter_accuracy": 0.70,
    "multi_tool": 0.60,
    "complex_params": 0.55,
}


def _get_deterministic_seed(task_id: str, model: str) -> int:
    """Generate deterministic seed for mock reproducibility."""
    combined = f"{task_id}:{model}"
    return int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)


def _get_api_config() -> tuple[str | None, str]:
    """Get API key and base URL from environment."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    api_base = os.environ.get("LLM_API_BASE", DEFAULT_OPENAI_API_BASE)
    return api_key, api_base


def call_llm_with_tools(
    prompt: str,
    tools: list[dict[str, Any]],
    model: str,
    parallel_tool_calls: bool = True,
) -> dict[str, Any]:
    """Call LLM with function calling support."""
    api_key, api_base = _get_api_config()

    if not api_key:
        print("Warning: No API key found.")
        return {"tool_calls": [], "message": ""}

    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": parallel_tool_calls,
        "temperature": 0.0,
        "max_tokens": 1000,
    }

    try:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=60) as response:  # noqa: S310
            result = json.loads(response.read().decode("utf-8"))
            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})

            tool_calls = []
            if message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    tool_calls.append(
                        {
                            "name": tc["function"]["name"],
                            "arguments": json.loads(tc["function"]["arguments"]),
                        }
                    )

            return {
                "tool_calls": tool_calls,
                "message": message.get("content", ""),
            }
    except (HTTPError, URLError, Exception) as e:
        print(f"LLM call error: {e}")
        return {"tool_calls": [], "message": ""}


# =============================================================================
# Mock Generation
# =============================================================================


def generate_mock_output(
    task_id: str,
    category: str,
    user_request: str,
    expected_tools: list[str],
    expected_parameters: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """Generate mock output based on model profiles."""
    seed = _get_deterministic_seed(task_id, model)
    rng = random.Random(seed)  # noqa: S311

    profile = MODEL_PROFILES.get(model, DEFAULT_PROFILE)

    # Determine success probability based on category
    if category == "single_tool":
        tool_prob = profile["tool_selection"]
        param_prob = profile["parameter_accuracy"]
    elif category == "multi_tool":
        tool_prob = profile["multi_tool"]
        param_prob = profile["parameter_accuracy"] * 0.9
    else:  # complex_params
        tool_prob = profile["tool_selection"]
        param_prob = profile["complex_params"]

    tool_correct = rng.random() < tool_prob
    params_correct = rng.random() < param_prob

    if tool_correct:
        selected_tools = expected_tools
    else:
        # Select wrong tool
        all_tool_names = [t["function"]["name"] for t in SAMPLE_TOOLS]
        wrong_tools = [t for t in all_tool_names if t not in expected_tools]
        selected_tools = [rng.choice(wrong_tools)] if wrong_tools else expected_tools

    if params_correct:
        parameters = expected_parameters
    else:
        # Generate incorrect parameters
        parameters = dict.fromkeys(expected_parameters, "wrong_value")

    tool_calls = [
        {"name": tool, "arguments": parameters.get(tool, {})} for tool in selected_tools
    ]

    return {
        "tool_calls": tool_calls,
        "tool_selection_correct": tool_correct,
        "parameters_correct": params_correct,
        "expected_tools": expected_tools,
        "selected_tools": selected_tools,
    }


def estimate_mock_cost(
    user_request: str, tools: list[dict[str, Any]], model: str
) -> float:
    """Estimate mock cost for the task."""
    # Tools add to input tokens
    tools_str = json.dumps(tools)
    input_tokens = (len(user_request) + len(tools_str)) // 4 + 100
    output_tokens = 200

    pricing = MODEL_COSTS.get(model, DEFAULT_MODEL_PRICING)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


# =============================================================================
# Metric Functions
# =============================================================================


def cost_metric(
    output: dict[str, Any] | None,
    expected: dict[str, Any],
    **kwargs: Any,
) -> float:
    """Calculate cost for the function calling task."""
    input_data = kwargs.get("input_data", {})
    config = kwargs.get("config", {})

    user_request = input_data.get("user_request", "")
    tools = input_data.get("available_tools", SAMPLE_TOOLS)
    model = config.get("model", DEFAULT_MODEL)

    return estimate_mock_cost(user_request, tools, model)


# =============================================================================
# Main Agent
# =============================================================================


@traigent.optimize(
    configuration_space=CONFIGURATION_SPACE,
    objectives=["tool_selection_accuracy", "parameter_accuracy", "cost"],
    metric_functions={
        "tool_selection_accuracy": tool_selection_accuracy,
        "parameter_accuracy": parameter_accuracy,
        "cost": cost_metric,
    },
    evaluation=EvaluationOptions(eval_dataset=str(DATASET_PATH)),
    execution=ExecutionOptions(execution_mode="edge_analytics"),
)
def function_calling_agent(
    task_id: str,
    category: str,
    user_request: str,
    available_tools: list[dict[str, Any]] | None = None,
    expected_tools: list[str] | None = None,
    expected_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Test function calling capabilities.

    This agent validates GPT-4.1's function calling reliability across:
    - Single tool selection
    - Multi-tool orchestration
    - Complex parameter schemas

    Args:
        task_id: Unique identifier for the task
        category: One of single_tool, multi_tool, complex_params
        user_request: The user's natural language request
        available_tools: List of tool definitions (defaults to SAMPLE_TOOLS)
        expected_tools: List of tool names that should be called
        expected_parameters: Dict mapping tool names to expected parameters

    Returns:
        Dict with tool_calls and accuracy metrics
    """
    config = traigent.get_config()
    model = config.get("model", DEFAULT_MODEL)
    parallel_tool_calls = config.get("parallel_tool_calls", True)

    tools = available_tools or SAMPLE_TOOLS
    expected_tools = expected_tools or []
    expected_parameters = expected_parameters or {}

    # Mock mode
    if is_mock_mode():
        return generate_mock_output(
            task_id=task_id,
            category=category,
            user_request=user_request,
            expected_tools=expected_tools,
            expected_parameters=expected_parameters,
            model=model,
        )

    # Real LLM call with tools
    result = call_llm_with_tools(user_request, tools, model, parallel_tool_calls)

    # Evaluate results
    selected_tools = [tc["name"] for tc in result["tool_calls"]]
    tool_correct = set(selected_tools) == set(expected_tools)

    # Check parameters
    params_correct = True
    for tc in result["tool_calls"]:
        expected = expected_parameters.get(tc["name"], {})
        if expected:
            for key, value in expected.items():
                if tc["arguments"].get(key) != value:
                    params_correct = False
                    break

    return {
        "tool_calls": result["tool_calls"],
        "tool_selection_correct": tool_correct,
        "parameters_correct": params_correct,
        "expected_tools": expected_tools,
        "selected_tools": selected_tools,
    }


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Run the function calling agent for testing."""
    parser = argparse.ArgumentParser(description="GPT-4.1 Function Calling Agent")
    parser.add_argument(
        "--max-trials", type=int, default=12, help="Max optimization trials"
    )
    parser.add_argument("--demo", action="store_true", help="Run demo task")
    args = parser.parse_args()

    if args.demo:
        print("Running demo task...")
        result = function_calling_agent(
            task_id="demo_001",
            category="single_tool",
            user_request="Search for all orders from last month in the orders table",
            expected_tools=["search_database"],
            expected_parameters={
                "search_database": {"table": "orders", "query": "last month"}
            },
        )
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print(f"Running optimization with {args.max_trials} trials...")


if __name__ == "__main__":
    main()
