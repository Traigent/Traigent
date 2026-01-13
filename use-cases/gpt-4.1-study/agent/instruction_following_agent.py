#!/usr/bin/env python3
"""Instruction Following Agent - Validates GPT-4.1 instruction following improvements.

This agent tests instruction following across multiple categories based on
OpenAI's GPT-4.1 announcement claims:
- Internal IF eval (hard): GPT-4.1 (49.1%) vs GPT-4o (29.2%)
- MultiChallenge: GPT-4.1 (38.3%) vs GPT-4o (27.8%)
- IFEval: GPT-4.1 (87.4%) vs GPT-4o (81.0%)

Test Categories (per OpenAI's internal eval):
- Format following (JSON, XML, YAML, Markdown)
- Negative instructions ("don't mention X")
- Ordered instructions ("first do A, then B")
- Content requirements ("must include X")
- Ranking/ordering output

Usage:
    # Mock mode (recommended for testing)
    export TRAIGENT_MOCK_LLM=true
    python use-cases/gpt-4.1-study/agent/instruction_following_agent.py

    # Real mode with OpenAI API
    export OPENAI_API_KEY=sk-...
    python use-cases/gpt-4.1-study/agent/instruction_following_agent.py --max-trials 30
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
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
_evaluator_path = Path(__file__).parent.parent / "eval" / "instruction_evaluator.py"
_spec = importlib.util.spec_from_file_location("instruction_evaluator", _evaluator_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load evaluator module from {_evaluator_path}")
_evaluator_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluator_module)

if TYPE_CHECKING:
    from types import ModuleType

    _evaluator_module: ModuleType

InstructionFollowingEvaluator = _evaluator_module.InstructionFollowingEvaluator
format_compliance = _evaluator_module.format_compliance
instruction_adherence = _evaluator_module.instruction_adherence

DATASET_PATH = (
    Path(__file__).parent.parent / "datasets" / "instruction_following_dataset.jsonl"
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
    "temperature": [0.0, 0.3],
}


# =============================================================================
# Instruction Categories
# =============================================================================

INSTRUCTION_CATEGORIES = {
    "format_following": "Output must be in specific format (JSON, XML, YAML, Markdown)",
    "negative_instructions": "Output must NOT include certain content",
    "ordered_instructions": "Steps must be performed in specific order",
    "content_requirements": "Output must include specific content",
    "ranking": "Output must be ordered/ranked in specific way",
}


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

# Model profiles for instruction following (based on blog claims)
MODEL_PROFILES = {
    MODEL_GPT_4_1: {
        "format_compliance": 0.90,  # Excellent format following
        "negative_instruction": 0.85,
        "ordered_instruction": 0.88,
        "content_requirement": 0.92,
        "ranking": 0.87,
    },
    MODEL_GPT_4_1_MINI: {
        "format_compliance": 0.82,
        "negative_instruction": 0.78,
        "ordered_instruction": 0.80,
        "content_requirement": 0.84,
        "ranking": 0.79,
    },
    MODEL_GPT_4_1_NANO: {
        "format_compliance": 0.65,
        "negative_instruction": 0.60,
        "ordered_instruction": 0.62,
        "content_requirement": 0.68,
        "ranking": 0.58,
    },
    MODEL_GPT_4O: {
        "format_compliance": 0.70,  # Lower based on claims
        "negative_instruction": 0.65,
        "ordered_instruction": 0.68,
        "content_requirement": 0.72,
        "ranking": 0.66,
    },
    MODEL_GPT_4O_MINI: {
        "format_compliance": 0.60,
        "negative_instruction": 0.55,
        "ordered_instruction": 0.58,
        "content_requirement": 0.62,
        "ranking": 0.54,
    },
}

DEFAULT_PROFILE = {
    "format_compliance": 0.65,
    "negative_instruction": 0.60,
    "ordered_instruction": 0.62,
    "content_requirement": 0.68,
    "ranking": 0.60,
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


def call_llm(prompt: str, model: str, temperature: float) -> str:
    """Call LLM via OpenAI-compatible API."""
    api_key, api_base = _get_api_config()

    if not api_key:
        print("Warning: No API key found.")
        return ""

    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 2000,
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
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except (HTTPError, URLError, Exception) as e:
        print(f"LLM call error: {e}")
        return ""


# =============================================================================
# Validation Functions
# =============================================================================


def validate_json_format(response: str) -> bool:
    """Check if response is valid JSON."""
    try:
        json.loads(response.strip())
        return True
    except json.JSONDecodeError:
        # Try extracting from code block
        match = re.search(r"```(?:json)?\n?(.*?)```", response, re.DOTALL)
        if match:
            try:
                json.loads(match.group(1).strip())
                return True
            except json.JSONDecodeError:
                pass
        return False


def validate_xml_format(response: str) -> bool:
    """Check if response is valid XML-like structure."""
    return bool(re.search(r"<\w+>.*</\w+>", response, re.DOTALL))


def validate_yaml_format(response: str) -> bool:
    """Check if response looks like valid YAML."""
    lines = response.strip().split("\n")
    yaml_patterns = [r"^\w+:", r"^- ", r"^  \w+:"]
    yaml_lines = sum(
        1 for line in lines if any(re.match(p, line) for p in yaml_patterns)
    )
    return yaml_lines >= len(lines) * 0.5


def validate_markdown_format(response: str) -> bool:
    """Check if response uses markdown formatting."""
    md_patterns = [r"^#{1,6} ", r"^\* ", r"^- ", r"^\d+\. ", r"\*\*.*\*\*", r"```"]
    return any(re.search(p, response, re.MULTILINE) for p in md_patterns)


def check_negative_instruction(response: str, forbidden_terms: list[str]) -> bool:
    """Check that forbidden terms are NOT present."""
    response_lower = response.lower()
    return not any(term.lower() in response_lower for term in forbidden_terms)


def check_content_requirement(response: str, required_terms: list[str]) -> bool:
    """Check that required terms ARE present."""
    response_lower = response.lower()
    return all(term.lower() in response_lower for term in required_terms)


def check_ordered_instruction(response: str, ordered_items: list[str]) -> bool:
    """Check that items appear in correct order."""
    positions = []
    response_lower = response.lower()
    for item in ordered_items:
        pos = response_lower.find(item.lower())
        if pos == -1:
            return False
        positions.append(pos)
    return positions == sorted(positions)


# =============================================================================
# Mock Generation
# =============================================================================


def generate_mock_output(
    task_id: str,
    category: str,
    prompt: str,
    expected_format: str | None,
    forbidden_terms: list[str] | None,
    required_terms: list[str] | None,
    ordered_items: list[str] | None,
    model: str,
) -> dict[str, Any]:
    """Generate mock output based on model profiles."""
    seed = _get_deterministic_seed(task_id, model)
    rng = random.Random(seed)  # noqa: S311

    profile = MODEL_PROFILES.get(model, DEFAULT_PROFILE)

    # Determine success based on category
    category_key = category.replace("-", "_").lower()
    if category_key == "format_following":
        success_prob = profile["format_compliance"]
    elif category_key == "negative_instructions":
        success_prob = profile["negative_instruction"]
    elif category_key == "ordered_instructions":
        success_prob = profile["ordered_instruction"]
    elif category_key == "content_requirements":
        success_prob = profile["content_requirement"]
    else:
        success_prob = profile.get("ranking", 0.65)

    success = rng.random() < success_prob

    # Generate appropriate mock response
    if success:
        response = _generate_compliant_response(
            expected_format, forbidden_terms, required_terms, ordered_items
        )
        format_ok = True
        negative_ok = True
        content_ok = True
        order_ok = True
    else:
        response = "Here is my response without following the instructions properly."
        format_ok = expected_format is None
        negative_ok = forbidden_terms is None
        content_ok = required_terms is None
        order_ok = ordered_items is None

    return {
        "response": response,
        "format_compliant": format_ok,
        "negative_instruction_followed": negative_ok,
        "content_requirements_met": content_ok,
        "order_followed": order_ok,
        "overall_compliance": success,
    }


def _generate_compliant_response(
    expected_format: str | None,
    forbidden_terms: list[str] | None,
    required_terms: list[str] | None,
    ordered_items: list[str] | None,
) -> str:
    """Generate a response that complies with all requirements."""
    if expected_format == "json":
        base = '{"result": "compliant response"'
        if required_terms:
            base += f', "included": {json.dumps(required_terms)}'
        return base + "}"
    elif expected_format == "xml":
        content = "compliant response"
        if required_terms:
            content = " ".join(required_terms)
        return f"<response>{content}</response>"
    elif expected_format == "yaml":
        lines = ["result: compliant response"]
        if required_terms:
            lines.append("included:")
            for term in required_terms:
                lines.append(f"  - {term}")
        return "\n".join(lines)
    elif expected_format == "markdown":
        lines = ["# Response", "", "This is a compliant response."]
        if required_terms:
            lines.append("")
            for term in required_terms:
                lines.append(f"- {term}")
        return "\n".join(lines)
    elif ordered_items:
        return " then ".join(ordered_items)
    elif required_terms:
        return f"Response including: {', '.join(required_terms)}"
    else:
        return "Compliant response following all instructions."


def estimate_mock_cost(prompt: str, model: str) -> float:
    """Estimate mock cost for the task."""
    input_tokens = len(prompt) // 4 + 50
    output_tokens = 300

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
    """Calculate cost for the instruction following task."""
    input_data = kwargs.get("input_data", {})
    config = kwargs.get("config", {})

    prompt = input_data.get("prompt", "")
    model = config.get("model", DEFAULT_MODEL)

    return estimate_mock_cost(prompt, model)


# =============================================================================
# Main Agent
# =============================================================================


@traigent.optimize(
    configuration_space=CONFIGURATION_SPACE,
    objectives=["format_compliance", "instruction_adherence", "cost"],
    metric_functions={
        "format_compliance": format_compliance,
        "instruction_adherence": instruction_adherence,
        "cost": cost_metric,
    },
    evaluation=EvaluationOptions(eval_dataset=str(DATASET_PATH)),
    execution=ExecutionOptions(execution_mode="edge_analytics"),
)
def instruction_following_agent(
    task_id: str,
    category: str,
    prompt: str,
    expected_format: str | None = None,
    forbidden_terms: list[str] | None = None,
    required_terms: list[str] | None = None,
    ordered_items: list[str] | None = None,
) -> dict[str, Any]:
    """Test instruction following capabilities.

    This agent validates GPT-4.1's claimed improvements in instruction following
    across multiple categories from OpenAI's internal evaluation.

    Args:
        task_id: Unique identifier for the task
        category: One of format_following, negative_instructions, ordered_instructions,
                  content_requirements, ranking
        prompt: The instruction prompt to follow
        expected_format: Expected output format (json, xml, yaml, markdown)
        forbidden_terms: Terms that must NOT appear in output
        required_terms: Terms that must appear in output
        ordered_items: Items that must appear in specific order

    Returns:
        Dict with response and compliance metrics
    """
    config = traigent.get_config()
    model = config.get("model", DEFAULT_MODEL)
    temperature = config.get("temperature", 0.0)

    # Mock mode
    if is_mock_mode():
        return generate_mock_output(
            task_id=task_id,
            category=category,
            prompt=prompt,
            expected_format=expected_format,
            forbidden_terms=forbidden_terms,
            required_terms=required_terms,
            ordered_items=ordered_items,
            model=model,
        )

    # Real LLM call
    response = call_llm(prompt, model, temperature)

    # Validate response
    format_ok = True
    if expected_format:
        validators = {
            "json": validate_json_format,
            "xml": validate_xml_format,
            "yaml": validate_yaml_format,
            "markdown": validate_markdown_format,
        }
        validator = validators.get(expected_format)
        format_ok = validator(response) if validator else True

    negative_ok = check_negative_instruction(response, forbidden_terms or [])
    content_ok = check_content_requirement(response, required_terms or [])
    order_ok = (
        check_ordered_instruction(response, ordered_items or [])
        if ordered_items
        else True
    )

    overall = format_ok and negative_ok and content_ok and order_ok

    return {
        "response": response,
        "format_compliant": format_ok,
        "negative_instruction_followed": negative_ok,
        "content_requirements_met": content_ok,
        "order_followed": order_ok,
        "overall_compliance": overall,
    }


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Run the instruction following agent for testing."""
    parser = argparse.ArgumentParser(description="GPT-4.1 Instruction Following Agent")
    parser.add_argument(
        "--max-trials", type=int, default=15, help="Max optimization trials"
    )
    parser.add_argument("--demo", action="store_true", help="Run demo task")
    args = parser.parse_args()

    if args.demo:
        print("Running demo task...")
        result = instruction_following_agent(
            task_id="demo_001",
            category="format_following",
            prompt="List 3 programming languages as a JSON array.",
            expected_format="json",
        )
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print(f"Running optimization with {args.max_trials} trials...")


if __name__ == "__main__":
    main()
