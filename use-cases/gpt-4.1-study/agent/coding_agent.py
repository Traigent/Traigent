#!/usr/bin/env python3
"""Coding Agent - Validates GPT-4.1 coding improvements.

This agent tests code generation and diff format reliability across
GPT-4.1, GPT-4.1-mini, GPT-4.1-nano, GPT-4o, and GPT-4o-mini models.

Based on claims from OpenAI's GPT-4.1 announcement:
- SWE-bench Verified: GPT-4.1 (54.6%) vs GPT-4o (33.2%)
- Aider Polyglot diff: GPT-4.1 (53%) vs GPT-4o (18%)
- Extraneous edits: GPT-4.1 (2%) vs GPT-4o (9%)

Usage:
    # Mock mode (recommended for testing)
    export TRAIGENT_MOCK_LLM=true
    python use-cases/gpt-4.1-study/agent/coding_agent.py

    # Real mode with OpenAI API
    export OPENAI_API_KEY=sk-...
    python use-cases/gpt-4.1-study/agent/coding_agent.py --max-trials 25
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

# Model constants
MODEL_GPT_4_1 = "gpt-4.1"
MODEL_GPT_4_1_MINI = "gpt-4.1-mini"
MODEL_GPT_4_1_NANO = "gpt-4.1-nano"
MODEL_GPT_4O = "gpt-4o"
MODEL_GPT_4O_MINI = "gpt-4o-mini"

DEFAULT_MODEL = MODEL_GPT_4O
DEFAULT_OUTPUT_FORMAT = "whole_file"

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

import traigent  # noqa: E402
from traigent.api.decorators import EvaluationOptions, ExecutionOptions  # noqa: E402

# Import evaluator
_evaluator_path = Path(__file__).parent.parent / "eval" / "coding_evaluator.py"
_spec = importlib.util.spec_from_file_location("coding_evaluator", _evaluator_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load evaluator module from {_evaluator_path}")
_evaluator_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluator_module)

if TYPE_CHECKING:
    from types import ModuleType

    _evaluator_module: ModuleType

CodingEvaluator = _evaluator_module.CodingEvaluator
coding_accuracy = _evaluator_module.coding_accuracy
diff_compliance = _evaluator_module.diff_compliance

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "coding_dataset.jsonl"


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
    "output_format": ["whole_file", "diff"],
    "temperature": [0.0, 0.3],
}


# =============================================================================
# Prompt Templates
# =============================================================================

GENERATION_PROMPT = """You are an expert Python programmer. Generate a complete Python function based on this specification.

Specification:
{specification}

Requirements:
- Write clean, readable Python code
- Include proper docstring
- Handle edge cases appropriately
- Follow PEP 8 style guidelines

Return ONLY the Python code, no explanations or markdown."""

DIFF_PROMPT = """You are an expert Python programmer. Modify the existing code according to the specification.

Original Code:
```python
{original_code}
```

Modification Required:
{specification}

Output Format: Use search/replace blocks as follows:
<<<<<<< SEARCH
[exact code to find]
=======
[replacement code]
>>>>>>> REPLACE

IMPORTANT:
- Output ONLY search/replace blocks
- Each SEARCH block must match the original code EXACTLY
- Make minimal changes - only modify what's necessary
- Do not add comments or explanations"""

BUG_FIX_PROMPT = """You are an expert Python programmer. Fix the bug in this code.

Buggy Code:
```python
{original_code}
```

Bug Description:
{specification}

Return the corrected Python code. Make ONLY the changes needed to fix the bug.
Do not refactor or improve unrelated code."""


# =============================================================================
# Mock Mode Detection
# =============================================================================


def is_mock_mode() -> bool:
    """Check if running in mock mode."""
    return os.environ.get("TRAIGENT_MOCK_LLM", "").lower() in ("true", "1", "yes")


# =============================================================================
# Model Cost Estimates (per 1M tokens, in USD)
# =============================================================================

MODEL_COSTS = {
    MODEL_GPT_4_1: {"input": 2.00, "output": 8.00},
    MODEL_GPT_4_1_MINI: {"input": 0.40, "output": 1.60},
    MODEL_GPT_4_1_NANO: {"input": 0.10, "output": 0.40},
    MODEL_GPT_4O: {"input": 2.50, "output": 10.00},
    MODEL_GPT_4O_MINI: {"input": 0.15, "output": 0.60},
}

DEFAULT_MODEL_PRICING = {"input": 2.50, "output": 10.00}

# Model performance profiles for mock mode (based on blog claims)
MODEL_PROFILES = {
    MODEL_GPT_4_1: {
        "task_completion": 0.85,  # Higher based on SWE-bench claims
        "diff_compliance": 0.90,  # Much higher diff format reliability
        "extraneous_rate": 0.02,  # 2% extraneous edits (blog claim)
    },
    MODEL_GPT_4_1_MINI: {
        "task_completion": 0.70,
        "diff_compliance": 0.75,
        "extraneous_rate": 0.05,
    },
    MODEL_GPT_4_1_NANO: {
        "task_completion": 0.55,
        "diff_compliance": 0.60,
        "extraneous_rate": 0.08,
    },
    MODEL_GPT_4O: {
        "task_completion": 0.65,  # Lower based on SWE-bench claims
        "diff_compliance": 0.45,  # Much lower diff format reliability
        "extraneous_rate": 0.09,  # 9% extraneous edits (blog claim)
    },
    MODEL_GPT_4O_MINI: {
        "task_completion": 0.45,
        "diff_compliance": 0.35,
        "extraneous_rate": 0.12,
    },
}

DEFAULT_PROFILE = {
    "task_completion": 0.60,
    "diff_compliance": 0.50,
    "extraneous_rate": 0.10,
}


# =============================================================================
# Helper Functions
# =============================================================================


def _get_deterministic_seed(task_id: str, model: str, output_format: str) -> int:
    """Generate deterministic seed for mock reproducibility."""
    combined = f"{task_id}:{model}:{output_format}"
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
        print("Warning: No API key found. Set OPENAI_API_KEY or LLM_API_KEY")
        return ""

    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert Python programmer.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 4000,
    }

    try:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=120) as response:  # noqa: S310
            result = json.loads(response.read().decode("utf-8"))
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except (HTTPError, URLError) as e:
        print(f"API error: {e}")
        return ""
    except Exception as e:
        print(f"LLM call error: {e}")
        return ""


def parse_diff_output(response: str) -> list[dict[str, str]]:
    """Parse search/replace diff blocks from LLM response."""
    blocks = []
    pattern = r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
    matches = re.findall(pattern, response, re.DOTALL)
    for search, replace in matches:
        blocks.append({"search": search, "replace": replace})
    return blocks


def apply_diff_blocks(original: str, blocks: list[dict[str, str]]) -> str:
    """Apply search/replace blocks to original code."""
    result = original
    for block in blocks:
        result = result.replace(block["search"], block["replace"], 1)
    return result


def extract_code_from_response(response: str) -> str:
    """Extract Python code from LLM response."""
    # Try to extract from markdown code block
    match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try plain code block
    match = re.search(r"```\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Return as-is if no code blocks found
    return response.strip()


# =============================================================================
# Mock Generation
# =============================================================================


def generate_mock_output(
    task_type: str,
    task_id: str,
    specification: str,
    original_code: str | None,
    expected_code: str,
    model: str,
    output_format: str,
) -> dict[str, Any]:
    """Generate mock output based on model profiles."""
    seed = _get_deterministic_seed(task_id, model, output_format)
    rng = random.Random(seed)  # noqa: S311

    profile = MODEL_PROFILES.get(model, DEFAULT_PROFILE)

    # Determine if task completes successfully
    task_success = rng.random() < profile["task_completion"]

    # Determine diff compliance (only relevant for diff format)
    diff_compliant = (
        output_format != "diff" or rng.random() < profile["diff_compliance"]
    )

    # Check for extraneous edits
    has_extraneous = rng.random() < profile["extraneous_rate"]

    if task_success and diff_compliant:
        generated_code = expected_code
        if has_extraneous:
            # Add a minor extraneous edit (comment or whitespace)
            generated_code = f"# Auto-generated\n{generated_code}"
    else:
        # Generate partial or incorrect output
        generated_code = "def placeholder():\n    pass  # TODO: implement"

    return {
        "generated_code": generated_code,
        "task_completed": task_success,
        "diff_compliant": diff_compliant,
        "has_extraneous_edits": has_extraneous,
        "output_format": output_format,
    }


def estimate_mock_cost(
    specification: str,
    original_code: str | None,
    model: str,
) -> float:
    """Estimate mock cost for the task."""
    # Estimate input tokens
    input_text = specification + (original_code or "")
    input_tokens = len(input_text) // 4 + 200  # prompt overhead

    # Estimate output tokens
    output_tokens = 500  # average code output

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
    """Calculate cost for the coding task."""
    input_data = kwargs.get("input_data", {})
    config = kwargs.get("config", {})

    specification = input_data.get("specification", "")
    original_code = input_data.get("original_code")
    model = config.get("model", DEFAULT_MODEL)

    return estimate_mock_cost(specification, original_code, model)


# =============================================================================
# Main Agent
# =============================================================================


@traigent.optimize(
    configuration_space=CONFIGURATION_SPACE,
    objectives=["task_completion", "diff_compliance", "cost"],
    metric_functions={
        "task_completion": coding_accuracy,
        "diff_compliance": diff_compliance,
        "cost": cost_metric,
    },
    evaluation=EvaluationOptions(eval_dataset=str(DATASET_PATH)),
    execution=ExecutionOptions(execution_mode="edge_analytics"),
)
def coding_agent(
    task_type: str,
    task_id: str,
    specification: str,
    original_code: str | None = None,
    expected_code: str = "",
) -> dict[str, Any]:
    """Generate or modify Python code based on specification.

    This agent validates GPT-4.1's claimed improvements in:
    - Task completion (SWE-bench style)
    - Diff format reliability (Aider style)
    - Avoiding extraneous edits

    Args:
        task_type: One of "generation", "modification", "bug_fix"
        task_id: Unique identifier for the task
        specification: Description of what to generate/modify/fix
        original_code: Existing code (for modification/bug_fix tasks)
        expected_code: Expected output for evaluation

    Returns:
        Dict with generated_code, task_completed, diff_compliant, etc.
    """
    config = traigent.get_config()
    model = config.get("model", DEFAULT_MODEL)
    output_format = config.get("output_format", DEFAULT_OUTPUT_FORMAT)
    temperature = config.get("temperature", 0.0)

    # Mock mode
    if is_mock_mode():
        return generate_mock_output(
            task_type=task_type,
            task_id=task_id,
            specification=specification,
            original_code=original_code,
            expected_code=expected_code,
            model=model,
            output_format=output_format,
        )

    # Select prompt based on task type and output format
    if task_type == "generation":
        prompt = GENERATION_PROMPT.format(specification=specification)
    elif task_type == "modification" and output_format == "diff":
        prompt = DIFF_PROMPT.format(
            original_code=original_code or "",
            specification=specification,
        )
    else:
        prompt = BUG_FIX_PROMPT.format(
            original_code=original_code or "",
            specification=specification,
        )

    # Call LLM
    response = call_llm(prompt, model, temperature)

    # Process response based on output format
    if output_format == "diff" and task_type == "modification":
        diff_blocks = parse_diff_output(response)
        diff_compliant = len(diff_blocks) > 0
        if diff_compliant and original_code:
            generated_code = apply_diff_blocks(original_code, diff_blocks)
        else:
            generated_code = extract_code_from_response(response)
    else:
        generated_code = extract_code_from_response(response)
        diff_compliant = True  # N/A for non-diff formats

    # Check for extraneous edits (compare with original if available)
    has_extraneous = False
    if original_code and generated_code:
        # Simple heuristic: check if unrelated lines were changed
        orig_lines = set(original_code.strip().split("\n"))
        gen_lines = set(generated_code.strip().split("\n"))
        # New lines that aren't part of expected changes
        unexpected_additions = gen_lines - orig_lines
        has_extraneous = len(unexpected_additions) > 5  # Threshold

    return {
        "generated_code": generated_code,
        "task_completed": bool(generated_code),
        "diff_compliant": diff_compliant,
        "has_extraneous_edits": has_extraneous,
        "output_format": output_format,
    }


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Run the coding agent for testing."""
    parser = argparse.ArgumentParser(description="GPT-4.1 Coding Agent")
    parser.add_argument(
        "--max-trials", type=int, default=10, help="Max optimization trials"
    )
    parser.add_argument("--demo", action="store_true", help="Run single demo task")
    args = parser.parse_args()

    if args.demo:
        print("Running demo task...")
        result = coding_agent(
            task_type="generation",
            task_id="demo_001",
            specification="Write a function that calculates the fibonacci sequence up to n terms",
            expected_code="def fibonacci(n): ...",
        )
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print(f"Running optimization with {args.max_trials} trials...")
        print("Note: Set TRAIGENT_MOCK_LLM=true for testing without API calls")


if __name__ == "__main__":
    main()
