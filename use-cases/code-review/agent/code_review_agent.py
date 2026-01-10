#!/usr/bin/env python3
"""Code Review Agent - Identifies code quality issues in Python functions.

This agent analyzes Python functions and identifies code quality issues
across 10 categories. It optimizes across multiple models and prompting
strategies to find the best configuration for accurate issue detection.

Usage:
    # Mock mode (recommended for testing)
    export TRAIGENT_MOCK_LLM=true
    python use-cases/code-review/agent/code_review_agent.py

    # Real mode with OpenAI API
    export OPENAI_API_KEY=sk-...
    python use-cases/code-review/agent/code_review_agent.py --max-trials 24

    # Demo mode (single function analysis)
    python use-cases/code-review/agent/code_review_agent.py --demo

    # Custom max trials
    python use-cases/code-review/agent/code_review_agent.py --max-trials 50
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import traigent
from traigent.api.decorators import EvaluationOptions, ExecutionOptions

# Import evaluator from sibling directory
_evaluator_path = Path(__file__).parent.parent / "eval" / "evaluator.py"
_spec = importlib.util.spec_from_file_location("code_review_evaluator", _evaluator_path)
_evaluator_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluator_module)
CodeReviewEvaluator = _evaluator_module.CodeReviewEvaluator
detection_f1_metric = _evaluator_module.detection_f1_metric

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "code_issues.jsonl"


# =============================================================================
# Issue Types
# =============================================================================

ISSUE_TYPES = {
    "MISSING_DOCS": "Incomplete or missing docstrings",
    "IMPLICIT_ASSUMPTION": "Unvalidated inputs, implicit assumptions",
    "SIDE_EFFECT": "Global state mutation, side effects",
    "COMPLEXITY": "Complex logic, deep nesting, potential bugs",
    "BROAD_EXCEPTION": "Overly broad exception handling (except Exception)",
    "PRINCIPLE_VIOLATION": "Violating coding principles (SRP, long functions)",
    "TODO_KNOWN_ISSUE": "TODO/FIXME/HACK comments, known issues",
    "TYPE_HANDLING": "Unsafe type operations without checks",
    "API_DESIGN": "Long parameter lists, unclear naming",
    "THREADING_ISSUE": "Race conditions, thread safety issues",
}


# =============================================================================
# Prompt Strategies
# =============================================================================

DIRECT_PROMPT = """Analyze this Python function for code quality issues.

Source file: {source_file}
Function: {function_name}

```python
{function_code}
```

Identify issues in these categories ONLY:
MISSING_DOCS, IMPLICIT_ASSUMPTION, SIDE_EFFECT, COMPLEXITY,
BROAD_EXCEPTION, PRINCIPLE_VIOLATION, TODO_KNOWN_ISSUE,
TYPE_HANDLING, API_DESIGN, THREADING_ISSUE

IMPORTANT: Return ONLY valid JSON. No explanations, no markdown.
Output format: [{{"issue_type": "CATEGORY", "description": "brief description"}}]
If no issues: []"""

CHAIN_OF_THOUGHT_PROMPT = """Analyze this Python function step-by-step for code quality issues.

Source file: {source_file}
Function: {function_name}

```python
{function_code}
```

Think through systematically:
1. Check documentation completeness
2. Check input validation and assumptions
3. Look for side effects and global state
4. Assess complexity and readability
5. Check exception handling patterns
6. Look for TODOs or known issues
7. Review type safety
8. Evaluate API design
9. Consider thread safety

After analysis, output ONLY valid JSON (no other text):
[{{"issue_type": "CATEGORY", "description": "brief description"}}]"""

CHECKLIST_PROMPT = """Review this Python function using the checklist.

Source file: {source_file}
Function: {function_name}

```python
{function_code}
```

Check each category:
- MISSING_DOCS: Has docstring? Params documented?
- IMPLICIT_ASSUMPTION: Inputs validated before use?
- SIDE_EFFECT: Modifies global state?
- COMPLEXITY: Deep nesting? Complex conditionals?
- BROAD_EXCEPTION: Bare except or except Exception?
- PRINCIPLE_VIOLATION: >50 lines? Multiple responsibilities?
- TODO_KNOWN_ISSUE: TODO/FIXME/HACK comments?
- TYPE_HANDLING: Types checked before operations?
- API_DESIGN: >5 params? Unclear naming?
- THREADING_ISSUE: Shared state without locks?

RESPOND WITH ONLY JSON. No explanation. No markdown.
[{{"issue_type": "CATEGORY", "description": "brief description"}}]"""

PROMPT_STRATEGIES = {
    "direct": DIRECT_PROMPT,
    "chain_of_thought": CHAIN_OF_THOUGHT_PROMPT,
    "checklist": CHECKLIST_PROMPT,
}


# =============================================================================
# Configuration Space
# =============================================================================

CONFIGURATION_SPACE = {
    # Free models available via GitHub Copilot CLI
    "model": ["gpt-4.1", "gpt-4o", "gpt-5-mini", "grok-code-fast-1"],
    "prompt_strategy": ["direct", "chain_of_thought", "checklist"],
}


# =============================================================================
# Helper Functions
# =============================================================================

def is_mock_mode() -> bool:
    """Check if mock mode is enabled via environment variable."""
    return os.environ.get("TRAIGENT_MOCK_LLM", "").lower() in ("true", "1", "yes")


def parse_json_issues(response: str) -> list[dict[str, str]]:
    """Parse JSON issues from LLM response with error handling.

    Args:
        response: Raw LLM response string

    Returns:
        List of issue dicts, empty list on parse failure
    """
    if not response:
        return []

    response = response.strip()

    # Try direct parse
    try:
        parsed = json.loads(response)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from response
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    return []


def call_llm_openai(prompt: str, model: str, temperature: float) -> str:
    """Call LLM via OpenAI-compatible API.

    Supports:
    - OpenAI API (OPENAI_API_KEY)
    - OpenRouter (OPENROUTER_API_KEY)
    - Any OpenAI-compatible endpoint (LLM_API_BASE + LLM_API_KEY)

    Args:
        prompt: The prompt to send
        model: Model name (e.g., gpt-4o, gpt-4.1)
        temperature: Sampling temperature

    Returns:
        LLM response string
    """
    # Determine API configuration
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY")
    api_base = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")

    if os.environ.get("OPENROUTER_API_KEY"):
        api_base = "https://openrouter.ai/api/v1"
        api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        print("Warning: No API key found. Set OPENAI_API_KEY, OPENROUTER_API_KEY, or LLM_API_KEY")
        return ""

    # Map model names to API-compatible names
    model_map = {
        "gpt-4.1": "gpt-4-1106-preview",
        "gpt-5-mini": "gpt-4o-mini",
        "grok-code-fast-1": "gpt-4o",  # Fallback
    }
    api_model = model_map.get(model, model)

    # Build request
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": "You are a code review expert. Respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 1000,
    }

    try:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except HTTPError as e:
        print(f"API error: {e.code} - {e.reason}")
        try:
            error_body = e.read().decode("utf-8")
            print(f"Error details: {error_body[:500]}")
        except Exception:
            pass
        return ""
    except URLError as e:
        print(f"Network error: {e}")
        return ""
    except Exception as e:
        print(f"LLM call error: {e}")
        return ""


def generate_mock_issues(function_code: str) -> list[dict[str, str]]:
    """Generate mock issues based on simple heuristics.

    Used in mock mode to simulate LLM responses without API calls.
    """
    issues = []
    code_lower = function_code.lower()

    # Check for missing docstring
    if '"""' not in function_code and "'''" not in function_code:
        issues.append({
            "issue_type": "MISSING_DOCS",
            "description": "Function lacks docstring",
        })

    # Check for broad exception handling
    if "except exception" in code_lower or "except:" in code_lower:
        issues.append({
            "issue_type": "BROAD_EXCEPTION",
            "description": "Overly broad exception handling",
        })

    # Check for TODO comments
    if "todo" in code_lower or "fixme" in code_lower or "hack" in code_lower:
        issues.append({
            "issue_type": "TODO_KNOWN_ISSUE",
            "description": "Contains TODO/FIXME/HACK comment",
        })

    # Check for global state modification
    if "global " in code_lower or "_global" in code_lower:
        issues.append({
            "issue_type": "SIDE_EFFECT",
            "description": "Modifies global state",
        })

    # Check for deep nesting (simple heuristic)
    if code_lower.count("    if ") > 3 or code_lower.count("        ") > 5:
        issues.append({
            "issue_type": "COMPLEXITY",
            "description": "Deep nesting detected",
        })

    # Check for long functions
    lines = function_code.strip().split("\n")
    if len(lines) > 50:
        issues.append({
            "issue_type": "PRINCIPLE_VIOLATION",
            "description": "Function exceeds 50 lines",
        })

    # Check for many parameters
    first_line = function_code.split("\n")[0] if function_code else ""
    param_count = first_line.count(",") + 1 if "(" in first_line else 0
    if param_count > 5:
        issues.append({
            "issue_type": "API_DESIGN",
            "description": f"Too many parameters ({param_count})",
        })

    return issues


# =============================================================================
# Main Agent
# =============================================================================

@traigent.optimize(
    configuration_space=CONFIGURATION_SPACE,
    objectives=["detection_f1", "cost"],
    metric_functions={"detection_f1": detection_f1_metric},
    evaluation=EvaluationOptions(eval_dataset=str(DATASET_PATH)),
    execution=ExecutionOptions(execution_mode="edge_analytics"),
)
def code_review_agent(
    function_code: str,
    function_name: str,
    source_file: str,
) -> list[dict[str, str]]:
    """Analyze a Python function for code quality issues.

    This agent reviews Python source code and identifies issues across
    10 categories: missing docs, implicit assumptions, side effects,
    complexity, broad exceptions, principle violations, TODOs, type handling,
    API design, and threading issues.

    Args:
        function_code: The complete Python function source code
        function_name: Name of the function being analyzed
        source_file: Path to the source file containing the function

    Returns:
        List of dicts: [{"issue_type": "CATEGORY", "description": "..."}]
    """
    # Get current configuration
    config = traigent.get_config()

    model = config.get("model", "gpt-4o")
    prompt_strategy = config.get("prompt_strategy", "direct")

    # Mock mode handling
    if is_mock_mode():
        return generate_mock_issues(function_code)

    # Build prompt using selected strategy
    prompt_template = PROMPT_STRATEGIES.get(prompt_strategy, DIRECT_PROMPT)
    prompt = prompt_template.format(
        function_code=function_code,
        function_name=function_name,
        source_file=source_file,
    )

    # Call LLM via OpenAI-compatible API (temperature fixed at 0.0 for determinism)
    response = call_llm_openai(prompt, model, temperature=0.0)

    # Parse response to list of issues
    return parse_json_issues(response)


# =============================================================================
# Optimization Runner
# =============================================================================

async def run_optimization(max_trials: int = 24, max_examples: int = 25) -> None:
    """Run the code review agent optimization.

    Args:
        max_trials: Maximum number of optimization trials
        max_examples: Maximum examples to evaluate per trial
    """
    print("=" * 60)
    print("Code Review Agent - Traigent Optimization")
    print("=" * 60)

    mock_mode = is_mock_mode()
    print(f"\nMock Mode: {'Enabled' if mock_mode else 'Disabled'}")

    if not mock_mode:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY")
        if api_key:
            print("\nAPI Key: Configured")
            api_base = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
            if os.environ.get("OPENROUTER_API_KEY"):
                api_base = "https://openrouter.ai/api/v1"
            print(f"API Base: {api_base}")
        else:
            print("\nWARNING: No API key found!")
            print("Set one of: OPENAI_API_KEY, OPENROUTER_API_KEY, or LLM_API_KEY")
            print("Falling back to mock mode...")
            os.environ["TRAIGENT_MOCK_LLM"] = "true"
            mock_mode = True

    print("\nConfiguration Space:")
    print("  - Models: gpt-4.1, gpt-4o, gpt-5-mini, grok-code-fast-1")
    print("  - Prompt Strategy: direct, chain_of_thought, checklist")
    print(f"\nMax Trials: {max_trials}")
    print(f"Max Examples: {max_examples}")
    print("\nObjectives: detection_f1, cost")
    print("-" * 60)

    # Check if dataset exists
    if not DATASET_PATH.exists():
        print(f"\nERROR: Dataset not found at {DATASET_PATH}")
        print("Please ensure the dataset file exists.")
        return

    print("\nStarting optimization...")

    try:
        results = await code_review_agent.optimize(
            algorithm="tpe",
            max_trials=max_trials,
            max_examples=max_examples,
        )

        print("\n" + "=" * 60)
        print("OPTIMIZATION RESULTS")
        print("=" * 60)
        print("\nBest Configuration:")
        for key, value in results.best_config.items():
            print(f"  {key}: {value}")
        print(f"\nBest F1 Score: {results.best_score:.4f}")

        # Apply best config
        code_review_agent.apply_best_config(results)
        print("\nBest configuration applied!")

    except Exception as e:
        print(f"\nOptimization error: {e}")
        import traceback
        traceback.print_exc()


def demo_agent() -> None:
    """Demonstrate the agent with a sample function."""
    print("=" * 60)
    print("Code Review Agent - Demo Mode")
    print("=" * 60)

    # Sample function with multiple issues
    sample_code = '''def process_data(data, config, options, flags, settings, extra):
    # TODO: Add proper validation
    global _CACHE
    try:
        for item in data:
            if item:
                if item.get("value"):
                    if item["value"] > 0:
                        _CACHE[item["id"]] = item
    except Exception as e:
        return None'''

    print("\nSample Function:")
    print("-" * 40)
    print(sample_code)
    print("-" * 40)

    print("\nAnalyzing (mock mode)...")
    issues = generate_mock_issues(sample_code)

    print(f"\nIssues Found ({len(issues)}):")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. [{issue['issue_type']}] {issue['description']}")


def test_api_connection() -> None:
    """Test the API connection with a simple prompt."""
    print("=" * 60)
    print("Testing API Connection")
    print("=" * 60)

    # Check environment
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        print("\nERROR: No API key found!")
        print("Set one of:")
        print("  - OPENAI_API_KEY (for OpenAI)")
        print("  - OPENROUTER_API_KEY (for OpenRouter)")
        print("  - LLM_API_KEY + LLM_API_BASE (for custom endpoint)")
        return

    api_base = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
    if os.environ.get("OPENROUTER_API_KEY"):
        api_base = "https://openrouter.ai/api/v1"

    print(f"\nAPI Key: {'*' * 10}{api_key[-4:]}")
    print(f"API Base: {api_base}")

    # Test prompt
    test_prompt = 'Return this exact JSON: [{"issue_type": "TEST", "description": "API working"}]'
    print(f"\nSending test prompt...")

    response = call_llm_openai(test_prompt, "gpt-4o", 0.0)

    if response:
        print(f"\nResponse received!")
        print(f"Raw: {response[:200]}...")
        issues = parse_json_issues(response)
        if issues:
            print(f"\nParsed issues: {issues}")
            print("\nAPI connection successful!")
        else:
            print(f"\nWarning: Could not parse response as JSON")
    else:
        print("\nERROR: No response received. Check API key and network.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Code Review Agent - LLM-powered code quality analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run optimization with mock mode
  TRAIGENT_MOCK_LLM=true python code_review_agent.py

  # Run optimization with OpenAI API
  OPENAI_API_KEY=sk-... python code_review_agent.py --max-trials 24

  # Test API connection
  OPENAI_API_KEY=sk-... python code_review_agent.py --test-api

  # Demo mode
  python code_review_agent.py --demo

Environment Variables:
  TRAIGENT_MOCK_LLM  Set to 'true' for mock mode (no API calls)
  OPENAI_API_KEY     OpenAI API key
  OPENROUTER_API_KEY OpenRouter API key
  LLM_API_KEY        Custom API key (with LLM_API_BASE)
  LLM_API_BASE       Custom API base URL
        """,
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=24,
        help="Maximum number of optimization trials (default: 24)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=25,
        help="Maximum examples to evaluate per trial (default: 25)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode with a sample function",
    )
    parser.add_argument(
        "--test-api",
        action="store_true",
        help="Test the API connection and exit",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.test_api:
        test_api_connection()
    elif args.demo:
        demo_agent()
    else:
        asyncio.run(run_optimization(
            max_trials=args.max_trials,
            max_examples=args.max_examples,
        ))


if __name__ == "__main__":
    main()
