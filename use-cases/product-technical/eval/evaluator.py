#!/usr/bin/env python3
"""
Evaluator for Product & Technical Agent (Code Generation)

This evaluator scores generated code on:
1. Test Pass Rate - Functional correctness via test execution
2. Code Quality - Readability, efficiency, style
3. Solution Efficiency - Token economy and code conciseness

Supports two modes:
- MOCK MODE (default): Uses test execution for objective evaluation
- REAL MODE: Uses LLM to generate code, then tests it

Usage:
  Mock mode: python evaluator.py  (default, uses heuristics)
  Real mode: TRAIGENT_MOCK_MODE=false python evaluator.py  (requires OPENAI_API_KEY)
"""

import ast
import os
import re
from dataclasses import dataclass
from typing import Any

# ============================================================================
# PROMPTS FOR REAL LLM MODE
# ============================================================================

# Prompt for the code generation agent
AGENT_PROMPT = """You are a Python code generator. Write a function that solves the given task.

TASK: {task}
FUNCTION NAME: {function_name}

Requirements:
1. Write clean, efficient Python code
2. Use type hints where appropriate
3. Handle edge cases
4. The function should be self-contained (no external dependencies except standard library)

Return ONLY the Python function code, nothing else. No explanations, no markdown.
Example output:
def {function_name}(n: int) -> bool:
    # implementation here
    return result
"""


@dataclass
class CodeEvaluationResult:
    """Result of code evaluation."""

    test_pass_rate: float  # 0-1 scale
    tests_passed: int
    tests_total: int
    code_quality: float  # 0-1 scale
    complexity_score: float  # 0-1 (1 = simple)
    readability_score: float  # 0-1
    efficiency_score: float  # 0-1
    overall_score: float
    errors: list[str]


class CodeEvaluator:
    """
    Evaluator for code generation agent.

    Evaluates generated code by running tests and analyzing quality.
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.weights = {
            "test_pass_rate": 0.5,
            "code_quality": 0.3,
            "efficiency": 0.2,
        }

    def __call__(
        self,
        prediction: dict[str, Any] | str,
        expected: str | None,
        input_data: dict[str, Any],
    ) -> dict[str, float]:
        """
        Evaluate generated code.

        Args:
            prediction: The generated code (dict or string)
            expected: Reference solution (if available)
            input_data: Input data containing test cases

        Returns:
            Dictionary of metric scores
        """
        # Parse prediction
        if isinstance(prediction, str):
            code = prediction
            function_name = input_data.get("function_name", "solution")
        else:
            code = prediction.get("code", "")
            function_name = prediction.get(
                "function_name", input_data.get("function_name", "solution")
            )

        # Get test cases
        test_cases = input_data.get("test_cases", [])
        reference_solution = input_data.get("reference_solution", expected)

        # Run tests
        test_result = self._run_tests(code, function_name, test_cases)

        # Evaluate code quality
        quality_score = self._evaluate_quality(code)

        # Evaluate efficiency
        efficiency_score = self._evaluate_efficiency(code, reference_solution)

        # Calculate overall score
        overall = (
            self.weights["test_pass_rate"] * test_result["pass_rate"]
            + self.weights["code_quality"] * quality_score
            + self.weights["efficiency"] * efficiency_score
        )

        return {
            "test_pass_rate": test_result["pass_rate"],
            "tests_passed": test_result["passed"],
            "tests_total": test_result["total"],
            "code_quality": quality_score,
            "efficiency": efficiency_score,
            "overall": overall,
        }

    def _run_tests(
        self,
        code: str,
        function_name: str,
        test_cases: list[dict],
    ) -> dict[str, Any]:
        """
        Execute test cases against the generated code.

        Returns:
            Dictionary with pass_rate, passed, total, and errors
        """
        if not test_cases:
            return {"pass_rate": 1.0, "passed": 0, "total": 0, "errors": []}

        # Create execution namespace
        namespace = {}
        errors = []

        # Try to execute the code
        try:
            exec(code, namespace)
        except SyntaxError as e:
            return {
                "pass_rate": 0.0,
                "passed": 0,
                "total": len(test_cases),
                "errors": [f"Syntax error: {e}"],
            }
        except Exception as e:
            return {
                "pass_rate": 0.0,
                "passed": 0,
                "total": len(test_cases),
                "errors": [f"Execution error: {e}"],
            }

        # Get the function
        if function_name not in namespace:
            return {
                "pass_rate": 0.0,
                "passed": 0,
                "total": len(test_cases),
                "errors": [f"Function '{function_name}' not found in code"],
            }

        func = namespace[function_name]

        # Run test cases
        passed = 0
        for i, test in enumerate(test_cases):
            try:
                input_args = test.get("input", [])
                expected = test.get("expected")

                # Call function with arguments
                if isinstance(input_args, list):
                    result = func(*input_args)
                else:
                    result = func(input_args)

                # Compare result
                if self._compare_results(result, expected):
                    passed += 1
                else:
                    errors.append(f"Test {i+1}: Expected {expected}, got {result}")

            except Exception as e:
                errors.append(f"Test {i+1} raised exception: {e}")

        return {
            "pass_rate": passed / len(test_cases) if test_cases else 1.0,
            "passed": passed,
            "total": len(test_cases),
            "errors": errors[:5],  # Limit to first 5 errors
        }

    def _compare_results(self, actual: Any, expected: Any) -> bool:
        """Compare actual and expected results with type flexibility."""
        # Direct equality
        if actual == expected:
            return True

        # Handle None/null comparison
        if actual is None and expected is None:
            return True
        if expected is None or actual is None:
            return False

        # Handle list comparison (order may matter)
        if isinstance(actual, list) and isinstance(expected, list):
            if len(actual) != len(expected):
                return False
            return all(self._compare_results(a, e) for a, e in zip(actual, expected))

        # Handle dict comparison
        if isinstance(actual, dict) and isinstance(expected, dict):
            if set(actual.keys()) != set(expected.keys()):
                return False
            return all(
                self._compare_results(actual[k], expected[k]) for k in actual.keys()
            )

        # Handle float comparison with tolerance
        if isinstance(actual, float) and isinstance(expected, float):
            return abs(actual - expected) < 1e-9

        # Handle set comparison
        if isinstance(actual, set) and isinstance(expected, (set, list)):
            return actual == set(expected)

        return False

    def _evaluate_quality(self, code: str) -> float:
        """
        Evaluate code quality.

        Considers:
        - Syntax validity
        - Code complexity
        - Naming conventions
        - Style

        Returns:
            Score between 0 and 1
        """
        if not code.strip():
            return 0.0

        score = 0.5  # Start at neutral

        # Check syntax validity
        try:
            ast.parse(code)
            score += 0.2
        except SyntaxError:
            return 0.1  # Syntax error is very bad

        # Check for good practices
        # Docstrings
        if '"""' in code or "'''" in code:
            score += 0.1

        # Type hints
        if "->" in code or ": " in code:
            score += 0.05

        # Reasonable line lengths
        lines = code.split("\n")
        long_lines = sum(1 for line in lines if len(line) > 100)
        if long_lines == 0:
            score += 0.05
        elif long_lines > len(lines) * 0.3:
            score -= 0.1

        # Check for common anti-patterns
        # Excessive nesting
        max_indent = (
            max((len(line) - len(line.lstrip())) // 4 for line in lines if line.strip())
            if lines
            else 0
        )
        if max_indent <= 3:
            score += 0.05
        elif max_indent > 5:
            score -= 0.1

        # Magic numbers (excluding 0, 1, 2)
        magic_numbers = re.findall(r"\b[3-9]\d*\b", code)
        # Filter out common acceptable numbers
        magic_numbers = [n for n in magic_numbers if n not in ["10", "100", "1000"]]
        if len(magic_numbers) <= 2:
            score += 0.05

        return max(0.0, min(1.0, score))

    def _evaluate_efficiency(
        self,
        code: str,
        reference: str | None,
    ) -> float:
        """
        Evaluate solution efficiency.

        Considers:
        - Code length vs reference
        - Algorithmic patterns

        Returns:
            Score between 0 and 1
        """
        if not code.strip():
            return 0.0

        score = 0.5

        # Count lines of actual code (not comments/blanks)
        code_lines = [
            line
            for line in code.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        num_lines = len(code_lines)

        # Prefer concise solutions
        if num_lines <= 5:
            score += 0.2
        elif num_lines <= 10:
            score += 0.1
        elif num_lines > 30:
            score -= 0.2

        # Compare to reference if available
        if reference:
            ref_lines = [
                line
                for line in reference.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            ref_num = len(ref_lines)

            # Ratio of code length
            if ref_num > 0:
                ratio = num_lines / ref_num
                if 0.8 <= ratio <= 1.2:
                    score += 0.15
                elif 0.5 <= ratio <= 2.0:
                    score += 0.05
                else:
                    score -= 0.1

        # Check for efficient patterns
        # List comprehensions
        if "[" in code and "for" in code and "]" in code:
            score += 0.05

        # Generator expressions
        if "(" in code and "for" in code and ")" in code:
            score += 0.05

        # Built-in functions usage
        builtins_used = [
            "sum(",
            "max(",
            "min(",
            "len(",
            "sorted(",
            "zip(",
            "enumerate(",
        ]
        if any(b in code for b in builtins_used):
            score += 0.05

        return max(0.0, min(1.0, score))


def is_mock_mode() -> bool:
    """Check if running in mock mode."""
    return os.environ.get("TRAIGENT_MOCK_MODE", "true").lower() == "true"


def run_optimization(num_configs: int = 5, num_examples: int = 10):
    """Run optimization testing different code generation configurations."""
    from openai import OpenAI

    client = OpenAI()
    dataset = load_dataset()[:num_examples]
    evaluator = CodeEvaluator()

    configs = [
        {
            "name": "baseline",
            "temperature": 0.0,
            "instruction": "Write clean Python code.",
        },
        {
            "name": "creative",
            "temperature": 0.7,
            "instruction": "Write elegant, creative Python code.",
        },
        {
            "name": "explicit",
            "temperature": 0.0,
            "instruction": "Write explicit, well-documented Python code with type hints.",
        },
        {
            "name": "minimal",
            "temperature": 0.2,
            "instruction": "Write minimal, concise Python code.",
        },
        {
            "name": "defensive",
            "temperature": 0.1,
            "instruction": "Write defensive Python code handling edge cases.",
        },
    ][:num_configs]

    print("\n" + "=" * 70)
    print("OPTIMIZATION RUN: Testing Different Code Generation Configs")
    print("=" * 70)
    print(
        f"\nConfigs: {num_configs}, Examples: {num_examples}, Total calls: {num_configs * num_examples}"
    )

    results = []
    for config in configs:
        print(f"\n--- Config: {config['name']} (temp={config['temperature']}) ---")
        scores = []

        for i, entry in enumerate(dataset):
            input_data = entry.get("input", {})
            task = input_data.get("task", "")
            func_name = input_data.get("function_name", "solution")
            test_cases = entry.get("test_cases", [])

            prompt = f"""{config['instruction']}

Task: {task}
Function name: {func_name}

Return ONLY the Python function code, no explanations."""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config["temperature"],
                )
                code = response.choices[0].message.content
                # Strip markdown
                if "```" in code:
                    code = re.sub(r"```python\n?", "", code)
                    code = re.sub(r"```\n?", "", code)

                result = evaluator(
                    {"code": code, "function_name": func_name},
                    None,
                    {"function_name": func_name, "test_cases": test_cases},
                )
                scores.append(result)
                passed = int(result.get("tests_passed", 0))
                total = int(result.get("tests_total", 0))
                print(
                    f"  [{i+1}/{num_examples}] tests={passed}/{total} quality={result['code_quality']:.2f}"
                )
            except Exception as e:
                print(f"  [{i+1}/{num_examples}] Error: {e}")
                scores.append({"test_pass_rate": 0, "code_quality": 0, "overall": 0})

        avg_pass = sum(s["test_pass_rate"] for s in scores) / len(scores)
        avg_quality = sum(s["code_quality"] for s in scores) / len(scores)
        results.append(
            {
                "config": config["name"],
                "temp": config["temperature"],
                "pass_rate": avg_pass,
                "quality": avg_quality,
                "overall": (avg_pass * 0.7 + avg_quality * 0.3),
            }
        )

    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print(
        f"\n{'Config':<12} {'Temp':<6} {'Pass Rate':<10} {'Quality':<10} {'Overall':<10}"
    )
    print("-" * 48)
    for r in sorted(results, key=lambda x: x["overall"], reverse=True):
        print(
            f"{r['config']:<12} {r['temp']:<6.1f} {r['pass_rate']*100:>5.0f}%      {r['quality']:.3f}      {r['overall']:.3f}"
        )

    best = max(results, key=lambda x: x["overall"])
    print("-" * 48)
    print(f"🏆 BEST: {best['config']} (score={best['overall']:.3f})")
    print("=" * 70)
    return results


def generate_code_with_llm(task: str, function_name: str) -> str:
    """Generate code using LLM (real mode only)."""
    try:
        from openai import OpenAI

        client = OpenAI()
        prompt = AGENT_PROMPT.format(task=task, function_name=function_name)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        code = response.choices[0].message.content
        # Strip markdown code blocks if present
        if code.startswith("```"):
            code = re.sub(r"^```\w*\n", "", code)
            code = re.sub(r"\n```$", "", code)
        return code.strip()
    except Exception as e:
        return f"# Error generating code: {e}"


def load_dataset() -> list[dict]:
    """Load the coding tasks dataset."""
    import json
    from pathlib import Path

    dataset_path = Path(__file__).parent.parent / "datasets" / "coding_tasks.jsonl"
    if not dataset_path.exists():
        return []

    entries = []
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def print_score_bar(label: str, score: float, max_score: float = 1.0, width: int = 20):
    """Print a visual score bar."""
    normalized = min(score / max_score, 1.0)
    filled = int(normalized * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = score * 100 if max_score == 1.0 else score
    print(f"  {label:<18} {bar} {pct:.0f}%")


def demo_evaluator():
    """Demo the Product & Technical Agent evaluator with clear input/output examples."""
    mock_mode = is_mock_mode()

    print("=" * 70)
    print("PRODUCT & TECHNICAL AGENT - Evaluator Demo")
    print("=" * 70)
    print(
        f"\nMODE: {'MOCK (test execution)' if mock_mode else 'REAL (LLM code generation)'}"
    )

    print(
        """
WHAT THIS AGENT DOES:
  A code generation agent that writes Python functions based on task
  descriptions. Given a task like "write a function to check if prime",
  it generates working Python code.

HOW IT'S EVALUATED:
  The evaluator ACTUALLY RUNS the generated code against test cases!
  This is the most objective evaluation - either the code works or it doesn't."""
    )
    if not mock_mode:
        print("  REAL MODE: LLM generates code, then we run tests against it.")
    print()

    # Load and show dataset info
    dataset = load_dataset()
    print(f"DATASET: {len(dataset)} coding tasks in coding_tasks.jsonl")

    if dataset:
        print("\n" + "-" * 70)
        print("SAMPLE DATA (first 2 entries):")
        print("-" * 70)
        for i, entry in enumerate(dataset[:2]):
            input_data = entry.get("input", {})
            task = input_data.get("task", "")
            func_name = input_data.get("function_name", "unknown")
            test_cases = entry.get("test_cases", [])
            ref_solution = entry.get("reference_solution", "")

            print(f"\n[Entry {i+1}]")
            print("  INPUT (task description):")
            print(f'    Task: "{task}"')
            print(f"    Function name: {func_name}()")
            print("\n  OUTPUT (expected code):")
            # Show first line of reference solution
            first_line = ref_solution.split("\n")[0] if ref_solution else "N/A"
            print(f"    {first_line}")
            print("    ... (full solution in dataset)")
            print(f"\n  TEST CASES ({len(test_cases)} tests):")
            for tc in test_cases[:2]:
                print(f"    {func_name}({tc['input']}) → {tc['expected']}")
            if len(test_cases) > 2:
                print(f"    ... and {len(test_cases) - 2} more tests")

    evaluator = CodeEvaluator()

    print("\n" + "=" * 70)
    print("HOW SCORING WORKS:")
    print("=" * 70)
    print(
        """
The evaluator measures:

  - Test Pass Rate:  How many test cases pass? (0-100%)
                     Code is executed in a sandbox - this is objective!

  - Code Quality:    Is the code well-structured? (checked via AST analysis)
                     - Valid syntax
                     - Has docstrings/type hints
                     - Not too deeply nested

  - Efficiency:      Is the solution concise vs the reference solution?
                     (Fewer lines is better, if tests still pass)
"""
    )

    print("=" * 70)
    print("EVALUATION EXAMPLES:")
    print("=" * 70)

    # Test case 1: Correct solution
    code1 = """def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True"""

    print("\n[CORRECT CODE] - All tests pass")
    print("-" * 70)
    print('Task: "Write a function to check if a number is prime"')
    print("\nGenerated Code:")
    for line in code1.strip().split("\n"):
        print(f"  {line}")

    result = evaluator(
        prediction={"code": code1, "function_name": "is_prime"},
        expected=None,
        input_data={
            "function_name": "is_prime",
            "test_cases": [
                {"input": [2], "expected": True},
                {"input": [3], "expected": True},
                {"input": [4], "expected": False},
                {"input": [17], "expected": True},
                {"input": [1], "expected": False},
            ],
        },
    )
    print(
        f"\nTest Results: {int(result['tests_passed'])}/{int(result['tests_total'])} passed"
    )
    print("  is_prime(2) = True  ✓")
    print("  is_prime(4) = False ✓")
    print("  is_prime(17) = True ✓")
    print("\nScores:")
    print_score_bar("Test Pass Rate", result["test_pass_rate"])
    print_score_bar("Code Quality", result["code_quality"])
    print_score_bar("Efficiency", result["efficiency"])
    print(f"  {'─' * 40}")
    print_score_bar("OVERALL", result["overall"])

    # Test case 2: Buggy solution
    code2 = "def is_prime(n: int) -> bool:\n    return n > 1  # Bug!"

    print("\n[BUGGY CODE] - Looks simple but fails edge cases")
    print("-" * 70)
    print('Task: "Write a function to check if a number is prime"')
    print("\nGenerated Code:")
    print("  def is_prime(n: int) -> bool:")
    print("      return n > 1  # Naive implementation!")

    result = evaluator(
        prediction={"code": code2, "function_name": "is_prime"},
        expected=None,
        input_data={
            "function_name": "is_prime",
            "test_cases": [
                {"input": [2], "expected": True},
                {"input": [4], "expected": False},
                {"input": [17], "expected": True},
            ],
        },
    )
    print(
        f"\nTest Results: {int(result['tests_passed'])}/{int(result['tests_total'])} passed"
    )
    print("  is_prime(2) = True  ✓")
    print("  is_prime(4) = True  ✗ Expected: False")
    print("  is_prime(17) = True ✓")
    print("\nScores:")
    print_score_bar("Test Pass Rate", result["test_pass_rate"])
    print("    ^ Test failures are the main issue!")
    print_score_bar("Code Quality", result["code_quality"])
    print_score_bar("Efficiency", result["efficiency"])
    print(f"  {'─' * 40}")
    print_score_bar("OVERALL", result["overall"])

    # Test case 3: Syntax error
    code3 = """def is_prime(n: int) -> bool
    return n > 1"""

    print("\n[SYNTAX ERROR] - Code won't compile")
    print("-" * 70)
    print('Task: "Write a function to check if a number is prime"')
    print("\nGenerated Code:")
    print("  def is_prime(n: int) -> bool    <- Missing colon!")
    print("      return n > 1")

    result = evaluator(
        prediction={"code": code3, "function_name": "is_prime"},
        expected=None,
        input_data={
            "function_name": "is_prime",
            "test_cases": [{"input": [2], "expected": True}],
        },
    )
    print("\nTest Results: FAILED - SyntaxError!")
    print("\nScores:")
    print_score_bar("Test Pass Rate", result["test_pass_rate"])
    print("    ^ Can't run tests on broken code")
    print_score_bar("Code Quality", result["code_quality"])
    print("    ^ Syntax error detected")
    print_score_bar("Efficiency", result["efficiency"])
    print(f"  {'─' * 40}")
    print_score_bar("OVERALL", result["overall"])

    # Test case 4: Verbose but correct
    code4 = """def is_prime(n: int) -> bool:
    \"\"\"Check if n is prime.\"\"\"
    if n is None:
        raise ValueError("Input cannot be None")
    if not isinstance(n, int):
        raise TypeError("Input must be integer")
    if n < 0:
        return False
    if n == 0:
        return False
    if n == 1:
        return False
    if n == 2:
        return True
    if n == 3:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, n, 2):
        if n % i == 0:
            return False
    return True"""

    print("\n[VERBOSE CODE] - Works but inefficient")
    print("-" * 70)
    print('Task: "Write a function to check if a number is prime"')
    print("\nGenerated Code: (20+ lines of overly defensive code)")
    print("  def is_prime(n: int) -> bool:")
    print('      """Check if n is prime."""')
    print("      if n is None: raise ValueError(...)")
    print("      if not isinstance(n, int): raise TypeError(...)")
    print("      ... (many more lines)")

    result = evaluator(
        prediction={"code": code4, "function_name": "is_prime"},
        expected=None,
        input_data={
            "function_name": "is_prime",
            "test_cases": [
                {"input": [2], "expected": True},
                {"input": [4], "expected": False},
                {"input": [17], "expected": True},
                {"input": [1], "expected": False},
            ],
        },
    )
    print(
        f"\nTest Results: {int(result['tests_passed'])}/{int(result['tests_total'])} passed"
    )
    print("\nScores:")
    print_score_bar("Test Pass Rate", result["test_pass_rate"])
    print("    ^ Tests pass - it works!")
    print_score_bar("Code Quality", result["code_quality"])
    print_score_bar("Efficiency", result["efficiency"])
    print("    ^ But it's way more code than needed")
    print(f"  {'─' * 40}")
    print_score_bar("OVERALL", result["overall"])

    # In real mode, run optimization
    if not mock_mode:
        run_optimization(num_configs=5, num_examples=10)

    print("\n" + "=" * 70)
    print("HOW TO RUN:")
    print("  Mock mode (heuristics): python evaluator.py  (default)")
    print(
        "  Real mode (LLM+optimize): TRAIGENT_MOCK_MODE=false OPENAI_API_KEY=sk-... python evaluator.py"
    )
    print("=" * 70)


if __name__ == "__main__":
    demo_evaluator()
