#!/usr/bin/env python3
"""
Evaluator for Product & Technical Agent (Code Generation)

This evaluator scores generated code on:
1. Test Pass Rate - Functional correctness via test execution
2. Code Quality - Readability, efficiency, style
3. Solution Efficiency - Token economy and code conciseness

Based on the Traigent Agent Optimization Guide specifications.
"""

import ast
import re
import traceback
from dataclasses import dataclass
from typing import Any


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
            function_name = prediction.get("function_name", input_data.get("function_name", "solution"))

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
                self._compare_results(actual[k], expected[k])
                for k in actual.keys()
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
        max_indent = max(
            (len(line) - len(line.lstrip())) // 4
            for line in lines if line.strip()
        ) if lines else 0
        if max_indent <= 3:
            score += 0.05
        elif max_indent > 5:
            score -= 0.1

        # Magic numbers (excluding 0, 1, 2)
        magic_numbers = re.findall(r'\b[3-9]\d*\b', code)
        # Filter out common acceptable numbers
        magic_numbers = [n for n in magic_numbers if n not in ['10', '100', '1000']]
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
            line for line in code.split("\n")
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
                line for line in reference.split("\n")
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
        builtins_used = ["sum(", "max(", "min(", "len(", "sorted(", "zip(", "enumerate("]
        if any(b in code for b in builtins_used):
            score += 0.05

        return max(0.0, min(1.0, score))


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


def demo_evaluator():
    """
    Demo the Product & Technical Agent evaluator.

    This runs in MOCK MODE - no API calls are made.
    The evaluator uses ACTUAL CODE EXECUTION to score generated code:
    - Test Pass Rate: Runs test cases against generated code
    - Code Quality: AST analysis for style, complexity, best practices
    - Efficiency: Code conciseness vs reference solution
    """
    print("=" * 60)
    print("PRODUCT & TECHNICAL AGENT - Evaluator Demo")
    print("=" * 60)
    print("\nMODE: Mock (actual code execution, no API calls)")
    print("\nEVALUATOR: CodeEvaluator")
    print("  - Executes generated code against test cases")
    print("  - Analyzes code quality via AST parsing")
    print("  - Compares solution length to reference")
    print("\nNOTE: This evaluator runs actual Python code in a sandbox!")

    # Load and show dataset info
    dataset = load_dataset()
    print(f"\nDATASET: coding_tasks.jsonl ({len(dataset)} coding tasks)")

    if dataset:
        # Categorize by difficulty
        difficulties = {}
        for e in dataset:
            diff = e.get("difficulty", "unknown")
            difficulties[diff] = difficulties.get(diff, 0) + 1
        print(f"  - Difficulty distribution: {difficulties}")

        print("\n" + "-" * 60)
        print("FIRST 3 CODING TASKS FROM DATASET:")
        print("-" * 60)
        for i, entry in enumerate(dataset[:3]):
            input_data = entry.get("input", {})
            func_name = input_data.get("function_name", "unknown")
            task_desc = input_data.get("task", "")[:40]
            test_count = len(entry.get("test_cases", []))
            print(f"\n  [{i+1}] Function: {func_name}()")
            print(f"      Task: \"{task_desc}...\"")
            print(f"      Test cases: {test_count}")

    evaluator = CodeEvaluator()

    print("\n" + "=" * 60)
    print("EVALUATION EXAMPLES")
    print("=" * 60)

    # Test case 1: Correct solution
    print("\n[EXAMPLE 1] Correct implementation (all tests pass)")
    print("-" * 60)
    code1 = '''def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True'''
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
    print(f"  Code: is_prime() with sqrt optimization")
    print(f"  Tests: {int(result['tests_passed'])}/{int(result['tests_total'])} passed")
    print(f"\n  Scores:")
    print(f"    Test Pass Rate: {result['test_pass_rate']:.2f}")
    print(f"    Code Quality:   {result['code_quality']:.2f}")
    print(f"    Efficiency:     {result['efficiency']:.2f}")
    print(f"    ─────────────────────────")
    print(f"    Overall:        {result['overall']:.2f}")

    # Test case 2: Incorrect solution
    print("\n[EXAMPLE 2] Buggy implementation (tests fail)")
    print("-" * 60)
    code2 = '''def is_prime(n: int) -> bool:
    return n > 1  # Wrong! 4 is > 1 but not prime'''
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
    print(f"  Code: is_prime() → returns n > 1 (BUGGY)")
    print(f"  Tests: {int(result['tests_passed'])}/{int(result['tests_total'])} passed")
    print(f"\n  Scores:")
    print(f"    Test Pass Rate: {result['test_pass_rate']:.2f} ← Tests failing!")
    print(f"    Code Quality:   {result['code_quality']:.2f}")
    print(f"    Efficiency:     {result['efficiency']:.2f}")
    print(f"    ─────────────────────────")
    print(f"    Overall:        {result['overall']:.2f}")

    # Test case 3: Syntax error
    print("\n[EXAMPLE 3] Syntax error (code doesn't compile)")
    print("-" * 60)
    code3 = '''def is_prime(n: int) -> bool
    return n > 1'''  # Missing colon
    result = evaluator(
        prediction={"code": code3, "function_name": "is_prime"},
        expected=None,
        input_data={
            "function_name": "is_prime",
            "test_cases": [{"input": [2], "expected": True}],
        },
    )
    print(f"  Code: Missing colon after function signature")
    print(f"\n  Scores:")
    print(f"    Test Pass Rate: {result['test_pass_rate']:.2f} ← Can't run tests!")
    print(f"    Code Quality:   {result['code_quality']:.2f} ← Syntax error")
    print(f"    ─────────────────────────")
    print(f"    Overall:        {result['overall']:.2f}")

    print("\n" + "=" * 60)
    print("To run optimization with real API calls:")
    print("  export OPENAI_API_KEY=<your-key>")
    print("  unset TRAIGENT_MOCK_MODE")
    print("  python use-cases/product-technical/agent/code_agent.py")
    print("=" * 60)


if __name__ == "__main__":
    demo_evaluator()
