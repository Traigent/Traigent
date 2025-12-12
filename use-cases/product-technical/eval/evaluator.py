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


def evaluate_sample():
    """Test the evaluator with sample code."""
    evaluator = CodeEvaluator()

    # Test case 1: Correct solution
    print("Test 1: Correct Solution")
    print("-" * 40)
    result = evaluator(
        prediction={
            "code": '''def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True''',
            "function_name": "is_prime",
        },
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
    print(f"Test Pass Rate: {result['test_pass_rate']:.2f}")
    print(f"Code Quality: {result['code_quality']:.2f}")
    print(f"Efficiency: {result['efficiency']:.2f}")
    print(f"Overall: {result['overall']:.2f}")

    # Test case 2: Incorrect solution
    print("\nTest 2: Incorrect Solution")
    print("-" * 40)
    result = evaluator(
        prediction={
            "code": '''def is_prime(n: int) -> bool:
    return n > 1  # Wrong!''',
            "function_name": "is_prime",
        },
        expected=None,
        input_data={
            "function_name": "is_prime",
            "test_cases": [
                {"input": [2], "expected": True},
                {"input": [3], "expected": True},
                {"input": [4], "expected": False},
                {"input": [17], "expected": True},
            ],
        },
    )
    print(f"Test Pass Rate: {result['test_pass_rate']:.2f}")
    print(f"Code Quality: {result['code_quality']:.2f}")
    print(f"Efficiency: {result['efficiency']:.2f}")
    print(f"Overall: {result['overall']:.2f}")

    # Test case 3: Syntax error
    print("\nTest 3: Syntax Error")
    print("-" * 40)
    result = evaluator(
        prediction={
            "code": '''def is_prime(n: int) -> bool
    return n > 1''',  # Missing colon
            "function_name": "is_prime",
        },
        expected=None,
        input_data={
            "function_name": "is_prime",
            "test_cases": [{"input": [2], "expected": True}],
        },
    )
    print(f"Test Pass Rate: {result['test_pass_rate']:.2f}")
    print(f"Code Quality: {result['code_quality']:.2f}")
    print(f"Overall: {result['overall']:.2f}")

    # Test case 4: Verbose but correct
    print("\nTest 4: Verbose Solution")
    print("-" * 40)
    result = evaluator(
        prediction={
            "code": '''def factorial(n: int) -> int:
    """
    Calculate the factorial of a non-negative integer.

    Args:
        n: A non-negative integer

    Returns:
        The factorial of n
    """
    # Handle base cases
    if n == 0:
        return 1
    if n == 1:
        return 1

    # Initialize result
    result = 1

    # Calculate factorial iteratively
    current = 2
    while current <= n:
        result = result * current
        current = current + 1

    return result''',
            "function_name": "factorial",
        },
        expected="def factorial(n: int) -> int:\n    return 1 if n <= 1 else n * factorial(n - 1)",
        input_data={
            "function_name": "factorial",
            "test_cases": [
                {"input": [0], "expected": 1},
                {"input": [5], "expected": 120},
            ],
            "reference_solution": "def factorial(n: int) -> int:\n    return 1 if n <= 1 else n * factorial(n - 1)",
        },
    )
    print(f"Test Pass Rate: {result['test_pass_rate']:.2f}")
    print(f"Code Quality: {result['code_quality']:.2f}")
    print(f"Efficiency: {result['efficiency']:.2f}")
    print(f"Overall: {result['overall']:.2f}")


if __name__ == "__main__":
    evaluate_sample()
