"""
Code Review Assistant Problem.

A challenging code analysis problem that tests the model's ability to:
1. Identify bugs, security issues, and performance problems
2. Suggest improvements for code quality and maintainability
3. Provide structured, actionable feedback
4. Balance thoroughness with practical priorities
"""

import sys
from typing import Any, Callable, Dict, List, Optional

import traigent
from traigent.evaluators.base import Dataset, EvaluationExample

try:
    from langchain.chains import LLMChain
    from langchain.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
except ImportError:
    print("Please install LangChain: pip install langchain langchain-openai")
    sys.exit(1)

from . import register_problem
from .base import BaseLangChainProblem, ProblemConfig, ProblemMetric


class CodeReviewProblem(BaseLangChainProblem):
    """
    Code review assistant problem.

    Tests the model's ability to perform comprehensive code analysis including
    bug detection, security issues, performance problems, and style improvements.
    """

    ISSUE_TYPES = [
        "bug",
        "security",
        "performance",
        "style",
        "maintainability",
        "documentation",
        "testing",
        "none",
    ]

    SEVERITY_LEVELS = ["critical", "high", "medium", "low", "info"]

    @classmethod
    def get_default_config(cls) -> ProblemConfig:
        """Get default configuration for this problem."""
        return ProblemConfig(
            name="code_review",
            description="Comprehensive code review assistant with bug detection and improvement suggestions",
            difficulty_level="Expert",
            dataset_size=15,
            model_configurations={
                "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
                "temperature": [0.1, 0.4],
                "max_tokens": [400, 600],
            },
            metrics=[
                ProblemMetric(
                    "issue_detection_rate",
                    "Rate of correctly identified issues",
                    True,
                    1.0,
                    ".1%",
                ),
                ProblemMetric(
                    "false_positive_rate",
                    "Rate of incorrectly flagged issues",
                    False,
                    0.8,
                    ".1%",
                ),
                ProblemMetric(
                    "severity_accuracy",
                    "Accuracy of severity assessment",
                    True,
                    0.9,
                    ".1%",
                ),
                ProblemMetric(
                    "suggestion_quality",
                    "Quality of improvement suggestions",
                    True,
                    1.1,
                    ".2f",
                ),
                ProblemMetric(
                    "coverage_completeness",
                    "Completeness of review coverage",
                    True,
                    0.7,
                    ".1%",
                ),
            ],
            optimization_objectives=["issue_detection_rate"],
            expected_model_ranking=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        )

    def __init__(self, config: Optional[ProblemConfig] = None):
        if config is None:
            config = self.get_default_config()
        super().__init__(config)

    def create_dataset(self) -> Dataset:
        """Create challenging code review dataset with various issue types."""
        examples_data = [
            # Security Issues
            {
                "code": """def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    cursor.execute(query)
    return cursor.fetchone() is not None""",
                "language": "python",
                "context": "User authentication function",
                "expected_issues": [
                    {
                        "type": "security",
                        "severity": "critical",
                        "description": "SQL injection vulnerability",
                        "suggestion": "Use parameterized queries to prevent SQL injection",
                    },
                    {
                        "type": "security",
                        "severity": "high",
                        "description": "Plain text password comparison",
                        "suggestion": "Hash passwords before storing and comparing",
                    },
                ],
                "difficulty": "medium",
            },
            # Performance Issues
            {
                "code": """def find_duplicates(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates""",
                "language": "python",
                "context": "Function to find duplicate items in a list",
                "expected_issues": [
                    {
                        "type": "performance",
                        "severity": "high",
                        "description": "O(n³) time complexity due to nested loops and 'in' check",
                        "suggestion": "Use a set or dictionary for O(n) time complexity",
                    },
                    {
                        "type": "style",
                        "severity": "low",
                        "description": "Could be more Pythonic",
                        "suggestion": "Consider using Counter from collections module",
                    },
                ],
                "difficulty": "hard",
            },
            # Memory/Resource Issues
            {
                "code": """def process_large_file(filename):
    with open(filename, 'r') as f:
        content = f.read()

    lines = content.split('\\n')
    processed = []
    for line in lines:
        if len(line) > 100:
            processed.append(line.upper())

    return processed""",
                "language": "python",
                "context": "Processing large text files",
                "expected_issues": [
                    {
                        "type": "performance",
                        "severity": "high",
                        "description": "Loads entire file into memory at once",
                        "suggestion": "Process file line by line for memory efficiency",
                    },
                    {
                        "type": "maintainability",
                        "severity": "medium",
                        "description": "Magic number 100 should be configurable",
                        "suggestion": "Make line length threshold a parameter",
                    },
                ],
                "difficulty": "medium",
            },
            # Null/Error Handling Issues
            {
                "code": """function calculateAverage(numbers) {
    let sum = 0;
    for (let i = 0; i < numbers.length; i++) {
        sum += numbers[i];
    }
    return sum / numbers.length;
}""",
                "language": "javascript",
                "context": "Calculate average of number array",
                "expected_issues": [
                    {
                        "type": "bug",
                        "severity": "medium",
                        "description": "Division by zero when array is empty",
                        "suggestion": "Check for empty array and handle appropriately",
                    },
                    {
                        "type": "bug",
                        "severity": "medium",
                        "description": "No validation for non-numeric values",
                        "suggestion": "Validate that array contains only numbers",
                    },
                ],
                "difficulty": "easy",
            },
            # Race Conditions/Concurrency
            {
                "code": """class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        current = self.count
        # Simulate some processing time
        time.sleep(0.001)
        self.count = current + 1

    def get_count(self):
        return self.count""",
                "language": "python",
                "context": "Thread-safe counter implementation",
                "expected_issues": [
                    {
                        "type": "bug",
                        "severity": "high",
                        "description": "Race condition in increment method",
                        "suggestion": "Use threading.Lock to ensure thread safety",
                    },
                    {
                        "type": "performance",
                        "severity": "low",
                        "description": "Unnecessary sleep in increment",
                        "suggestion": "Remove artificial delay unless needed for testing",
                    },
                ],
                "difficulty": "hard",
            },
            # Resource Management
            {
                "code": """def save_data(data, filename):
    file = open(filename, 'w')
    json.dump(data, file)
    # File is not closed explicitly""",
                "language": "python",
                "context": "Save data to JSON file",
                "expected_issues": [
                    {
                        "type": "bug",
                        "severity": "medium",
                        "description": "File handle not properly closed",
                        "suggestion": "Use 'with' statement for automatic file closure",
                    },
                    {
                        "type": "maintainability",
                        "severity": "low",
                        "description": "Missing error handling",
                        "suggestion": "Add try-except block for file operations",
                    },
                ],
                "difficulty": "easy",
            },
            # Complex Logic Bug
            {
                "code": """def binary_search(arr, target):
    left, right = 0, len(arr)

    while left < right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid

    return -1""",
                "language": "python",
                "context": "Binary search implementation",
                "expected_issues": [
                    {
                        "type": "bug",
                        "severity": "medium",
                        "description": "Incorrect initialization of right boundary",
                        "suggestion": "Initialize right = len(arr) - 1 for proper bounds",
                    },
                    {
                        "type": "documentation",
                        "severity": "low",
                        "description": "Missing docstring and type hints",
                        "suggestion": "Add documentation for parameters and return value",
                    },
                ],
                "difficulty": "very_hard",
            },
            # API Design Issues
            {
                "code": """class UserService:
    def get_user(self, user_id):
        user = db.query(f"SELECT * FROM users WHERE id = {user_id}")
        if user:
            return {
                'id': user[0],
                'name': user[1],
                'email': user[2],
                'password': user[3]  # Exposed password
            }
        return None""",
                "language": "python",
                "context": "User service API",
                "expected_issues": [
                    {
                        "type": "security",
                        "severity": "critical",
                        "description": "Password field exposed in response",
                        "suggestion": "Never return password in user data",
                    },
                    {
                        "type": "security",
                        "severity": "critical",
                        "description": "SQL injection vulnerability",
                        "suggestion": "Use parameterized queries",
                    },
                    {
                        "type": "maintainability",
                        "severity": "medium",
                        "description": "Magic index numbers for database fields",
                        "suggestion": "Use named fields or ORM mapping",
                    },
                ],
                "difficulty": "medium",
            },
            # No Issues (Good Code)
            {
                "code": """def calculate_factorial(n: int) -> int:
    \"\"\"Calculate factorial of a non-negative integer.

    Args:
        n: Non-negative integer

    Returns:
        Factorial of n

    Raises:
        ValueError: If n is negative
    \"\"\"
    if not isinstance(n, int) or n < 0:
        raise ValueError("Input must be a non-negative integer")

    if n <= 1:
        return 1

    result = 1
    for i in range(2, n + 1):
        result *= i

    return result""",
                "language": "python",
                "context": "Factorial calculation function",
                "expected_issues": [],
                "difficulty": "easy",
            },
        ]

        examples = []
        for i, data in enumerate(examples_data):
            example = EvaluationExample(
                input_data={
                    "code": data["code"],
                    "language": data["language"],
                    "context": data["context"],
                },
                expected_output={
                    "issues": data["expected_issues"],
                    "has_issues": len(data["expected_issues"]) > 0,
                },
                metadata={
                    "difficulty": data["difficulty"],
                    "example_id": f"code_{i+1:03d}",
                    "issue_count": len(data["expected_issues"]),
                    "language": data["language"],
                },
            )
            examples.append(example)

        return Dataset(
            examples=examples,
            name="Code Review Assistant",
            description=f"Code review with {len(examples)} examples covering security, performance, and maintainability issues",
        )

    def create_function(self) -> Callable:
        """Create the base code review function."""

        def code_reviewer(code: str, language: str, context: str) -> str:
            """Review code and identify issues with suggestions for improvement."""
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                model_kwargs={"max_tokens": 500},
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are an expert code reviewer. Analyze code for:
1. Bugs and logical errors
2. Security vulnerabilities
3. Performance issues
4. Style and maintainability problems
5. Documentation needs

For each issue found, specify:
- Type: bug, security, performance, style, maintainability, documentation, testing
- Severity: critical, high, medium, low, info
- Description: Brief explanation of the issue
- Suggestion: Specific improvement recommendation

If no issues found, respond with "No issues detected - code looks good!"

Format your response as structured text.""",
                    ),
                    (
                        "human",
                        """Review this {language} code:

Context: {context}

Code:
```{language}
{code}
```

Provide a thorough code review with specific, actionable feedback.""",
                    ),
                ]
            )

            chain = LLMChain(llm=llm, prompt=prompt_template)
            result = chain.invoke(
                {"code": code, "language": language, "context": context}
            )["text"]

            return result.strip()

        return code_reviewer

    def create_optimized_function(self) -> Callable:
        """Create the optimized code reviewer."""

        @traigent.optimize(
            eval_dataset=self.create_temporary_dataset_file(),
            objectives=self.get_optimization_objectives(),
            configuration_space=self.get_configuration_space(),
            auto_override_frameworks=True,
            framework_targets=["langchain_openai.ChatOpenAI"],
            execution_mode="edge_analytics",
        )
        def code_reviewer_optimized(code: str, language: str, context: str) -> str:
            """Optimized code reviewer."""
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # Will be overridden by TraiGent
                temperature=0.1,  # Will be overridden by TraiGent
                model_kwargs={"max_tokens": 500},  # Will be overridden by TraiGent
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are an expert code reviewer. Analyze code for:
1. Bugs and logical errors
2. Security vulnerabilities
3. Performance issues
4. Style and maintainability problems
5. Documentation needs

For each issue found, specify:
- Type: bug, security, performance, style, maintainability, documentation, testing
- Severity: critical, high, medium, low, info
- Description: Brief explanation of the issue
- Suggestion: Specific improvement recommendation

If no issues found, respond with "No issues detected - code looks good!"

Format your response as structured text.""",
                    ),
                    (
                        "human",
                        """Review this {language} code:

Context: {context}

Code:
```{language}
{code}
```

Provide a thorough code review with specific, actionable feedback.""",
                    ),
                ]
            )

            chain = LLMChain(llm=llm, prompt=prompt_template)
            result = chain.invoke(
                {"code": code, "language": language, "context": context}
            )["text"]

            return result.strip()

        return code_reviewer_optimized

    def evaluate_custom_metrics(
        self,
        outputs: List[Any],
        expected_outputs: List[Any],
        errors: List[Optional[str]],
    ) -> Dict[str, float]:
        """Compute code review specific metrics."""
        metrics = {}

        dataset = self.get_dataset()

        # Issue detection rate
        detected_issues = 0
        total_issues = 0
        false_positives = 0
        total_outputs = 0

        for i, (output, expected, error) in enumerate(
            zip(outputs, expected_outputs, errors)
        ):
            if (
                error is None
                and expected is not None
                and output
                and i < len(dataset.examples)
            ):
                total_outputs += 1
                expected_issues = expected.get("issues", [])
                total_issues += len(expected_issues)

                output_str = str(output).lower()

                # Count detected issues (simplified keyword matching)
                issue_keywords = {
                    "security": [
                        "security",
                        "sql injection",
                        "vulnerability",
                        "password",
                        "authentication",
                    ],
                    "performance": [
                        "performance",
                        "complexity",
                        "memory",
                        "optimization",
                        "slow",
                    ],
                    "bug": [
                        "bug",
                        "error",
                        "exception",
                        "null",
                        "undefined",
                        "race condition",
                    ],
                    "style": ["style", "pythonic", "readable", "naming"],
                    "maintainability": [
                        "maintainability",
                        "documentation",
                        "magic number",
                        "hardcoded",
                    ],
                }

                detected_count = 0
                for expected_issue in expected_issues:
                    issue_type = expected_issue["type"]
                    keywords = issue_keywords.get(issue_type, [issue_type])

                    # Check if any keyword for this issue type appears in output
                    if any(keyword in output_str for keyword in keywords):
                        detected_count += 1

                detected_issues += detected_count

                # Count false positives (issues detected when none expected)
                if len(expected_issues) == 0 and "no issues" not in output_str:
                    false_positives += 1

        metrics["issue_detection_rate"] = (
            detected_issues / total_issues if total_issues > 0 else 1.0
        )
        metrics["false_positive_rate"] = (
            false_positives / total_outputs if total_outputs > 0 else 0.0
        )

        # Severity accuracy (simplified)
        severity_correct = 0
        severity_total = 0
        for output, expected, error in zip(outputs, expected_outputs, errors):
            if error is None and expected is not None and output:
                expected_issues = expected.get("issues", [])
                output_str = str(output).lower()

                for issue in expected_issues:
                    severity_total += 1
                    if issue["severity"] in output_str:
                        severity_correct += 1

        metrics["severity_accuracy"] = (
            severity_correct / severity_total if severity_total > 0 else 0.0
        )

        # Suggestion quality (based on keyword presence)
        suggestion_scores = []
        for output, expected, error in zip(outputs, expected_outputs, errors):
            if error is None and expected is not None and output:
                output_str = str(output).lower()
                quality_indicators = [
                    "use",
                    "implement",
                    "consider",
                    "add",
                    "remove",
                    "replace",
                    "parameterized",
                    "validation",
                    "error handling",
                    "documentation",
                ]

                quality_score = sum(
                    1 for indicator in quality_indicators if indicator in output_str
                )
                suggestion_scores.append(min(quality_score / 3.0, 1.0))  # Normalize

        metrics["suggestion_quality"] = (
            sum(suggestion_scores) / len(suggestion_scores)
            if suggestion_scores
            else 0.0
        )

        # Coverage completeness
        metrics["coverage_completeness"] = metrics["issue_detection_rate"]

        return metrics


# Register this problem
register_problem("code_review", CodeReviewProblem)
