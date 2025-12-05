"""
Example Validator for TraiGent SDK.

This module provides validation for generated examples to ensure they meet
quality standards and have proper structure for each problem type.
"""

from typing import Any, Dict, List, Tuple


class ExampleValidator:
    """Validates generated examples for quality and correctness."""

    def __init__(self):
        """Initialize the validator with problem type requirements."""
        self.problem_type_requirements = {
            "classification": {
                "required_input_fields": ["text"],
                "output_type": str,
                "output_validation": self._validate_classification_output,
            },
            "generation": {
                "required_input_fields": ["text"],
                "output_type": str,
                "output_validation": self._validate_generation_output,
            },
            "reasoning": {
                "required_input_fields": ["problem"],
                "output_type": (dict, str),
                "output_validation": self._validate_reasoning_output,
            },
            "question_answering": {
                "required_input_fields": ["question"],
                "optional_input_fields": ["context"],
                "output_type": str,
                "output_validation": self._validate_qa_output,
            },
            "information_extraction": {
                "required_input_fields": ["text"],
                "output_type": dict,
                "output_validation": self._validate_extraction_output,
            },
            "summarization": {
                "required_input_fields": ["text"],
                "output_type": str,
                "output_validation": self._validate_summarization_output,
            },
            "ranking_retrieval": {
                "required_input_fields": ["query"],
                "output_type": (list, dict),
                "output_validation": self._validate_ranking_output,
            },
            "translation_transformation": {
                "required_input_fields": ["text"],
                "output_type": str,
                "output_validation": self._validate_transformation_output,
            },
            "code_generation": {
                "required_input_fields": ["description"],
                "output_type": str,
                "output_validation": self._validate_code_output,
            },
        }

        # Invalid placeholders that indicate incomplete examples
        self.invalid_placeholders = [
            "to be determined",
            "tbd",
            "todo",
            "placeholder",
            "sample output",
            "not specified",
            "[your output here]",
            "[expected output here]",
            "...",
        ]

    def validate_example(
        self, example: Dict[str, Any], problem_type: str, strict: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Validate a single example.

        Args:
            example: The example to validate
            problem_type: The problem type (e.g., "classification", "reasoning")
            strict: Whether to apply strict validation rules

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check basic structure
        if not isinstance(example, dict):
            return False, ["Example must be a dictionary"]

        # Check required fields
        if "input_data" not in example:
            issues.append("Missing 'input_data' field")

        if "expected_output" not in example:
            issues.append("Missing 'expected_output' field")

        if issues:
            return False, issues

        # Validate input data
        input_issues = self._validate_input_data(example["input_data"], problem_type)
        issues.extend(input_issues)

        # Validate output
        output_issues = self._validate_output(example["expected_output"], problem_type)
        issues.extend(output_issues)

        # Check for placeholders
        if self._contains_placeholder(example["expected_output"]):
            issues.append("Output contains invalid placeholder text")

        # Validate metadata if present
        if "metadata" in example:
            metadata_issues = self._validate_metadata(example["metadata"])
            issues.extend(metadata_issues)

        # Problem type specific validation
        if problem_type in self.problem_type_requirements:
            req = self.problem_type_requirements[problem_type]
            if "output_validation" in req:
                type_specific_issues = req["output_validation"](
                    example["expected_output"], example.get("input_data", {})
                )
                issues.extend(type_specific_issues)

        return len(issues) == 0, issues

    def validate_batch(
        self, examples: List[Dict[str, Any]], problem_type: str, min_examples: int = 1
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a batch of examples.

        Args:
            examples: List of examples to validate
            problem_type: The problem type
            min_examples: Minimum number of valid examples required

        Returns:
            Tuple of (is_valid, validation_report)
        """
        if not examples:
            return False, {
                "valid": False,
                "total_examples": 0,
                "valid_examples": 0,
                "issues": ["No examples provided"],
            }

        valid_examples = []
        invalid_examples = []
        all_issues = []

        for i, example in enumerate(examples):
            is_valid, issues = self.validate_example(example, problem_type)

            if is_valid:
                valid_examples.append(i)
            else:
                invalid_examples.append(i)
                all_issues.append({"example_index": i, "issues": issues})

        report = {
            "valid": len(valid_examples) >= min_examples,
            "total_examples": len(examples),
            "valid_examples": len(valid_examples),
            "invalid_examples": len(invalid_examples),
            "valid_indices": valid_examples,
            "invalid_indices": invalid_examples,
            "issues_by_example": all_issues,
        }

        return report["valid"], report

    def _validate_input_data(self, input_data: Any, problem_type: str) -> List[str]:
        """Validate input data structure."""
        issues = []

        if not isinstance(input_data, dict):
            return ["Input data must be a dictionary"]

        if problem_type in self.problem_type_requirements:
            req = self.problem_type_requirements[problem_type]

            # Check required fields
            for field in req.get("required_input_fields", []):
                if field not in input_data:
                    issues.append(f"Missing required input field: '{field}'")
                elif not input_data[field]:
                    issues.append(f"Empty value for required field: '{field}'")

        return issues

    def _validate_output(self, output: Any, problem_type: str) -> List[str]:
        """Validate output structure and type."""
        issues = []

        if output is None:
            return ["Output cannot be None"]

        if problem_type in self.problem_type_requirements:
            req = self.problem_type_requirements[problem_type]
            expected_type = req.get("output_type")

            if expected_type and not isinstance(output, expected_type):
                issues.append(
                    f"Output type mismatch: expected {expected_type}, "
                    f"got {type(output)}"
                )

        return issues

    def _validate_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """Validate metadata structure."""
        issues = []

        if not isinstance(metadata, dict):
            issues.append("Metadata must be a dictionary")

        return issues

    def _contains_placeholder(self, value: Any) -> bool:
        """Check if value contains placeholder text."""
        if isinstance(value, str):
            value_lower = value.lower().strip()
            return any(
                placeholder in value_lower for placeholder in self.invalid_placeholders
            )
        elif isinstance(value, dict):
            return any(self._contains_placeholder(v) for v in value.values())
        elif isinstance(value, list):
            return any(self._contains_placeholder(item) for item in value)
        return False

    # Problem type specific validation methods

    def _validate_classification_output(
        self, output: Any, input_data: Dict[str, Any]
    ) -> List[str]:
        """Validate classification output."""
        issues = []

        if not isinstance(output, str):
            issues.append("Classification output must be a string category")
        elif len(output) < 2:
            issues.append("Category name too short")
        elif len(output) > 50:
            issues.append("Category name too long (max 50 chars)")
        elif " " in output.strip():
            # Check if it looks like a sentence instead of a category
            if len(output.split()) > 3:
                issues.append("Output appears to be a sentence, not a category name")

        # Check for common invalid patterns
        if isinstance(output, str):
            output_lower = output.lower()
            if any(
                phrase in output_lower
                for phrase in ["the category is", "classified as", "this is"]
            ):
                issues.append(
                    "Output contains explanation text instead of just the category"
                )

        return issues

    def _validate_generation_output(
        self, output: Any, input_data: Dict[str, Any]
    ) -> List[str]:
        """Validate generation output."""
        issues = []

        if not isinstance(output, str):
            issues.append("Generation output must be a string")
        elif len(output) < 10:
            issues.append("Generated text too short (minimum 10 characters)")

        return issues

    def _validate_reasoning_output(
        self, output: Any, input_data: Dict[str, Any]
    ) -> List[str]:
        """Validate reasoning output."""
        issues = []

        if isinstance(output, dict):
            # Structured reasoning output
            if "answer" not in output:
                issues.append("Reasoning output missing 'answer' field")
            else:
                # Validate answer field
                answer = output["answer"]
                if not answer or (isinstance(answer, str) and len(answer.strip()) == 0):
                    issues.append("Answer field is empty")

            if "steps" in output:
                if not isinstance(output["steps"], list):
                    issues.append("'steps' must be a list")
                elif len(output["steps"]) == 0:
                    issues.append("'steps' list cannot be empty")
                elif len(output["steps"]) < 2:
                    issues.append("Reasoning should have at least 2 steps")
                else:
                    # Validate each step
                    for i, step in enumerate(output["steps"]):
                        if not isinstance(step, str):
                            issues.append(f"Step {i+1} must be a string")
                        elif len(step.strip()) < 5:
                            issues.append(f"Step {i+1} is too short")
            else:
                issues.append("Reasoning output missing 'steps' field")

            # Check for explanation if present
            if "explanation" in output and output["explanation"]:
                if not isinstance(output["explanation"], str):
                    issues.append("'explanation' must be a string")
        elif isinstance(output, str):
            # Simple string answer - should be dict for reasoning
            issues.append(
                "Reasoning output should be a structured object with steps, not a simple string"
            )
        else:
            issues.append("Reasoning output must be dict")

        return issues

    def _validate_qa_output(self, output: Any, input_data: Dict[str, Any]) -> List[str]:
        """Validate question-answering output."""
        issues = []

        if not isinstance(output, str):
            issues.append("Q&A output must be a string answer")
        elif len(output) < 2:
            issues.append("Answer too short")
        elif len(output) > 500:
            issues.append("Answer too long (max 500 chars)")

        # Check if answer is derived from context (if context provided)
        if (
            isinstance(input_data, dict)
            and "context" in input_data
            and "question" in input_data
        ):
            input_data["context"]
            question = input_data["question"]

            # Boolean questions should have yes/no answers
            if isinstance(question, str):
                question_lower = question.lower()
                if any(
                    q in question_lower
                    for q in ["is ", "are ", "does ", "do ", "can ", "will ", "should "]
                ):
                    # Likely a yes/no question
                    output_lower = output.lower()
                    if (
                        not any(ans in output_lower for ans in ["yes", "no"])
                        and len(output.split()) > 5
                    ):
                        issues.append(
                            "Boolean question should have a concise yes/no answer"
                        )

        return issues

    def _validate_extraction_output(
        self, output: Any, input_data: Dict[str, Any]
    ) -> List[str]:
        """Validate information extraction output."""
        issues = []

        if not isinstance(output, dict):
            issues.append("Extraction output must be a dictionary")
        elif len(output) == 0:
            issues.append("No information extracted")
        else:
            # Validate extracted fields
            for key, value in output.items():
                if not isinstance(key, str):
                    issues.append(f"Field name must be string, got {type(key)}")
                elif not value:
                    issues.append(f"Field '{key}' has empty value")
                elif isinstance(value, list) and len(value) == 0:
                    issues.append(f"Field '{key}' has empty list")

            # Check if values seem to be extracted from input
            if isinstance(input_data, dict) and "text" in input_data:
                input_text = str(input_data["text"]).lower()
                # At least some values should appear in the input
                values_found = 0
                for value in output.values():
                    if isinstance(value, str) and len(value) > 2:
                        if value.lower() in input_text:
                            values_found += 1

                if values_found == 0 and len(output) > 2:
                    issues.append(
                        "Extracted values don't appear to come from the input text"
                    )

        return issues

    def _validate_summarization_output(
        self, output: Any, input_data: Dict[str, Any]
    ) -> List[str]:
        """Validate summarization output."""
        issues = []

        if not isinstance(output, str):
            issues.append("Summary must be a string")
        elif len(output) < 10:
            issues.append("Summary too short (minimum 10 characters)")

        # Check if summary is shorter than input (if input is text)
        if isinstance(input_data, dict) and "text" in input_data:
            input_text = input_data["text"]
            if isinstance(input_text, str) and len(output) >= len(input_text):
                issues.append("Summary should be shorter than input text")

        return issues

    def _validate_ranking_output(
        self, output: Any, input_data: Dict[str, Any]
    ) -> List[str]:
        """Validate ranking/retrieval output."""
        issues = []

        if isinstance(output, list):
            if len(output) == 0:
                issues.append("Ranking output cannot be empty")
        elif isinstance(output, dict):
            if "results" not in output and "rankings" not in output:
                issues.append("Ranking output must contain 'results' or 'rankings'")
        else:
            issues.append("Ranking output must be list or dict")

        return issues

    def _validate_transformation_output(
        self, output: Any, input_data: Dict[str, Any]
    ) -> List[str]:
        """Validate translation/transformation output."""
        issues = []

        if not isinstance(output, str):
            issues.append("Transformation output must be a string")
        elif len(output) < 2:
            issues.append("Transformed text too short")

        return issues

    def _validate_code_output(
        self, output: Any, input_data: Dict[str, Any]
    ) -> List[str]:
        """Validate code generation output."""
        issues = []

        if not isinstance(output, str):
            issues.append("Generated code must be a string")
        elif len(output) < 5:
            issues.append("Generated code too short")

        # Check for common code patterns
        if isinstance(output, str):
            # Very basic check - at least has some code-like content
            has_code_patterns = any(
                [
                    "=" in output,  # Assignment
                    "(" in output,  # Function call
                    "{" in output,  # Block
                    "def " in output,  # Python function
                    "function " in output,  # JS function
                    "SELECT " in output.upper(),  # SQL
                    ";" in output,  # Statement terminator
                ]
            )

            if not has_code_patterns:
                issues.append("Output doesn't appear to contain valid code")

        return issues
