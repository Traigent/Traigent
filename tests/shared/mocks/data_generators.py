"""Test data generators for consistent test fixtures.

This module provides functions to generate common test data structures
used across the TraiGent SDK test suite.
"""

from typing import Any, Dict, List, Optional

from traigent.api.types import TrialResult, TrialStatus
from traigent.evaluators.base import Dataset, EvaluationExample


def create_test_dataset(
    name: str = "test_dataset", size: int = 5, domain: str = "general"
) -> Dataset:
    """Create a test dataset with realistic examples.

    Args:
        name: Dataset name
        size: Number of examples to generate
        domain: Domain type (general, math, code, qa)

    Returns:
        Dataset with generated examples
    """
    examples = []

    if domain == "math":
        math_problems = [
            ("What is 2+2?", "4"),
            ("What is 5*7?", "35"),
            ("What is 12/3?", "4"),
            ("What is 8-3?", "5"),
            ("What is 6+9?", "15"),
        ]
        for i in range(min(size, len(math_problems))):
            question, answer = math_problems[i]
            examples.append(
                EvaluationExample(
                    input_data={"question": question}, expected_output=answer
                )
            )

    elif domain == "qa":
        qa_pairs = [
            ("What is the capital of France?", "Paris"),
            ("Who invented the telephone?", "Alexander Graham Bell"),
            ("What year did World War II end?", "1945"),
            ("What is the largest planet?", "Jupiter"),
            ("Who wrote Romeo and Juliet?", "William Shakespeare"),
        ]
        for i in range(min(size, len(qa_pairs))):
            question, answer = qa_pairs[i]
            examples.append(
                EvaluationExample(
                    input_data={"question": question}, expected_output=answer
                )
            )

    elif domain == "code":
        code_problems = [
            (
                "Write a function to reverse a string",
                "def reverse_string(s): return s[::-1]",
            ),
            ("How do you sort a list in Python?", "list.sort() or sorted(list)"),
            ("What is a lambda function?", "Anonymous function defined with lambda"),
            (
                "How to read a file in Python?",
                "with open('file.txt', 'r') as f: content = f.read()",
            ),
            ("What is list comprehension?", "[expression for item in iterable]"),
        ]
        for i in range(min(size, len(code_problems))):
            question, answer = code_problems[i]
            examples.append(
                EvaluationExample(
                    input_data={"question": question}, expected_output=answer
                )
            )

    else:  # general domain
        general_examples = [
            (
                "Describe the weather today",
                "It's a pleasant day with mild temperatures",
            ),
            (
                "What is artificial intelligence?",
                "AI is computer systems that can perform tasks requiring human intelligence",
            ),
            (
                "Explain machine learning",
                "ML is a subset of AI that learns patterns from data",
            ),
            (
                "What is deep learning?",
                "Deep learning uses neural networks with multiple layers",
            ),
            (
                "How does natural language processing work?",
                "NLP enables computers to understand and process human language",
            ),
        ]
        for i in range(min(size, len(general_examples))):
            question, answer = general_examples[i]
            examples.append(
                EvaluationExample(
                    input_data={"question": question}, expected_output=answer
                )
            )

    # Pad with generic examples if needed
    while len(examples) < size:
        i = len(examples)
        examples.append(
            EvaluationExample(
                input_data={"question": f"Generic question {i+1}"},
                expected_output=f"Generic answer {i+1}",
            )
        )

    return Dataset(name=name, examples=examples[:size])


def create_config_space(
    complexity: str = "simple", include_advanced: bool = False
) -> Dict[str, List[Any]]:
    """Create a configuration space for optimization testing.

    Args:
        complexity: simple, medium, or complex
        include_advanced: Include advanced parameters

    Returns:
        Configuration space dictionary
    """
    if complexity == "simple":
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.1, 0.5, 0.9],
        }
    elif complexity == "medium":
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"],
            "temperature": [0.0, 0.3, 0.5, 0.7, 1.0],
            "max_tokens": [100, 500, 1000],
        }
    else:  # complex
        config_space = {
            "model": [
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4o-mini",
                "claude-3-sonnet",
                "claude-3-opus",
            ],
            "temperature": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            "max_tokens": [50, 100, 250, 500, 1000, 2000],
            "top_p": [0.1, 0.5, 0.9, 1.0],
            "frequency_penalty": [0.0, 0.5, 1.0],
            "presence_penalty": [0.0, 0.5, 1.0],
        }

    if include_advanced:
        config_space.update(
            {
                "stop_sequences": [None, ["\\n"], ["END"], ["\\n", "END"]],
                "stream": [True, False],
                "tools": [
                    None,
                    [{"type": "function", "function": {"name": "calculator"}}],
                ],
            }
        )

    return config_space


def create_evaluation_examples(
    count: int = 3, input_keys: Optional[List[str]] = None, output_type: str = "string"
) -> List[EvaluationExample]:
    """Create evaluation examples with specified structure.

    Args:
        count: Number of examples to create
        input_keys: Keys for input data (default: ["question"])
        output_type: Type of expected output

    Returns:
        List of evaluation examples
    """
    if input_keys is None:
        input_keys = ["question"]

    examples = []
    for i in range(count):
        input_data = {}
        for key in input_keys:
            input_data[key] = f"Sample {key} {i+1}"

        if output_type == "string":
            expected_output = f"Sample output {i+1}"
        elif output_type == "number":
            expected_output = str(i + 1)
        elif output_type == "boolean":
            expected_output = str(i % 2 == 0).lower()
        elif output_type == "list":
            expected_output = [f"item_{i+1}_1", f"item_{i+1}_2"]
        else:
            expected_output = {"result": f"structured_output_{i+1}"}

        examples.append(
            EvaluationExample(input_data=input_data, expected_output=expected_output)
        )

    return examples


def create_mock_trial_results(
    count: int = 5,
    status: TrialStatus = TrialStatus.COMPLETED,
    config_space: Optional[Dict[str, List[Any]]] = None,
) -> List[TrialResult]:
    """Create mock trial results for testing.

    Args:
        count: Number of trial results to create
        status: Status for all trials
        config_space: Configuration space to sample from

    Returns:
        List of trial results
    """
    if config_space is None:
        config_space = create_config_space("simple")

    results = []
    for i in range(count):
        # Sample a configuration
        config = {}
        for param, values in config_space.items():
            config[param] = values[i % len(values)]

        # Generate realistic metrics
        accuracy = 0.7 + (i * 0.05) + (0.1 if config.get("model") == "gpt-4" else 0)
        accuracy = min(accuracy, 1.0)

        cost = 0.01 + (i * 0.002) + (0.005 if config.get("model") == "gpt-4" else 0)
        latency = 1.0 + (i * 0.2) + (0.3 if config.get("model") == "gpt-4" else 0)

        trial_result = TrialResult(
            trial_id=f"trial_{i+1}",
            config=config,
            metrics={
                "accuracy": round(accuracy, 3),
                "cost": round(cost, 4),
                "latency": round(latency, 2),
            },
            status=status,
            duration=latency,
            error_message=(
                None if status == TrialStatus.COMPLETED else f"Error in trial {i+1}"
            ),
        )
        results.append(trial_result)

    return results


def create_large_dataset(size: int = 1000, domain: str = "general") -> Dataset:
    """Create a large dataset for performance testing.

    Args:
        size: Number of examples (can be large)
        domain: Domain type

    Returns:
        Large dataset
    """
    examples = []

    # Create examples in batches to avoid memory issues
    batch_size = 100
    for batch_start in range(0, size, batch_size):
        batch_end = min(batch_start + batch_size, size)
        batch_size_actual = batch_end - batch_start

        batch_examples = create_evaluation_examples(
            count=batch_size_actual,
            input_keys=["question", "context"] if domain == "qa" else ["question"],
        )
        examples.extend(batch_examples)

    return Dataset(name=f"large_{domain}_dataset", examples=examples)
