#!/usr/bin/env python3
"""Example: Evaluation Datasets - Providing Test Data for Optimization."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# --- Setup for running from repo without installation ---
# Add repo root to path so we can import examples.utils and traigent
_module_path = Path(__file__).resolve()
for _depth in range(1, 7):
    try:
        _repo_root = _module_path.parents[_depth]
        if (_repo_root / "traigent").is_dir() and (_repo_root / "examples").is_dir():
            if str(_repo_root) not in sys.path:
                sys.path.insert(0, str(_repo_root))
            break
    except IndexError:
        continue
from examples.utils.langchain_compat import ChatOpenAI, HumanMessage

try:
    import traigent
except ImportError:  # pragma: no cover - support IDE execution paths
    import importlib

    module_path = Path(__file__).resolve()
    for depth in (2, 3):
        try:
            sys.path.append(str(module_path.parents[depth]))
        except IndexError:
            continue
    traigent = importlib.import_module("traigent")

from traigent.evaluators.base import Dataset, EvaluationExample

# Create JSONL dataset file
DATASET_FILE = os.path.join(os.path.dirname(__file__), "classification_tasks.jsonl")


def create_jsonl_dataset() -> str:
    """Create sample classification dataset."""
    dataset = [
        {"input": {"text": "The product arrived damaged"}, "output": "complaint"},
        {"input": {"text": "Amazing service, highly recommend!"}, "output": "feedback"},
        {"input": {"text": "How do I track my order?"}, "output": "inquiry"},
        {"input": {"text": "Thank you for the quick delivery"}, "output": "feedback"},
        {"input": {"text": "Missing items in my package"}, "output": "complaint"},
    ]

    # Write to JSONL file
    with open(DATASET_FILE, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    return DATASET_FILE


# Create the dataset file
create_jsonl_dataset()


# Helper to convert raw records into a Dataset instance
def _records_to_dataset(records: list[dict[str, Any]], name: str) -> Dataset:
    examples = []
    for record in records:
        input_data = record.get("input", {})
        expected_output = record.get("output") or record.get("expected_output")
        metadata = {
            k: v
            for k, v in record.items()
            if k not in {"input", "output", "expected_output"}
        }
        examples.append(
            EvaluationExample(
                input_data=(
                    input_data
                    if isinstance(input_data, dict)
                    else {"value": input_data}
                ),
                expected_output=expected_output,
                metadata=metadata or None,
            )
        )
    return Dataset(
        examples=examples, name=name, description=f"In-memory dataset: {name}"
    )


# Example 1: Using JSONL file dataset
@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.1, 0.5, 0.9],
    },
    eval_dataset=DATASET_FILE,  # JSONL file
    objectives=["accuracy", "cost"],
    execution_mode="edge_analytics",
    max_trials=10,
)
def classifier_with_jsonl(text: str) -> str:
    """Classifier using JSONL dataset for evaluation."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    prompt = (
        "Classify the text into one of these categories: complaint, feedback, inquiry.\n"
        f"Respond with only the category word.\nText: {text}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return str(getattr(response, "content", response))


# Example 2: Using Python list dataset
def _python_dataset_records() -> list[dict[str, Any]]:
    return [
        {"input": {"text": "The product arrived damaged"}, "output": "complaint"},
        {"input": {"text": "Amazing service, highly recommend!"}, "output": "feedback"},
        {"input": {"text": "How do I track my order?"}, "output": "inquiry"},
        {"input": {"text": "Request for refund processing"}, "output": "request"},
        {"input": {"text": "Thank you for the quick response"}, "output": "feedback"},
    ]


def create_python_dataset() -> Dataset:
    """Create evaluation dataset as an in-memory Dataset object."""
    return _records_to_dataset(_python_dataset_records(), name="support_in_memory")


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.1, 0.3, 0.5],
        "prompt_style": ["direct", "detailed", "structured"],
    },
    eval_dataset=create_python_dataset(),  # Python list
    objectives=["accuracy", "consistency"],
    execution_mode="edge_analytics",
    max_trials=10,
)
def classifier_with_python_list(text: str) -> str:
    """Classifier using Python list dataset."""
    current = traigent.get_config()
    config: dict[str, Any] = current if isinstance(current, dict) else {}

    prompt_styles = {
        "direct": f"Classify: {text}",
        "detailed": f"Analyze the following text and classify it into the appropriate category:\nText: {text}\nCategory:",
        "structured": f"Task: Text Classification\nInput: {text}\nOutput the category:",
    }

    llm = ChatOpenAI(
        model=config.get("model", "gpt-3.5-turbo"),
        temperature=config.get("temperature", 0.3),
    )

    style = str(config.get("prompt_style", "direct"))
    prompt = prompt_styles.get(style, prompt_styles["direct"])

    response = llm.invoke([HumanMessage(content=prompt)])
    return str(getattr(response, "content", response))


# Example 3: Dynamic dataset generation
class DynamicDatasetGenerator:
    """Generate evaluation datasets dynamically."""

    @staticmethod
    def generate_math_problems(count: int = 10) -> Dataset:
        """Generate math problem dataset."""
        import random

        problems: list[dict[str, Any]] = []
        for _ in range(count):
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            operation = random.choice(["+", "-", "*"])

            if operation == "+":
                answer = a + b
            elif operation == "-":
                answer = a - b
            else:  # multiplication
                answer = a * b

            problems.append(
                {
                    "input": {"problem": f"What is {a} {operation} {b}?"},
                    "expected_output": {"answer": str(answer)},
                }
            )

        return _records_to_dataset(problems, name="math_problems")

    @staticmethod
    def generate_translation_pairs(count: int = 10) -> Dataset:
        """Generate translation dataset."""
        translations = [
            ("Hello", "Bonjour"),
            ("Thank you", "Merci"),
            ("Good morning", "Bonjour"),
            ("Goodbye", "Au revoir"),
            ("Please", "S'il vous plaît"),
            ("Yes", "Oui"),
            ("No", "Non"),
            ("Sorry", "Désolé"),
            ("Excuse me", "Excusez-moi"),
            ("How are you?", "Comment allez-vous?"),
        ]

        dataset: list[dict[str, Any]] = []
        for english, french in translations[:count]:
            dataset.append(
                {
                    "input": {"text": english, "target_language": "French"},
                    "expected_output": {"translation": french},
                }
            )

        return _records_to_dataset(dataset, name="translation_pairs")


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.1, 0.2],  # Low temp for math
        "reasoning_steps": [True, False],
    },
    eval_dataset=DynamicDatasetGenerator.generate_math_problems(10),  # Dynamic
    objectives=["accuracy"],
    execution_mode="edge_analytics",
    max_trials=10,
)
def math_solver(problem: str) -> str:
    """Math problem solver with dynamic dataset."""
    current = traigent.get_config()
    config: dict[str, Any] = current if isinstance(current, dict) else {}

    if config.get("reasoning_steps", False):
        prompt = f"Solve step by step: {problem}"
    else:
        prompt = f"Solve: {problem}"

    llm = ChatOpenAI(
        model=config.get("model", "gpt-3.5-turbo"),
        temperature=config.get("temperature", 0.1),
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return str(getattr(response, "content", response))


def demonstrate_dataset_formats() -> None:
    """Show different dataset formats and structures."""

    print("TraiGent Evaluation Dataset Formats")
    print("=" * 50)

    # JSONL format
    print("\n1. JSONL Format (classification_tasks.jsonl):")
    print(
        '{"input": {"text": "Great product!"}, "expected_output": {"sentiment": "positive"}}'
    )
    print(
        '{"input": {"text": "Terrible service"}, "expected_output": {"sentiment": "negative"}}'
    )

    # Python list format
    print("\n2. Python List Format:")
    python_dataset = create_python_dataset()
    print(f"Dataset with {len(python_dataset)} examples")
    first_example = {
        "input": python_dataset[0].input_data,
        "expected_output": python_dataset[0].expected_output,
    }
    print(f"First example: {json.dumps(first_example, indent=2)}")

    # Dynamic generation
    print("\n3. Dynamic Dataset Generation:")
    math_dataset = DynamicDatasetGenerator.generate_math_problems(3)
    for i, problem in enumerate(math_dataset, 1):
        expected = problem.expected_output or {}
        answer = None
        if isinstance(expected, dict):
            answer = expected.get("answer")
        print(f"  Problem {i}: {problem.input_data.get('problem')} = {answer}")

    # Dataset requirements
    print("\n4. Dataset Structure Requirements:")
    print("  - Must have 'input' field (dict)")
    print("  - Must have 'expected_output' field (dict)")
    print("  - Optional: 'metadata' field for additional info")
    print("  - Minimum 5-10 examples recommended")


def analyze_dataset_characteristics() -> None:
    """Analyze dataset characteristics for optimization."""

    print("\n" + "=" * 50)
    print("Dataset Best Practices")
    print("=" * 50)

    best_practices = {
        "Size": {
            "minimum": "5-10 examples for basic optimization",
            "recommended": "20-50 examples for robust results",
            "maximum": "100+ for complex scenarios",
        },
        "Diversity": {
            "coverage": "Include edge cases and typical cases",
            "balance": "Balanced representation of categories",
            "difficulty": "Mix of easy and challenging examples",
        },
        "Quality": {
            "accuracy": "Verified correct expected outputs",
            "consistency": "Consistent format and structure",
            "relevance": "Representative of production data",
        },
    }

    for category, details in best_practices.items():
        print(f"\n{category}:")
        for key, value in details.items():
            print(f"  {key:12} - {value}")


if __name__ == "__main__":
    print("=" * 60)
    print("TraiGent Core Concepts: Evaluation Datasets")
    print("=" * 60)

    # Demonstrate dataset formats
    demonstrate_dataset_formats()

    # Analyze dataset characteristics
    analyze_dataset_characteristics()

    print("\n" + "=" * 50)
    print("Dataset Examples Created")
    print("=" * 50)
    print("1. JSONL file dataset (classification_tasks.jsonl)")
    print("2. Python list dataset (in-memory)")
    print("3. Dynamic dataset generation (math problems)")
    print("\nEach format has its use cases and advantages!")
