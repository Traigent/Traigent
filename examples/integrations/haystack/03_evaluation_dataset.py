#!/usr/bin/env python3
"""Example: Evaluation Dataset Creation with Traigent.

This example demonstrates how to create and manage evaluation datasets
for Haystack pipeline optimization.

Coverage: Epic 3, Story 3.1 (Define Evaluation Dataset Format)
"""

from __future__ import annotations


def example_from_dicts():
    """Demonstrate creating dataset from dictionaries."""
    print("=" * 60)
    print("Example 1: Create Dataset from Dictionaries")
    print("=" * 60)

    from traigent.integrations.haystack import EvaluationDataset

    # Create dataset from list of dicts
    dataset = EvaluationDataset.from_dicts(
        [
            {
                "input": {"query": "What is machine learning?"},
                "expected": "Machine learning is a subset of AI...",
            },
            {
                "input": {"query": "Explain neural networks"},
                "expected": "Neural networks are computing systems...",
            },
            {
                "input": {"query": "What is deep learning?"},
                "expected": "Deep learning uses multiple layers...",
                "metadata": {"difficulty": "medium", "category": "AI"},
            },
        ]
    )

    print(f"\nCreated dataset with {len(dataset)} examples")
    for i, example in enumerate(dataset.examples):
        print(f"\n  Example {i + 1}:")
        print(f"    Input: {example.input}")
        print(f"    Expected: {example.expected[:50]}...")
        if example.metadata:
            print(f"    Metadata: {example.metadata}")

    print("\n")


def example_from_json():
    """Demonstrate creating dataset from JSON data."""
    print("=" * 60)
    print("Example 2: Create Dataset from JSON Data")
    print("=" * 60)

    import json
    import tempfile

    from traigent.integrations.haystack import EvaluationDataset

    # Create a temporary JSON file
    data = [
        {"input": {"query": "Q1"}, "expected": "A1", "id": "test-001"},
        {"input": {"query": "Q2"}, "expected": "A2", "id": "test-002"},
        {"input": {"query": "Q3"}, "expected": "A3", "id": "test-003"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        json_path = f.name

    # Load JSON and create dataset using from_dicts
    with open(json_path) as f:
        loaded_data = json.load(f)
    dataset = EvaluationDataset.from_dicts(loaded_data)
    print(f"\nLoaded {len(dataset)} examples from JSON")

    for example in dataset.examples:
        example_id = example.metadata.get("id", "N/A") if example.metadata else "N/A"
        print(f"  - {example_id}: {example.input['query']} -> {example.expected}")

    print("\n")


def example_from_csv():
    """Demonstrate creating dataset from CSV data."""
    print("=" * 60)
    print("Example 3: Create Dataset from CSV Data")
    print("=" * 60)

    import csv
    import tempfile

    from traigent.integrations.haystack import EvaluationDataset

    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=["query", "expected_answer", "category"])
        writer.writeheader()
        writer.writerow(
            {
                "query": "What is Python?",
                "expected_answer": "A programming language",
                "category": "programming",
            }
        )
        writer.writerow(
            {
                "query": "What is Java?",
                "expected_answer": "A programming language",
                "category": "programming",
            }
        )
        writer.writerow(
            {
                "query": "What is Linux?",
                "expected_answer": "An operating system",
                "category": "systems",
            }
        )
        csv_path = f.name

    # Load CSV and convert to from_dicts format
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        data = [
            {
                "input": {"query": row["query"]},
                "expected": row["expected_answer"],
                "metadata": {"category": row["category"]},
            }
            for row in reader
        ]

    dataset = EvaluationDataset.from_dicts(data)
    print(f"\nLoaded {len(dataset)} examples from CSV")
    for example in dataset.examples:
        print(f"  - {example.input} -> {example.expected}")
        cat = example.metadata.get("category") if example.metadata else "N/A"
        print(f"    Category: {cat}")

    print("\n")


def example_dataset_operations():
    """Demonstrate dataset operations."""
    print("=" * 60)
    print("Example 4: Dataset Operations")
    print("=" * 60)

    from traigent.integrations.haystack import EvaluationDataset

    # Create dataset
    dataset = EvaluationDataset.from_dicts(
        [
            {"input": {"query": f"Question {i}"}, "expected": f"Answer {i}"}
            for i in range(10)
        ]
    )

    print(f"Original dataset size: {len(dataset)}")

    # Slice dataset
    subset = dataset[:5]
    print(f"Subset (first 5): {len(subset)} examples")

    # Iterate over examples
    print("\nIterating over first 3 examples:")
    for i, example in enumerate(dataset.examples[:3]):
        print(f"  {i}: {example.input['query']}")

    # Convert to core dataset format (for orchestrator compatibility)
    core_dataset = dataset.to_core_dataset()
    print(f"\nConverted to core Dataset with {len(core_dataset.examples)} examples")
    print(f"  Core example type: {type(core_dataset.examples[0]).__name__}")

    print("\n")


def example_with_complex_inputs():
    """Demonstrate dataset with complex input structures."""
    print("=" * 60)
    print("Example 5: Complex Input Structures")
    print("=" * 60)

    from traigent.integrations.haystack import EvaluationDataset

    # RAG-style inputs with query and context
    dataset = EvaluationDataset.from_dicts(
        [
            {
                "input": {
                    "query": "What is the capital of France?",
                    "documents": [
                        {
                            "content": "Paris is the capital of France.",
                            "meta": {"source": "wiki"},
                        },
                        {
                            "content": "France is a country in Europe.",
                            "meta": {"source": "wiki"},
                        },
                    ],
                },
                "expected": "Paris",
                "metadata": {"type": "factual", "difficulty": "easy"},
            },
            {
                "input": {
                    "query": "Who wrote Romeo and Juliet?",
                    "documents": [
                        {
                            "content": "William Shakespeare wrote many plays.",
                            "meta": {"source": "lit"},
                        },
                    ],
                },
                "expected": "William Shakespeare",
                "metadata": {"type": "factual", "difficulty": "easy"},
            },
        ]
    )

    print(f"\nCreated RAG dataset with {len(dataset)} examples")
    for example in dataset.examples:
        print(f"\n  Query: {example.input['query']}")
        print(f"  Documents: {len(example.input.get('documents', []))} provided")
        print(f"  Expected: {example.expected}")
        print(f"  Metadata: {example.metadata}")

    print("\n")


if __name__ == "__main__":
    example_from_dicts()
    example_from_json()
    example_from_csv()
    example_dataset_operations()
    example_with_complex_inputs()

    print("All evaluation dataset examples completed successfully!")
