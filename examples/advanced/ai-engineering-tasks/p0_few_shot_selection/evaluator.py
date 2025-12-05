"""
Few-Shot Example Selection Evaluator
===================================

Evaluation functions for few-shot example selection optimization.
"""

from __future__ import annotations

import numpy as np
from selection_config import ExampleSelectionConfig


def _select_random_examples(example_pool: list[dict], n_examples: int) -> list[dict]:
    indices = np.random.choice(
        len(example_pool), min(n_examples, len(example_pool)), replace=False
    )
    return [example_pool[i] for i in indices]


def _tokenize(text: str) -> set[str]:
    return set(text.lower().split())


def _jaccard_similarity(words_a: set[str], words_b: set[str]) -> float:
    denominator = words_a | words_b
    if not denominator:
        return 0.0
    return len(words_a & words_b) / len(denominator)


def _select_semantic_knn(
    query: str, example_pool: list[dict], n_examples: int
) -> list[dict]:
    query_words = _tokenize(query)
    similarities = [
        _jaccard_similarity(query_words, _tokenize(example["input"]))
        for example in example_pool
    ]
    top_indices = np.argsort(similarities)[-n_examples:]
    return [example_pool[i] for i in reversed(top_indices)]


def _select_semantic_diverse(
    query: str, example_pool: list[dict], n_examples: int
) -> list[dict]:
    query_words = _tokenize(query)
    similarities = [
        (_jaccard_similarity(query_words, _tokenize(example["input"])), idx)
        for idx, example in enumerate(example_pool)
    ]
    similarities.sort(reverse=True)
    if not similarities:
        return []
    selected_indices = [similarities[0][1]]
    remaining = set(range(len(example_pool))) - {selected_indices[0]}

    while len(selected_indices) < n_examples and remaining:
        best_idx = None
        best_diversity = -1.0
        for candidate_idx in list(remaining):
            candidate_words = _tokenize(example_pool[candidate_idx]["input"])
            min_similarity = min(
                _jaccard_similarity(
                    candidate_words, _tokenize(example_pool[sel]["input"])
                )
                for sel in selected_indices
            )
            diversity = 1 - min_similarity
            if diversity > best_diversity:
                best_diversity = diversity
                best_idx = candidate_idx
        if best_idx is None:
            break
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [example_pool[i] for i in selected_indices]


def select_examples(
    query: str, config: ExampleSelectionConfig, example_pool: list[dict] | None = None
) -> list[dict]:
    """Select few-shot examples based on configuration."""

    if example_pool is None:
        # Default example pool for evaluation
        example_pool = [
            {
                "input": "What is machine learning?",
                "output": "Machine learning is a subset of AI that enables computers to learn from data.",
            },
            {
                "input": "Define neural networks",
                "output": "Neural networks are computing systems inspired by biological neural networks.",
            },
            {
                "input": "Explain deep learning",
                "output": "Deep learning uses multiple layers to model complex patterns in data.",
            },
            {
                "input": "What is NLP?",
                "output": "Natural language processing helps computers understand human language.",
            },
            {
                "input": "Describe computer vision",
                "output": "Computer vision enables machines to interpret visual information.",
            },
            {
                "input": "What is supervised learning?",
                "output": "Supervised learning uses labeled data to train predictive models.",
            },
            {
                "input": "Define unsupervised learning",
                "output": "Unsupervised learning finds patterns in data without labels.",
            },
            {
                "input": "Explain reinforcement learning",
                "output": "Reinforcement learning trains agents through rewards and penalties.",
            },
        ]

    if config.n_examples == 0:
        return []

    selected_examples = []

    if config.selection_method == "random":
        selected_examples = _select_random_examples(example_pool, config.n_examples)
    elif config.selection_method == "semantic_knn":
        selected_examples = _select_semantic_knn(query, example_pool, config.n_examples)
    elif config.selection_method == "semantic_diverse":
        selected_examples = _select_semantic_diverse(
            query, example_pool, config.n_examples
        )
    else:
        selected_examples = example_pool[: config.n_examples]

    return selected_examples


def evaluate_with_examples(
    query: str, examples: list[dict], ground_truth: str | None = None
) -> dict[str, float]:
    """Evaluate performance with given examples."""

    # Simulate LLM performance with examples
    # In real implementation, would call actual LLM

    # Base performance
    base_accuracy = 0.7

    # Example quality factor
    n_examples = len(examples)
    example_quality = min(1.0, n_examples * 0.1)  # Each example adds 10% quality

    # Diversity bonus (simplified)
    if n_examples > 1:
        unique_words = set()
        for ex in examples:
            unique_words.update(ex["input"].lower().split())
        diversity_bonus = min(0.2, len(unique_words) * 0.01)
    else:
        diversity_bonus = 0

    # Relevance bonus
    query_words = set(query.lower().split())
    relevance_scores = []
    for ex in examples:
        ex_words = set(ex["input"].lower().split())
        relevance = (
            len(query_words & ex_words) / len(query_words | ex_words)
            if query_words | ex_words
            else 0
        )
        relevance_scores.append(relevance)

    relevance_bonus = np.mean(relevance_scores) * 0.15 if relevance_scores else 0

    # Calculate final metrics
    accuracy = min(
        1.0, base_accuracy + example_quality + diversity_bonus + relevance_bonus
    )

    # Add some realistic variance
    accuracy += np.random.normal(0, 0.05)
    accuracy = max(0.0, min(1.0, accuracy))

    # Other metrics
    f1_score = accuracy * np.random.uniform(0.9, 1.1)
    f1_score = max(0.0, min(1.0, f1_score))

    latency_ms = 50 + n_examples * 20  # Base latency + example processing
    latency_ms += np.random.uniform(-10, 10)

    return {
        "accuracy": accuracy,
        "f1_score": f1_score,
        "latency_ms": latency_ms,
        "n_examples_used": n_examples,
        "relevance_score": np.mean(relevance_scores) if relevance_scores else 0,
        "diversity_score": diversity_bonus / 0.2 if diversity_bonus > 0 else 0,
    }
