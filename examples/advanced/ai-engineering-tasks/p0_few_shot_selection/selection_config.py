"""
Few-Shot Example Selection Configuration
========================================

Configuration dataclass defining the search space for few-shot example optimization.
Based on the Few-Shot Example Selection Strategies use case specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ExampleSelectionConfig:
    """Configuration for few-shot example selection optimization.

    This dataclass defines the complete search space for optimizing
    few-shot example selection, covering modern 2024/2025 approaches.
    """

    # Selection algorithms (2024/2025 methods)
    selection_method: str  # See SELECTION_METHODS below

    # Number and composition
    n_examples: int  # 0, 1, 3, 5, 8

    # Ordering strategies
    example_ordering: str  # See ORDERING_STRATEGIES below

    # Formatting approaches
    example_format: str  # See FORMAT_APPROACHES below

    # Dynamic vs static
    dynamic_selection: bool  # True for query-specific selection
    cache_selections: bool  # True to cache for performance

    # Prompt integration
    example_placement: str  # "before_instruction", "after_instruction", "interleaved"

    # Model parameters
    temperature: float  # 0.0, 0.3, 0.7


# Search space definition for TraiGent optimization
EXAMPLE_SELECTION_SEARCH_SPACE = {
    "selection_method": [
        "random",  # Baseline
        "semantic_knn",  # Nearest neighbors
        "semantic_diverse",  # Maximize coverage
        "mmr",  # Maximal Marginal Relevance
        "cluster_centroids",  # Representative examples
        "influence_based",  # High-influence examples
        "contrastive",  # Positive and negative
        "curriculum",  # Easy to hard
        "uncertainty_based",  # Model-uncertain examples
    ],
    "n_examples": [0, 1, 3, 5, 8],
    "example_ordering": [
        "random",
        "similarity_desc",  # Most similar first
        "similarity_asc",  # Least similar first
        "difficulty_progression",  # Easy to hard
        "diversity_first",  # Diverse then similar
        "alternating",  # Alternate strategies
    ],
    "example_format": [
        "input_output",  # Simple I/O pairs
        "input_output_explanation",  # With reasoning
        "structured_fields",  # Field-by-field
        "conversational",  # Dialog format
        "xml_wrapped",  # XML-tagged sections
    ],
    "dynamic_selection": [True, False],
    "cache_selections": [True, False],
    "example_placement": ["before_instruction", "after_instruction", "interleaved"],
    "temperature": [0.0, 0.3, 0.7],
}


def create_selection_config(**kwargs) -> ExampleSelectionConfig:
    """Create an ExampleSelectionConfig with validation."""
    return ExampleSelectionConfig(
        selection_method=kwargs.get("selection_method", "random"),
        n_examples=kwargs.get("n_examples", 0),
        example_ordering=kwargs.get("example_ordering", "random"),
        example_format=kwargs.get("example_format", "input_output"),
        dynamic_selection=kwargs.get("dynamic_selection", False),
        cache_selections=kwargs.get("cache_selections", False),
        example_placement=kwargs.get("example_placement", "before_instruction"),
        temperature=kwargs.get("temperature", 0.0),
    )


class ExampleSelector:
    """Base class for example selection strategies."""

    def __init__(self, config: ExampleSelectionConfig):
        self.config = config
        self._cache = {} if config.cache_selections else None

    def select(
        self,
        query: str | dict[str, Any],
        example_pool: list[Any],
        embeddings: np.ndarray | None = None,
    ) -> list[Any]:
        """Select examples based on configuration."""

        # Check cache if enabled
        if self._cache is not None:
            cache_key = self._get_cache_key(query)
            if cache_key in self._cache:
                return self._cache[cache_key]

        method_handlers = {
            "random": lambda: self._random_selection(example_pool),
            "semantic_knn": lambda: self._semantic_knn_selection(
                query, example_pool, embeddings
            ),
            "semantic_diverse": lambda: self._semantic_diverse_selection(
                query, example_pool, embeddings
            ),
            "mmr": lambda: self._mmr_selection(query, example_pool, embeddings),
            "cluster_centroids": lambda: self._cluster_centroid_selection(
                example_pool, embeddings
            ),
            "influence_based": lambda: self._influence_based_selection(
                query, example_pool
            ),
            "contrastive": lambda: self._contrastive_selection(query, example_pool),
            "curriculum": lambda: self._curriculum_selection(query, example_pool),
            "uncertainty_based": lambda: self._uncertainty_based_selection(
                query, example_pool
            ),
        }
        handler = method_handlers.get(
            self.config.selection_method, method_handlers["random"]
        )
        selected = handler()

        # Apply ordering
        selected = self._apply_ordering(selected, query, embeddings)

        # Cache if enabled
        if self._cache is not None:
            self._cache[cache_key] = selected

        return selected

    def _random_selection(self, example_pool: list[Any]) -> list[Any]:
        """Random selection baseline."""
        import random

        n = min(self.config.n_examples, len(example_pool))
        return random.sample(example_pool, n) if n > 0 else []

    def _semantic_knn_selection(
        self, query: Any, example_pool: list[Any], embeddings: np.ndarray | None
    ) -> list[Any]:
        """K-nearest neighbors based on semantic similarity."""
        if embeddings is None or len(embeddings) == 0:
            return self._random_selection(example_pool)

        # Simulate similarity computation
        # In real implementation, compute cosine similarity between query and examples
        n = min(self.config.n_examples, len(example_pool))

        # For demo, return first n examples
        return example_pool[:n] if n > 0 else []

    def _semantic_diverse_selection(
        self, query: Any, example_pool: list[Any], embeddings: np.ndarray | None
    ) -> list[Any]:
        """Select diverse examples that maximize coverage."""
        if embeddings is None or len(embeddings) == 0:
            return self._random_selection(example_pool)

        n = min(self.config.n_examples, len(example_pool))
        if n == 0:
            return []

        # Simplified diverse selection for demo
        # In real implementation, use clustering or MMR algorithm
        step = max(1, len(example_pool) // n)
        return example_pool[::step][:n]

    def _mmr_selection(
        self, query: Any, example_pool: list[Any], embeddings: np.ndarray | None
    ) -> list[Any]:
        """Maximal Marginal Relevance selection."""
        # Balance relevance to query with diversity among selected examples
        n = min(self.config.n_examples, len(example_pool))
        if n == 0:
            return []

        # Simplified MMR for demo
        selected = []
        remaining = example_pool.copy()

        # Select most relevant first
        if remaining:
            selected.append(remaining.pop(0))

        # Select diverse examples
        while len(selected) < n and remaining:
            # In real implementation, compute MMR score
            selected.append(remaining.pop(len(remaining) // 2))

        return selected

    def _cluster_centroid_selection(
        self, example_pool: list[Any], embeddings: np.ndarray | None
    ) -> list[Any]:
        """Select cluster centroids as representative examples."""
        n = min(self.config.n_examples, len(example_pool))
        if n == 0:
            return []

        # Simplified clustering for demo
        # In real implementation, use K-means or other clustering
        step = max(1, len(example_pool) // n)
        return example_pool[step // 2 :: step][:n]

    def _influence_based_selection(
        self, query: Any, example_pool: list[Any]
    ) -> list[Any]:
        """Select high-influence examples based on impact."""
        # In real implementation, compute influence scores
        n = min(self.config.n_examples, len(example_pool))
        return example_pool[:n] if n > 0 else []

    def _contrastive_selection(self, query: Any, example_pool: list[Any]) -> list[Any]:
        """Select positive and negative examples."""
        n = min(self.config.n_examples, len(example_pool))
        if n == 0:
            return []

        # Select half positive, half negative (simplified)
        positive_n = n // 2 + n % 2
        negative_n = n // 2

        selected = example_pool[:positive_n]
        if negative_n > 0:
            selected.extend(example_pool[-negative_n:])

        return selected

    def _curriculum_selection(self, query: Any, example_pool: list[Any]) -> list[Any]:
        """Select examples in curriculum order (easy to hard)."""
        n = min(self.config.n_examples, len(example_pool))
        # Assume examples are ordered by difficulty
        return example_pool[:n] if n > 0 else []

    def _uncertainty_based_selection(
        self, query: Any, example_pool: list[Any]
    ) -> list[Any]:
        """Select examples where model is uncertain."""
        # In real implementation, compute uncertainty scores
        n = min(self.config.n_examples, len(example_pool))
        return example_pool[:n] if n > 0 else []

    def _apply_ordering(
        self, selected: list[Any], query: Any, embeddings: np.ndarray | None
    ) -> list[Any]:
        """Apply ordering strategy to selected examples."""
        if not selected:
            return selected

        if self.config.example_ordering == "random":
            import random

            random.shuffle(selected)
        elif self.config.example_ordering == "similarity_desc":
            # Most similar first (no change for demo)
            pass
        elif self.config.example_ordering == "similarity_asc":
            # Least similar first
            selected.reverse()
        elif self.config.example_ordering == "difficulty_progression":
            # Easy to hard (assume already ordered)
            pass
        elif self.config.example_ordering == "diversity_first":
            # Put diverse examples first
            if len(selected) > 1:
                # Simple reordering for demo
                len(selected) // 2
                selected = selected[::2] + selected[1::2]
        elif self.config.example_ordering == "alternating":
            # Alternate between strategies
            if len(selected) > 1:
                selected = [selected[i] for i in [0, -1] * (len(selected) // 2)][
                    : len(selected)
                ]

        return selected

    def _get_cache_key(self, query: Any) -> str:
        """Generate cache key for query."""
        if isinstance(query, str):
            return query[:100]  # Use first 100 chars
        else:
            return str(query)[:100]


def format_examples(examples: list[Any], config: ExampleSelectionConfig) -> str:
    """Format selected examples according to configuration."""

    if not examples:
        return ""

    formatted = []

    for i, example in enumerate(examples):
        if config.example_format == "input_output":
            formatted.append(
                f"Example {i+1}:\nInput: {example.get('input', '')}\nOutput: {example.get('output', '')}"
            )

        elif config.example_format == "input_output_explanation":
            formatted.append(
                f"Example {i+1}:\n"
                f"Input: {example.get('input', '')}\n"
                f"Output: {example.get('output', '')}\n"
                f"Explanation: {example.get('explanation', 'The model processed the input to produce this output.')}"
            )

        elif config.example_format == "structured_fields":
            fields = "\n".join([f"  {k}: {v}" for k, v in example.items()])
            formatted.append(f"Example {i+1}:\n{fields}")

        elif config.example_format == "conversational":
            formatted.append(
                f"Example {i+1}:\n"
                f"User: {example.get('input', '')}\n"
                f"Assistant: {example.get('output', '')}"
            )

        elif config.example_format == "xml_wrapped":
            formatted.append(
                f"<example_{i+1}>\n"
                f"  <input>{example.get('input', '')}</input>\n"
                f"  <output>{example.get('output', '')}</output>\n"
                f"</example_{i+1}>"
            )

    return "\n\n".join(formatted)
