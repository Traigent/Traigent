"""Smart dataset subset selection algorithms for cost optimization."""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import hashlib
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

# Optional dependencies for advanced subset selection
try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

    # Provide fallback classes
    class KMeans:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("scikit-learn not available for clustering") from None

    class TfidfVectorizer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("scikit-learn not available for TF-IDF") from None

    def cosine_similarity(*args, **kwargs) -> None:
        raise ImportError(
            "scikit-learn not available for similarity calculation"
        ) from None


from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


def _compute_reduction_ratio(original_size: int, selected_size: int) -> float:
    """Safely compute reduction ratio, guarding against division by zero."""
    if original_size <= 0:
        return 0.0
    return 1 - (selected_size / original_size)


@dataclass
class SubsetSelectionResult:
    """Result from subset selection algorithm."""

    selected_examples: list[EvaluationExample]
    selection_strategy: str
    original_size: int
    selected_size: int
    reduction_ratio: float
    diversity_score: float
    confidence_score: float


class BaseSubsetSelector(ABC):
    """Abstract base class for dataset subset selection algorithms."""

    @abstractmethod
    async def select_subset(
        self, dataset: Dataset, target_size: int, **kwargs: Any
    ) -> SubsetSelectionResult:
        """Select optimal subset of examples from dataset.

        Args:
            dataset: Original dataset
            target_size: Target number of examples
            **kwargs: Algorithm-specific parameters

        Returns:
            SubsetSelectionResult with selected examples
        """
        pass


class DiverseSampling(BaseSubsetSelector):
    """Diverse sampling strategy using clustering and TF-IDF similarity."""

    def __init__(self, random_seed: int = 42) -> None:
        """Initialize diverse sampling selector.

        Args:
            random_seed: Random seed for reproducible results
        """
        self.random_seed = random_seed

    async def select_subset(
        self, dataset: Dataset, target_size: int, **kwargs: Any
    ) -> SubsetSelectionResult:
        """Select diverse examples using clustering or similarity-based sampling.

        Args:
            dataset: Original dataset
            target_size: Target number of examples
            **kwargs: Supports ``use_clustering`` (bool) to enable k-means clustering.

        Returns:
            SubsetSelectionResult with diverse examples
        """
        use_clustering_value = kwargs.get("use_clustering", True)
        if not isinstance(use_clustering_value, bool):
            raise TypeError("use_clustering must be a boolean") from None
        use_clustering = use_clustering_value

        if not dataset.examples:
            return SubsetSelectionResult(
                selected_examples=[],
                selection_strategy="diverse_sampling",
                original_size=0,
                selected_size=0,
                reduction_ratio=0.0,
                diversity_score=0.0,
                confidence_score=0.0,
            )

        if target_size >= len(dataset.examples):
            logger.warning("Target size >= dataset size, returning full dataset")
            return SubsetSelectionResult(
                selected_examples=dataset.examples,
                selection_strategy="diverse_sampling",
                original_size=len(dataset.examples),
                selected_size=len(dataset.examples),
                reduction_ratio=_compute_reduction_ratio(
                    len(dataset.examples), len(dataset.examples)
                ),
                diversity_score=1.0,
                confidence_score=1.0,
            )

        # Extract text features from examples
        text_features = self._extract_text_features(dataset.examples)

        if use_clustering and len(text_features) > target_size:
            selected_indices = await self._cluster_based_selection(
                text_features, target_size
            )
        else:
            selected_indices = await self._similarity_based_selection(
                text_features, target_size
            )

        selected_examples = [dataset.examples[i] for i in selected_indices]

        # Calculate diversity score
        diversity_score = self._calculate_diversity_score(
            text_features, selected_indices
        )

        return SubsetSelectionResult(
            selected_examples=selected_examples,
            selection_strategy="diverse_sampling",
            original_size=len(dataset.examples),
            selected_size=len(selected_examples),
            reduction_ratio=_compute_reduction_ratio(
                len(dataset.examples), len(selected_examples)
            ),
            diversity_score=diversity_score,
            confidence_score=0.8,  # High confidence in diverse sampling
        )

    def _extract_text_features(self, examples: list[EvaluationExample]) -> list[str]:
        """Extract text content from examples for feature analysis."""
        text_features = []

        for example in examples:
            # Combine input and output text
            input_text = (
                json.dumps(example.input_data)
                if isinstance(example.input_data, dict)
                else str(example.input_data)
            )

            output_text = str(example.expected_output)
            combined_text = f"{input_text} {output_text}"
            text_features.append(combined_text)

        return text_features

    async def _cluster_based_selection(
        self, text_features: list[str], target_size: int
    ) -> list[int]:
        """Select examples using k-means clustering for diversity."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, using random sampling")
            return random.sample(range(len(text_features)), target_size)

        # Vectorize text features
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        try:
            feature_matrix = vectorizer.fit_transform(text_features)
        except (ValueError, ImportError):
            # Fallback to random sampling if TF-IDF fails
            logger.warning("TF-IDF vectorization failed, using random sampling")
            return random.sample(range(len(text_features)), target_size)

        # Perform clustering
        n_clusters = min(target_size, len(text_features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed)
        cluster_labels = kmeans.fit_predict(feature_matrix)

        # Select one example from each cluster
        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                # Select example closest to cluster center
                cluster_features = feature_matrix[cluster_indices]
                center = kmeans.cluster_centers_[cluster_id]
                distances = cosine_similarity(cluster_features, center.reshape(1, -1))
                best_idx = cluster_indices[np.argmax(distances)]
                selected_indices.append(best_idx)

        # If we need more examples, add random ones
        while len(selected_indices) < target_size:
            remaining_indices = set(range(len(text_features))) - set(selected_indices)
            if remaining_indices:
                selected_indices.append(random.choice(list(remaining_indices)))
            else:
                break

        return selected_indices[:target_size]

    async def _similarity_based_selection(
        self, text_features: list[str], target_size: int
    ) -> list[int]:
        """Select examples using similarity-based diverse sampling."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, using random sampling")
            return random.sample(range(len(text_features)), target_size)

        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        try:
            feature_matrix = vectorizer.fit_transform(text_features)
        except (ValueError, ImportError):
            # Fallback to random sampling
            return random.sample(range(len(text_features)), target_size)

        selected_indices = []
        remaining_indices = set(range(len(text_features)))

        # Start with random example
        first_idx = random.choice(list(remaining_indices))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Iteratively select most diverse examples
        while len(selected_indices) < target_size and remaining_indices:
            max_min_distance: float = 0.0
            best_idx = None

            for candidate_idx in remaining_indices:
                # Calculate minimum distance to already selected examples
                min_distance = float("inf")
                for selected_idx in selected_indices:
                    similarity = cosine_similarity(
                        feature_matrix[candidate_idx], feature_matrix[selected_idx]
                    )[0, 0]
                    distance = 1 - similarity
                    min_distance = min(min_distance, distance)

                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = candidate_idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break

        return selected_indices

    def _calculate_diversity_score(
        self, text_features: list[str], selected_indices: list[int]
    ) -> float:
        """Calculate diversity score for selected examples."""
        if len(selected_indices) < 2 or not text_features:
            return 1.0

        if not SKLEARN_AVAILABLE:
            return 0.5  # Default moderate diversity score

        try:
            vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
            feature_matrix = vectorizer.fit_transform(text_features)

            selected_features = feature_matrix[selected_indices]
            similarity_matrix = cosine_similarity(selected_features)

            # Average pairwise distance (1 - similarity)
            distances = 1 - similarity_matrix
            avg_distance = np.mean(distances[np.triu_indices(len(distances), k=1)])

            return cast(float, min(1.0, avg_distance))
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.5  # Default moderate diversity score


class RepresentativeSampling(BaseSubsetSelector):
    """Representative sampling strategy using stratified sampling."""

    def __init__(self, random_seed: int = 42) -> None:
        """Initialize representative sampling selector."""
        self.random_seed = random_seed
        self._random = random.Random(random_seed)

    async def select_subset(
        self, dataset: Dataset, target_size: int, **kwargs: Any
    ) -> SubsetSelectionResult:
        """Select representative examples maintaining output distribution.

        Args:
            dataset: Original dataset
            target_size: Target number of examples
            **kwargs: Supports ``balance_outputs`` (bool) to maintain output distribution.

        Returns:
            SubsetSelectionResult with representative examples
        """
        balance_outputs_value = kwargs.get("balance_outputs", True)
        if not isinstance(balance_outputs_value, bool):
            raise TypeError("balance_outputs must be a boolean") from None
        balance_outputs = balance_outputs_value

        if not dataset.examples:
            return SubsetSelectionResult(
                selected_examples=[],
                selection_strategy="representative_sampling",
                original_size=0,
                selected_size=0,
                reduction_ratio=0.0,
                diversity_score=0.0,
                confidence_score=0.0,
            )

        if target_size >= len(dataset.examples):
            return SubsetSelectionResult(
                selected_examples=dataset.examples,
                selection_strategy="representative_sampling",
                original_size=len(dataset.examples),
                selected_size=len(dataset.examples),
                reduction_ratio=_compute_reduction_ratio(
                    len(dataset.examples), len(dataset.examples)
                ),
                diversity_score=1.0,
                confidence_score=1.0,
            )

        if balance_outputs:
            selected_examples = self._stratified_sampling(dataset.examples, target_size)
        else:
            selected_examples = self._random.sample(dataset.examples, target_size)

        return SubsetSelectionResult(
            selected_examples=selected_examples,
            selection_strategy="representative_sampling",
            original_size=len(dataset.examples),
            selected_size=len(selected_examples),
            reduction_ratio=_compute_reduction_ratio(
                len(dataset.examples), len(selected_examples)
            ),
            diversity_score=0.7,  # Moderate diversity
            confidence_score=0.9,  # High confidence in representation
        )

    def _stratified_sampling(
        self, examples: list[EvaluationExample], target_size: int
    ) -> list[EvaluationExample]:
        """Perform stratified sampling to maintain output distribution."""
        if not examples:
            return []

        # Group by expected output
        output_groups: dict[str, list[EvaluationExample]] = {}
        for example in examples:
            output_key = str(example.expected_output)
            if output_key not in output_groups:
                output_groups[output_key] = []
            output_groups[output_key].append(example)

        # Calculate proportional sizes
        total_examples = len(examples)
        selected_examples = []

        for _output_key, group_examples in output_groups.items():
            group_proportion = len(group_examples) / total_examples
            group_target = max(1, int(target_size * group_proportion))
            group_target = min(group_target, len(group_examples))

            group_selected = self._random.sample(group_examples, group_target)
            selected_examples.extend(group_selected)

        # Adjust if we have too many or too few
        if len(selected_examples) > target_size:
            selected_examples = self._random.sample(selected_examples, target_size)
        elif len(selected_examples) < target_size:
            # Add random examples from remaining pool
            remaining = [ex for ex in examples if ex not in selected_examples]
            needed = target_size - len(selected_examples)
            if remaining:
                additional = self._random.sample(remaining, min(needed, len(remaining)))
                selected_examples.extend(additional)

        return selected_examples


class HighConfidenceSampling(BaseSubsetSelector):
    """High confidence sampling using uncertainty and error analysis."""

    def __init__(self, random_seed: int = 42) -> None:
        """Initialize high confidence sampling selector."""
        self.random_seed = random_seed

    async def select_subset(
        self, dataset: Dataset, target_size: int, **kwargs: Any
    ) -> SubsetSelectionResult:
        """Select high-confidence examples for reliable optimization.

        Args:
            dataset: Original dataset
            target_size: Target number of examples
            **kwargs: Supports ``prioritize_difficult`` (bool) to prioritize challenging examples.

        Returns:
            SubsetSelectionResult with high-confidence examples
        """
        prioritize_difficult_value = kwargs.get("prioritize_difficult", True)
        if not isinstance(prioritize_difficult_value, bool):
            raise TypeError("prioritize_difficult must be a boolean") from None
        prioritize_difficult = prioritize_difficult_value

        if not dataset.examples:
            return SubsetSelectionResult(
                selected_examples=[],
                selection_strategy="high_confidence_sampling",
                original_size=0,
                selected_size=0,
                reduction_ratio=0.0,
                diversity_score=0.0,
                confidence_score=0.0,
            )

        if target_size >= len(dataset.examples):
            return SubsetSelectionResult(
                selected_examples=dataset.examples,
                selection_strategy="high_confidence_sampling",
                original_size=len(dataset.examples),
                selected_size=len(dataset.examples),
                reduction_ratio=_compute_reduction_ratio(
                    len(dataset.examples), len(dataset.examples)
                ),
                diversity_score=0.8,
                confidence_score=1.0,
            )

        # Score examples by difficulty/importance
        example_scores = self._score_examples(dataset.examples)

        # Sort by score (high to low if prioritizing difficult)
        sorted_examples = sorted(
            zip(dataset.examples, example_scores, strict=False),
            key=lambda x: x[1],
            reverse=prioritize_difficult,
        )

        selected_examples = [ex for ex, _ in sorted_examples[:target_size]]

        return SubsetSelectionResult(
            selected_examples=selected_examples,
            selection_strategy="high_confidence_sampling",
            original_size=len(dataset.examples),
            selected_size=len(selected_examples),
            reduction_ratio=_compute_reduction_ratio(
                len(dataset.examples), len(selected_examples)
            ),
            diversity_score=0.6,  # Lower diversity, higher confidence
            confidence_score=0.95,  # Very high confidence
        )

    def _score_examples(self, examples: list[EvaluationExample]) -> list[float]:
        """Score examples by difficulty/importance."""
        scores = []

        for example in examples:
            score = 0.0

            # Length-based scoring (longer = more complex)
            input_length = len(str(example.input_data))
            output_length = len(str(example.expected_output))
            score += (input_length + output_length) / 1000

            # Keyword-based scoring (certain keywords indicate difficulty)
            text = f"{example.input_data} {example.expected_output}".lower()
            difficulty_keywords = [
                "complex",
                "difficult",
                "challenging",
                "edge",
                "corner",
                "exception",
                "error",
                "fail",
                "ambiguous",
                "unclear",
            ]
            for keyword in difficulty_keywords:
                if keyword in text:
                    score += 0.5

            # Uniqueness scoring (unique outputs are more valuable)
            output_hash = hashlib.sha256(
                str(example.expected_output).encode()
            ).hexdigest()
            score += len(set(output_hash)) / 16  # Normalized by max hex chars

            scores.append(score)

        return scores


class SmartSubsetSelector:
    """Intelligent subset selector that combines multiple strategies."""

    def __init__(self) -> None:
        """Initialize smart subset selector."""
        self.diverse_sampler = DiverseSampling()
        self.representative_sampler = RepresentativeSampling()
        self.confidence_sampler = HighConfidenceSampling()

    async def select_optimal_subset(
        self, dataset: Dataset, target_reduction: float = 0.65, strategy: str = "auto"
    ) -> Dataset:
        """Select optimal subset using intelligent strategy selection.

        Args:
            dataset: Original dataset
            target_reduction: Target reduction ratio (0.0-1.0)
            strategy: Selection strategy ("auto", "diverse", "representative", "confident")

        Returns:
            New Dataset with selected subset
        """
        target_size = max(1, int(len(dataset.examples) * (1 - target_reduction)))

        if not dataset.examples:
            logger.warning("Dataset is empty; returning original dataset unchanged.")
            return dataset

        if strategy == "auto":
            strategy = self._choose_optimal_strategy(dataset, target_size)

        # Select subset using chosen strategy
        if strategy == "diverse":
            result = await self.diverse_sampler.select_subset(dataset, target_size)
        elif strategy == "representative":
            result = await self.representative_sampler.select_subset(
                dataset, target_size
            )
        elif strategy == "confident":
            result = await self.confidence_sampler.select_subset(dataset, target_size)
        else:
            # Default to diverse sampling
            result = await self.diverse_sampler.select_subset(dataset, target_size)

        logger.info(
            f"Selected {result.selected_size}/{result.original_size} examples "
            f"using {result.selection_strategy} strategy "
            f"(diversity: {result.diversity_score:.2f}, confidence: {result.confidence_score:.2f})"
        )

        return Dataset(
            examples=result.selected_examples, name=f"{dataset.name}_subset_{strategy}"
        )

    def _choose_optimal_strategy(self, dataset: Dataset, target_size: int) -> str:
        """Choose optimal strategy based on dataset characteristics."""
        num_examples = len(dataset.examples)
        if num_examples == 0:
            return "diverse"

        # Analyze output diversity
        unique_outputs = len({str(ex.expected_output) for ex in dataset.examples})
        output_diversity = unique_outputs / num_examples if num_examples else 0.0

        # Analyze input complexity
        input_lengths = [len(str(ex.input_data)) for ex in dataset.examples]
        avg_input_length = float(np.mean(input_lengths)) if input_lengths else 0.0

        # Strategy selection logic
        if output_diversity > 0.8:
            # High output diversity -> use diverse sampling
            return "diverse"
        elif output_diversity < 0.3:
            # Low output diversity -> use representative sampling
            return "representative"
        elif avg_input_length > 500:
            # Complex inputs -> use confident sampling
            return "confident"
        else:
            # Default to diverse sampling
            return "diverse"
