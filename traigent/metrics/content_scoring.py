"""Content-based example scoring using TF-IDF and cosine similarity.

This module computes content-based quality metrics for dataset examples by analyzing
the text content. Requires scikit-learn for TF-IDF vectorization.

Metrics Computed:
- Uniqueness: How different each example is from all other examples (1.0 = unique, 0.0 = duplicate)
- Novelty: How different each example is from the dataset average (1.0 = novel, 0.0 = average)

Thread Safety Warning:
    ContentScorer instances are NOT thread-safe. The TfidfVectorizer maintains
    internal state that can be corrupted by concurrent access.

    Safe Usage Patterns:
    1. Per-trial instantiation (recommended for parallel trials):
       def evaluate_trial(config):
           scorer = ContentScorer()  # New instance per trial
           scores = scorer.compute_uniqueness_scores(inputs)

    2. Shared instance with external locking:
       scorer = ContentScorer()
       lock = threading.Lock()
       with lock:
           scores = scorer.compute_uniqueness_scores(inputs)

    3. Thread-local storage:
       import threading
       thread_local = threading.local()
       def get_scorer():
           if not hasattr(thread_local, 'scorer'):
               thread_local.scorer = ContentScorer()
           return thread_local.scorer
"""

import threading
from typing import Dict, List, Optional

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    np = None  # type: ignore


class ContentScorer:
    """Compute content-based example scores.

    Thread Safety: NOT thread-safe. Instantiate per-thread or use external locking.
    See module docstring for safe usage patterns.
    """

    def __init__(self):
        """Initialize content scorer.

        Raises:
            ImportError: If scikit-learn is not installed
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for content scoring. "
                "Install with: pip install traigent[analytics]"
            )

        self.tfidf_vectorizer: Optional[TfidfVectorizer] = TfidfVectorizer(max_features=500)
        self._lock = threading.Lock()  # For external thread-safe usage

    def compute_uniqueness_scores(self, example_inputs: List[str]) -> Dict[int, float]:
        """Compute uniqueness score for each example.

        Uniqueness measures how different each example is from all other examples
        using TF-IDF vectorization and cosine similarity.

        Args:
            example_inputs: List of input text strings (one per example)

        Returns:
            Dict mapping example_index -> uniqueness_score (0.0-1.0)
            - 1.0 = completely unique (no similar examples)
            - 0.0 = duplicate (identical to another example)

        Example:
            >>> scorer = ContentScorer()
            >>> inputs = ["Hello world", "Goodbye world", "Hello there"]
            >>> scores = scorer.compute_uniqueness_scores(inputs)
            >>> scores[0]  # Uniqueness of "Hello world"
            0.73

        Thread Safety: NOT thread-safe. See module docstring.
        """
        if not SKLEARN_AVAILABLE or len(example_inputs) < 2:
            # Fallback to neutral score if sklearn unavailable or too few examples
            return {i: 0.5 for i in range(len(example_inputs))}

        try:
            # Vectorize inputs using TF-IDF
            assert self.tfidf_vectorizer is not None
            vectors = self.tfidf_vectorizer.fit_transform(example_inputs)
        except Exception:
            # If vectorization fails (e.g., empty strings), return neutral scores
            return {i: 0.5 for i in range(len(example_inputs))}

        # Compute pairwise similarities
        similarities = cosine_similarity(vectors)

        # For each example, uniqueness = 1 - max_similarity_to_others
        uniqueness_scores = {}
        for i in range(len(example_inputs)):
            # Get similarities to all other examples (exclude self)
            other_sims = [similarities[i][j] for j in range(len(example_inputs)) if j != i]
            max_sim = max(other_sims) if other_sims else 0.0
            # Clamp to [0, 1] to handle floating point precision issues
            uniqueness_scores[i] = float(max(0.0, min(1.0, 1.0 - max_sim)))

        return uniqueness_scores

    def compute_novelty_scores(self, example_inputs: List[str]) -> Dict[int, float]:
        """Compute coverage/novelty score for each example.

        Novelty measures how different each example is from the dataset average
        (centroid). Examples far from the centroid are novel; those near it are typical.

        Args:
            example_inputs: List of input text strings (one per example)

        Returns:
            Dict mapping example_index -> novelty_score (0.0-1.0)
            - 1.0 = very novel (far from average)
            - 0.0 = typical (near average)

        Example:
            >>> scorer = ContentScorer()
            >>> inputs = ["Hello world", "Goodbye world", "Hello there"]
            >>> scores = scorer.compute_novelty_scores(inputs)
            >>> scores[0]  # Novelty of "Hello world"
            0.42

        Thread Safety: NOT thread-safe. See module docstring.
        """
        if not SKLEARN_AVAILABLE or len(example_inputs) < 2:
            return {i: 0.5 for i in range(len(example_inputs))}

        try:
            assert self.tfidf_vectorizer is not None
            vectors = self.tfidf_vectorizer.fit_transform(example_inputs)
            # Compute dataset centroid (average vector)
            centroid = vectors.mean(axis=0)

            novelty_scores = {}
            for i in range(len(example_inputs)):
                # Distance from centroid (lower similarity = higher novelty)
                distance = 1.0 - cosine_similarity(vectors[i], centroid)[0][0]
                # Clamp to [0, 1] range
                novelty_scores[i] = float(min(1.0, max(0.0, distance)))

            return novelty_scores
        except Exception:
            return {i: 0.5 for i in range(len(example_inputs))}

    def compute_all_scores(self, example_inputs: List[str]) -> Dict[str, Dict[int, float]]:
        """Compute all content-based scores at once.

        Convenience method to compute both uniqueness and novelty scores in one call.

        Args:
            example_inputs: List of input text strings (one per example)

        Returns:
            Dict with keys "uniqueness" and "novelty", each mapping to
            example_index -> score dict

        Example:
            >>> scorer = ContentScorer()
            >>> inputs = ["Hello world", "Goodbye world"]
            >>> scores = scorer.compute_all_scores(inputs)
            >>> scores["uniqueness"][0]
            0.73
            >>> scores["novelty"][0]
            0.42

        Thread Safety: NOT thread-safe. See module docstring.
        """
        return {
            "uniqueness": self.compute_uniqueness_scores(example_inputs),
            "novelty": self.compute_novelty_scores(example_inputs),
        }
