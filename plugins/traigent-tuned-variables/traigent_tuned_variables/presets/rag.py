"""RAG (Retrieval-Augmented Generation) parameter presets.

Provides pre-configured parameter ranges for RAG optimization including
retrieval depth, chunk size, and chunk overlap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from traigent.api.parameter_ranges import IntRange, Range


class RAGPresets:
    """Pre-configured parameter ranges for RAG optimization.

    These presets encode domain knowledge about sensible parameter ranges
    for retrieval-augmented generation workflows.
    """

    @staticmethod
    def k_retrieval(*, max_k: int = 10) -> IntRange:
        """Number of documents to retrieve.

        Args:
            max_k: Maximum number of documents to retrieve

        Returns:
            IntRange instance configured for retrieval depth
        """
        from traigent.api.parameter_ranges import IntRange

        return IntRange(1, max_k, default=3, name="k")

    @staticmethod
    def chunk_size(
        *,
        min_size: int = 100,
        max_size: int = 1000,
        step: int = 100,
    ) -> IntRange:
        """Document chunk size for RAG.

        Args:
            min_size: Minimum chunk size in characters
            max_size: Maximum chunk size in characters
            step: Step size for chunk size optimization

        Returns:
            IntRange instance configured for chunk size
        """
        from traigent.api.parameter_ranges import IntRange

        return IntRange(min_size, max_size, step=step, default=500, name="chunk_size")

    @staticmethod
    def chunk_overlap(
        *,
        min_overlap: int = 0,
        max_overlap: int = 200,
        step: int = 25,
    ) -> IntRange:
        """Document chunk overlap for RAG.

        Args:
            min_overlap: Minimum overlap in characters
            max_overlap: Maximum overlap in characters
            step: Step size for overlap optimization

        Returns:
            IntRange instance configured for chunk overlap
        """
        from traigent.api.parameter_ranges import IntRange

        return IntRange(
            min_overlap, max_overlap, step=step, default=50, name="chunk_overlap"
        )

    @staticmethod
    def similarity_threshold(
        *,
        min_threshold: float = 0.0,
        max_threshold: float = 1.0,
    ) -> Range:
        """Similarity threshold for filtering retrieved documents.

        Args:
            min_threshold: Minimum similarity threshold
            max_threshold: Maximum similarity threshold

        Returns:
            Range instance configured for similarity threshold
        """
        from traigent.api.parameter_ranges import Range

        return Range(
            min_threshold,
            max_threshold,
            default=0.5,
            name="similarity_threshold",
        )

    @staticmethod
    def mmr_lambda(
        *,
        min_lambda: float = 0.0,
        max_lambda: float = 1.0,
    ) -> Range:
        """MMR (Maximal Marginal Relevance) lambda parameter.

        Controls the trade-off between relevance and diversity:
        - lambda=1.0: Pure relevance (most similar)
        - lambda=0.0: Pure diversity (most different from already selected)

        Args:
            min_lambda: Minimum lambda value
            max_lambda: Maximum lambda value

        Returns:
            Range instance configured for MMR lambda
        """
        from traigent.api.parameter_ranges import Range

        return Range(
            min_lambda,
            max_lambda,
            default=0.5,
            name="mmr_lambda",
        )
