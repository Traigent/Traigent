"""Built-in retrieval functions for RAG.

Provides common retrieval strategies that can be used as tunable callables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from traigent.api.parameter_ranges import Choices

from .callable import TunedCallable


class Retrievers:
    """Built-in retrieval functions for RAG optimization.

    Example:
        ```python
        @traigent.optimize(
            retriever=Retrievers.as_choices(),
            k=IntRange(1, 10),
        )
        def rag_agent(query: str) -> str:
            config = traigent.get_config()
            docs = Retrievers.invoke(
                config["retriever"],
                vector_store,
                query,
                k=config["k"]
            )
            ...
        ```
    """

    @staticmethod
    def similarity(vector_store: Any, query: str, k: int = 3) -> list:
        """Similarity-based retrieval.

        Args:
            vector_store: Vector store with similarity_search method
            query: Query string
            k: Number of documents to retrieve

        Returns:
            List of retrieved documents
        """
        if hasattr(vector_store, "similarity_search"):
            return vector_store.similarity_search(query, k=k)
        raise AttributeError("vector_store must have similarity_search method")

    @staticmethod
    def mmr(
        vector_store: Any,
        query: str,
        k: int = 3,
        lambda_mult: float = 0.5,
    ) -> list:
        """Maximal Marginal Relevance retrieval.

        Balances relevance with diversity in retrieved documents.

        Args:
            vector_store: Vector store with max_marginal_relevance_search method
            query: Query string
            k: Number of documents to retrieve
            lambda_mult: Balance between relevance (1.0) and diversity (0.0)

        Returns:
            List of retrieved documents
        """
        if hasattr(vector_store, "max_marginal_relevance_search"):
            return vector_store.max_marginal_relevance_search(
                query, k=k, lambda_mult=lambda_mult
            )
        # Fallback to similarity search
        return Retrievers.similarity(vector_store, query, k=k)

    @staticmethod
    def similarity_with_score(
        vector_store: Any,
        query: str,
        k: int = 3,
        score_threshold: float = 0.0,
    ) -> list:
        """Similarity retrieval with score filtering.

        Args:
            vector_store: Vector store with similarity_search_with_score method
            query: Query string
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of retrieved documents above threshold
        """
        if hasattr(vector_store, "similarity_search_with_score"):
            results = vector_store.similarity_search_with_score(query, k=k)
            # Filter by threshold and return just documents
            return [doc for doc, score in results if score >= score_threshold]
        return Retrievers.similarity(vector_store, query, k=k)

    @classmethod
    def as_tuned_callable(cls) -> TunedCallable:
        """Get as TunedCallable with per-retriever parameters.

        Returns:
            TunedCallable configured for retriever selection
        """
        from traigent.api.parameter_ranges import Range

        return TunedCallable(
            name="retriever",
            callables={
                "similarity": cls.similarity,
                "mmr": cls.mmr,
                "similarity_with_score": cls.similarity_with_score,
            },
            parameters={
                "mmr": {"lambda_mult": Range(0.0, 1.0, default=0.5)},
                "similarity_with_score": {
                    "score_threshold": Range(0.0, 1.0, default=0.5)
                },
            },
            description="Document retrieval strategies for RAG",
        )

    @classmethod
    def as_choices(cls) -> Choices:
        """Get as Choices for configuration space.

        Returns:
            Choices instance with retriever options
        """
        return cls.as_tuned_callable().as_choices()

    @classmethod
    def invoke(cls, name: str, *args: Any, **kwargs: Any) -> list:
        """Invoke a retriever by name.

        Args:
            name: Retriever name (similarity, mmr, similarity_with_score)
            *args: Positional arguments (vector_store, query)
            **kwargs: Keyword arguments (k, lambda_mult, etc.)

        Returns:
            List of retrieved documents
        """
        return cls.as_tuned_callable().invoke(name, *args, **kwargs)
