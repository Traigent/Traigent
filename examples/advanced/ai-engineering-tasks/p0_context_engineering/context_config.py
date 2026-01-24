"""
Context Engineering and RAG Configuration
========================================

Configuration dataclass defining the search space for context engineering optimization.
Based on the Context Engineering and RAG Optimization use case specification.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ContextConfig:
    """Configuration for context engineering and RAG optimization.

    This dataclass defines the complete search space for optimizing
    context assembly in RAG systems, covering modern 2024/2025 approaches.
    """

    # Retrieval strategies (2024/2025 state-of-art)
    retrieval_method: str  # See RETRIEVAL_METHODS below

    # Embedding models (if dense/hybrid)
    embedding_model: str | None  # See EMBEDDING_MODELS below

    # Chunking strategies
    chunk_size: int  # 256, 512, 1024, 2048
    chunk_overlap: int  # 0, 64, 128, 256
    chunking_method: str  # See CHUNKING_METHODS below

    # Reranking approaches
    reranker: str | None  # See RERANKERS below
    rerank_top_k: int  # 20, 50, 100

    # Context assembly
    context_ordering: str  # See ORDERING_STRATEGIES below

    # Token budget allocation (percentages must sum to 1.0)
    budget_allocation: dict[str, float]

    # Few-shot example selection
    example_selection: str  # See EXAMPLE_STRATEGIES below
    n_examples: int  # 0, 1, 3, 5


# Search space definition for Traigent optimization
CONTEXT_ENGINEERING_SEARCH_SPACE = {
    "retrieval_method": [
        "bm25",  # Sparse retrieval
        "dense_embedding",  # Semantic search
        "hybrid_weighted",  # Weighted combination
        "hyde",  # Hypothetical document embedding
        "multi_query_expansion",  # Query decomposition
    ],
    "embedding_model": ["openai_ada_002", "e5_large_v2", "bge_large_en", "voyage_02"],
    "chunk_size": [256, 512, 1024, 2048],
    "chunk_overlap": [0, 64, 128, 256],
    "chunking_method": [
        "fixed_tokens",
        "sentence_boundary",
        "semantic_segments",
        "markdown_sections",
    ],
    "reranker": [
        "none",
        "cohere_rerank_v2",
        "bge_reranker_large",
        "cross_encoder",
        "llm_scorer",
    ],
    "rerank_top_k": [20, 50, 100],
    "context_ordering": [
        "relevance_score",
        "diversity_first",
        "temporal_order",
        "hierarchical",
    ],
    # CUSTOM LOGIC - NOT PART OF TRAIGENT:
    # Token budget allocation is a custom feature implemented in this example.
    # Traigent does not have built-in budget allocation - this is custom logic.
    # Budget allocation ranges (will be normalized to sum to 1.0)
    "budget_retrieved_docs": [0.4, 0.5, 0.6, 0.7, 0.8],
    "budget_few_shot": [0.0, 0.1, 0.2, 0.3],
    "budget_history": [0.0, 0.1, 0.2],
    "budget_system": [0.05, 0.1, 0.15],
    "example_selection": [
        "none",
        "random",
        "semantic_similar",
        "diverse_coverage",
        "difficulty_based",
    ],
    "n_examples": [0, 1, 3, 5],
}


def create_context_config(**kwargs) -> ContextConfig:
    """Create a ContextConfig with validation and normalization."""

    # Normalize budget allocation to sum to 1.0
    budget_allocation = {
        "retrieved_docs": kwargs.get("budget_retrieved_docs", 0.6),
        "few_shot_examples": kwargs.get("budget_few_shot", 0.1),
        "conversation_history": kwargs.get("budget_history", 0.1),
        "system_prompt": kwargs.get("budget_system", 0.1),
        "buffer": 0.1,  # Always reserve some buffer
    }

    # Normalize to sum to 1.0
    total = sum(budget_allocation.values())
    if total > 0:
        budget_allocation = {k: v / total for k, v in budget_allocation.items()}

    return ContextConfig(
        retrieval_method=kwargs.get("retrieval_method", "dense_embedding"),
        embedding_model=(
            kwargs.get("embedding_model", "openai_ada_002")
            if kwargs.get("retrieval_method") != "bm25"
            else None
        ),
        chunk_size=kwargs.get("chunk_size", 512),
        chunk_overlap=kwargs.get("chunk_overlap", 64),
        chunking_method=kwargs.get("chunking_method", "sentence_boundary"),
        reranker=kwargs.get("reranker", "none"),
        rerank_top_k=kwargs.get("rerank_top_k", 50),
        context_ordering=kwargs.get("context_ordering", "relevance_score"),
        budget_allocation=budget_allocation,
        example_selection=kwargs.get("example_selection", "none"),
        n_examples=kwargs.get("n_examples", 0),
    )


class DocumentCorpus:
    """Represents a document corpus for retrieval."""

    def __init__(self, documents: list[str], metadata: list[dict] | None = None):
        self.documents = documents
        self.metadata = metadata or [{} for _ in documents]
        self.chunks = []
        self.chunk_metadata = []
        self.embeddings = None

    def chunk_documents(self, config: ContextConfig) -> None:
        """Chunk documents according to configuration."""
        self.chunks = []
        self.chunk_metadata = []

        for doc_idx, doc in enumerate(self.documents):
            doc_chunks = self._chunk_single_document(doc, config)
            self.chunks.extend(doc_chunks)

            # Add metadata for each chunk
            for chunk_idx, _ in enumerate(doc_chunks):
                meta = self.metadata[doc_idx].copy()
                meta["doc_idx"] = doc_idx
                meta["chunk_idx"] = chunk_idx
                self.chunk_metadata.append(meta)

    def _chunk_single_document(self, doc: str, config: ContextConfig) -> list[str]:
        """Chunk a single document based on configuration."""

        handlers = {
            "fixed_tokens": self._chunk_fixed_tokens,
            "sentence_boundary": self._chunk_sentence_boundary,
            "semantic_segments": self._chunk_semantic_segments,
            "markdown_sections": self._chunk_markdown_sections,
        }
        handler = handlers.get(config.chunking_method, lambda d, _: [d])
        return handler(doc, config)

    def _chunk_fixed_tokens(self, doc: str, config: ContextConfig) -> list[str]:
        words = doc.split()
        chunk_size_words = max(1, config.chunk_size // 4)
        overlap_words = max(0, config.chunk_overlap // 4)

        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size_words, len(words))
            chunks.append(" ".join(words[start:end]))
            next_start = start + chunk_size_words - overlap_words
            if next_start <= start:
                break
            start = next_start

        return chunks if chunks else [doc]

    def _chunk_sentence_boundary(self, doc: str, config: ContextConfig) -> list[str]:
        import re

        sentences = [s.strip() for s in re.split(r"[.!?]+", doc) if s.strip()]
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence.split()) * 4
            if current_chunk and current_size + sentence_size > config.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_size = self._apply_sentence_overlap(
                    current_chunk, config.chunk_overlap
                )
            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks if chunks else [doc]

    def _apply_sentence_overlap(
        self, sentences: list[str], overlap_tokens: int
    ) -> tuple[list[str], int]:
        if overlap_tokens <= 0 or not sentences:
            return [], 0
        overlap_sentences = max(1, overlap_tokens // 50)
        kept = sentences[-overlap_sentences:]
        size = sum(len(s.split()) * 4 for s in kept)
        return kept, size

    def _chunk_semantic_segments(self, doc: str, config: ContextConfig) -> list[str]:
        paragraphs = [p.strip() for p in doc.split("\n\n") if p.strip()]
        return paragraphs if paragraphs else [doc]

    def _chunk_markdown_sections(self, doc: str, config: ContextConfig) -> list[str]:
        import re

        sections = [s.strip() for s in re.split(r"^#+\s", doc, flags=re.MULTILINE)]
        return [s for s in sections if s]

    def compute_embeddings(self, config: ContextConfig) -> None:
        """Compute embeddings for chunks based on embedding model."""
        if config.retrieval_method in ["dense_embedding", "hybrid_weighted", "hyde"]:
            # Simulate embedding computation
            # In real implementation, use actual embedding models
            self.embeddings = np.random.randn(
                len(self.chunks), 768
            )  # Standard embedding size


@dataclass
class ContextResult:
    """Result of context assembly."""

    context: str
    metadata: dict[str, Any]
    token_count: int
    retrieved_chunks: list[str]
    selected_examples: list[dict]


class ContextAssembler:
    """Assembles context based on configuration."""

    def __init__(self, config: ContextConfig):
        self.config = config

    def assemble(
        self,
        query: str,
        corpus: DocumentCorpus,
        token_budget: int = 4000,
        conversation_history: list[dict] | None = None,
    ) -> ContextResult:
        """Assemble optimal context for query."""

        # Retrieve relevant chunks
        retrieved_chunks = self._retrieve_chunks(query, corpus)

        # Rerank if configured
        if self.config.reranker != "none":
            retrieved_chunks = self._rerank_chunks(query, retrieved_chunks)

        # Select examples if configured
        selected_examples = (
            self._select_examples(query) if self.config.n_examples > 0 else []
        )

        # Allocate token budget
        allocated_budgets = self._allocate_budget(token_budget)

        # Assemble context components
        context_parts = []

        # System prompt
        if allocated_budgets["system_prompt"] > 0:
            context_parts.append("System: You are a helpful assistant.")

        # Few-shot examples
        if selected_examples and allocated_budgets["few_shot_examples"] > 0:
            examples_text = self._format_examples(selected_examples)
            context_parts.append(f"Examples:\n{examples_text}")

        # Retrieved documents
        if retrieved_chunks and allocated_budgets["retrieved_docs"] > 0:
            # Truncate to budget
            max_chunks = int(
                allocated_budgets["retrieved_docs"] / 100
            )  # Rough estimate
            chunks_text = "\n\n".join(retrieved_chunks[:max_chunks])
            context_parts.append(f"Context:\n{chunks_text}")

        # Conversation history
        if conversation_history and allocated_budgets["conversation_history"] > 0:
            history_text = self._format_history(conversation_history)
            context_parts.append(f"Previous conversation:\n{history_text}")

        # Assemble final context
        context = "\n\n".join(context_parts)

        # Apply ordering strategy
        context = self._apply_ordering(context, context_parts)

        return ContextResult(
            context=context,
            metadata={
                "retrieval_method": self.config.retrieval_method,
                "n_retrieved": len(retrieved_chunks),
                "n_examples": len(selected_examples),
                "ordering": self.config.context_ordering,
            },
            token_count=len(context.split()) * 1.3,  # Rough token estimate
            retrieved_chunks=retrieved_chunks,
            selected_examples=selected_examples,
        )

    def _retrieve_chunks(self, query: str, corpus: DocumentCorpus) -> list[str]:
        """Retrieve relevant chunks based on retrieval method."""

        if not corpus.chunks:
            return []

        if self.config.retrieval_method == "bm25":
            # Simulate BM25 retrieval
            # In real implementation, use actual BM25 algorithm
            import random

            n_retrieve = min(10, len(corpus.chunks))
            return random.sample(corpus.chunks, n_retrieve)

        elif self.config.retrieval_method == "dense_embedding":
            # Simulate dense retrieval
            # In real implementation, compute query embedding and find nearest neighbors
            if corpus.embeddings is not None:
                # Return top-k chunks (simplified)
                n_retrieve = min(10, len(corpus.chunks))
                return corpus.chunks[:n_retrieve]
            else:
                return corpus.chunks[:10]

        elif self.config.retrieval_method == "hybrid_weighted":
            # Combine sparse and dense retrieval
            # Simplified for demo
            n_retrieve = min(10, len(corpus.chunks))
            return corpus.chunks[:n_retrieve]

        elif self.config.retrieval_method == "hyde":
            # Hypothetical Document Embedding
            # Generate hypothetical answer and retrieve similar documents
            # Simplified for demo
            n_retrieve = min(10, len(corpus.chunks))
            return corpus.chunks[:n_retrieve]

        elif self.config.retrieval_method == "multi_query_expansion":
            # Expand query into multiple queries
            # Simplified for demo
            n_retrieve = min(15, len(corpus.chunks))
            return corpus.chunks[:n_retrieve]

        else:
            return corpus.chunks[:10]

    def _rerank_chunks(self, query: str, chunks: list[str]) -> list[str]:
        """Rerank chunks based on reranker configuration."""

        if self.config.reranker == "none":
            return chunks

        # Simulate reranking
        # In real implementation, use actual reranking models

        if self.config.reranker == "cohere_rerank_v2":
            # Simulate Cohere reranking
            # Would use Cohere API in practice
            pass
        elif self.config.reranker == "bge_reranker_large":
            # Simulate BGE reranker
            pass
        elif self.config.reranker == "cross_encoder":
            # Simulate cross-encoder reranking
            pass
        elif self.config.reranker == "llm_scorer":
            # Use LLM to score relevance
            pass

        # Return top-k after reranking
        return chunks[: self.config.rerank_top_k]

    def _select_examples(self, query: str) -> list[dict]:
        """Select few-shot examples based on strategy."""

        # Simulate example selection
        # In real implementation, would have example pool

        examples = []
        for i in range(self.config.n_examples):
            examples.append(
                {"input": f"Example input {i+1}", "output": f"Example output {i+1}"}
            )

        return examples

    def _allocate_budget(self, total_budget: int) -> dict[str, int]:
        """Allocate token budget to different components."""

        allocated = {}
        for component, percentage in self.config.budget_allocation.items():
            allocated[component] = int(total_budget * percentage)

        return allocated

    def _format_examples(self, examples: list[dict]) -> str:
        """Format examples for inclusion in context."""
        formatted = []
        for i, ex in enumerate(examples):
            formatted.append(
                f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
            )
        return "\n\n".join(formatted)

    def _format_history(self, history: list[dict]) -> str:
        """Format conversation history."""
        formatted = []
        for turn in history[-3:]:  # Last 3 turns
            role = turn.get("role", "user")
            content = turn.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def _apply_ordering(self, context: str, parts: list[str]) -> str:
        """Apply ordering strategy to context."""

        if self.config.context_ordering == "relevance_score":
            # Already ordered by relevance
            return context
        elif self.config.context_ordering == "diversity_first":
            # Put diverse content first
            # Simplified for demo
            return context
        elif self.config.context_ordering == "temporal_order":
            # Order by time (if available)
            return context
        elif self.config.context_ordering == "hierarchical":
            # Hierarchical ordering (overview first, then details)
            return context

        return context
