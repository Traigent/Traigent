"""Canonical TVAR name → ParameterRange preset mapping.

Maps detection-canonical names (e.g. ``"temperature"``, ``"max_tokens"``)
to the rich factory presets already defined in
``traigent.api.parameter_ranges``.
"""

from __future__ import annotations

from typing import Any


def get_preset_range(canonical_name: str) -> dict[str, Any] | None:
    """Return range type + kwargs for a canonical TVAR name, or *None*.

    Returns
    -------
    dict with ``"range_type"`` and ``"kwargs"`` keys, matching the format
    used by ``SuggestedRange`` in the detection module.
    """
    return _CANONICAL_PRESETS.get(canonical_name)


def has_preset(canonical_name: str) -> bool:
    """Check whether a canonical preset exists for *canonical_name*."""
    return canonical_name in _CANONICAL_PRESETS


def all_canonical_names() -> frozenset[str]:
    """Return all canonical names that have a preset range."""
    return frozenset(_CANONICAL_PRESETS.keys())


# ---------------------------------------------------------------------------
# Preset catalog
#
# Each entry mirrors the factory methods on Range / IntRange / Choices in
# traigent/api/parameter_ranges.py but stored as plain dicts so that the
# config_generator module does not import heavyweight ParameterRange objects
# at module level.
# ---------------------------------------------------------------------------

_CANONICAL_PRESETS: dict[str, dict[str, Any]] = {
    # --- Continuous float ranges (Range) ---
    "temperature": {
        "range_type": "Range",
        "kwargs": {"low": 0.0, "high": 1.0, "default": 0.7},
    },
    "top_p": {
        "range_type": "Range",
        "kwargs": {"low": 0.1, "high": 1.0, "default": 0.9},
    },
    "frequency_penalty": {
        "range_type": "Range",
        "kwargs": {"low": 0.0, "high": 2.0, "default": 0.0},
    },
    "presence_penalty": {
        "range_type": "Range",
        "kwargs": {"low": 0.0, "high": 2.0, "default": 0.0},
    },
    "similarity_threshold": {
        "range_type": "Range",
        "kwargs": {"low": 0.0, "high": 1.0},
    },
    "mmr_lambda": {
        "range_type": "Range",
        "kwargs": {"low": 0.0, "high": 1.0},
    },
    "chunk_overlap_ratio": {
        "range_type": "Range",
        "kwargs": {"low": 0.0, "high": 0.5},
    },
    # --- Integer ranges (IntRange) ---
    "max_tokens": {
        "range_type": "IntRange",
        "kwargs": {"low": 256, "high": 4096},
    },
    "k": {
        "range_type": "IntRange",
        "kwargs": {"low": 1, "high": 10},
    },
    "chunk_size": {
        "range_type": "IntRange",
        "kwargs": {"low": 100, "high": 1000},
    },
    "chunk_overlap": {
        "range_type": "IntRange",
        "kwargs": {"low": 0, "high": 200},
    },
    "few_shot_count": {
        "range_type": "IntRange",
        "kwargs": {"low": 0, "high": 10},
    },
    "batch_size": {
        "range_type": "IntRange",
        "kwargs": {"low": 1, "high": 64, "default": 16},
    },
    "n": {
        "range_type": "IntRange",
        "kwargs": {"low": 1, "high": 5},
    },
    "seed": {
        "range_type": "IntRange",
        "kwargs": {"low": 0, "high": 1000},
    },
    "top_k": {
        "range_type": "IntRange",
        "kwargs": {"low": 1, "high": 100},
    },
    # --- Categorical (Choices) ---
    "model": {
        "range_type": "Choices",
        "kwargs": {
            "values": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano"],
        },
    },
    "prompting_strategy": {
        "range_type": "Choices",
        "kwargs": {
            "values": ["direct", "chain_of_thought", "react", "self_consistency"],
        },
    },
    "context_format": {
        "range_type": "Choices",
        "kwargs": {
            "values": ["bullet", "numbered", "xml", "markdown", "json"],
        },
    },
    "retriever_type": {
        "range_type": "Choices",
        "kwargs": {
            "values": ["similarity", "mmr", "bm25", "hybrid"],
        },
    },
    "embedding_model": {
        "range_type": "Choices",
        "kwargs": {
            "values": [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ],
        },
    },
    "reranker_model": {
        "range_type": "Choices",
        "kwargs": {
            "values": [
                "none",
                "cohere-rerank-v3",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "llm-rerank",
            ],
        },
    },
}
