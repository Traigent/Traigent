"""
Configuration for token budget optimization.

This module defines the search space for optimizing token allocation across
different context components while maintaining task performance.
"""

# CUSTOM LOGIC - NOT PART OF TRAIGENT:
# This example implements custom budget allocation logic.
# Token budget management is not a built-in TraiGent feature.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AllocationStrategy(Enum):
    """Token allocation strategies."""

    FIXED_PERCENTAGES = "fixed_percentages"
    DYNAMIC_PRIORITY = "dynamic_priority"
    IMPORTANCE_WEIGHTED = "importance_weighted"
    QUERY_TYPE_BASED = "query_type_based"
    ITERATIVE_FITTING = "iterative_fitting"


class TruncationStrategy(Enum):
    """Text truncation methods."""

    SIMPLE_CUTOFF = "simple_cutoff"
    SENTENCE_BOUNDARY = "sentence_boundary"
    SEMANTIC_UNITS = "semantic_units"
    IMPORTANCE_SCORING = "importance_scoring"
    RECURSIVE_SUMMARIZATION = "recursive_summarization"


class ContextSelection(Enum):
    """Context selection strategies."""

    RECENCY_FIRST = "recency_first"
    RELEVANCE_FIRST = "relevance_first"
    DIVERSITY_BALANCED = "diversity_balanced"
    INFORMATION_DENSITY = "information_density"


@dataclass
class TokenBudgetConfig:
    """Configuration for token budget optimization."""

    # Allocation strategy
    allocation_strategy: str = "importance_weighted"

    # Budget allocation percentages
    system_prompt_pct: float = 0.10
    examples_pct: float = 0.20
    retrieved_context_pct: float = 0.55
    conversation_history_pct: float = 0.10
    buffer_pct: float = 0.05

    # Truncation method
    truncation_strategy: str = "sentence_boundary"

    # Context selection approach
    context_selection: str = "relevance_first"

    # Adaptive features
    adaptive_reallocation: bool = True
    query_classification: bool = True

    # Output buffer for generation
    output_buffer: int = 512

    # Advanced options
    preserve_critical_info: bool = True
    smart_compression: bool = False
    priority_weighting: dict[str, float] | None = None

    def __post_init__(self):
        """Initialize default priority weighting if not provided."""
        if self.priority_weighting is None:
            self.priority_weighting = {
                "system_prompt": 1.0,
                "examples": 0.8,
                "retrieved_context": 0.9,
                "conversation_history": 0.6,
                "buffer": 1.0,
            }

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "allocation_strategy": self.allocation_strategy,
            "system_prompt_pct": self.system_prompt_pct,
            "examples_pct": self.examples_pct,
            "retrieved_context_pct": self.retrieved_context_pct,
            "conversation_history_pct": self.conversation_history_pct,
            "buffer_pct": self.buffer_pct,
            "truncation_strategy": self.truncation_strategy,
            "context_selection": self.context_selection,
            "adaptive_reallocation": self.adaptive_reallocation,
            "query_classification": self.query_classification,
            "output_buffer": self.output_buffer,
            "preserve_critical_info": self.preserve_critical_info,
            "smart_compression": self.smart_compression,
            "priority_weighting": self.priority_weighting,
        }

    def get_total_allocation(self) -> float:
        """Get total allocation percentage."""
        return (
            self.system_prompt_pct
            + self.examples_pct
            + self.retrieved_context_pct
            + self.conversation_history_pct
            + self.buffer_pct
        )

    def normalize_allocations(self) -> None:
        """Normalize allocations to sum to 1.0."""
        total = self.get_total_allocation()
        if total > 0:
            self.system_prompt_pct /= total
            self.examples_pct /= total
            self.retrieved_context_pct /= total
            self.conversation_history_pct /= total
            self.buffer_pct /= total


# TraiGent search space for token budget optimization
TOKEN_BUDGET_SEARCH_SPACE = {
    # Allocation strategies
    "allocation_strategy": [
        "fixed_percentages",  # Simple fixed allocation
        "dynamic_priority",  # Priority-based dynamic allocation
        "importance_weighted",  # Content importance weighting
        "query_type_based",  # Adapt based on query type
        "iterative_fitting",  # Iterative budget fitting
    ],
    # Component budget allocations (percentages)
    "system_prompt_pct": [0.05, 0.10, 0.15, 0.20],
    "examples_pct": [0.0, 0.10, 0.20, 0.30, 0.40],
    "retrieved_context_pct": [0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    "conversation_history_pct": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
    "buffer_pct": [0.05, 0.10, 0.15],
    # Truncation methods
    "truncation_strategy": [
        "simple_cutoff",  # Cut at token limit
        "sentence_boundary",  # Preserve sentence boundaries
        "semantic_units",  # Preserve semantic coherence
        "importance_scoring",  # Keep most important content
        "recursive_summarization",  # Summarize excess content
    ],
    # Context selection strategies
    "context_selection": [
        "recency_first",  # Newest content first
        "relevance_first",  # Most relevant content first
        "diversity_balanced",  # Balance relevance and diversity
        "information_density",  # Highest information content first
    ],
    # Adaptive features
    "adaptive_reallocation": [True, False],
    "query_classification": [True, False],
    # Output buffer sizes
    "output_buffer": [256, 512, 768, 1024],
    # Advanced options
    "preserve_critical_info": [True, False],
    "smart_compression": [True, False],
}


def create_token_budget_config(**config_params) -> TokenBudgetConfig:
    """Create a TokenBudgetConfig from parameter dictionary."""
    # Normalize allocations if provided
    config = TokenBudgetConfig(**config_params)
    if config.get_total_allocation() != 1.0:
        config.normalize_allocations()
    return config


def calculate_token_allocation(
    total_budget: int, config: TokenBudgetConfig
) -> dict[str, int]:
    """Calculate token allocation for each component."""

    # Reserve output buffer first
    available_tokens = total_budget - config.output_buffer

    if available_tokens <= 0:
        return {
            "system_prompt": 0,
            "examples": 0,
            "retrieved_context": 0,
            "conversation_history": 0,
            "buffer": 0,
            "output_buffer": total_budget,
        }

    # Allocate based on percentages
    allocation = {
        "system_prompt": int(available_tokens * config.system_prompt_pct),
        "examples": int(available_tokens * config.examples_pct),
        "retrieved_context": int(available_tokens * config.retrieved_context_pct),
        "conversation_history": int(available_tokens * config.conversation_history_pct),
        "buffer": int(available_tokens * config.buffer_pct),
        "output_buffer": config.output_buffer,
    }

    # Handle rounding errors
    allocated_total = sum(allocation.values())
    if allocated_total < total_budget:
        # Add remaining tokens to retrieved_context (usually largest component)
        allocation["retrieved_context"] += total_budget - allocated_total
    elif allocated_total > total_budget:
        # Remove excess from buffer or retrieved_context
        excess = allocated_total - total_budget
        if allocation["buffer"] >= excess:
            allocation["buffer"] -= excess
        else:
            allocation["retrieved_context"] -= excess

    return allocation


def get_budget_scenarios() -> list[dict[str, Any]]:
    """Get different budget scenarios for testing."""

    return [
        {
            "name": "tight_budget",
            "total_tokens": 2000,
            "description": "Tight budget scenario - requires aggressive optimization",
        },
        {
            "name": "standard_budget",
            "total_tokens": 4000,
            "description": "Standard budget scenario - balanced allocation",
        },
        {
            "name": "generous_budget",
            "total_tokens": 8000,
            "description": "Generous budget scenario - focus on quality",
        },
        {
            "name": "enterprise_budget",
            "total_tokens": 16000,
            "description": "Enterprise budget scenario - comprehensive context",
        },
    ]


def get_baseline_configs() -> list[dict[str, Any]]:
    """Get baseline token budget configurations."""

    return [
        {
            "name": "naive_allocation",
            "allocation_strategy": "fixed_percentages",
            "system_prompt_pct": 0.05,
            "examples_pct": 0.10,
            "retrieved_context_pct": 0.70,
            "conversation_history_pct": 0.10,
            "buffer_pct": 0.05,
            "truncation_strategy": "simple_cutoff",
            "context_selection": "recency_first",
            "adaptive_reallocation": False,
            "query_classification": False,
            "output_buffer": 512,
            "preserve_critical_info": False,
            "smart_compression": False,
        },
        {
            "name": "balanced_allocation",
            "allocation_strategy": "importance_weighted",
            "system_prompt_pct": 0.10,
            "examples_pct": 0.15,
            "retrieved_context_pct": 0.60,
            "conversation_history_pct": 0.10,
            "buffer_pct": 0.05,
            "truncation_strategy": "sentence_boundary",
            "context_selection": "relevance_first",
            "adaptive_reallocation": True,
            "query_classification": False,
            "output_buffer": 512,
            "preserve_critical_info": True,
            "smart_compression": False,
        },
        {
            "name": "smart_allocation",
            "allocation_strategy": "query_type_based",
            "system_prompt_pct": 0.08,
            "examples_pct": 0.12,
            "retrieved_context_pct": 0.65,
            "conversation_history_pct": 0.10,
            "buffer_pct": 0.05,
            "truncation_strategy": "importance_scoring",
            "context_selection": "information_density",
            "adaptive_reallocation": True,
            "query_classification": True,
            "output_buffer": 512,
            "preserve_critical_info": True,
            "smart_compression": True,
        },
    ]


def estimate_token_usage(text: str, model_type: str = "gpt") -> int:
    """Estimate token usage for text."""
    # Simple approximation: ~1.3 words per token for English
    if model_type.startswith("gpt"):
        return int(len(text.split()) * 1.3)
    elif model_type.startswith("claude"):
        return int(len(text.split()) * 1.2)
    else:
        return int(len(text.split()) * 1.25)


def calculate_cost_savings(
    baseline_tokens: int, optimized_tokens: int, cost_per_1k_tokens: float = 0.002
) -> dict[str, float]:
    """Calculate cost savings from token optimization."""

    baseline_cost = (baseline_tokens / 1000) * cost_per_1k_tokens
    optimized_cost = (optimized_tokens / 1000) * cost_per_1k_tokens

    savings = baseline_cost - optimized_cost
    savings_percentage = (savings / baseline_cost) * 100 if baseline_cost > 0 else 0

    return {
        "baseline_cost": baseline_cost,
        "optimized_cost": optimized_cost,
        "savings": savings,
        "savings_percentage": savings_percentage,
        "token_reduction": baseline_tokens - optimized_tokens,
        "token_reduction_percentage": (
            ((baseline_tokens - optimized_tokens) / baseline_tokens) * 100
            if baseline_tokens > 0
            else 0
        ),
    }
