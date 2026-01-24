"""
Token budget evaluation system.

This module provides evaluation functions for testing token allocation strategies
and measuring the cost-performance tradeoffs of different budget configurations.
"""

# CUSTOM LOGIC - NOT PART OF TRAIGENT:
# This example implements custom budget allocation logic.
# Token budget management is not a built-in Traigent feature.

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

from budget_config import (
    TokenBudgetConfig,
    calculate_token_allocation,
    estimate_token_usage,
)


@dataclass
class TokenBudgetResult:
    """Result of token budget allocation."""

    allocated_tokens: dict[str, int]
    actual_usage: dict[str, int]
    performance_score: float
    cost_per_query: float
    truncation_losses: dict[str, float]
    processing_time_ms: float


@dataclass
class ContextComponent:
    """A component of the context with content and metadata."""

    component_type: str
    content: str
    priority: float
    is_critical: bool = False
    token_estimate: int = 0

    def __post_init__(self):
        if self.token_estimate == 0:
            self.token_estimate = estimate_token_usage(self.content)


def create_mock_context_components(
    task_type: str = "qa", complexity: str = "medium"
) -> dict[str, list[ContextComponent]]:
    """Create mock context components for different task types."""

    components = {
        "system_prompt": [],
        "examples": [],
        "retrieved_context": [],
        "conversation_history": [],
        "buffer": [],
    }

    # System prompt
    if task_type == "qa":
        system_content = "You are a helpful AI assistant that answers questions accurately and concisely. Always provide sources when possible and admit when you don't know something."
    elif task_type == "code":
        system_content = "You are an expert programmer who helps with coding tasks. Write clean, well-documented code and explain your reasoning. Follow best practices for the specific programming language."
    else:
        system_content = "You are a helpful AI assistant. Be accurate, helpful, and honest in all responses."

    components["system_prompt"].append(
        ContextComponent(
            component_type="system_prompt",
            content=system_content,
            priority=1.0,
            is_critical=True,
        )
    )

    # Examples (few-shot)
    if complexity in ["medium", "complex"]:
        example_contents = [
            "Q: What is the capital of France?\nA: The capital of France is Paris.",
            "Q: How do you implement binary search?\nA: Binary search works by repeatedly dividing the search interval in half...",
            "Q: What causes climate change?\nA: Climate change is primarily caused by greenhouse gas emissions from human activities...",
        ]

        for _i, content in enumerate(
            example_contents[: 2 if complexity == "medium" else 3]
        ):
            components["examples"].append(
                ContextComponent(
                    component_type="examples",
                    content=content,
                    priority=0.8,
                    is_critical=False,
                )
            )

    # Retrieved context (most variable component)
    context_docs = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or classifications based on those patterns.",
        "Deep learning is a subfield of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. These networks are inspired by the structure and function of the human brain.",
        "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in a valuable way.",
        "Computer vision is an AI field that enables machines to interpret and understand visual information from the world, such as images and videos. It combines machine learning, image processing, and pattern recognition.",
        "Reinforcement learning is a type of machine learning where agents learn to make decisions by interacting with an environment and receiving feedback in the form of rewards or penalties.",
    ]

    num_docs = 2 if complexity == "simple" else 3 if complexity == "medium" else 5
    for i, content in enumerate(context_docs[:num_docs]):
        components["retrieved_context"].append(
            ContextComponent(
                component_type="retrieved_context",
                content=content,
                priority=0.9 - (i * 0.1),  # Decreasing priority
                is_critical=(i == 0),  # First document is critical
            )
        )

    # Conversation history
    if task_type != "simple":
        history_items = [
            "User: Hello, can you help me understand AI concepts?",
            "Assistant: Of course! I'd be happy to explain AI concepts. What specific area would you like to learn about?",
            "User: I'm particularly interested in machine learning.",
        ]

        for i, content in enumerate(history_items):
            components["conversation_history"].append(
                ContextComponent(
                    component_type="conversation_history",
                    content=content,
                    priority=0.6 + (i * 0.1),  # More recent = higher priority
                    is_critical=False,
                )
            )

    return components


def apply_token_budget(
    components: dict[str, list[ContextComponent]],
    config: TokenBudgetConfig,
    total_budget: int,
) -> TokenBudgetResult:
    """Apply token budget allocation to context components."""

    start_time = time.time()

    # Calculate allocation
    allocation = calculate_token_allocation(total_budget, config)

    # Apply allocation to each component type
    actual_usage = {}
    truncation_losses = {}

    for component_type, allocated_tokens in allocation.items():
        if component_type == "output_buffer":
            actual_usage[component_type] = allocated_tokens
            truncation_losses[component_type] = 0.0
            continue

        component_list = components.get(component_type, [])

        if not component_list:
            actual_usage[component_type] = 0
            truncation_losses[component_type] = 0.0
            continue

        # Apply selection and truncation strategy
        selected_components, used_tokens, loss = select_and_truncate_components(
            component_list,
            allocated_tokens,
            config.truncation_strategy,
            config.context_selection,
            config.preserve_critical_info,
        )

        actual_usage[component_type] = used_tokens
        truncation_losses[component_type] = loss

    # Calculate performance score based on truncation and content quality
    performance_score = calculate_performance_score(
        components, actual_usage, truncation_losses, config
    )

    # Calculate cost per query (simplified)
    total_tokens_used = sum(actual_usage.values())
    cost_per_query = total_tokens_used * 0.000002  # $0.002 per 1K tokens

    processing_time_ms = (time.time() - start_time) * 1000

    return TokenBudgetResult(
        allocated_tokens=allocation,
        actual_usage=actual_usage,
        performance_score=performance_score,
        cost_per_query=cost_per_query,
        truncation_losses=truncation_losses,
        processing_time_ms=processing_time_ms,
    )


def select_and_truncate_components(
    components: list[ContextComponent],
    token_budget: int,
    truncation_strategy: str,
    selection_strategy: str,
    preserve_critical: bool,
) -> tuple[list[ContextComponent], int, float]:
    """Select and truncate components to fit within budget."""

    if not components or token_budget <= 0:
        return [], 0, 0.0

    # Sort components based on selection strategy
    if selection_strategy == "relevance_first":
        sorted_components = sorted(components, key=lambda c: c.priority, reverse=True)
    elif selection_strategy == "information_density":
        # Estimate information density (tokens per priority)
        sorted_components = sorted(
            components,
            key=lambda c: c.priority / max(1, c.token_estimate),
            reverse=True,
        )
    elif selection_strategy == "recency_first":
        # For conversation history, newer is better
        sorted_components = list(reversed(components))
    else:  # diversity_balanced
        sorted_components = components.copy()
        random.shuffle(sorted_components)

    # Preserve critical components first
    if preserve_critical:
        critical_components = [c for c in sorted_components if c.is_critical]
        non_critical_components = [c for c in sorted_components if not c.is_critical]
        sorted_components = critical_components + non_critical_components

    # Select components to fit budget
    selected_components = []
    used_tokens = 0
    total_original_tokens = sum(c.token_estimate for c in components)

    for component in sorted_components:
        component_tokens = component.token_estimate

        if used_tokens + component_tokens <= token_budget:
            # Component fits completely
            selected_components.append(component)
            used_tokens += component_tokens
        elif used_tokens < token_budget and truncation_strategy != "simple_cutoff":
            # Partial component can be included
            remaining_budget = token_budget - used_tokens
            truncated_component = truncate_component(
                component, remaining_budget, truncation_strategy
            )
            if truncated_component:
                selected_components.append(truncated_component)
                used_tokens += remaining_budget
            break
        else:
            break

    # Calculate truncation loss
    selected_tokens = sum(c.token_estimate for c in selected_components)
    truncation_loss = (
        max(0.0, (total_original_tokens - selected_tokens) / total_original_tokens)
        if total_original_tokens > 0
        else 0.0
    )

    return selected_components, used_tokens, truncation_loss


def _truncate_simple_cutoff(content: str, max_tokens: int) -> str:

    words = content.split()

    target_words = max(1, int(max_tokens / 1.3))

    return " ".join(words[:target_words])


def _truncate_sentence_boundary(content: str, max_tokens: int) -> str:

    sentences = content.split(". ")

    truncated_sentences: list[str] = []

    estimated_tokens = 0

    for sentence in sentences:

        sentence_tokens = estimate_token_usage(sentence)

        if estimated_tokens + sentence_tokens <= max_tokens:

            truncated_sentences.append(sentence)

            estimated_tokens += sentence_tokens

        else:

            break

    truncated_content = ". ".join(truncated_sentences)

    if truncated_content and not truncated_content.endswith("."):

        truncated_content += "."

    return truncated_content


def _truncate_importance_scoring(content: str, max_tokens: int) -> str:

    sentences = content.split(". ")

    if len(sentences) <= 1:

        return content[: int(max_tokens * 4)]

    important_sentences = [sentences[0]]

    if len(sentences) > 1:

        important_sentences.append(sentences[-1])

    middle_sentences = sentences[1:-1] if len(sentences) > 2 else []

    for sentence in middle_sentences:

        candidate = ". ".join(
            important_sentences[:-1] + [sentence, important_sentences[-1]]
        )

        if estimate_token_usage(candidate) <= max_tokens:

            important_sentences.insert(-1, sentence)

        else:

            break

    return ". ".join(important_sentences)


def _truncate_recursive_summarization(content: str, max_tokens: int) -> str:

    words = content.split()

    target_words = max(1, int(max_tokens / 1.3))

    if len(words) > target_words:

        return " ".join(words[: target_words - 5]) + " [content continues...]"

    return content


def truncate_component(
    component: ContextComponent, max_tokens: int, strategy: str
) -> ContextComponent | None:
    """Truncate a component using the specified strategy."""

    if max_tokens <= 0:
        return None

    if component.token_estimate <= max_tokens:
        return component

    content = component.content
    strategy_handlers = {
        "simple_cutoff": _truncate_simple_cutoff,
        "sentence_boundary": _truncate_sentence_boundary,
        "importance_scoring": _truncate_importance_scoring,
        "recursive_summarization": _truncate_recursive_summarization,
    }
    truncated_content = strategy_handlers.get(strategy, _truncate_simple_cutoff)(
        content, max_tokens
    )

    # Create truncated component
    return ContextComponent(
        component_type=component.component_type,
        content=truncated_content,
        priority=component.priority,
        is_critical=component.is_critical,
        token_estimate=min(max_tokens, estimate_token_usage(truncated_content)),
    )


def calculate_performance_score(
    original_components: dict[str, list[ContextComponent]],
    actual_usage: dict[str, int],
    truncation_losses: dict[str, float],
    config: TokenBudgetConfig,
) -> float:
    """Calculate performance score based on budget utilization and content preservation."""

    # Base score from content preservation
    content_preservation_score = 0.0
    total_weight = 0.0

    for component_type, loss in truncation_losses.items():
        if component_type == "output_buffer":
            continue

        # Weight by importance
        weight = config.priority_weighting.get(component_type, 0.5)
        preservation = 1.0 - loss
        content_preservation_score += preservation * weight
        total_weight += weight

    if total_weight > 0:
        content_preservation_score /= total_weight

    # Budget utilization efficiency
    total_budget = sum(actual_usage.values())
    allocated_budget = sum(actual_usage.values())  # Same in this case
    budget_efficiency = (
        min(1.0, allocated_budget / total_budget) if total_budget > 0 else 1.0
    )

    # Critical information preservation bonus
    critical_bonus = 0.0
    if config.preserve_critical_info:
        for component_type, components in original_components.items():
            critical_components = [c for c in components if c.is_critical]
            if critical_components:
                loss = truncation_losses.get(component_type, 0.0)
                if loss < 0.1:  # Less than 10% loss for critical content
                    critical_bonus += 0.1

    # Combine scores
    performance_score = (
        content_preservation_score * 0.6
        + budget_efficiency * 0.3
        + critical_bonus * 0.1
    )

    return min(1.0, performance_score)


def evaluate_token_budget_task(
    task: dict[str, Any], config: TokenBudgetConfig, budget_scenario: str = "standard"
) -> TokenBudgetResult:
    """Evaluate token budget configuration on a specific task."""

    # Get budget for scenario
    budget_map = {
        "tight": 2000,
        "standard": 4000,
        "generous": 8000,
        "enterprise": 16000,
    }

    total_budget = budget_map.get(budget_scenario, 4000)

    # Create context components for task
    components = create_mock_context_components(
        task_type=task.get("type", "qa"), complexity=task.get("complexity", "medium")
    )

    # Apply budget allocation
    result = apply_token_budget(components, config, total_budget)

    return result


def calculate_budget_metrics(results: list[TokenBudgetResult]) -> dict[str, float]:
    """Calculate aggregated metrics from budget evaluation results."""

    if not results:
        return {
            "avg_performance": 0.0,
            "avg_cost_per_query": 1.0,
            "token_efficiency": 0.0,
            "content_preservation": 0.0,
            "processing_speed": 0.0,
        }

    # Calculate averages
    avg_performance = sum(r.performance_score for r in results) / len(results)
    avg_cost = sum(r.cost_per_query for r in results) / len(results)
    avg_processing_time = sum(r.processing_time_ms for r in results) / len(results)

    # Token efficiency (performance per dollar)
    token_efficiency = avg_performance / avg_cost if avg_cost > 0 else 0

    # Content preservation (inverse of truncation losses)
    avg_truncation_loss = sum(
        (
            sum(r.truncation_losses.values()) / len(r.truncation_losses)
            if r.truncation_losses
            else 0
        )
        for r in results
    ) / len(results)
    content_preservation = max(0.0, 1.0 - avg_truncation_loss)

    # Processing speed score
    processing_speed = max(0.0, 1.0 - (avg_processing_time / 1000))  # Normalize to 0-1

    return {
        "avg_performance": avg_performance,
        "avg_cost_per_query": avg_cost,
        "token_efficiency": token_efficiency,
        "content_preservation": content_preservation,
        "processing_speed": processing_speed,
        "avg_processing_time_ms": avg_processing_time,
        # Additional metrics
        "total_evaluations": len(results),
        "performance_variance": sum(
            (r.performance_score - avg_performance) ** 2 for r in results
        )
        / len(results),
    }
