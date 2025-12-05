"""
Dataset generation for token budget optimization testing.

This module creates tasks across different complexities and domains to test
token allocation strategies under various budget constraints.
"""

# CUSTOM LOGIC - NOT PART OF TRAIGENT:
# This example implements custom budget allocation logic.
# Token budget management is not a built-in TraiGent feature.

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any


class TaskType(Enum):
    """Task types with different token usage patterns."""

    SIMPLE_QA = "simple_qa"  # Basic questions
    RESEARCH_QA = "research_qa"  # Research-intensive queries
    CODE_GENERATION = "code_generation"  # Programming tasks
    COMPLEX_ANALYSIS = "complex_analysis"  # Multi-step reasoning
    CREATIVE_WRITING = "creative_writing"  # Content generation
    SUMMARIZATION = "summarization"  # Text summarization


class BudgetScenario(Enum):
    """Budget constraint scenarios."""

    TIGHT = "tight"  # 2K tokens - aggressive optimization needed
    STANDARD = "standard"  # 4K tokens - balanced approach
    GENEROUS = "generous"  # 8K tokens - quality focus
    ENTERPRISE = "enterprise"  # 16K tokens - comprehensive context


@dataclass
class TokenBudgetTask:
    """A task for token budget optimization testing."""

    id: str
    task_type: TaskType
    description: str
    query: str
    context_requirements: dict[str, Any]
    expected_token_usage: dict[str, int]  # Expected usage per component
    quality_metrics: dict[str, Any]
    success_criteria: dict[str, Any]
    priority_components: list[str]  # Which components are critical


def generate_simple_qa_tasks() -> list[TokenBudgetTask]:
    """Generate simple Q&A tasks that don't need much context."""

    tasks = [
        TokenBudgetTask(
            id="qa_001",
            task_type=TaskType.SIMPLE_QA,
            description="Basic factual question with minimal context needs",
            query="What is the capital of France?",
            context_requirements={
                "system_prompt": "high",
                "examples": "low",
                "retrieved_context": "minimal",
                "conversation_history": "none",
                "complexity": "simple",
            },
            expected_token_usage={
                "system_prompt": 50,
                "examples": 0,
                "retrieved_context": 100,
                "conversation_history": 0,
                "output_buffer": 20,
            },
            quality_metrics={
                "accuracy_required": 0.95,
                "response_length": "short",
                "factual_correctness": 1.0,
            },
            success_criteria={"token_efficiency": True, "answer_quality": "high"},
            priority_components=["system_prompt", "retrieved_context"],
        ),
        TokenBudgetTask(
            id="qa_002",
            task_type=TaskType.SIMPLE_QA,
            description="Mathematical calculation with examples",
            query="Calculate the compound interest on $1000 at 5% for 3 years",
            context_requirements={
                "system_prompt": "medium",
                "examples": "high",
                "retrieved_context": "low",
                "conversation_history": "none",
                "complexity": "simple",
            },
            expected_token_usage={
                "system_prompt": 80,
                "examples": 200,
                "retrieved_context": 50,
                "conversation_history": 0,
                "output_buffer": 100,
            },
            quality_metrics={"calculation_accuracy": 1.0, "step_by_step": True},
            success_criteria={
                "mathematical_correctness": True,
                "clear_explanation": True,
            },
            priority_components=["examples", "system_prompt"],
        ),
    ]

    return tasks


def generate_research_qa_tasks() -> list[TokenBudgetTask]:
    """Generate research-intensive Q&A tasks requiring substantial context."""

    tasks = [
        TokenBudgetTask(
            id="research_001",
            task_type=TaskType.RESEARCH_QA,
            description="Complex question requiring multiple sources",
            query="Compare the economic impacts of renewable energy policies in Germany vs Denmark from 2015-2020",
            context_requirements={
                "system_prompt": "high",
                "examples": "medium",
                "retrieved_context": "critical",
                "conversation_history": "low",
                "complexity": "complex",
            },
            expected_token_usage={
                "system_prompt": 150,
                "examples": 300,
                "retrieved_context": 2000,
                "conversation_history": 100,
                "output_buffer": 400,
            },
            quality_metrics={
                "source_coverage": 0.8,
                "comparative_analysis": True,
                "data_accuracy": 0.9,
            },
            success_criteria={
                "comprehensive_comparison": True,
                "factual_evidence": True,
            },
            priority_components=["retrieved_context", "system_prompt", "examples"],
        ),
        TokenBudgetTask(
            id="research_002",
            task_type=TaskType.RESEARCH_QA,
            description="Technical research with follow-up questions",
            query="Explain the current state of quantum computing and its implications for cryptography",
            context_requirements={
                "system_prompt": "high",
                "examples": "medium",
                "retrieved_context": "critical",
                "conversation_history": "medium",
                "complexity": "complex",
            },
            expected_token_usage={
                "system_prompt": 120,
                "examples": 250,
                "retrieved_context": 1800,
                "conversation_history": 200,
                "output_buffer": 500,
            },
            quality_metrics={
                "technical_accuracy": 0.9,
                "current_information": True,
                "implications_coverage": 0.8,
            },
            success_criteria={"technical_depth": True, "practical_implications": True},
            priority_components=[
                "retrieved_context",
                "system_prompt",
                "conversation_history",
            ],
        ),
    ]

    return tasks


def generate_code_generation_tasks() -> list[TokenBudgetTask]:
    """Generate programming tasks with varying context needs."""

    tasks = [
        TokenBudgetTask(
            id="code_001",
            task_type=TaskType.CODE_GENERATION,
            description="Simple function implementation",
            query="Write a Python function to find the maximum element in a list",
            context_requirements={
                "system_prompt": "high",
                "examples": "critical",
                "retrieved_context": "low",
                "conversation_history": "none",
                "complexity": "simple",
            },
            expected_token_usage={
                "system_prompt": 100,
                "examples": 400,
                "retrieved_context": 100,
                "conversation_history": 0,
                "output_buffer": 150,
            },
            quality_metrics={
                "code_correctness": 1.0,
                "follows_best_practices": True,
                "includes_docstring": True,
            },
            success_criteria={"functional_correctness": True, "code_quality": True},
            priority_components=["examples", "system_prompt"],
        ),
        TokenBudgetTask(
            id="code_002",
            task_type=TaskType.CODE_GENERATION,
            description="Complex algorithm with multiple constraints",
            query="Implement a thread-safe LRU cache in Python with TTL support and metrics tracking",
            context_requirements={
                "system_prompt": "high",
                "examples": "critical",
                "retrieved_context": "high",
                "conversation_history": "medium",
                "complexity": "complex",
            },
            expected_token_usage={
                "system_prompt": 150,
                "examples": 800,
                "retrieved_context": 600,
                "conversation_history": 150,
                "output_buffer": 600,
            },
            quality_metrics={
                "thread_safety": True,
                "performance_requirements": True,
                "code_organization": 0.9,
            },
            success_criteria={"all_requirements_met": True, "production_ready": True},
            priority_components=["examples", "retrieved_context", "system_prompt"],
        ),
    ]

    return tasks


def generate_complex_analysis_tasks() -> list[TokenBudgetTask]:
    """Generate tasks requiring multi-step reasoning and analysis."""

    tasks = [
        TokenBudgetTask(
            id="analysis_001",
            task_type=TaskType.COMPLEX_ANALYSIS,
            description="Multi-step business analysis",
            query="Analyze the potential market opportunity for a B2B SaaS product targeting mid-size manufacturing companies, including competitive landscape, pricing strategy, and go-to-market approach",
            context_requirements={
                "system_prompt": "high",
                "examples": "high",
                "retrieved_context": "critical",
                "conversation_history": "high",
                "complexity": "complex",
            },
            expected_token_usage={
                "system_prompt": 200,
                "examples": 600,
                "retrieved_context": 2500,
                "conversation_history": 300,
                "output_buffer": 800,
            },
            quality_metrics={
                "analysis_depth": 0.9,
                "market_insights": True,
                "actionable_recommendations": 0.8,
            },
            success_criteria={
                "comprehensive_analysis": True,
                "strategic_insights": True,
            },
            priority_components=[
                "retrieved_context",
                "conversation_history",
                "examples",
            ],
        )
    ]

    return tasks


def generate_creative_writing_tasks() -> list[TokenBudgetTask]:
    """Generate creative content tasks with style and tone requirements."""

    tasks = [
        TokenBudgetTask(
            id="creative_001",
            task_type=TaskType.CREATIVE_WRITING,
            description="Blog post with specific style requirements",
            query="Write a 500-word blog post about the future of remote work, targeting tech professionals, with an optimistic but realistic tone",
            context_requirements={
                "system_prompt": "critical",
                "examples": "high",
                "retrieved_context": "medium",
                "conversation_history": "low",
                "complexity": "medium",
            },
            expected_token_usage={
                "system_prompt": 200,
                "examples": 500,
                "retrieved_context": 400,
                "conversation_history": 100,
                "output_buffer": 600,
            },
            quality_metrics={
                "tone_consistency": 0.9,
                "target_audience_alignment": 0.8,
                "content_structure": True,
            },
            success_criteria={
                "engaging_content": True,
                "meets_length_requirement": True,
            },
            priority_components=["system_prompt", "examples", "retrieved_context"],
        )
    ]

    return tasks


def generate_summarization_tasks() -> list[TokenBudgetTask]:
    """Generate text summarization tasks with varying input lengths."""

    tasks = [
        TokenBudgetTask(
            id="summary_001",
            task_type=TaskType.SUMMARIZATION,
            description="Long document summarization",
            query="Summarize this 10,000-word research paper on climate change impacts in a 200-word executive summary",
            context_requirements={
                "system_prompt": "high",
                "examples": "medium",
                "retrieved_context": "critical",  # The document content
                "conversation_history": "none",
                "complexity": "medium",
            },
            expected_token_usage={
                "system_prompt": 150,
                "examples": 300,
                "retrieved_context": 3000,  # Truncated document
                "conversation_history": 0,
                "output_buffer": 250,
            },
            quality_metrics={
                "key_points_coverage": 0.8,
                "length_adherence": True,
                "readability": 0.9,
            },
            success_criteria={"comprehensive_summary": True, "executive_level": True},
            priority_components=["retrieved_context", "system_prompt"],
        )
    ]

    return tasks


def create_token_budget_dataset(
    total_tasks: int = 50, task_distribution: dict[TaskType, float] | None = None
) -> list[TokenBudgetTask]:
    """Create a comprehensive token budget optimization dataset."""

    if task_distribution is None:
        task_distribution = {
            TaskType.SIMPLE_QA: 0.20,
            TaskType.RESEARCH_QA: 0.25,
            TaskType.CODE_GENERATION: 0.20,
            TaskType.COMPLEX_ANALYSIS: 0.15,
            TaskType.CREATIVE_WRITING: 0.10,
            TaskType.SUMMARIZATION: 0.10,
        }

    # Generate base tasks by type
    all_tasks = []
    all_tasks.extend(generate_simple_qa_tasks())
    all_tasks.extend(generate_research_qa_tasks())
    all_tasks.extend(generate_code_generation_tasks())
    all_tasks.extend(generate_complex_analysis_tasks())
    all_tasks.extend(generate_creative_writing_tasks())
    all_tasks.extend(generate_summarization_tasks())

    # Expand dataset to reach desired size
    while len(all_tasks) < total_tasks:
        # Create variations of existing tasks
        base_task = random.choice(all_tasks[: len(all_tasks) // 2])

        variation = TokenBudgetTask(
            id=f"{base_task.id}_var_{len(all_tasks)}",
            task_type=base_task.task_type,
            description=f"{base_task.description} (variation)",
            query=f"{base_task.query} [variant]",
            context_requirements=base_task.context_requirements.copy(),
            expected_token_usage=base_task.expected_token_usage.copy(),
            quality_metrics=base_task.quality_metrics.copy(),
            success_criteria=base_task.success_criteria.copy(),
            priority_components=base_task.priority_components.copy(),
        )

        all_tasks.append(variation)

    return all_tasks[:total_tasks]


def get_budget_scenarios() -> list[dict[str, Any]]:
    """Get different budget constraint scenarios for testing."""

    return [
        {
            "name": "tight_budget",
            "total_tokens": 2000,
            "description": "Aggressive optimization required",
            "challenges": [
                "context_truncation",
                "quality_vs_cost",
                "critical_selection",
            ],
        },
        {
            "name": "standard_budget",
            "total_tokens": 4000,
            "description": "Balanced allocation approach",
            "challenges": ["efficient_allocation", "component_prioritization"],
        },
        {
            "name": "generous_budget",
            "total_tokens": 8000,
            "description": "Quality-focused with some constraints",
            "challenges": ["quality_optimization", "waste_prevention"],
        },
        {
            "name": "enterprise_budget",
            "total_tokens": 16000,
            "description": "Comprehensive context support",
            "challenges": ["full_utilization", "performance_optimization"],
        },
    ]


def analyze_task_token_requirements(tasks: list[TokenBudgetTask]) -> dict[str, Any]:
    """Analyze token requirements across the dataset."""

    # Calculate statistics by task type
    type_stats = {}
    for task_type in TaskType:
        type_tasks = [t for t in tasks if t.task_type == task_type]
        if type_tasks:
            total_tokens = [sum(t.expected_token_usage.values()) for t in type_tasks]
            type_stats[task_type.value] = {
                "count": len(type_tasks),
                "avg_tokens": sum(total_tokens) / len(total_tokens),
                "min_tokens": min(total_tokens),
                "max_tokens": max(total_tokens),
            }

    # Overall statistics
    all_token_usage = [sum(t.expected_token_usage.values()) for t in tasks]

    return {
        "total_tasks": len(tasks),
        "by_type": type_stats,
        "overall": {
            "avg_tokens": sum(all_token_usage) / len(all_token_usage),
            "min_tokens": min(all_token_usage),
            "max_tokens": max(all_token_usage),
        },
        "component_usage": {
            "system_prompt": sum(
                t.expected_token_usage.get("system_prompt", 0) for t in tasks
            )
            / len(tasks),
            "examples": sum(t.expected_token_usage.get("examples", 0) for t in tasks)
            / len(tasks),
            "retrieved_context": sum(
                t.expected_token_usage.get("retrieved_context", 0) for t in tasks
            )
            / len(tasks),
            "conversation_history": sum(
                t.expected_token_usage.get("conversation_history", 0) for t in tasks
            )
            / len(tasks),
            "output_buffer": sum(
                t.expected_token_usage.get("output_buffer", 0) for t in tasks
            )
            / len(tasks),
        },
    }
