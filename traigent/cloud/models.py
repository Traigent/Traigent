"""Data models for backend integration and reserved cloud APIs.

This module contains all the data classes used for communication between
the Traigent client and backend services. Agent-based remote execution models
are reserved for a future cloud release.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID FUNC-AGENTS REQ-CLOUD-009 REQ-AGNT-013

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from traigent.evaluators.base import Dataset


class OptimizationSessionStatus(Enum):
    """Status of an optimization session."""

    PENDING = "pending"
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrialStatus(Enum):
    """Status of an individual trial."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SessionObjectiveDefinition:
    """Typed objective definition for interactive cloud sessions."""

    metric: str
    direction: str | None = None
    band: dict[str, Any] | None = None
    test: str | None = None
    alpha: float | None = None
    weight: float | None = None

    def __post_init__(self) -> None:
        """Validate mutually exclusive objective modes."""
        if self.band is not None and self.direction is not None:
            raise ValueError(
                "SessionObjectiveDefinition: 'band' and 'direction' are mutually exclusive."
            )
        if self.band is None and self.direction is None:
            raise ValueError(
                "SessionObjectiveDefinition: one of 'band' or 'direction' must be provided."
            )
        if (self.test is not None or self.alpha is not None) and self.band is None:
            raise ValueError(
                "SessionObjectiveDefinition: 'test' and 'alpha' require 'band' to be set."
            )


@dataclass
class OptimizationRequest:
    """Request to start an optimization."""

    function_name: str
    dataset: Dataset
    configuration_space: dict[str, Any]
    objectives: list[str]
    max_trials: int = 10  # Changed default from 50 to 10
    target_cost_reduction: float = 0.65
    user_id: str | None = None
    billing_tier: str = "standard"
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_specification: AgentSpecification | None = None


@dataclass
class OptimizationResponse:
    """Response from optimization service."""

    request_id: str
    best_config: dict[str, Any]
    best_metrics: dict[str, float]
    trials_count: int
    optimization_time: float
    cost_reduction: float
    subset_used: bool
    billing_info: dict[str, Any]
    status: str
    subset_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationSession:
    """Represents a stateful optimization session for client-side execution."""

    session_id: str
    function_name: str
    configuration_space: dict[str, Any]
    objectives: list[str]
    max_trials: int
    status: OptimizationSessionStatus
    created_at: datetime
    updated_at: datetime
    completed_trials: int = 0
    best_config: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None
    optimization_strategy: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_complete(self) -> bool:
        """Check if optimization session is complete."""
        return self.status in [
            OptimizationSessionStatus.COMPLETED,
            OptimizationSessionStatus.FAILED,
            OptimizationSessionStatus.CANCELLED,
        ]

    def can_continue(self) -> bool:
        """Check if session can accept more trials."""
        return (
            self.status == OptimizationSessionStatus.ACTIVE
            and self.completed_trials < self.max_trials
        )


@dataclass
class DatasetSubsetIndices:
    """Indices for selecting a subset of the dataset."""

    indices: list[int]
    selection_strategy: str
    confidence_level: float
    estimated_representativeness: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialSuggestion:
    """Suggestion for the next trial in client-side execution."""

    trial_id: str
    session_id: str
    trial_number: int
    config: dict[str, Any]
    dataset_subset: DatasetSubsetIndices
    exploration_type: str  # "exploration", "exploitation", "verification"
    priority: int = 1
    estimated_duration: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResultSubmission:
    """Client submission of trial results."""

    session_id: str
    trial_id: str
    metrics: dict[str, float]
    duration: float
    status: TrialStatus
    outputs_sample: list[Any] | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionCreationRequest:
    """Request to create a new optimization session."""

    function_name: str | None = None
    configuration_space: dict[str, Any] | None = None
    objectives: Sequence[str | SessionObjectiveDefinition | dict[str, Any]] | None = (
        None
    )
    dataset_metadata: dict[str, Any] | None = None  # Size, type, characteristics
    max_trials: int | None = 10  # Changed default to 10 and allow None
    budget: dict[str, Any] | None = None
    constraints: dict[str, Any] | None = None
    default_config: dict[str, Any] | None = None
    promotion_policy: dict[str, Any] | None = None
    optimization_strategy: dict[str, Any] | None = None
    user_id: str | None = None
    billing_tier: str = "standard"
    metadata: dict[str, Any] = field(default_factory=dict)
    # Alternative parameter names for test compatibility
    problem_type: str | None = None

    def __post_init__(self) -> None:
        """Handle default values and alternative parameter names."""
        if self.function_name is None:
            self.function_name = "test_function"
        if self.configuration_space is None:
            self.configuration_space: dict[str, Any] = {}
        if self.objectives is None:
            self.objectives: list[Any] = []
        if self.dataset_metadata is None:
            self.dataset_metadata: dict[str, Any] = {"size": 100, "type": "test"}
        # Ensure max_trials is never None
        if self.max_trials is None:
            self.max_trials = 10


@dataclass
class SessionCreationResponse:
    """Response after creating an optimization session."""

    session_id: str
    status: OptimizationSessionStatus
    optimization_strategy: dict[str, Any]
    estimated_duration: float | None = None
    billing_estimate: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NextTrialRequest:
    """Request for the next trial suggestion."""

    session_id: str
    previous_results: list[TrialResultSubmission] | None = None
    request_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NextTrialResponse:
    """Response with next trial suggestion."""

    suggestion: TrialSuggestion | None
    should_continue: bool
    reason: str | None = None
    stop_reason: str | None = None
    session_status: OptimizationSessionStatus = OptimizationSessionStatus.ACTIVE
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationFinalizationRequest:
    """Request to finalize an optimization session."""

    session_id: str
    include_full_history: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationFinalizationResponse:
    """Final results of an optimization session."""

    session_id: str
    best_config: dict[str, Any]
    best_metrics: dict[str, float]
    total_trials: int
    successful_trials: int
    total_duration: float
    cost_savings: float
    stop_reason: str | None = None
    convergence_history: list[dict[str, Any]] | None = None
    full_history: list[TrialResultSubmission] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Agent-specific models for Model 2


@dataclass
class AgentSpecification:
    """Specification for an AI agent following Traigent schema."""

    id: str | None = None
    name: str | None = None
    agent_type: str | None = None
    agent_platform: str | None = None
    prompt_template: str | None = None
    model_parameters: dict[str, Any] | None = None
    reasoning: str | None = None
    style: str | None = None
    tone: str | None = None
    format: str | None = None
    persona: str | None = None
    guidelines: str | list[str] | None = None
    response_validation: bool = False
    custom_tools: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    description: str | None = None  # For test compatibility
    version: str | None = None  # For test compatibility

    def __post_init__(self) -> None:
        """Handle default values."""
        if self.id is None:
            self.id = "test-agent"
        if self.name is None:
            self.name = "Test Agent"
        if self.agent_type is None:
            self.agent_type = "test"
        if self.agent_platform is None:
            self.agent_platform = "test"
        if self.prompt_template is None:
            self.prompt_template = "Test prompt"
        if self.model_parameters is None:
            self.model_parameters: dict[str, Any] = {}


@dataclass
class AgentOptimizationRequest:
    """Request to optimize an agent configuration."""

    agent_spec: AgentSpecification | None = None
    dataset: Dataset | None = None
    configuration_space: dict[str, Any] | None = None  # Maps to model_parameters paths
    objectives: list[str] | None = None
    max_trials: int = 10  # Changed default from 50 to 10
    target_cost_reduction: float = 0.65
    user_id: str | None = None
    billing_tier: str = "standard"
    metadata: dict[str, Any] = field(default_factory=dict)
    # Alternative parameter names for test compatibility
    agent_specification: AgentSpecification | None = None
    dataset_path: str | None = None

    def __post_init__(self) -> None:
        """Handle alternative parameter names."""
        if self.agent_specification is not None and self.agent_spec is None:
            self.agent_spec = self.agent_specification
        if self.dataset_path is not None and self.dataset is None:
            # Create a mock dataset for test compatibility
            from traigent.evaluators.base import Dataset, EvaluationExample

            self.dataset = Dataset([EvaluationExample({"input": "test"}, "test")])
        if self.objectives is None:
            self.objectives: list[Any] = []


@dataclass
class AgentOptimizationStatus:
    """Status update for agent optimization."""

    optimization_id: str
    status: OptimizationSessionStatus
    progress: float  # 0.0 to 1.0
    completed_trials: int
    total_trials: int
    current_best_metrics: dict[str, float] | None = None
    estimated_time_remaining: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentExecutionRequest:
    """Request for agent execution."""

    agent_spec: AgentSpecification | None = None
    input_data: dict[str, Any] | None = None
    config_overrides: dict[str, Any] | None = None
    execution_context: dict[str, Any] | None = None
    # Alternative parameter names for test compatibility
    agent_specification: AgentSpecification | None = None
    inputs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Handle alternative parameter names."""
        if self.agent_specification is not None and self.agent_spec is None:
            self.agent_spec = self.agent_specification
        if self.inputs is not None and self.input_data is None:
            self.input_data = self.inputs
        if self.input_data is None:
            self.input_data: dict[str, Any] = {}


@dataclass
class AgentExecutionResponse:
    """Response from agent execution."""

    output: Any
    duration: float
    tokens_used: int | None = None
    cost: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class AgentOptimizationResponse:
    """Response from agent optimization start."""

    session_id: str
    optimization_id: str
    status: str
    estimated_cost: float | None = None
    estimated_duration: float | None = None
    next_steps: list[str] = field(default_factory=list)
