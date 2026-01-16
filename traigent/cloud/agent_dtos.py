"""Multi-agent workflow cost tracking DTOs.

This module provides validated data transfer objects for tracking costs
and token usage in multi-agent workflows, with built-in arithmetic validation
to prevent data corruption.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-EVAL-METRICS REQ-EVAL-005

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentCostBreakdown:
    """Per-agent cost breakdown with arithmetic validation.

    Validates token and cost arithmetic in __post_init__ to ensure:
    - total_tokens = input_tokens + output_tokens
    - total_cost = input_cost + output_cost
    - All values are non-negative
    - No NaN or Inf values

    Attributes:
        agent_id: Unique identifier for the agent
        agent_name: Human-readable agent name
        input_tokens: Number of input tokens consumed
        output_tokens: Number of output tokens generated
        total_tokens: Total tokens (must equal input + output)
        input_cost: Cost in USD for input tokens
        output_cost: Cost in USD for output tokens
        total_cost: Total cost in USD (must equal input + output)
        model_used: Model identifier (e.g., "gpt-4o-mini")

    Raises:
        ValueError: If validation fails (invalid arithmetic, negative values, etc.)
    """

    agent_id: str
    agent_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    model_used: str

    def __post_init__(self) -> None:
        """Validate all fields for data integrity.

        Raises:
            ValueError: If any validation check fails
        """
        errors: list[str] = []

        # Validate identifiers
        if not self.agent_id:
            errors.append("agent_id cannot be empty")
        if not self.agent_name:
            errors.append("agent_name cannot be empty")
        if not self.model_used:
            errors.append("model_used cannot be empty")

        # Validate token counts (non-negative)
        if self.input_tokens < 0:
            errors.append(f"input_tokens cannot be negative: {self.input_tokens}")
        if self.output_tokens < 0:
            errors.append(f"output_tokens cannot be negative: {self.output_tokens}")
        if self.total_tokens < 0:
            errors.append(f"total_tokens cannot be negative: {self.total_tokens}")

        # Validate token arithmetic
        expected_total_tokens = self.input_tokens + self.output_tokens
        if self.total_tokens != expected_total_tokens:
            errors.append(
                f"total_tokens ({self.total_tokens}) must equal "
                f"input_tokens ({self.input_tokens}) + "
                f"output_tokens ({self.output_tokens}) = {expected_total_tokens}"
            )

        # Validate costs (non-negative, no NaN/Inf)
        for field_name, value in [
            ("input_cost", self.input_cost),
            ("output_cost", self.output_cost),
            ("total_cost", self.total_cost),
        ]:
            if value < 0:
                errors.append(f"{field_name} cannot be negative: {value}")
            if math.isnan(value):
                errors.append(f"{field_name} cannot be NaN")
            if math.isinf(value):
                errors.append(f"{field_name} cannot be Inf")

        # Validate cost arithmetic (with floating point tolerance)
        expected_total_cost = self.input_cost + self.output_cost
        if abs(self.total_cost - expected_total_cost) > 0.0001:
            errors.append(
                f"total_cost ({self.total_cost:.8f}) must equal "
                f"input_cost ({self.input_cost:.8f}) + "
                f"output_cost ({self.output_cost:.8f}) = {expected_total_cost:.8f}"
            )

        if errors:
            raise ValueError(
                "AgentCostBreakdown validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary with all agent cost fields
        """
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "model_used": self.model_used,
        }


@dataclass
class WorkflowCostSummary:
    """Aggregated cost summary across multiple agents.

    Automatically aggregates token and cost totals from agent breakdowns
    and validates the aggregation arithmetic.

    Attributes:
        workflow_id: Unique identifier for the workflow
        workflow_name: Human-readable workflow name
        agent_breakdowns: List of per-agent cost breakdowns
        total_input_tokens: Sum of all agent input tokens
        total_output_tokens: Sum of all agent output tokens
        total_tokens: Sum of all agent total tokens
        total_input_cost: Sum of all agent input costs
        total_output_cost: Sum of all agent output costs
        total_cost: Sum of all agent total costs

    Raises:
        ValueError: If validation fails (empty workflow, invalid identifiers)
    """

    workflow_id: str
    workflow_name: str
    agent_breakdowns: list[AgentCostBreakdown]
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_input_cost: float = 0.0
    total_output_cost: float = 0.0
    total_cost: float = 0.0

    def __post_init__(self) -> None:
        """Validate and aggregate from agent breakdowns.

        Raises:
            ValueError: If validation fails
        """
        errors: list[str] = []

        if not self.workflow_id:
            errors.append("workflow_id cannot be empty")
        if not self.workflow_name:
            errors.append("workflow_name cannot be empty")
        if not self.agent_breakdowns:
            errors.append("agent_breakdowns cannot be empty")

        if errors:
            raise ValueError(
                "WorkflowCostSummary validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        # Aggregate from agents (override any provided values)
        self.total_input_tokens = sum(a.input_tokens for a in self.agent_breakdowns)
        self.total_output_tokens = sum(a.output_tokens for a in self.agent_breakdowns)
        self.total_tokens = sum(a.total_tokens for a in self.agent_breakdowns)
        self.total_input_cost = sum(a.input_cost for a in self.agent_breakdowns)
        self.total_output_cost = sum(a.output_cost for a in self.agent_breakdowns)
        self.total_cost = sum(a.total_cost for a in self.agent_breakdowns)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary with workflow summary and agent breakdowns
        """
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_input_cost": self.total_input_cost,
            "total_output_cost": self.total_output_cost,
            "total_cost": self.total_cost,
            "agent_breakdowns": [a.to_dict() for a in self.agent_breakdowns],
        }
