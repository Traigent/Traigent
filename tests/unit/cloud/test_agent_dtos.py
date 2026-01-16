"""Tests for multi-agent workflow cost tracking DTOs.

This module tests AgentCostBreakdown and WorkflowCostSummary DTOs
to ensure they properly validate arithmetic and prevent data corruption.
"""

import math

import pytest

from traigent.cloud.agent_dtos import AgentCostBreakdown, WorkflowCostSummary


class TestAgentCostBreakdown:
    """Tests for AgentCostBreakdown validation."""

    def test_valid_breakdown_constructs_successfully(self):
        """Valid breakdown should construct successfully."""
        breakdown = AgentCostBreakdown(
            agent_id="agent-001",
            agent_name="Researcher",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
            model_used="gpt-4o-mini",
        )
        assert breakdown.agent_id == "agent-001"
        assert breakdown.agent_name == "Researcher"
        assert breakdown.total_tokens == 150
        assert breakdown.total_cost == 0.003
        assert breakdown.model_used == "gpt-4o-mini"

    def test_rejects_empty_agent_id(self):
        """Empty agent_id should raise ValueError."""
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            AgentCostBreakdown(
                agent_id="",
                agent_name="Researcher",
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                input_cost=0.001,
                output_cost=0.002,
                total_cost=0.003,
                model_used="gpt-4o",
            )

    def test_rejects_empty_agent_name(self):
        """Empty agent_name should raise ValueError."""
        with pytest.raises(ValueError, match="agent_name cannot be empty"):
            AgentCostBreakdown(
                agent_id="agent-001",
                agent_name="",
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                input_cost=0.001,
                output_cost=0.002,
                total_cost=0.003,
                model_used="gpt-4o",
            )

    def test_rejects_empty_model_used(self):
        """Empty model_used should raise ValueError."""
        with pytest.raises(ValueError, match="model_used cannot be empty"):
            AgentCostBreakdown(
                agent_id="agent-001",
                agent_name="Researcher",
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                input_cost=0.001,
                output_cost=0.002,
                total_cost=0.003,
                model_used="",
            )

    def test_rejects_negative_input_tokens(self):
        """Negative input_tokens should raise ValueError."""
        with pytest.raises(ValueError, match="input_tokens cannot be negative"):
            AgentCostBreakdown(
                agent_id="agent-001",
                agent_name="Researcher",
                input_tokens=-10,
                output_tokens=50,
                total_tokens=40,
                input_cost=0.001,
                output_cost=0.002,
                total_cost=0.003,
                model_used="gpt-4o",
            )

    def test_rejects_negative_output_tokens(self):
        """Negative output_tokens should raise ValueError."""
        with pytest.raises(ValueError, match="output_tokens cannot be negative"):
            AgentCostBreakdown(
                agent_id="agent-001",
                agent_name="Researcher",
                input_tokens=100,
                output_tokens=-50,
                total_tokens=50,
                input_cost=0.001,
                output_cost=0.002,
                total_cost=0.003,
                model_used="gpt-4o",
            )

    def test_rejects_negative_total_tokens(self):
        """Negative total_tokens should raise ValueError."""
        with pytest.raises(ValueError, match="total_tokens cannot be negative"):
            AgentCostBreakdown(
                agent_id="agent-001",
                agent_name="Researcher",
                input_tokens=100,
                output_tokens=50,
                total_tokens=-150,
                input_cost=0.001,
                output_cost=0.002,
                total_cost=0.003,
                model_used="gpt-4o",
            )

    def test_rejects_token_arithmetic_mismatch(self):
        """Token arithmetic errors should raise ValueError."""
        with pytest.raises(ValueError, match="total_tokens.*must equal"):
            AgentCostBreakdown(
                agent_id="agent-001",
                agent_name="Researcher",
                input_tokens=100,
                output_tokens=50,
                total_tokens=999,  # Wrong! Should be 150
                input_cost=0.001,
                output_cost=0.002,
                total_cost=0.003,
                model_used="gpt-4o",
            )

    def test_rejects_negative_input_cost(self):
        """Negative input_cost should raise ValueError."""
        with pytest.raises(ValueError, match="input_cost cannot be negative"):
            AgentCostBreakdown(
                agent_id="agent-001",
                agent_name="Researcher",
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                input_cost=-0.001,
                output_cost=0.002,
                total_cost=0.001,
                model_used="gpt-4o",
            )

    def test_rejects_negative_output_cost(self):
        """Negative output_cost should raise ValueError."""
        with pytest.raises(ValueError, match="output_cost cannot be negative"):
            AgentCostBreakdown(
                agent_id="agent-001",
                agent_name="Researcher",
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                input_cost=0.001,
                output_cost=-0.002,
                total_cost=-0.001,
                model_used="gpt-4o",
            )

    def test_rejects_negative_total_cost(self):
        """Negative total_cost should raise ValueError."""
        with pytest.raises(ValueError, match="total_cost cannot be negative"):
            AgentCostBreakdown(
                agent_id="agent-001",
                agent_name="Researcher",
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                input_cost=0.001,
                output_cost=0.002,
                total_cost=-0.003,
                model_used="gpt-4o",
            )

    def test_rejects_nan_cost(self):
        """NaN cost values should raise ValueError."""
        with pytest.raises(ValueError, match="input_cost cannot be NaN"):
            AgentCostBreakdown(
                agent_id="agent-001",
                agent_name="Researcher",
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                input_cost=math.nan,
                output_cost=0.002,
                total_cost=0.003,
                model_used="gpt-4o",
            )

    def test_rejects_inf_cost(self):
        """Inf cost values should raise ValueError."""
        with pytest.raises(ValueError, match="total_cost cannot be Inf"):
            AgentCostBreakdown(
                agent_id="agent-001",
                agent_name="Researcher",
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                input_cost=0.001,
                output_cost=0.002,
                total_cost=math.inf,
                model_used="gpt-4o",
            )

    def test_rejects_cost_arithmetic_mismatch(self):
        """Cost arithmetic errors should raise ValueError."""
        with pytest.raises(ValueError, match="total_cost.*must equal"):
            AgentCostBreakdown(
                agent_id="agent-001",
                agent_name="Researcher",
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                input_cost=0.001,
                output_cost=0.002,
                total_cost=0.999,  # Wrong! Should be 0.003
                model_used="gpt-4o",
            )

    def test_accepts_zero_tokens_and_costs(self):
        """Zero tokens and costs are valid."""
        breakdown = AgentCostBreakdown(
            agent_id="agent-001",
            agent_name="Researcher",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
            model_used="gpt-4o-mini",
        )
        assert breakdown.total_tokens == 0
        assert breakdown.total_cost == 0.0

    def test_to_dict_serialization(self):
        """to_dict should serialize all fields correctly."""
        breakdown = AgentCostBreakdown(
            agent_id="agent-001",
            agent_name="Researcher",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
            model_used="gpt-4o-mini",
        )

        result = breakdown.to_dict()

        assert result == {
            "agent_id": "agent-001",
            "agent_name": "Researcher",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "input_cost": 0.001,
            "output_cost": 0.002,
            "total_cost": 0.003,
            "model_used": "gpt-4o-mini",
        }


class TestWorkflowCostSummary:
    """Tests for WorkflowCostSummary validation and aggregation."""

    def test_valid_workflow_aggregates_correctly(self):
        """Workflow should aggregate costs from multiple agents correctly."""
        agent1 = AgentCostBreakdown(
            agent_id="agent-001",
            agent_name="Researcher",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
            model_used="gpt-4o-mini",
        )
        agent2 = AgentCostBreakdown(
            agent_id="agent-002",
            agent_name="Writer",
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
            input_cost=0.002,
            output_cost=0.004,
            total_cost=0.006,
            model_used="gpt-4o",
        )

        workflow = WorkflowCostSummary(
            workflow_id="workflow-001",
            workflow_name="Research + Write",
            agent_breakdowns=[agent1, agent2],
        )

        # Verify aggregated totals
        assert workflow.total_input_tokens == 300
        assert workflow.total_output_tokens == 150
        assert workflow.total_tokens == 450
        assert workflow.total_input_cost == 0.003
        assert workflow.total_output_cost == 0.006
        assert abs(workflow.total_cost - 0.009) < 0.0001

    def test_rejects_empty_workflow_id(self):
        """Empty workflow_id should raise ValueError."""
        agent = AgentCostBreakdown(
            agent_id="agent-001",
            agent_name="Researcher",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
            model_used="gpt-4o-mini",
        )

        with pytest.raises(ValueError, match="workflow_id cannot be empty"):
            WorkflowCostSummary(
                workflow_id="",
                workflow_name="Test Workflow",
                agent_breakdowns=[agent],
            )

    def test_rejects_empty_workflow_name(self):
        """Empty workflow_name should raise ValueError."""
        agent = AgentCostBreakdown(
            agent_id="agent-001",
            agent_name="Researcher",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
            model_used="gpt-4o-mini",
        )

        with pytest.raises(ValueError, match="workflow_name cannot be empty"):
            WorkflowCostSummary(
                workflow_id="workflow-001",
                workflow_name="",
                agent_breakdowns=[agent],
            )

    def test_rejects_empty_agent_breakdowns(self):
        """Empty agent_breakdowns should raise ValueError."""
        with pytest.raises(ValueError, match="agent_breakdowns cannot be empty"):
            WorkflowCostSummary(
                workflow_id="workflow-001",
                workflow_name="Test Workflow",
                agent_breakdowns=[],
            )

    def test_single_agent_workflow(self):
        """Workflow with single agent should work correctly."""
        agent = AgentCostBreakdown(
            agent_id="agent-001",
            agent_name="Solo Agent",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
            model_used="gpt-4o-mini",
        )

        workflow = WorkflowCostSummary(
            workflow_id="workflow-001",
            workflow_name="Solo Workflow",
            agent_breakdowns=[agent],
        )

        # Totals should match single agent
        assert workflow.total_tokens == 150
        assert workflow.total_cost == 0.003

    def test_many_agents_workflow(self):
        """Workflow with many agents should aggregate correctly."""
        agents = [
            AgentCostBreakdown(
                agent_id=f"agent-{i:03d}",
                agent_name=f"Agent {i}",
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                input_cost=0.001,
                output_cost=0.002,
                total_cost=0.003,
                model_used="gpt-4o-mini",
            )
            for i in range(10)
        ]

        workflow = WorkflowCostSummary(
            workflow_id="workflow-001",
            workflow_name="Multi-Agent Workflow",
            agent_breakdowns=agents,
        )

        # Verify totals (10 agents * 150 tokens = 1500)
        assert workflow.total_tokens == 1500
        assert abs(workflow.total_cost - 0.030) < 0.0001

    def test_to_dict_serialization(self):
        """to_dict should serialize workflow and agent breakdowns."""
        agent1 = AgentCostBreakdown(
            agent_id="agent-001",
            agent_name="Researcher",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
            model_used="gpt-4o-mini",
        )
        agent2 = AgentCostBreakdown(
            agent_id="agent-002",
            agent_name="Writer",
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
            input_cost=0.002,
            output_cost=0.004,
            total_cost=0.006,
            model_used="gpt-4o",
        )

        workflow = WorkflowCostSummary(
            workflow_id="workflow-001",
            workflow_name="Research + Write",
            agent_breakdowns=[agent1, agent2],
        )

        result = workflow.to_dict()

        assert result["workflow_id"] == "workflow-001"
        assert result["workflow_name"] == "Research + Write"
        assert result["total_tokens"] == 450
        assert abs(result["total_cost"] - 0.009) < 0.0001
        assert len(result["agent_breakdowns"]) == 2
        assert result["agent_breakdowns"][0]["agent_id"] == "agent-001"
        assert result["agent_breakdowns"][1]["agent_id"] == "agent-002"
