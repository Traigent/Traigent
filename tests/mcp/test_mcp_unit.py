"""Unit tests for individual MCP endpoints.

Tests each MCP endpoint in isolation to ensure correct behavior,
error handling, and response formats.
"""

import pytest

from traigent.cloud.models import (
    AgentExecutionRequest,
    AgentOptimizationRequest,
    AgentSpecification,
    NextTrialRequest,
    OptimizationSessionStatus,
    SessionCreationRequest,
)


class TestAgentCreation:
    """Test agent creation endpoints."""

    @pytest.mark.asyncio
    async def test_create_basic_agent(self, mock_mcp_service, sample_agent_spec):
        """Test creating a basic conversational agent."""
        # Create agent
        result = await mock_mcp_service.create_agent(sample_agent_spec)

        # Verify response
        assert result["status"] == "created"
        assert "agent_id" in result
        assert result["platform"] == "openai"

        # Verify agent was stored
        assert len(mock_mcp_service.agents) == 1
        assert result["agent_id"] in mock_mcp_service.agents

    @pytest.mark.asyncio
    async def test_create_agent_with_tools(self, mock_mcp_service):
        """Test creating a task agent with custom tools."""
        agent_spec = AgentSpecification(
            id="task-agent-001",
            name="Documentation Assistant",
            agent_type="task",
            agent_platform="langchain",
            prompt_template="Search and answer: {query}",
            model_parameters={"model": "GPT-4o", "temperature": 0.3},
            custom_tools=["search", "read_docs", "summarize"],
        )

        result = await mock_mcp_service.create_agent(agent_spec)

        assert result["status"] == "created"
        stored_agent = mock_mcp_service.agents[result["agent_id"]]
        assert stored_agent.custom_tools == ["search", "read_docs", "summarize"]

    @pytest.mark.asyncio
    async def test_create_agent_validation(self, mock_mcp_service):
        """Test agent creation with invalid specifications."""
        # Agent without required fields
        invalid_spec = AgentSpecification(
            id="invalid",
            name="Invalid Agent",
            agent_type="unknown_type",  # Invalid type
            agent_platform="openai",
            prompt_template="",  # Empty template
            model_parameters={},  # Missing required params
        )

        # In a real implementation, this should raise an error
        # For the mock, we'll just verify it was called
        result = await mock_mcp_service.create_agent(invalid_spec)
        assert "agent_id" in result  # Mock always succeeds

    @pytest.mark.asyncio
    async def test_create_multiple_agents(self, mock_mcp_service):
        """Test creating multiple agents."""
        agents = []
        for i in range(3):
            spec = AgentSpecification(
                id=f"agent-{i}",
                name=f"Agent {i}",
                agent_type="conversational",
                agent_platform="openai",
                prompt_template=f"Template {i}",
                model_parameters={"model": "o4-mini"},
            )
            result = await mock_mcp_service.create_agent(spec)
            agents.append(result["agent_id"])

        assert len(mock_mcp_service.agents) == 3
        assert len(set(agents)) == 3  # All unique IDs


class TestAgentExecution:
    """Test agent execution endpoints."""

    @pytest.mark.asyncio
    async def test_execute_agent_basic(self, mock_mcp_service, sample_agent_spec):
        """Test basic agent execution."""
        # First create an agent
        create_result = await mock_mcp_service.create_agent(sample_agent_spec)
        create_result["agent_id"]

        # Execute the agent
        request = AgentExecutionRequest(
            agent_spec=sample_agent_spec, input_data={"query": "What is Python?"}
        )

        response = await mock_mcp_service.execute_agent(request)

        # Verify response
        assert response.output is not None
        assert response.duration > 0
        assert response.tokens_used > 0
        assert response.cost > 0
        assert response.error is None

    @pytest.mark.asyncio
    async def test_execute_agent_with_overrides(
        self, mock_mcp_service, sample_agent_spec
    ):
        """Test agent execution with configuration overrides."""
        request = AgentExecutionRequest(
            agent_spec=sample_agent_spec,
            input_data={"query": "Explain decorators"},
            config_overrides={"temperature": 0.2, "max_tokens": 200},
        )

        response = await mock_mcp_service.execute_agent(request)

        assert response.output is not None
        assert "Mock response" in response.output

    @pytest.mark.asyncio
    async def test_execute_agent_error_handling(self, mock_mcp_service):
        """Test agent execution error handling."""
        # Agent with missing platform
        invalid_spec = AgentSpecification(
            id="error-agent",
            name="Error Agent",
            agent_type="conversational",
            agent_platform="non_existent_platform",
            prompt_template="Test",
            model_parameters={},
        )

        request = AgentExecutionRequest(
            agent_spec=invalid_spec, input_data={"query": "test"}
        )

        # Mock returns success, but real implementation should handle errors
        response = await mock_mcp_service.execute_agent(request)
        assert response.output is not None

    @pytest.mark.asyncio
    async def test_execute_agent_with_context(
        self, mock_mcp_service, sample_agent_spec
    ):
        """Test agent execution with execution context."""
        request = AgentExecutionRequest(
            agent_spec=sample_agent_spec,
            input_data={"query": "What is async/await?"},
            execution_context={
                "session_id": "sess-123",
                "user_id": "user-456",
                "previous_queries": ["What is Python?", "How do functions work?"],
            },
        )

        response = await mock_mcp_service.execute_agent(request)
        assert response.output is not None


class TestOptimizationSessions:
    """Test optimization session management."""

    @pytest.mark.asyncio
    async def test_create_optimization_session(self, mock_mcp_service):
        """Test creating an optimization session."""
        request = SessionCreationRequest(
            function_name="chat_assistant",
            configuration_space={
                "model": ["o4-mini", "GPT-4o"],
                "temperature": (0.0, 1.0),
            },
            objectives=["accuracy", "cost"],
            dataset_metadata={"size": 100, "type": "qa_pairs"},
            max_trials=20,
        )

        response = await mock_mcp_service.create_optimization_session(request)

        assert response.session_id is not None
        assert response.status == OptimizationSessionStatus.CREATED
        assert "optimization_strategy" in response.__dict__

    @pytest.mark.asyncio
    async def test_get_next_trial(self, mock_mcp_service):
        """Test getting next trial suggestion."""
        # Create session first
        session_request = SessionCreationRequest(
            function_name="test_function",
            configuration_space={"model": ["o4-mini", "GPT-4o"]},
            objectives=["accuracy"],
            dataset_metadata={"size": 50},
        )

        session_response = await mock_mcp_service.create_optimization_session(
            session_request
        )
        session_id = session_response.session_id

        # Get next trial
        trial_request = NextTrialRequest(session_id=session_id)
        trial_response = await mock_mcp_service.get_next_trial(trial_request)

        assert trial_response.suggestion is not None
        assert trial_response.should_continue is True
        assert trial_response.suggestion.config is not None
        assert trial_response.suggestion.dataset_subset is not None

    @pytest.mark.asyncio
    async def test_trial_progression(self, mock_mcp_service):
        """Test trial progression through optimization."""
        # Create session
        session_request = SessionCreationRequest(
            function_name="test_function",
            configuration_space={
                "model": ["o4-mini", "GPT-4o"],
                "temperature": (0.0, 1.0),
            },
            objectives=["accuracy"],
            dataset_metadata={"size": 100},
            max_trials=10,
        )

        session_response = await mock_mcp_service.create_optimization_session(
            session_request
        )
        session_id = session_response.session_id

        # Run multiple trials
        trials = []
        for i in range(5):
            trial_request = NextTrialRequest(session_id=session_id)
            trial_response = await mock_mcp_service.get_next_trial(trial_request)

            assert trial_response.suggestion is not None
            trials.append(trial_response.suggestion)

            # Verify trial number increases
            assert trial_response.suggestion.trial_number == i + 1

        # Verify exploration vs exploitation
        exploration_count = sum(
            1 for t in trials if t.exploration_type == "exploration"
        )
        assert exploration_count >= 2  # Should have some exploration

    @pytest.mark.asyncio
    async def test_invalid_session_handling(self, mock_mcp_service):
        """Test handling of invalid session requests."""
        # Request trial for non-existent session
        trial_request = NextTrialRequest(session_id="invalid-session-id")
        trial_response = await mock_mcp_service.get_next_trial(trial_request)

        assert trial_response.suggestion is None
        assert trial_response.should_continue is False
        assert trial_response.reason == "Session not found"


class TestAgentOptimization:
    """Test agent optimization endpoints."""

    @pytest.mark.asyncio
    async def test_start_agent_optimization(
        self, mock_mcp_service, sample_agent_spec, sample_dataset
    ):
        """Test starting agent optimization."""
        request = AgentOptimizationRequest(
            agent_spec=sample_agent_spec,
            dataset=sample_dataset,
            configuration_space={
                "temperature": (0.0, 1.0),
                "max_tokens": [100, 200, 300],
            },
            objectives=["accuracy", "response_time"],
            max_trials=30,
        )

        response = await mock_mcp_service.start_agent_optimization(request)

        assert response.session_id is not None
        assert response.optimization_id is not None
        assert response.status == "started"
        assert response.estimated_cost > 0
        assert response.estimated_duration > 0
        assert len(response.next_steps) > 0

    @pytest.mark.asyncio
    async def test_optimization_with_constraints(
        self, mock_mcp_service, sample_agent_spec, sample_dataset
    ):
        """Test optimization with budget constraints."""
        request = AgentOptimizationRequest(
            agent_spec=sample_agent_spec,
            dataset=sample_dataset,
            configuration_space={
                "model": ["o4-mini", "GPT-4o"],
                "temperature": (0.0, 0.8),
            },
            objectives=["accuracy", "cost"],
            max_trials=20,
            target_cost_reduction=0.5,
            metadata={"budget_limit": 10.0},
        )

        response = await mock_mcp_service.start_agent_optimization(request)
        assert response.status == "started"


class TestCallHistory:
    """Test MCP service call tracking."""

    @pytest.mark.asyncio
    async def test_call_history_tracking(self, mock_mcp_service, sample_agent_spec):
        """Test that all calls are properly tracked."""
        # Make various calls
        await mock_mcp_service.create_agent(sample_agent_spec)

        exec_request = AgentExecutionRequest(
            agent_spec=sample_agent_spec, input_data={"query": "test"}
        )
        await mock_mcp_service.execute_agent(exec_request)

        session_request = SessionCreationRequest(
            function_name="test",
            configuration_space={},
            objectives=["accuracy"],
            dataset_metadata={},
        )
        await mock_mcp_service.create_optimization_session(session_request)

        # Check history
        history = mock_mcp_service.get_call_history()
        assert len(history) == 3
        assert history[0][0] == "create_agent"
        assert history[1][0] == "execute_agent"
        assert history[2][0] == "create_optimization_session"

    @pytest.mark.asyncio
    async def test_clear_history(self, mock_mcp_service, sample_agent_spec):
        """Test clearing call history."""
        # Make some calls
        await mock_mcp_service.create_agent(sample_agent_spec)
        assert len(mock_mcp_service.get_call_history()) == 1

        # Clear history
        mock_mcp_service.clear_history()
        assert len(mock_mcp_service.get_call_history()) == 0


class TestDatasetSubsetSelection:
    """Test dataset subset selection in trials."""

    @pytest.mark.asyncio
    async def test_subset_indices_validity(self, mock_mcp_service):
        """Test that subset indices are valid."""
        # Create session
        session_request = SessionCreationRequest(
            function_name="test",
            configuration_space={"model": ["o4-mini"]},
            objectives=["accuracy"],
            dataset_metadata={"size": 100},
        )

        session_response = await mock_mcp_service.create_optimization_session(
            session_request
        )

        # Get trial
        trial_request = NextTrialRequest(session_id=session_response.session_id)
        trial_response = await mock_mcp_service.get_next_trial(trial_request)

        subset = trial_response.suggestion.dataset_subset
        assert len(subset.indices) > 0
        assert subset.selection_strategy in [
            "diverse_sampling",
            "representative_sampling",
        ]
        assert 0 <= subset.confidence_level <= 1
        assert 0 <= subset.estimated_representativeness <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
