"""Unit tests for agent support in cloud client."""

from unittest.mock import AsyncMock

import pytest

from traigent.cloud.client import CloudServiceError, TraiGentCloudClient
from traigent.cloud.models import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    AgentOptimizationRequest,
    AgentOptimizationResponse,
    AgentOptimizationStatus,
    AgentSpecification,
    OptimizationSessionStatus,
)
from traigent.evaluators.base import Dataset, EvaluationExample


@pytest.fixture
def mock_aiohttp_session():
    """Create mock aiohttp session."""
    from unittest.mock import MagicMock

    session = MagicMock()
    return session


@pytest.fixture
def sample_agent_spec():
    """Create sample agent specification."""
    return AgentSpecification(
        id="test-agent",
        name="Test Agent",
        agent_type="conversational",
        agent_platform="openai",
        prompt_template="Answer this question: {question}",
        model_parameters={
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 500,
        },
        persona="helpful assistant",
        guidelines=["Be concise", "Be accurate"],
        metadata={"version": "1.0"},
    )


@pytest.fixture
def sample_dataset():
    """Create sample dataset."""
    examples = [
        EvaluationExample(
            input_data={"question": "What is AI?"},
            expected_output="Artificial Intelligence",
        ),
        EvaluationExample(
            input_data={"question": "What is ML?"}, expected_output="Machine Learning"
        ),
    ]
    return Dataset(examples)


@pytest.fixture
def agent_client(mock_aiohttp_session):
    """Create agent client with mocked session."""
    # Create a valid format API key (tg_ + 61 characters = 64 total)
    valid_api_key = "tg_" + "1234567890abcdef" * 3 + "1234567890abc"  # 64 chars total
    client = TraiGentCloudClient(
        base_url="https://api.traigent.ai", api_key=valid_api_key
    )
    client._session = mock_aiohttp_session
    return client


class TestAgentOptimization:
    """Test agent optimization functionality."""

    @pytest.mark.asyncio
    async def test_optimize_agent_success(
        self, agent_client, sample_agent_spec, sample_dataset
    ):
        """Test successful agent optimization."""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "session_id": "session_123",
            "optimization_id": "opt_456",
            "status": "started",
            "estimated_cost": 5.0,
            "estimated_duration": 300.0,
            "next_steps": ["Monitor progress", "Check results"],
        }

        agent_client._session.post.return_value.__aenter__.return_value = mock_response

        # Test optimization request
        config_space = {
            "model": ["gpt-4o-mini", "GPT-4o"],
            "temperature": (0.1, 0.9),
            "max_tokens": [100, 500, 1000],
        }

        result = await agent_client.optimize_agent(
            agent_spec=sample_agent_spec,
            dataset=sample_dataset,
            configuration_space=config_space,
            objectives=["accuracy", "cost"],
            max_trials=20,
        )

        # Verify result
        assert isinstance(result, AgentOptimizationResponse)
        assert result.session_id == "session_123"
        assert result.optimization_id == "opt_456"
        assert result.status == "started"
        assert result.estimated_cost == 5.0
        assert result.estimated_duration == 300.0
        assert "Monitor progress" in result.next_steps

        # Verify request was made correctly
        agent_client._session.post.assert_called_once()
        call_args = agent_client._session.post.call_args
        assert call_args[1]["json"]["configuration_space"] == config_space
        assert call_args[1]["json"]["objectives"] == ["accuracy", "cost"]
        assert call_args[1]["json"]["max_trials"] == 20

    @pytest.mark.asyncio
    async def test_optimize_agent_default_objectives(
        self, agent_client, sample_agent_spec, sample_dataset
    ):
        """Test agent optimization with default objectives."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "session_id": "session_123",
            "optimization_id": "opt_456",
            "status": "started",
        }

        agent_client._session.post.return_value.__aenter__.return_value = mock_response

        # Test with no objectives specified (should default to ["accuracy"])
        await agent_client.optimize_agent(
            agent_spec=sample_agent_spec,
            dataset=sample_dataset,
            configuration_space={"temperature": (0.1, 0.9)},
        )

        # Verify default objectives were used
        call_args = agent_client._session.post.call_args
        assert call_args[1]["json"]["objectives"] == ["accuracy"]

    @pytest.mark.asyncio
    async def test_optimize_agent_http_error(
        self, agent_client, sample_agent_spec, sample_dataset
    ):
        """Test agent optimization with HTTP error."""
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text.return_value = "Invalid request"

        agent_client._session.post.return_value.__aenter__.return_value = mock_response

        with pytest.raises(
            CloudServiceError, match="Failed to start agent optimization: HTTP 400"
        ):
            await agent_client.optimize_agent(
                agent_spec=sample_agent_spec,
                dataset=sample_dataset,
                configuration_space={"temperature": (0.1, 0.9)},
            )

    @pytest.mark.asyncio
    async def test_optimize_agent_no_session(self, sample_agent_spec, sample_dataset):
        """Test agent optimization without initialized session."""
        from traigent.utils.exceptions import AuthenticationError

        client = TraiGentCloudClient(
            base_url="https://api.traigent.ai", api_key="test-key"
        )
        # Don't set _session - this will cause authentication to fail

        with pytest.raises(
            AuthenticationError, match="Not authenticated with TraiGent Cloud Service"
        ):
            await client.optimize_agent(
                agent_spec=sample_agent_spec,
                dataset=sample_dataset,
                configuration_space={"temperature": (0.1, 0.9)},
            )


class TestAgentExecution:
    """Test direct agent execution functionality."""

    @pytest.mark.asyncio
    async def test_execute_agent_success(self, agent_client, sample_agent_spec):
        """Test successful agent execution."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "output": "Artificial Intelligence is a field of computer science.",
            "duration": 2.5,
            "tokens_used": 45,
            "cost": 0.002,
            "metadata": {"model": "gpt-4o-mini"},
            "error": None,
        }

        agent_client._session.post.return_value.__aenter__.return_value = mock_response

        # Test execution
        input_data = {"question": "What is AI?"}
        config_overrides = {"temperature": 0.8}

        result = await agent_client.execute_agent(
            agent_spec_or_request=sample_agent_spec,
            input_data=input_data,
            config_overrides=config_overrides,
        )

        # Verify result
        assert isinstance(result, AgentExecutionResponse)
        assert (
            result.output == "Artificial Intelligence is a field of computer science."
        )
        assert result.duration == 2.5
        assert result.tokens_used == 45
        assert result.cost == 0.002
        assert result.metadata["model"] == "gpt-4o-mini"
        assert result.error is None

        # Verify request
        call_args = agent_client._session.post.call_args
        assert call_args[1]["json"]["input_data"] == input_data
        assert call_args[1]["json"]["config_overrides"] == config_overrides

    @pytest.mark.asyncio
    async def test_execute_agent_with_error(self, agent_client, sample_agent_spec):
        """Test agent execution with error response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "output": None,
            "duration": 0.5,
            "tokens_used": None,
            "cost": None,
            "metadata": {},
            "error": "Model not available",
        }

        agent_client._session.post.return_value.__aenter__.return_value = mock_response

        result = await agent_client.execute_agent(
            agent_spec_or_request=sample_agent_spec,
            input_data={"question": "What is AI?"},
        )

        assert result.output is None
        assert result.error == "Model not available"

    @pytest.mark.asyncio
    async def test_execute_agent_http_error(self, agent_client, sample_agent_spec):
        """Test agent execution with HTTP error."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text.return_value = "Internal server error"

        agent_client._session.post.return_value.__aenter__.return_value = mock_response

        with pytest.raises(
            CloudServiceError, match="Failed to execute agent: HTTP 500"
        ):
            await agent_client.execute_agent(
                agent_spec_or_request=sample_agent_spec,
                input_data={"question": "What is AI?"},
            )


class TestAgentOptimizationStatus:
    """Test agent optimization status monitoring."""

    @pytest.mark.asyncio
    async def test_get_optimization_status_success(self, agent_client):
        """Test successful status retrieval."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "optimization_id": "opt_456",
            "status": "active",
            "progress": 0.4,
            "completed_trials": 8,
            "total_trials": 20,
            "current_best_metrics": {"accuracy": 0.85, "cost": 0.002},
            "estimated_time_remaining": 120.0,
            "metadata": {"strategy": "bayesian"},
        }

        agent_client._session.get.return_value.__aenter__.return_value = mock_response

        result = await agent_client.get_agent_optimization_status("opt_456")

        assert isinstance(result, AgentOptimizationStatus)
        assert result.optimization_id == "opt_456"
        assert result.status == OptimizationSessionStatus.ACTIVE
        assert result.progress == 0.4
        assert result.completed_trials == 8
        assert result.total_trials == 20
        assert result.current_best_metrics["accuracy"] == 0.85
        assert result.estimated_time_remaining == 120.0

    @pytest.mark.asyncio
    async def test_get_optimization_status_not_found(self, agent_client):
        """Test status retrieval for non-existent optimization."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text.return_value = "Optimization not found"

        agent_client._session.get.return_value.__aenter__.return_value = mock_response

        with pytest.raises(
            CloudServiceError, match="Failed to get optimization status: HTTP 404"
        ):
            await agent_client.get_agent_optimization_status("opt_nonexistent")


class TestAgentOptimizationCancellation:
    """Test agent optimization cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_optimization_success(self, agent_client):
        """Test successful optimization cancellation."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "optimization_id": "opt_456",
            "status": "cancelled",
            "message": "Optimization cancelled successfully",
        }

        agent_client._session.post.return_value.__aenter__.return_value = mock_response

        result = await agent_client.cancel_agent_optimization("opt_456")

        assert result["optimization_id"] == "opt_456"
        assert result["status"] == "cancelled"
        assert "cancelled successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_cancel_optimization_not_found(self, agent_client):
        """Test cancellation of non-existent optimization."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text.return_value = "Optimization not found"

        agent_client._session.post.return_value.__aenter__.return_value = mock_response

        with pytest.raises(
            CloudServiceError, match="Failed to cancel optimization: HTTP 404"
        ):
            await agent_client.cancel_agent_optimization("opt_nonexistent")


class TestAgentSerialization:
    """Test agent serialization methods."""

    def test_serialize_agent_spec(self, agent_client, sample_agent_spec):
        """Test agent specification serialization."""
        serialized = agent_client._serialize_agent_spec(sample_agent_spec)

        assert serialized["id"] == "test-agent"
        assert serialized["name"] == "Test Agent"
        assert serialized["agent_type"] == "conversational"
        assert serialized["agent_platform"] == "openai"
        assert serialized["prompt_template"] == "Answer this question: {question}"
        assert serialized["model_parameters"]["model"] == "gpt-4o-mini"
        assert serialized["persona"] == "helpful assistant"
        assert serialized["guidelines"] == ["Be concise", "Be accurate"]
        assert serialized["metadata"]["version"] == "1.0"

    def test_serialize_agent_optimization_request(
        self, agent_client, sample_agent_spec, sample_dataset
    ):
        """Test agent optimization request serialization."""
        request = AgentOptimizationRequest(
            agent_spec=sample_agent_spec,
            dataset=sample_dataset,
            configuration_space={"temperature": (0.1, 0.9)},
            objectives=["accuracy"],
            max_trials=10,
            target_cost_reduction=0.5,
            user_id="user_123",
            billing_tier="premium",
        )

        serialized = agent_client._serialize_agent_optimization_request(request)

        assert "agent_spec" in serialized
        assert "dataset" in serialized
        assert serialized["configuration_space"] == {"temperature": (0.1, 0.9)}
        assert serialized["objectives"] == ["accuracy"]
        assert serialized["max_trials"] == 10
        assert serialized["target_cost_reduction"] == 0.5
        assert serialized["user_id"] == "user_123"
        assert serialized["billing_tier"] == "premium"

    def test_serialize_agent_execution_request(self, agent_client, sample_agent_spec):
        """Test agent execution request serialization."""
        request = AgentExecutionRequest(
            agent_spec=sample_agent_spec,
            input_data={"question": "What is AI?"},
            config_overrides={"temperature": 0.8},
            execution_context={"timeout": 30},
        )

        serialized = agent_client._serialize_agent_execution_request(request)

        assert "agent_spec" in serialized
        assert serialized["input_data"] == {"question": "What is AI?"}
        assert serialized["config_overrides"] == {"temperature": 0.8}
        assert serialized["execution_context"] == {"timeout": 30}


class TestAgentDeserialization:
    """Test agent deserialization methods."""

    def test_deserialize_agent_optimization_response(self, agent_client):
        """Test agent optimization response deserialization."""
        data = {
            "session_id": "session_123",
            "optimization_id": "opt_456",
            "status": "started",
            "estimated_cost": 5.0,
            "estimated_duration": 300.0,
            "next_steps": ["Monitor progress", "Check results"],
        }

        response = agent_client._deserialize_agent_optimization_response(data)

        assert isinstance(response, AgentOptimizationResponse)
        assert response.session_id == "session_123"
        assert response.optimization_id == "opt_456"
        assert response.status == "started"
        assert response.estimated_cost == 5.0
        assert response.estimated_duration == 300.0
        assert response.next_steps == ["Monitor progress", "Check results"]

    def test_deserialize_agent_execution_response(self, agent_client):
        """Test agent execution response deserialization."""
        data = {
            "output": "AI is artificial intelligence",
            "duration": 2.5,
            "tokens_used": 45,
            "cost": 0.002,
            "metadata": {"model": "gpt-4o-mini"},
            "error": None,
        }

        response = agent_client._deserialize_agent_execution_response(data)

        assert isinstance(response, AgentExecutionResponse)
        assert response.output == "AI is artificial intelligence"
        assert response.duration == 2.5
        assert response.tokens_used == 45
        assert response.cost == 0.002
        assert response.metadata["model"] == "gpt-4o-mini"
        assert response.error is None

    def test_deserialize_agent_optimization_status(self, agent_client):
        """Test agent optimization status deserialization."""
        data = {
            "optimization_id": "opt_456",
            "status": "active",
            "progress": 0.6,
            "completed_trials": 12,
            "total_trials": 20,
            "current_best_metrics": {"accuracy": 0.88},
            "estimated_time_remaining": 90.0,
            "metadata": {"strategy": "grid"},
        }

        status = agent_client._deserialize_agent_optimization_status(data)

        assert isinstance(status, AgentOptimizationStatus)
        assert status.optimization_id == "opt_456"
        assert status.status == OptimizationSessionStatus.ACTIVE
        assert status.progress == 0.6
        assert status.completed_trials == 12
        assert status.total_trials == 20
        assert status.current_best_metrics["accuracy"] == 0.88
        assert status.estimated_time_remaining == 90.0
        assert status.metadata["strategy"] == "grid"


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_network_error_optimize_agent(
        self, agent_client, sample_agent_spec, sample_dataset
    ):
        """Test network error during agent optimization."""
        # Mock network error
        agent_client._session.post.side_effect = Exception("Connection failed")

        with pytest.raises(
            CloudServiceError, match="Network error starting agent optimization"
        ):
            await agent_client.optimize_agent(
                agent_spec=sample_agent_spec,
                dataset=sample_dataset,
                configuration_space={"temperature": (0.1, 0.9)},
            )

    @pytest.mark.asyncio
    async def test_network_error_execute_agent(self, agent_client, sample_agent_spec):
        """Test network error during agent execution."""
        agent_client._session.post.side_effect = Exception("Connection timeout")

        with pytest.raises(CloudServiceError, match="Network error executing agent"):
            await agent_client.execute_agent(
                agent_spec_or_request=sample_agent_spec,
                input_data={"question": "What is AI?"},
            )

    @pytest.mark.asyncio
    async def test_network_error_get_status(self, agent_client):
        """Test network error during status retrieval."""
        agent_client._session.get.side_effect = Exception("Network timeout")

        with pytest.raises(
            CloudServiceError, match="Network error getting optimization status"
        ):
            await agent_client.get_agent_optimization_status("opt_456")

    @pytest.mark.asyncio
    async def test_network_error_cancel_optimization(self, agent_client):
        """Test network error during optimization cancellation."""
        agent_client._session.post.side_effect = Exception("Connection reset")

        with pytest.raises(
            CloudServiceError, match="Network error canceling optimization"
        ):
            await agent_client.cancel_agent_optimization("opt_456")


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple operations."""

    @pytest.mark.asyncio
    async def test_full_agent_optimization_workflow(
        self, agent_client, sample_agent_spec, sample_dataset
    ):
        """Test complete agent optimization workflow."""
        # Mock optimization start
        start_response = AsyncMock()
        start_response.status = 200
        start_response.json.return_value = {
            "session_id": "session_123",
            "optimization_id": "opt_456",
            "status": "started",
        }

        # Mock status check
        status_response = AsyncMock()
        status_response.status = 200
        status_response.json.return_value = {
            "optimization_id": "opt_456",
            "status": "completed",
            "progress": 1.0,
            "completed_trials": 20,
            "total_trials": 20,
            "current_best_metrics": {"accuracy": 0.95},
        }

        # Mock cancellation
        cancel_response = AsyncMock()
        cancel_response.status = 200
        cancel_response.json.return_value = {
            "optimization_id": "opt_456",
            "status": "cancelled",
        }

        # Set up mock responses
        agent_client._session.post.return_value.__aenter__.side_effect = [
            start_response,
            cancel_response,
        ]
        agent_client._session.get.return_value.__aenter__.return_value = status_response

        # 1. Start optimization
        opt_result = await agent_client.optimize_agent(
            agent_spec=sample_agent_spec,
            dataset=sample_dataset,
            configuration_space={"temperature": (0.1, 0.9)},
        )

        assert opt_result.optimization_id == "opt_456"

        # 2. Check status
        status = await agent_client.get_agent_optimization_status("opt_456")
        assert status.status == OptimizationSessionStatus.COMPLETED
        assert status.progress == 1.0

        # 3. Cancel (even though completed, for testing)
        cancel_result = await agent_client.cancel_agent_optimization("opt_456")
        assert cancel_result["status"] == "cancelled"
