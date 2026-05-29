"""Unit tests for agent executor base class."""

from unittest.mock import AsyncMock

import pytest

from traigent.agents.executor import AgentExecutionResult, AgentExecutor
from traigent.cloud.models import AgentSpecification
from traigent.utils.exceptions import AgentExecutionError


class MockAgentExecutor(AgentExecutor):
    """Mock implementation for testing."""

    async def _platform_initialize(self) -> None:
        """Mock platform initialization."""
        self.platform_initialized = True

    async def _execute_agent(self, agent_spec, input_data, config):
        """Mock agent execution."""
        return {
            "output": f"Mock output for {input_data.get('query', 'test')}",
            "tokens_used": 100,
            "cost": 0.002,
            "metadata": {"mock": True},
        }

    def _validate_platform_spec(self, agent_spec):
        """Mock platform validation."""
        if agent_spec.agent_platform != "mock":
            raise ValueError("Invalid platform for mock executor")

    async def _validate_platform_config(self, config):
        """Mock config validation."""
        errors = []
        warnings = []

        if config.get("invalid_param"):
            errors.append("invalid_param is not allowed")

        if config.get("deprecated_param"):
            warnings.append("deprecated_param is deprecated")

        return {"errors": errors, "warnings": warnings}

    def _get_platform_capabilities(self):
        """Mock capabilities."""
        return ["test", "mock", "async"]


@pytest.fixture
def mock_executor():
    """Create a mock executor."""
    return MockAgentExecutor(platform_config={"test": True})


@pytest.fixture
def sample_agent_spec():
    """Create a sample agent specification."""
    return AgentSpecification(
        id="test-agent",
        name="Test Agent",
        agent_type="conversational",
        agent_platform="mock",
        prompt_template="Answer this: {query}",
        model_parameters={"temperature": 0.7, "max_tokens": 100},
    )


class TestAgentExecutor:
    """Test AgentExecutor base functionality."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_executor):
        """Test executor initialization."""
        assert not mock_executor._initialized

        await mock_executor.initialize()

        assert mock_executor._initialized
        assert hasattr(mock_executor, "platform_initialized")
        assert mock_executor.platform_initialized is True

        # Second initialization should be no-op
        mock_executor.platform_initialized = False
        await mock_executor.initialize()
        assert mock_executor.platform_initialized is False  # Not re-initialized

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_executor, sample_agent_spec):
        """Test successful agent execution."""
        input_data = {"query": "What is AI?"}

        result = await mock_executor.execute(sample_agent_spec, input_data)

        assert isinstance(result, AgentExecutionResult)
        assert result.output == "Mock output for What is AI?"
        assert result.tokens_used == 100
        assert result.cost == 0.002
        assert result.duration > 0
        assert result.error is None
        assert result.metadata["mock"] is True

    @pytest.mark.asyncio
    async def test_execute_with_config_overrides(
        self, mock_executor, sample_agent_spec
    ):
        """Test execution with configuration overrides."""
        input_data = {"query": "test"}
        config_overrides = {"temperature": 0.9, "new_param": "value"}

        # Spy on _execute_agent to check merged config
        original_execute = mock_executor._execute_agent
        called_config = None

        async def spy_execute(spec, data, config):
            nonlocal called_config
            called_config = config
            return await original_execute(spec, data, config)

        mock_executor._execute_agent = spy_execute

        await mock_executor.execute(sample_agent_spec, input_data, config_overrides)

        assert called_config["temperature"] == 0.9  # Override applied
        assert called_config["max_tokens"] == 100  # Original preserved
        assert called_config["new_param"] == "value"  # New param added

    @pytest.mark.asyncio
    async def test_execute_validation_error(self, mock_executor):
        """Test execution with invalid agent specification."""
        # Agent with no prompt template
        invalid_agent = AgentSpecification(
            id="invalid",
            name="Invalid",
            agent_type="task",
            agent_platform="mock",
            prompt_template="",  # Empty template
            model_parameters={},
        )

        result = await mock_executor.execute(invalid_agent, {})

        assert result.output is None
        assert result.error is not None
        assert "prompt template" in result.error

    @pytest.mark.asyncio
    async def test_execute_platform_error(self, mock_executor, sample_agent_spec):
        """Test execution with platform error."""
        # Make _execute_agent raise an error
        mock_executor._execute_agent = AsyncMock(
            side_effect=Exception("Platform error")
        )

        result = await mock_executor.execute(sample_agent_spec, {})

        assert result.output is None
        assert result.error == "Platform error"
        assert result.metadata["error_type"] == "Exception"

    @pytest.mark.asyncio
    async def test_batch_execute(self, mock_executor, sample_agent_spec):
        """Test batch execution."""
        input_batch = [
            {"query": "Question 1"},
            {"query": "Question 2"},
            {"query": "Question 3"},
        ]

        results = await mock_executor.batch_execute(
            sample_agent_spec, input_batch, max_concurrent=2
        )

        assert len(results) == 3
        assert all(isinstance(r, AgentExecutionResult) for r in results)
        assert results[0].output == "Mock output for Question 1"
        assert results[1].output == "Mock output for Question 2"
        assert results[2].output == "Mock output for Question 3"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invalid_concurrency", [0, -1])
    async def test_batch_execute_rejects_non_positive_concurrency(
        self, mock_executor, sample_agent_spec, invalid_concurrency
    ):
        """Ensure batch execution validates concurrency limit."""
        with pytest.raises(ValueError, match="positive integer"):
            await mock_executor.batch_execute(
                sample_agent_spec,
                [{"query": "Question"}],
                max_concurrent=invalid_concurrency,
            )

    def test_merge_configurations(self, mock_executor):
        """Test configuration merging."""
        base_config = {
            "temperature": 0.7,
            "max_tokens": 100,
            "nested": {"param1": "value1", "param2": "value2"},
        }

        overrides = {
            "temperature": 0.9,
            "new_param": "new_value",
            "nested": {"param2": "overridden", "param3": "added"},
        }

        merged = mock_executor._merge_configurations(base_config, overrides)

        assert merged["temperature"] == 0.9
        assert merged["max_tokens"] == 100
        assert merged["new_param"] == "new_value"
        assert merged["nested"]["param1"] == "value1"
        assert merged["nested"]["param2"] == "overridden"
        assert merged["nested"]["param3"] == "added"

    def test_validate_agent_spec(self, mock_executor):
        """Test agent specification validation."""
        # Valid spec
        valid_spec = AgentSpecification(
            id="valid",
            name="Valid",
            agent_type="task",
            agent_platform="mock",
            prompt_template="Test prompt",
            model_parameters={},
        )

        mock_executor._validate_agent_spec(valid_spec)  # Should not raise

        # No prompt template
        with pytest.raises(AgentExecutionError, match="prompt template"):
            invalid_spec = AgentSpecification(
                id="invalid",
                name="Invalid",
                agent_type="task",
                agent_platform="mock",
                prompt_template="",
                model_parameters={},
            )
            mock_executor._validate_agent_spec(invalid_spec)

        # No platform
        with pytest.raises(AgentExecutionError, match="platform"):
            invalid_spec = AgentSpecification(
                id="invalid",
                name="Invalid",
                agent_type="task",
                agent_platform="",
                prompt_template="Test",
                model_parameters={},
            )
            mock_executor._validate_agent_spec(invalid_spec)

    @pytest.mark.asyncio
    async def test_validate_configuration(self, mock_executor):
        """Test configuration validation."""
        # Valid config
        valid_config = {"temperature": 0.7}
        result = await mock_executor.validate_configuration(valid_config)

        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert len(result["warnings"]) == 0

        # Config with errors
        invalid_config = {"invalid_param": True, "temperature": 0.7}
        result = await mock_executor.validate_configuration(invalid_config)

        assert result["valid"] is False
        assert len(result["errors"]) == 1
        assert "invalid_param" in result["errors"][0]

        # Config with warnings
        warning_config = {"deprecated_param": True}
        result = await mock_executor.validate_configuration(warning_config)

        assert result["valid"] is True
        assert len(result["warnings"]) == 1
        assert "deprecated_param" in result["warnings"][0]

    def test_get_platform_info(self, mock_executor):
        """Test getting platform information."""
        info = mock_executor.get_platform_info()

        assert info["platform"] == "MockAgentExecutor"
        assert info["initialized"] is False
        assert "test" in info["capabilities"]
        assert info["config"]["test"] is True

    @pytest.mark.asyncio
    async def test_estimate_cost(self, mock_executor, sample_agent_spec):
        """Test cost estimation."""
        input_data = {"query": "test"}

        with pytest.raises(NotImplementedError, match="cost estimation"):
            await mock_executor.estimate_cost(sample_agent_spec, input_data)

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_executor):
        """Test executor cleanup."""
        # Initialize executor
        await mock_executor.initialize()
        assert mock_executor._initialized is True
        assert mock_executor._cleanup_done is False

        # Cleanup
        await mock_executor.cleanup()

        assert mock_executor._initialized is False
        assert mock_executor._cleanup_done is True

        # Second cleanup should be a no-op
        await mock_executor.cleanup()
        assert mock_executor._cleanup_done is True

    @pytest.mark.asyncio
    async def test_cleanup_without_initialization(self, mock_executor):
        """Test cleanup works even without prior initialization."""
        assert mock_executor._initialized is False

        # Should not raise an error
        await mock_executor.cleanup()

        assert mock_executor._cleanup_done is True

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_executor, sample_agent_spec):
        """Test executor as async context manager."""
        input_data = {"query": "context test"}

        async with mock_executor as executor:
            # Executor should be initialized
            assert executor._initialized is True

            # Execute operation
            result = await executor.execute(sample_agent_spec, input_data)
            assert result.output == "Mock output for context test"

        # After exiting context, cleanup should have been called
        assert mock_executor._cleanup_done is True
        assert mock_executor._initialized is False

    @pytest.mark.asyncio
    async def test_async_context_manager_with_exception(
        self, mock_executor, sample_agent_spec
    ):
        """Test cleanup happens even when exception is raised."""
        input_data = {"query": "error test"}

        # Make execute raise an exception
        async def failing_execute(*args, **kwargs):
            raise ValueError("Intentional test error")

        mock_executor._execute_agent = failing_execute

        try:
            async with mock_executor as executor:
                # This should raise ValueError
                await executor.execute(sample_agent_spec, input_data)
        except ValueError:
            pass  # Expected

        # Cleanup should still have been called
        assert mock_executor._cleanup_done is True
        assert mock_executor._initialized is False


class TestAgentExecutionResult:
    """Test AgentExecutionResult dataclass."""

    def test_result_creation(self):
        """Test creating execution result."""
        result = AgentExecutionResult(
            output="Test output", duration=1.5, tokens_used=50, cost=0.001
        )

        assert result.output == "Test output"
        assert result.duration == 1.5
        assert result.tokens_used == 50
        assert result.cost == 0.001
        assert result.error is None
        assert result.metadata == {}

    def test_result_with_error(self):
        """Test result with error."""
        result = AgentExecutionResult(
            output=None, duration=0.1, error="Execution failed"
        )

        assert result.output is None
        assert result.error == "Execution failed"
        assert result.tokens_used is None
        assert result.cost is None

    def test_result_with_metadata(self):
        """Test result with metadata."""
        metadata = {"model": "GPT-4o", "prompt_tokens": 30, "completion_tokens": 20}

        result = AgentExecutionResult(
            output="Response", duration=2.0, metadata=metadata
        )

        assert result.metadata["model"] == "GPT-4o"
        assert result.metadata["prompt_tokens"] == 30
