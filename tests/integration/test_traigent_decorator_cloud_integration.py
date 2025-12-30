#!/usr/bin/env python3
"""Tests demonstrating Traigent decorator with cloud modules integration.

This test suite validates that the @traigent.optimize decorator works correctly
with our cloud modules, including:
- Cloud client initialization
- Authentication handling
- Session management
- Both local and cloud execution modes
- Framework override integration
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.api.decorators import optimize
from traigent.cloud.auth import AuthManager
from traigent.cloud.client import CloudServiceError, TraigentCloudClient
from traigent.cloud.models import (
    DatasetSubsetIndices,
    NextTrialResponse,
    OptimizationFinalizationResponse,
    OptimizationSessionStatus,
    SessionCreationResponse,
    TrialSuggestion,
)
from traigent.config.context import set_config
from traigent.config.types import TraigentConfig
from traigent.evaluators.base import Dataset, EvaluationExample


class TestTraigentDecoratorCloudIntegration:
    """Test Traigent decorator integration with cloud modules."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear any existing configuration
        set_config(None)

        # Create test dataset
        self.test_dataset = Dataset(
            name="test_dataset",
            examples=[
                EvaluationExample(
                    input_data={"question": "What is 2+2?"}, expected_output="4"
                ),
                EvaluationExample(
                    input_data={"question": "What is 3+3?"}, expected_output="6"
                ),
            ],
        )

        # Configuration space for testing
        self.config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.1, 0.5, 0.9],
            "max_tokens": [100, 500],
        }

    @patch("openai.OpenAI")
    def test_basic_decorator_with_cloud_mode(self, mock_openai):
        """Test basic decorator functionality with cloud mode enabled."""

        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "The answer is 4"
        mock_client.chat.completions.create.return_value = mock_response

        @optimize(
            eval_dataset=self.test_dataset,
            objectives=["accuracy"],
            configuration_space=self.config_space,
            execution_mode="cloud",  # Enable cloud mode
        )
        def simple_llm_function(question: str) -> str:
            """Simple function that uses LLM."""
            # Simulate LLM call - framework override will inject parameters
            from openai import OpenAI

            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": question}],
                temperature=0.7,
            )
            return response.choices[0].message.content

        # Verify the function is wrapped properly
        assert hasattr(simple_llm_function, "optimize")
        assert hasattr(simple_llm_function, "get_best_config")
        assert hasattr(simple_llm_function, "set_config")
        assert hasattr(simple_llm_function, "reset_optimization")

        # Verify cloud mode flag is enabled
        assert simple_llm_function.execution_mode == "cloud"

        # Test regular function call works
        result = simple_llm_function("What is 2+2?")
        assert isinstance(result, str)
        assert "answer" in result.lower()

    @patch("traigent.cloud.client.aiohttp")
    @patch("traigent.cloud.auth.AuthManager.is_authenticated")
    @pytest.mark.asyncio
    async def test_decorator_with_cloud_optimization(self, mock_auth, mock_aiohttp):
        """Test decorator optimization using cloud service."""

        # Mock cloud client responses
        mock_auth.return_value = True

        # Mock aiohttp session
        mock_session = AsyncMock()
        mock_aiohttp.ClientSession.return_value = mock_session

        # Mock optimization response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "best_config": {"model": "gpt-4", "temperature": 0.1},
            "best_metrics": {"accuracy": 0.92},
            "trials_count": 10,
        }
        mock_session.post.return_value.__aenter__.return_value = mock_response

        @optimize(
            eval_dataset=self.test_dataset,
            objectives=["accuracy"],
            configuration_space=self.config_space,
            execution_mode="cloud",
        )
        def cloud_optimized_function(question: str) -> str:
            """Function optimized via cloud service."""
            return f"Answer to: {question}"

        # Test cloud optimization
        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
            result = await cloud_optimized_function.optimize()

            # Verify optimization results
            assert result is not None
            assert hasattr(result, "best_config")

            # Verify best config is applied
            best_config = cloud_optimized_function.get_best_config()
            assert best_config is not None

    @patch("traigent.cloud.auth.AuthManager.is_authenticated")
    @pytest.mark.asyncio
    async def test_decorator_with_cloud_fallback(self, mock_auth):
        """Test decorator fallback to local optimization when cloud fails."""

        # Mock authentication failure to trigger fallback
        mock_auth.return_value = False

        @optimize(
            eval_dataset=self.test_dataset,
            objectives=["accuracy"],
            configuration_space=self.config_space,
            execution_mode="cloud",
        )
        def fallback_function(question: str) -> str:
            """Function that falls back to local optimization."""
            return f"Local answer: {question}"

        # Test that function is properly configured for fallback
        # Note: Actual fallback would be tested in the optimization logic
        assert fallback_function.execution_mode == "cloud"

        # We can verify that the function can be called
        result = fallback_function("test question")
        assert "Local answer:" in result

    @pytest.mark.asyncio
    async def test_decorator_with_session_management(self):
        """Test decorator with cloud session management."""

        @optimize(
            eval_dataset=self.test_dataset,
            objectives=["accuracy"],
            configuration_space=self.config_space,
            execution_mode="cloud",
        )
        def session_managed_function(input_text: str) -> str:
            """Function with session management."""
            return f"Processed: {input_text}"

        # Mock cloud client
        with patch("traigent.cloud.client.TraigentCloudClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock session creation
            mock_client.create_optimization_session.return_value = (
                SessionCreationResponse(
                    session_id="test-session-123",
                    status=OptimizationSessionStatus.ACTIVE,
                    optimization_strategy={},
                    estimated_duration=120.0,
                    billing_estimate=0.05,
                    metadata={},
                )
            )

            # Mock trial suggestions
            mock_client.get_next_trial.return_value = NextTrialResponse(
                suggestion=TrialSuggestion(
                    trial_id="trial-1",
                    session_id="test-session-123",
                    trial_number=1,
                    config={"model": "gpt-4", "temperature": 0.1},
                    dataset_subset=DatasetSubsetIndices(
                        indices=[0, 1],
                        selection_strategy="random",
                        confidence_level=0.95,
                        estimated_representativeness=0.9,
                        metadata={},
                    ),
                    exploration_type="exploration",
                    priority=1,
                    estimated_duration=30.0,
                    metadata={},
                ),
                should_continue=True,
                reason="Continue optimization",
                session_status=OptimizationSessionStatus.ACTIVE,
                metadata={},
            )

            # Mock finalization
            mock_client.finalize_optimization.return_value = (
                OptimizationFinalizationResponse(
                    session_id="test-session-123",
                    best_config={"model": "gpt-4", "temperature": 0.1},
                    best_metrics={"accuracy": 0.95},
                    total_trials=5,
                    successful_trials=5,
                    total_duration=150.0,
                    cost_savings=0.3,
                    convergence_history=[],
                    full_history=None,
                    metadata={},
                )
            )

            # Test session-managed optimization concept
            # Note: Actual optimization would require real implementation
            # This test verifies the configuration is set up correctly
            assert session_managed_function.execution_mode == "cloud"

    def test_decorator_with_framework_override_and_cloud(self):
        """Test decorator with framework override and cloud integration."""

        @optimize(
            eval_dataset=self.test_dataset,
            objectives=["accuracy"],
            configuration_space=self.config_space,
            execution_mode="cloud",
            auto_override_frameworks=True,
            framework_targets=["openai.OpenAI", "langchain_openai.ChatOpenAI"],
        )
        def framework_override_function(question: str) -> str:
            """Function with framework override enabled."""
            # Just return a mock response for testing
            return f"Framework override response for: {question}"

        # Verify framework override is configured
        assert framework_override_function.auto_override_frameworks
        assert "openai.OpenAI" in framework_override_function.framework_targets
        assert (
            "langchain_openai.ChatOpenAI"
            in framework_override_function.framework_targets
        )

    @pytest.mark.asyncio
    async def test_decorator_authentication_flow(self):
        """Test decorator with authentication flow."""

        @optimize(
            eval_dataset=self.test_dataset,
            objectives=["accuracy"],
            configuration_space=self.config_space,
            execution_mode="cloud",
        )
        def authenticated_function(input_data: str) -> str:
            """Function requiring authentication."""
            return f"Authenticated processing: {input_data}"

        # Mock authentication manager
        with patch("traigent.cloud.auth.AuthManager") as mock_auth_class:
            mock_auth = AsyncMock()
            mock_auth_class.return_value = mock_auth

            # Test successful authentication
            mock_auth.is_authenticated.return_value = True
            mock_auth.get_headers.return_value = {"Authorization": "Bearer test-token"}

            # Verify authentication can be checked
            auth_manager = AuthManager("test-api-key")
            assert auth_manager is not None

    def test_decorator_with_private_mode(self):
        """Test decorator with privacy-first execution mode."""

        @optimize(
            eval_dataset=self.test_dataset,
            objectives=["accuracy"],
            configuration_space=self.config_space,
            execution_mode="privacy",  # Data never leaves local environment
        )
        def private_function(sensitive_data: str) -> str:
            """Function that processes sensitive data locally."""
            return f"Privately processed: {sensitive_data}"

        # Verify privacy mode configuration
        assert hasattr(private_function, "kwargs")
        # Commercial mode should be disabled for privacy
        assert private_function.execution_mode == "privacy"

    def test_decorator_mode_flags(self):
        """Execution mode controls whether commercial features are active."""

        @optimize(
            eval_dataset=self.test_dataset,
            objectives=["accuracy", "cost"],
            configuration_space=self.config_space,
            execution_mode="edge_analytics",
        )
        def local_function(request: str) -> str:
            return f"Local response: {request}"

        assert local_function.execution_mode == "edge_analytics"

        @optimize(
            eval_dataset=self.test_dataset,
            objectives=["accuracy", "cost"],
            configuration_space=self.config_space,
            execution_mode="standard",
            auto_override_frameworks=True,
        )
        def hybrid_commercial_function(request: str) -> str:
            return f"Hybrid commercial response: {request}"

        assert hybrid_commercial_function.execution_mode == "standard"
        assert hybrid_commercial_function.auto_override_frameworks

        @optimize(
            eval_dataset=self.test_dataset,
            objectives=["accuracy", "cost"],
            configuration_space=self.config_space,
            execution_mode="cloud",
        )
        def pure_cloud_function(request: str) -> str:
            return f"Cloud response: {request}"

        assert pure_cloud_function.execution_mode == "cloud"

    @pytest.mark.asyncio
    async def test_decorator_error_handling_with_cloud(self):
        """Test decorator error handling with cloud service errors."""

        @optimize(
            eval_dataset=self.test_dataset,
            objectives=["accuracy"],
            configuration_space=self.config_space,
            execution_mode="cloud",
        )
        def error_prone_function(input_val: str) -> str:
            """Function that may encounter cloud errors."""
            return f"Result: {input_val}"

        # Mock cloud service error
        with patch("traigent.cloud.client.TraigentCloudClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Simulate cloud service error
            mock_client.optimize_function.side_effect = CloudServiceError(
                "Service unavailable"
            )

            # Mock fallback optimization
            with patch(
                "traigent.optimizers.registry.get_optimizer"
            ) as mock_get_optimizer:
                mock_optimizer = AsyncMock()
                mock_optimizer.optimize.return_value = Mock(
                    best_config={"model": "gpt-3.5-turbo"},
                    best_metrics={"accuracy": 0.8},
                    trials=[],
                )
                mock_get_optimizer.return_value = mock_optimizer

                # Test that errors are handled gracefully
                result = await error_prone_function.optimize()
                assert result is not None  # Should fallback to local

    def test_decorator_configuration_validation(self):
        """Test decorator with various configuration validations."""

        # Test with minimal configuration
        @optimize(
            eval_dataset=self.test_dataset,
            objectives=["accuracy"],
            configuration_space={"model": ["gpt-3.5-turbo"]},  # Minimal config space
        )
        def minimal_config_function(data: str) -> str:
            return f"Minimal: {data}"

        assert minimal_config_function is not None

        # Test with comprehensive configuration
        @optimize(
            eval_dataset=self.test_dataset,
            objectives=["accuracy", "cost", "latency"],
            configuration_space=self.config_space,
            default_config={"model": "gpt-3.5-turbo", "temperature": 0.5},
            constraints=[
                lambda cfg: cfg["temperature"] < 0.8,
                lambda cfg, metrics: metrics.get("cost", 0) <= 0.10,
            ],
            execution_mode="cloud",
            auto_override_frameworks=True,
            framework_targets=["openai.OpenAI"],
            parallel_config={"trial_concurrency": 4},
        )
        def comprehensive_config_function(input_str: str) -> str:
            return f"Comprehensive: {input_str}"

        assert comprehensive_config_function is not None
        assert len(comprehensive_config_function.objectives) == 3
        assert comprehensive_config_function.execution_mode == "cloud"

    @pytest.mark.asyncio
    async def test_decorator_with_real_cloud_client_mock(self):
        """Test decorator with realistic cloud client mocking."""

        @optimize(
            eval_dataset=self.test_dataset,
            objectives=["accuracy"],
            configuration_space=self.config_space,
            execution_mode="cloud",
        )
        def realistic_cloud_function(query: str) -> str:
            """Function with realistic cloud integration."""
            # Mock LangChain usage for testing
            return f"LangChain response for: {query}"

        # Create realistic cloud client mock
        with patch("traigent.cloud.client.TraigentCloudClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Setup realistic cloud responses
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None

            mock_client.optimize_function.return_value = Mock(
                best_config={"model": "gpt-4", "temperature": 0.1, "max_tokens": 500},
                best_metrics={"accuracy": 0.94, "cost": 0.08, "latency": 1.2},
                trials_count=15,
                cost_reduction=0.67,
                optimization_time=180.0,
                subset_used=True,
                subset_size=8,
            )

            # Test realistic optimization
            async with mock_client:
                # This would trigger cloud optimization in real scenario
                pass

    def test_decorator_with_config_context_integration(self):
        """Test decorator integration with Traigent configuration context."""

        # Set global configuration
        global_config = TraigentConfig(
            model="gpt-4",
            temperature=0.3,
            max_tokens=1000,
            custom_params={
                "stream": True,
                "tools": [{"type": "function", "function": {"name": "calculator"}}],
            },
        )
        set_config(global_config)

        @optimize(
            eval_dataset=self.test_dataset,
            objectives=["accuracy"],
            configuration_space=self.config_space,
            execution_mode="cloud",
        )
        def context_integrated_function(problem: str) -> str:
            """Function that uses context configuration."""
            # Configuration should be automatically available
            from traigent.config.context import get_config

            current_config = get_config()
            assert current_config is not None
            assert current_config.model == "gpt-4"
            return f"Context result: {problem}"

        # Test that function works with context
        result = context_integrated_function("test problem")
        assert "Context result:" in result

        # Clean up
        set_config(None)


class TestCloudModulesIntegration:
    """Test individual cloud modules integration."""

    @pytest.mark.asyncio
    async def test_cloud_client_initialization(self):
        """Test TraigentCloudClient initialization."""

        # Test without aiohttp (fallback mode)
        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", False):
            client = TraigentCloudClient(api_key="test-key")
            assert client is not None
            assert client.enable_fallback

        # Test with aiohttp available
        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
            client = TraigentCloudClient(
                api_key="test-key",
                base_url="https://api.test.com",
                enable_fallback=True,
                max_retries=5,
                timeout=60.0,
            )
            assert client.base_url == "https://api.test.com"
            assert client.max_retries == 5
            assert client.timeout == 60.0

    @pytest.mark.asyncio
    async def test_auth_manager_integration(self):
        """Test AuthManager integration."""

        auth = AuthManager("test-api-key")
        assert auth is not None

        # Test that we can check authentication status
        is_auth = await auth.is_authenticated()
        assert isinstance(is_auth, bool)

        # Test basic functionality without requiring real authentication
        assert hasattr(auth, "get_headers")

    def test_cloud_models_creation(self):
        """Test cloud models can be created correctly."""

        from traigent.cloud.models import (
            OptimizationRequest,
            SessionCreationRequest,
            TrialResultSubmission,
            TrialStatus,
        )

        # Test basic models
        request = OptimizationRequest(
            function_name="test_func",
            dataset=Dataset(name="test", examples=[]),
            configuration_space={"model": ["gpt-3.5-turbo"]},
            objectives=["accuracy"],
        )
        assert request.function_name == "test_func"

        # Test session request
        session_req = SessionCreationRequest(
            function_name="test_session",
            configuration_space={"temp": [0.1, 0.5]},
            objectives=["accuracy"],
        )
        assert session_req.function_name == "test_session"

        # Test trial result
        trial_result = TrialResultSubmission(
            session_id="session-123",
            trial_id="trial-456",
            metrics={"accuracy": 0.92},
            duration=45.0,
            status=TrialStatus.COMPLETED,
        )
        assert trial_result.session_id == "session-123"
        assert trial_result.status == TrialStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_end_to_end_decorator_cloud_workflow(self):
        """Test complete end-to-end decorator with cloud workflow."""

        # Create a comprehensive example
        @optimize(
            eval_dataset=Dataset(
                name="math_qa",
                examples=[
                    EvaluationExample(
                        input_data={"question": "What is 5*7?"}, expected_output="35"
                    ),
                    EvaluationExample(
                        input_data={"question": "What is 12/3?"}, expected_output="4"
                    ),
                ],
            ),
            objectives=["accuracy", "cost"],
            configuration_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.0, 0.3, 0.7],
                "max_tokens": [50, 100, 200],
            },
            execution_mode="cloud",
            auto_override_frameworks=True,
            framework_targets=[
                "openai.OpenAI"
            ],  # Removed langchain_openai.ChatOpenAI due to Pydantic v2 compatibility issues
        )
        def math_qa_agent(question: str) -> str:
            """Mathematical Q&A agent with cloud optimization."""
            # This would use LangChain or OpenAI in practice
            return f"Math answer for: {question}"

        # Verify the decorator configuration
        assert math_qa_agent.execution_mode == "cloud"
        assert math_qa_agent.auto_override_frameworks
        assert "openai.OpenAI" in math_qa_agent.framework_targets
        assert len(math_qa_agent.objectives) == 2
        assert "accuracy" in math_qa_agent.objectives
        assert "cost" in math_qa_agent.objectives

        # Test function execution
        result = math_qa_agent("What is 2+2?")
        assert "Math answer for:" in result

        # Verify optimization interface exists
        assert hasattr(math_qa_agent, "optimize")
        assert hasattr(math_qa_agent, "get_best_config")
        assert hasattr(math_qa_agent, "set_config")
        assert hasattr(math_qa_agent, "reset_optimization")


if __name__ == "__main__":
    pytest.main([__file__])
