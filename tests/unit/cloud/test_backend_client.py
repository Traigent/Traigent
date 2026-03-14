"""Tests for Traigent Cloud Backend Client."""

import time
from datetime import UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import traigent.cloud.backend_client
from traigent.cloud.backend_client import (
    BackendClientConfig,
    BackendIntegratedClient,
    CloudServiceError,
    get_backend_client,
)
from traigent.cloud.models import (
    AgentSpecification,
    OptimizationSessionStatus,
    SessionCreationRequest,
    TrialResultSubmission,
    TrialStatus,
)
from traigent.config.backend_config import BackendConfig
from traigent.evaluators.base import Dataset, EvaluationExample


@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing."""
    examples = [
        EvaluationExample(
            input_data={"query": "What is AI?"},
            expected_output="AI is artificial intelligence.",
        ),
        EvaluationExample(
            input_data={"query": "What is ML?"},
            expected_output="ML is machine learning.",
        ),
        EvaluationExample(
            input_data={"query": "What is DL?"}, expected_output="DL is deep learning."
        ),
    ]
    return Dataset(examples=examples, name="test_dataset")


@pytest.fixture
def agent_specification():
    """Create agent specification for testing."""
    return AgentSpecification(
        id="agent_123",
        name="Test Agent",
        agent_type="conversational",
        agent_platform="openai",
        prompt_template="Test prompt: {input}",
        model_parameters={"temperature": 0.7, "max_tokens": 150},
    )


@pytest.fixture
def backend_config():
    """Create backend configuration for testing."""
    return BackendClientConfig(
        backend_base_url="http://localhost:5000",
        use_mcp=True,
        mcp_server_path="/tmp/test_server",
        enable_session_sync=True,
        session_sync_interval=5.0,
    )


@pytest.fixture
def backend_client(backend_config):
    """Create backend integrated client for testing."""
    return BackendIntegratedClient(
        api_key="tg_test_" + "x" * 56,  # pragma: allowlist secret
        base_url="http://localhost:8000",
        backend_config=backend_config,
        enable_fallback=True,
        max_retries=3,
        timeout=30.0,
    )


class TestBackendClientConfig:
    """Test backend client configuration."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = BackendClientConfig()

        from traigent.config.backend_config import BackendConfig

        assert config.backend_base_url == BackendConfig.get_backend_url().rstrip("/")
        assert config.use_mcp is False
        assert config.mcp_server_path is None
        assert config.enable_session_sync is True
        assert config.session_sync_interval == 5.0

    def test_custom_config(self):
        """Test custom configuration creation."""
        config = BackendClientConfig(
            backend_base_url="https://backend.example.com",
            use_mcp=True,
            mcp_server_path="/custom/path",
            enable_session_sync=False,
            session_sync_interval=10.0,
        )

        assert config.backend_base_url == "https://backend.example.com"
        assert config.use_mcp is True
        assert config.mcp_server_path == "/custom/path"
        assert config.enable_session_sync is False
        assert config.session_sync_interval == 10.0


class TestBackendIntegratedClient:
    """Test backend integrated client functionality."""

    def test_client_initialization(self, backend_config):
        """Test client initialization."""
        client = BackendIntegratedClient(
            api_key="test_key",  # pragma: allowlist secret
            base_url="https://api.test.com",
            backend_config=backend_config,
            enable_fallback=False,
            max_retries=5,
            timeout=60.0,
        )

        assert client.base_url == "https://api.test.com"
        assert client.backend_config == backend_config
        assert client.enable_fallback is False
        assert client.max_retries == 5
        assert client.timeout == 60.0
        assert client._session is None
        assert len(client._active_sessions) == 0

    def test_client_default_initialization(self, monkeypatch):
        """Test client with default configuration."""
        # Ensure consistent environment for test
        monkeypatch.delenv("TRAIGENT_BACKEND_URL", raising=False)
        monkeypatch.delenv("TRAIGENT_API_URL", raising=False)
        monkeypatch.setenv("TRAIGENT_ENV", "production")

        with patch(
            "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
            return_value=None,
        ):
            client = BackendIntegratedClient()

        assert client.base_url == BackendConfig.get_backend_url()
        assert isinstance(client.backend_config, BackendClientConfig)
        assert client.backend_config.api_base_url == BackendConfig.get_backend_api_url()
        assert client.enable_fallback is True
        assert client.max_retries == 3
        assert client.timeout == 30.0

    def test_client_url_normalization(self):
        """Test URL normalization on initialization."""
        client = BackendIntegratedClient(base_url="https://api.test.com/")
        assert client.base_url == "https://api.test.com"

    def test_explicit_base_url_overrides_default_backend_config(self, monkeypatch):
        """Explicit base_url should propagate through backend and API URLs."""
        monkeypatch.delenv("TRAIGENT_BACKEND_URL", raising=False)
        monkeypatch.delenv("TRAIGENT_API_URL", raising=False)

        with patch(
            "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
            return_value=None,
        ):
            client = BackendIntegratedClient(base_url="https://api.test.com")

        assert client.base_url == "https://api.test.com"
        assert client.backend_config.backend_base_url == "https://api.test.com"
        assert client.backend_config.api_base_url == "https://api.test.com/api/v1"

    @patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", False)
    def test_client_without_aiohttp(self, backend_config):
        """Test client initialization without aiohttp."""
        with patch("traigent.cloud.backend_client.logger") as mock_logger:
            client = BackendIntegratedClient(backend_config=backend_config)
            # Check that the aiohttp warning was logged (among possibly other warnings)
            warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
            assert any("aiohttp not available" in msg for msg in warning_calls), (
                f"Expected 'aiohttp not available' warning, but got: {warning_calls}"
            )
            assert client is not None

    def test_async_context_manager(self, backend_client):
        """Test async context manager functionality."""

        async def run_test():
            async with backend_client as client:
                assert client is backend_client
                # Session initialization is mocked, so we can't test actual session

            # Session should be None after exit (if it was created)
            assert backend_client._session is None

        import asyncio

        asyncio.run(run_test())

    @patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", False)
    def test_async_context_manager_no_aiohttp(self, backend_client):
        """Test async context manager without aiohttp."""

        async def run_test():
            async with backend_client as client:
                assert client is backend_client
                assert client._session is None

        import asyncio

        asyncio.run(run_test())

    @patch("requests.post")
    def test_upload_example_features_posts_to_analytics_endpoint(
        self, mock_post, backend_client
    ):
        """Example features upload uses the analytics feature endpoint."""
        mock_response = MagicMock(status_code=200, text="ok")
        mock_post.return_value = mock_response
        with patch.object(
            backend_client.auth_manager.auth,
            "get_headers",
            AsyncMock(return_value={"Authorization": "Bearer test-token"}),
        ) as mock_get_headers:
            result = backend_client.upload_example_features(
                "run_123",
                "simhash_v1",
                [{"example_id": "ex_1", "feature": "0f0f"}],
            )

        assert result is True
        mock_get_headers.assert_awaited_once_with(target="backend")
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.args[0].endswith(
            "/api/v1/analytics/example-scoring/run_123/features"
        )
        assert call_args.kwargs["json"]["feature_kind"] == "simhash_v1"
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer test-token"

    @patch("requests.post")
    def test_upload_example_features_rejects_unsupported_feature_kind(
        self, mock_post, backend_client
    ):
        """Example feature upload only accepts supported feature kinds."""
        result = backend_client.upload_example_features(
            "run_123",
            "unknown_v1",
            [{"example_id": "ex_1", "feature": "0f0f"}],
        )

        assert result is False
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_upload_example_features_url_encodes_run_id(
        self, mock_post, backend_client
    ):
        """Run IDs are path-encoded before constructing the upload URL."""
        mock_response = MagicMock(status_code=200, text="ok")
        mock_post.return_value = mock_response
        with patch.object(
            backend_client.auth_manager.auth,
            "get_headers",
            AsyncMock(return_value={"Authorization": "Bearer test-token"}),
        ):
            result = backend_client.upload_example_features(
                "run/../unsafe",
                "simhash_v1",
                [{"example_id": "ex_1", "feature": "0f0f"}],
            )

        assert result is True
        assert (
            mock_post.call_args.args[0]
            == (
                f"{backend_client.api_base_url}/analytics/example-scoring/"
                "run%2F..%2Funsafe/features"
            )
        )


class TestPrivacyFirstOptimization:
    """Test privacy-first optimization functionality."""

    def test_create_privacy_optimization_session(self, backend_client):
        """Test creating privacy-first optimization session."""

        async def run_test():
            with (
                patch.object(backend_client, "_create_cloud_session") as mock_cloud,
                patch.object(
                    backend_client, "_create_traigent_session_via_api"
                ) as mock_session_api,
                patch("traigent.cloud.backend_client.bridge") as mock_bridge,
            ):
                # Mock responses
                mock_cloud.return_value = MagicMock(
                    session_id="session_123",
                    optimization_strategy={"algorithm": "bayesian"},
                    metadata={"created_at": time.time()},
                )
                # Mock the session API to return the expected session_id
                mock_session_api.return_value = ("session_123", "exp_456", "run_789")
                mock_bridge.create_session_mapping.return_value = MagicMock()

                (
                    session_id,
                    exp_id,
                    run_id,
                ) = await backend_client._deprecated_create_privacy_optimization_session(
                    function_name="test_function",
                    configuration_space={"param": [1, 2, 3]},
                    objectives=["accuracy"],
                    dataset_metadata={"size": 1000, "type": "qa"},
                    max_trials=25,
                    user_id="test_user",
                )

                assert session_id == "session_123"
                assert exp_id == "exp_456"
                assert run_id == "run_789"

                # Verify session creation call
                mock_cloud.assert_called_once()
                session_request = mock_cloud.call_args[0][0]
                assert session_request.function_name == "test_function"
                assert session_request.billing_tier == "privacy"

                # Verify session API call
                mock_session_api.assert_called_once()

                # Verify session mapping creation
                mock_bridge.create_session_mapping.assert_called_once()

                # Verify active session storage
                assert session_id in backend_client._active_sessions
                session = backend_client._active_sessions[session_id]
                assert session.function_name == "test_function"
                assert session.status == OptimizationSessionStatus.ACTIVE

        import asyncio

        asyncio.run(run_test())

    def test_create_privacy_session_error_handling(self, backend_client):
        """Test error handling in privacy session creation."""

        async def run_test():
            with patch.object(
                backend_client,
                "_create_cloud_session",
                side_effect=Exception("Network error"),
            ):
                with pytest.raises(CloudServiceError, match="Failed to create session"):
                    await (
                        backend_client._deprecated_create_privacy_optimization_session(
                            function_name="test_function",
                            configuration_space={"param": [1, 2, 3]},
                            objectives=["accuracy"],
                            dataset_metadata={"size": 1000},
                        )
                    )

        import asyncio

        asyncio.run(run_test())

    def test_get_next_privacy_trial(self, backend_client):
        """Test getting next trial for privacy optimization."""

        async def run_test():
            session_id = "session_123"

            with (
                patch.object(
                    backend_client, "_get_cloud_trial_suggestion"
                ) as mock_cloud,
                patch("traigent.cloud.backend_client.bridge") as mock_bridge,
            ):
                # Setup mock response
                from traigent.cloud.models import (
                    DatasetSubsetIndices,
                    NextTrialResponse,
                    TrialSuggestion,
                )

                mock_suggestion = TrialSuggestion(
                    trial_id="trial_123",
                    session_id=session_id,
                    trial_number=1,
                    config={"param": 2},
                    dataset_subset=DatasetSubsetIndices(
                        indices=[0, 1, 2],
                        selection_strategy="random",
                        confidence_level=0.8,
                        estimated_representativeness=0.9,
                    ),
                    exploration_type="exploration",
                )

                mock_cloud.return_value = NextTrialResponse(
                    suggestion=mock_suggestion, should_continue=True
                )

                mock_bridge.get_session_mapping.return_value = MagicMock(
                    experiment_run_id="run_789"
                )
                mock_bridge.trial_suggestion_to_config_run.return_value = MagicMock(
                    config_run_id="config_123"
                )

                # Add active session
                backend_client._active_sessions[session_id] = MagicMock(
                    completed_trials=0, updated_at=time.time()
                )

                suggestion = await backend_client.get_next_privacy_trial(session_id)

                assert suggestion is not None
                assert suggestion.trial_id == "trial_123"
                assert suggestion.config == {"param": 2}

                # Verify cloud call
                mock_cloud.assert_called_once()

                # Verify trial mapping (no backend config run creation anymore)
                mock_bridge.add_trial_mapping.assert_called_once()

                # Verify session update
                assert backend_client._active_sessions[session_id].completed_trials == 1

        import asyncio

        asyncio.run(run_test())

    def test_get_next_privacy_trial_no_session_mapping(self, backend_client):
        """Test getting next trial without session mapping."""

        async def run_test():
            session_id = "session_123"

            with (
                patch.object(
                    backend_client, "_get_cloud_trial_suggestion"
                ) as mock_cloud,
                patch("traigent.cloud.backend_client.bridge") as mock_bridge,
            ):
                from traigent.cloud.models import NextTrialResponse

                mock_cloud.return_value = NextTrialResponse(
                    suggestion=MagicMock(trial_id="trial_123"), should_continue=True
                )
                mock_bridge.get_session_mapping.return_value = None

                suggestion = await backend_client.get_next_privacy_trial(session_id)

                assert suggestion is not None
                # Should not create backend config run without mapping
                mock_bridge.add_trial_mapping.assert_not_called()

        import asyncio

        asyncio.run(run_test())

    def test_get_next_privacy_trial_error_handling(self, backend_client):
        """Test error handling in getting next trial."""

        async def run_test():
            with patch.object(
                backend_client,
                "_get_cloud_trial_suggestion",
                side_effect=Exception("Network error"),
            ):
                suggestion = await backend_client.get_next_privacy_trial("session_123")
                assert suggestion is None

        import asyncio

        asyncio.run(run_test())

    def test_submit_privacy_trial_results(self, backend_client):
        """Test submitting privacy trial results."""

        async def run_test():
            session_id = "session_123"
            trial_id = "trial_456"

            with (
                patch.object(
                    backend_client,
                    "_submit_cloud_trial_results",
                    new_callable=AsyncMock,
                ) as mock_cloud,
                patch.object(
                    backend_client,
                    "_submit_trial_result_via_session",
                    new_callable=AsyncMock,
                ) as mock_session,
                patch("traigent.cloud.backend_client.bridge") as mock_bridge,
            ):
                mock_cloud.return_value = True  # Mock cloud submission success
                mock_session.return_value = True  # Mock session submission success
                mock_bridge.get_trial_mapping.return_value = "config_789"

                success = await backend_client.submit_privacy_trial_results(
                    session_id=session_id,
                    trial_id=trial_id,
                    config={"param1": 0.5},  # Add required config parameter
                    metrics={"accuracy": 0.85, "latency": 1.2},
                    duration=30.5,
                )

                assert success is True

                # Verify cloud submission
                mock_cloud.assert_called_once()
                submission = mock_cloud.call_args[0][0]
                assert submission.session_id == session_id
                assert submission.trial_id == trial_id
                assert submission.metrics == {"accuracy": 0.85, "latency": 1.2}
                assert submission.duration == 30.5
                assert submission.status == TrialStatus.COMPLETED

                # Verify session submission
                mock_session.assert_called_once()
                args = mock_session.call_args[0]
                assert args[0] == session_id
                assert args[1] == trial_id
                assert args[2] == {"param1": 0.5}  # config
                assert args[3] == {"accuracy": 0.85, "latency": 1.2}  # metrics
                assert args[4] == "COMPLETED"  # status
                assert args[5] is None  # error_message

        import asyncio

        asyncio.run(run_test())

    def test_submit_privacy_trial_results_with_error(self, backend_client):
        """Test submitting privacy trial results with error."""

        async def run_test():
            with (
                patch.object(
                    backend_client,
                    "_submit_cloud_trial_results",
                    new_callable=AsyncMock,
                ) as mock_cloud,
                patch.object(
                    backend_client,
                    "_submit_trial_result_via_session",
                    new_callable=AsyncMock,
                ) as mock_session,
                patch("traigent.cloud.backend_client.bridge") as mock_bridge,
            ):
                mock_cloud.return_value = True  # Mock cloud submission success
                mock_session.return_value = True  # Mock session submission success
                mock_bridge.get_trial_mapping.return_value = "config_789"

                success = await backend_client.submit_privacy_trial_results(
                    session_id="session_123",
                    trial_id="trial_456",
                    config={"param1": 0.5},  # Add required config parameter
                    metrics={"accuracy": 0.0},
                    duration=5.0,
                    error_message="Execution failed",
                )

                assert success is True

                # Verify error status
                submission = mock_cloud.call_args[0][0]
                assert submission.status == TrialStatus.FAILED
                assert submission.error_message == "Execution failed"

                # Verify session submission with error
                session_args = mock_session.call_args[0]
                assert session_args[4] == "FAILED"  # status
                assert session_args[5] == "Execution failed"  # error_message

        import asyncio

        asyncio.run(run_test())

    def test_submit_privacy_trial_results_error_handling(self, backend_client):
        """Test error handling in submitting trial results."""

        async def run_test():
            with patch.object(
                backend_client,
                "_submit_cloud_trial_results",
                side_effect=Exception("Network error"),
            ):
                success = await backend_client.submit_privacy_trial_results(
                    session_id="session_123",
                    trial_id="trial_456",
                    config={"param1": 0.5},  # Add required config parameter
                    metrics={"accuracy": 0.85},
                    duration=30.0,
                )
                assert success is False

        import asyncio

        asyncio.run(run_test())


class TestCloudSaaSOptimization:
    """Test cloud SaaS optimization functionality."""

    def test_start_agent_optimization(
        self, backend_client, agent_specification, sample_dataset
    ):
        """Test starting agent optimization."""

        async def run_test():
            with (
                patch.object(
                    backend_client, "_create_backend_agent_experiment"
                ) as mock_backend,
                patch.object(
                    backend_client, "_submit_agent_optimization"
                ) as mock_cloud,
                patch("traigent.cloud.backend_client.bridge") as mock_bridge,
            ):
                # Mock responses
                mock_backend.return_value = ("exp_123", "run_456")
                mock_cloud.return_value = MagicMock(
                    session_id="agent_session_789",
                    optimization_id="opt_123",
                    status="started",
                )

                response = await backend_client.start_agent_optimization(
                    agent_spec=agent_specification,
                    dataset=sample_dataset,
                    configuration_space={"temperature": [0.3, 0.7, 1.0]},
                    objectives=["accuracy", "cost"],
                    max_trials=30,
                    user_id="test_user",
                )

                assert response.session_id == "agent_session_789"
                assert response.optimization_id == "opt_123"
                assert response.status == "started"

                # Verify backend experiment creation
                mock_backend.assert_called_once()
                backend_args = mock_backend.call_args[0]
                assert backend_args[0] == agent_specification
                assert backend_args[1] == sample_dataset

                # Verify cloud submission
                mock_cloud.assert_called_once()
                cloud_request = mock_cloud.call_args[0][0]
                assert cloud_request.agent_spec == agent_specification
                assert cloud_request.billing_tier == "cloud"

                # Verify session mapping
                mock_bridge.create_session_mapping.assert_called_once()

        import asyncio

        asyncio.run(run_test())

    def test_start_agent_optimization_error_handling(
        self, backend_client, agent_specification, sample_dataset
    ):
        """Test error handling in agent optimization."""

        async def run_test():
            with patch.object(
                backend_client,
                "_create_backend_agent_experiment",
                side_effect=Exception("Backend error"),
            ):
                with pytest.raises(
                    CloudServiceError, match="Failed to start optimization"
                ):
                    await backend_client.start_agent_optimization(
                        agent_spec=agent_specification,
                        dataset=sample_dataset,
                        configuration_space={"temperature": [0.7]},
                        objectives=["accuracy"],
                    )

        import asyncio

        asyncio.run(run_test())

    def test_execute_agent(self, backend_client, agent_specification):
        """Test agent execution."""

        async def run_test():
            with patch.object(backend_client, "_execute_cloud_agent") as mock_execute:
                mock_execute.return_value = MagicMock(
                    output="Test response", duration=2.5, tokens_used=75, cost=0.002
                )

                response = await backend_client.execute_agent(
                    agent_spec=agent_specification,
                    input_data={"query": "What is AI?"},
                    config_overrides={"temperature": 0.8},
                )

                assert response.output == "Test response"
                assert response.duration == 2.5
                assert response.tokens_used == 75
                assert response.cost == 0.002

                # Verify execution request
                mock_execute.assert_called_once()
                execution_request = mock_execute.call_args[0][0]
                assert execution_request.agent_spec == agent_specification
                assert execution_request.input_data == {"query": "What is AI?"}
                assert execution_request.config_overrides == {"temperature": 0.8}

        import asyncio

        asyncio.run(run_test())

    def test_execute_agent_error_handling(self, backend_client, agent_specification):
        """Test error handling in agent execution."""

        async def run_test():
            with patch.object(
                backend_client,
                "_execute_cloud_agent",
                side_effect=Exception("Execution error"),
            ):
                with pytest.raises(CloudServiceError, match="Failed to execute agent"):
                    await backend_client.execute_agent(
                        agent_spec=agent_specification, input_data={"query": "test"}
                    )

        import asyncio

        asyncio.run(run_test())


class TestBackendAPIIntegration:
    """Test backend API integration methods."""

    def test_create_backend_experiment_tracking(self, backend_client):
        """Test that backend experiment tracking creation is deprecated."""

        async def run_test():
            session_request = SessionCreationRequest(
                function_name="test_func",
                configuration_space={"param": [1, 2, 3]},
                objectives=["accuracy"],
                dataset_metadata={"size": 1000},
                user_id="test_user",
            )

            # Should raise NotImplementedError since this method is deprecated
            with pytest.raises(
                NotImplementedError, match="SDK must use session endpoints only"
            ):
                await backend_client._create_backend_experiment_tracking(
                    session_request
                )

        import asyncio

        asyncio.run(run_test())

    def test_create_backend_agent_experiment(
        self, backend_client, agent_specification, sample_dataset
    ):
        """Test that backend agent experiment creation is deprecated."""

        async def run_test():
            # Should raise NotImplementedError since this method is deprecated
            with patch("traigent.cloud.backend_client.bridge") as mock_bridge:
                with pytest.raises(
                    NotImplementedError, match="SDK must use session endpoints only"
                ):
                    await backend_client._create_backend_agent_experiment(
                        agent_spec=agent_specification,
                        dataset=sample_dataset,
                        configuration_space={"temperature": [0.7]},
                        objectives=["accuracy"],
                        max_trials=25,
                    )

                    # Verify agent data override
                    mock_bridge.agent_specification_to_backend.assert_called_once_with(
                        agent_specification
                    )

        import asyncio

        asyncio.run(run_test())

    def test_backend_api_placeholders(self, backend_client):
        """Test that backend API placeholder methods are deprecated."""

        async def run_test():
            # All these methods should raise NotImplementedError since they're deprecated

            # Test experiment creation
            with pytest.raises(
                NotImplementedError, match="SDK must use session endpoints only"
            ):
                await backend_client._create_backend_experiment_via_api(
                    MagicMock(name="Test Experiment", experiment_id="exp_123")
                )

            # Test experiment run creation
            session_request = SessionCreationRequest(
                function_name="test_func",
                configuration_space={},
                objectives=["accuracy"],
                dataset_metadata={},
            )
            with pytest.raises(
                NotImplementedError, match="SDK must use session endpoints only"
            ):
                await backend_client._create_backend_experiment_run_via_api(
                    "exp_123", session_request
                )

            # Test config run creation
            with pytest.raises(
                NotImplementedError, match="SDK must use session endpoints only"
            ):
                await backend_client._create_backend_config_run(
                    MagicMock(config_run_id="config_123")
                )

            # Test config run update
            with pytest.raises(
                NotImplementedError, match="SDK must use session endpoints only"
            ):
                await backend_client._update_backend_config_run_results(
                    "config_123", {"accuracy": 0.8}, "completed", None
                )

        import asyncio

        asyncio.run(run_test())


class TestCloudServiceIntegration:
    """Test cloud service integration placeholder methods."""

    def test_cloud_service_placeholders(self, backend_client):
        """Test cloud service placeholder methods."""

        async def run_test():
            # Test session creation
            session_request = SessionCreationRequest(
                function_name="test_func",
                configuration_space={},
                objectives=["accuracy"],
                dataset_metadata={},
            )
            session_response = await backend_client._create_cloud_session(
                session_request
            )
            assert "session_" in session_response.session_id
            assert session_response.status == OptimizationSessionStatus.CREATED

            # Test trial suggestion
            from traigent.cloud.models import NextTrialRequest

            trial_request = NextTrialRequest(session_id="session_123")
            trial_response = await backend_client._get_cloud_trial_suggestion(
                trial_request
            )
            assert trial_response.suggestion is not None
            assert trial_response.should_continue is True

            # Test trial results submission
            trial_submission = TrialResultSubmission(
                session_id="session_123",
                trial_id="trial_456",
                metrics={"accuracy": 0.8},
                duration=30.0,
                status=TrialStatus.COMPLETED,
            )
            await backend_client._submit_cloud_trial_results(trial_submission)

            # Test agent optimization submission
            from traigent.cloud.models import AgentOptimizationRequest

            agent_request = AgentOptimizationRequest(
                agent_spec=MagicMock(),
                dataset=MagicMock(),
                configuration_space={},
                objectives=["accuracy"],
            )
            agent_response = await backend_client._submit_agent_optimization(
                agent_request
            )
            assert "agent_session_" in agent_response.session_id
            assert agent_response.status == "started"

            # Test agent execution
            from traigent.cloud.models import AgentExecutionRequest

            exec_request = AgentExecutionRequest(
                agent_spec=MagicMock(), input_data={"query": "test"}
            )
            exec_response = await backend_client._execute_cloud_agent(exec_request)
            assert exec_response.output == "Mock agent response"
            assert exec_response.duration == 1.5

        import asyncio

        asyncio.run(run_test())


class TestUtilityMethods:
    """Test utility methods."""

    def test_get_active_sessions(self, backend_client):
        """Test getting active sessions."""
        # Add a test session
        from traigent.cloud.models import OptimizationSession

        test_session = OptimizationSession(
            session_id="session_123",
            function_name="test_func",
            configuration_space={},
            objectives=["accuracy"],
            max_trials=25,
            status=OptimizationSessionStatus.ACTIVE,
            created_at=time.time(),
            updated_at=time.time(),
        )
        backend_client._active_sessions["session_123"] = test_session

        active_sessions = backend_client.get_active_sessions()
        assert "session_123" in active_sessions
        assert active_sessions["session_123"] == test_session

        # Ensure it's a copy
        assert active_sessions is not backend_client._active_sessions

    def test_get_session_mapping(self, backend_client):
        """Test getting session mapping."""
        with patch("traigent.cloud.backend_client.bridge") as mock_bridge:
            mock_mapping = MagicMock()
            mock_bridge.get_session_mapping.return_value = mock_mapping

            mapping = backend_client.get_session_mapping("session_123")
            assert mapping == mock_mapping
            mock_bridge.get_session_mapping.assert_called_once_with("session_123")

    def test_finalize_session(self, backend_client):
        """Test session finalization."""

        async def run_test():
            # Add active session
            backend_client._active_sessions["session_123"] = MagicMock()

            response = await backend_client.finalize_session(
                "session_123", include_full_history=True
            )

            assert response.session_id == "session_123"
            assert "session_123" not in backend_client._active_sessions

        import asyncio

        asyncio.run(run_test())

    def test_submit_result_updates_session(self, backend_client):
        """submit_result should increment trial count and refresh timestamps."""

        from datetime import datetime

        from traigent.cloud.models import OptimizationSession

        session_id = "session_submit"
        created_at = datetime.now(UTC)

        backend_client._active_sessions[session_id] = OptimizationSession(
            session_id=session_id,
            function_name="test_func",
            configuration_space={"param": [1, 2]},
            objectives=["accuracy"],
            max_trials=10,
            status=OptimizationSessionStatus.ACTIVE,
            created_at=created_at,
            updated_at=created_at,
        )

        backend_client.submit_result(
            session_id=session_id,
            config={"param": 1},
            score=0.9,
            metadata={"source": "unit-test"},
        )

        session = backend_client._active_sessions[session_id]
        assert session.completed_trials == 1
        assert session.updated_at is not None

    def test_submit_result_local_storage_fallback(self, backend_client):
        """submit_result should persist results to local storage when enabled."""

        from datetime import datetime

        from traigent.cloud.models import OptimizationSession

        session_id = "session_fallback"
        backend_client.enable_fallback = True
        backend_client.local_storage = MagicMock()

        backend_client._active_sessions[session_id] = OptimizationSession(
            session_id=session_id,
            function_name="test_func",
            configuration_space={"param": [1, 2]},
            objectives=["accuracy"],
            max_trials=5,
            status=OptimizationSessionStatus.ACTIVE,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        backend_client.submit_result(
            session_id=session_id,
            config={"param": 2},
            score=0.75,
            metadata={"source": "local"},
        )

        session = backend_client._active_sessions[session_id]
        assert session.completed_trials == 1
        backend_client.local_storage.add_trial_result.assert_called_once_with(
            session_id=session_id,
            config={"param": 2},
            score=0.75,
            metadata={"source": "local"},
        )


class TestGlobalClientInstance:
    """Test global client instance management."""

    def test_get_backend_client_creates_instance(self):
        """Test that get_backend_client creates a new instance."""
        # Clear any existing global instance

        traigent.cloud.backend_client._backend_client = None

        client = get_backend_client(api_key="test_key")
        assert isinstance(client, BackendIntegratedClient)

        # Second call should return same instance
        client2 = get_backend_client(api_key="different_key")  # Should be ignored
        assert client is client2

    def test_get_backend_client_with_kwargs(self):
        """Explicit base_url should override default backend routing."""
        # Clear global instance

        traigent.cloud.backend_client._backend_client = None

        backend_config = BackendClientConfig(
            backend_base_url="https://custom.backend.com"
        )
        client = get_backend_client(
            api_key="test_key",  # pragma: allowlist secret
            base_url="https://custom.api.com",
            backend_config=backend_config,
            enable_fallback=False,
        )

        assert client.base_url == "https://custom.api.com"
        assert client.backend_config.backend_base_url == "https://custom.api.com"
        assert client.backend_config.api_base_url == "https://custom.api.com/api/v1"
        assert client.enable_fallback is False


class TestCloudServiceErrorHandling:
    """Test CloudServiceError exception."""

    def test_cloud_service_error_creation(self):
        """Test CloudServiceError creation."""
        error = CloudServiceError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_cloud_service_error_inheritance(self):
        """Test CloudServiceError inheritance."""
        error = CloudServiceError("Test error")
        assert isinstance(error, Exception)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_session_operations_without_active_session(self, backend_client):
        """Test operations without active sessions."""

        async def run_test():
            # Test get_next_privacy_trial without active session
            with patch.object(
                backend_client, "_get_cloud_trial_suggestion"
            ) as mock_cloud:
                from traigent.cloud.models import NextTrialResponse

                mock_cloud.return_value = NextTrialResponse(
                    suggestion=MagicMock(trial_id="trial_123"), should_continue=True
                )

                # Should work but not update session state
                suggestion = await backend_client.get_next_privacy_trial(
                    "nonexistent_session"
                )
                assert suggestion is not None

        import asyncio

        asyncio.run(run_test())

    def test_trial_mapping_without_session_mapping(self, backend_client):
        """Test trial operations without session mapping."""

        async def run_test():
            with (
                patch("traigent.cloud.backend_client.bridge") as mock_bridge,
                patch.object(
                    backend_client,
                    "_submit_cloud_trial_results",
                    new_callable=AsyncMock,
                ) as mock_cloud,
                patch.object(
                    backend_client,
                    "_submit_trial_result_via_session",
                    new_callable=AsyncMock,
                ) as mock_session,
            ):
                mock_bridge.get_trial_mapping.return_value = None
                mock_cloud.return_value = True
                mock_session.return_value = True

                # Should succeed but not update backend
                success = await backend_client.submit_privacy_trial_results(
                    session_id="session_123",
                    trial_id="trial_456",
                    config={"temperature": 0.7, "max_tokens": 150},
                    metrics={"accuracy": 0.8},
                    duration=30.0,
                )
                assert success is True

        import asyncio

        asyncio.run(run_test())

    def test_empty_dataset_handling(self, backend_client, agent_specification):
        """Test handling of empty datasets."""

        async def run_test():
            empty_dataset = Dataset(examples=[], name="empty")

            with (
                patch.object(
                    backend_client, "_create_backend_agent_experiment"
                ) as mock_backend,
                patch.object(
                    backend_client, "_submit_agent_optimization"
                ) as mock_cloud,
            ):
                mock_backend.return_value = ("exp_123", "run_456")
                mock_cloud.return_value = MagicMock(session_id="session_123")

                response = await backend_client.start_agent_optimization(
                    agent_spec=agent_specification,
                    dataset=empty_dataset,
                    configuration_space={"temperature": [0.7]},
                    objectives=["accuracy"],
                )

                assert response.session_id == "session_123"

                # Verify empty dataset was passed
                backend_args = mock_backend.call_args[0]
                assert len(backend_args[1].examples) == 0

        import asyncio

        asyncio.run(run_test())
