"""Unified test coverage for authentication headers in all HTTP methods.

This module ensures 100% coverage of all HTTP methods in both TraigentCloudClient
and BackendIntegratedClient, verifying authentication headers are always included.
Combines tests from multiple authentication test files into a single comprehensive suite.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud.backend_client import BackendClientConfig, BackendIntegratedClient
from traigent.cloud.client import CloudServiceError, TraigentCloudClient


@pytest.fixture
def valid_api_key():
    """Valid API key for testing."""
    return "tg_" + "a" * 61  # 64 chars total


@pytest.fixture
def mock_aiohttp_session():
    """Create a properly mocked aiohttp session."""
    with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
        with patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True):
            # Create mock session
            mock_session = Mock()

            # Create mock response
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "status": "active",
                    "optimization_id": "opt-123",
                    "result": "test-result",
                    "output": "test-output",
                    "duration": 1.5,
                    "tokens_used": 100,
                    "cost": 0.01,
                    "metadata": {},
                    "execution_id": "exec-123",
                    "session_id": "session-123",
                    "trial_id": "trial-123",
                    "progress": {"completed_trials": 5, "total_trials": 10},
                    "completed_trials": 5,
                    "total_trials": 10,
                    "current_best": {
                        "config": {"temp": 0.5},
                        "metrics": {"accuracy": 0.9},
                    },
                    "estimated_time_remaining": 300.0,
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T01:00:00Z",
                    "configuration": {"param": 0.5},
                    "should_continue": True,
                    "next_trial_available": True,
                    "best_config": {"param": 0.7},
                    "best_metrics": {"accuracy": 0.95},
                    "successful_trials": 8,
                    "total_duration": 120.5,
                    "cost_savings": 0.65,
                }
            )
            mock_response.text = AsyncMock(return_value="OK")

            # Create mock context manager
            mock_context = Mock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)

            # Setup session methods
            mock_session.post = Mock(return_value=mock_context)
            mock_session.get = Mock(return_value=mock_context)
            mock_session.put = Mock(return_value=mock_context)
            mock_session.delete = Mock(return_value=mock_context)
            mock_session.close = AsyncMock()

            # Mock ClientSession constructor
            with patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs_client:
                with patch(
                    "traigent.cloud.backend_client.aiohttp.ClientSession"
                ) as mock_cs_backend:
                    mock_cs_client.return_value = mock_session
                    mock_cs_backend.return_value = mock_session

                    # Mock ClientTimeout
                    with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                        with patch(
                            "traigent.cloud.backend_client.aiohttp.ClientTimeout"
                        ):
                            yield mock_session


class TestTraigentCloudClientCore:
    """Core tests for TraigentCloudClient HTTP methods."""

    @pytest.mark.asyncio
    async def test_ensure_session_creates_session_with_headers(
        self, valid_api_key, mock_aiohttp_session
    ):
        """Test _ensure_session creates session with authentication headers."""
        with patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs:
            mock_session = Mock()
            mock_cs.return_value = mock_session

            with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                client = TraigentCloudClient(api_key=valid_api_key)

                # Call _ensure_session directly
                await client._ensure_session()

                # Verify session was created with headers
                mock_cs.assert_called_once()
                call_kwargs = mock_cs.call_args[1]
                assert "headers" in call_kwargs
                headers = call_kwargs["headers"]
                assert "Authorization" in headers
                assert headers["Authorization"].startswith("Bearer ")
                assert "X-Traigent-Client" in headers
                assert "Content-Type" in headers

    @pytest.mark.asyncio
    async def test_headers_included_without_context_manager(self, valid_api_key):
        """Test that headers are included even when not using context manager."""
        # This is the unique test from test_authentication_headers_simple.py
        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
            with patch("traigent.cloud.client.aiohttp") as mock_aiohttp:
                # Create a properly configured mock session
                mock_session = Mock()
                mock_post_context = Mock()
                mock_response = Mock()
                mock_response.status = 201
                mock_response.json = AsyncMock(
                    return_value={
                        "session_id": "test",
                        "status": "created",
                        "metadata": {},
                    }
                )
                mock_post_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_post_context.__aexit__ = AsyncMock(return_value=None)
                mock_session.post = Mock(return_value=mock_post_context)

                # Mock ClientSession to return our mock session
                mock_aiohttp.ClientSession = Mock(return_value=mock_session)
                mock_aiohttp.ClientTimeout = Mock()

                # Create client WITHOUT using context manager
                client = TraigentCloudClient(api_key=valid_api_key)

                # Mock the auth.get_headers to return valid headers
                client.auth.get_headers = AsyncMock(
                    return_value={
                        "Authorization": f"Bearer {valid_api_key}",
                        "X-Traigent-Client": "test",
                        "Content-Type": "application/json",
                    }
                )

                # Call create_optimization_session
                from traigent.cloud.models import SessionCreationRequest

                request = SessionCreationRequest(
                    function_name="test_function",
                    configuration_space={"param": [1, 2, 3]},
                    objectives=["accuracy"],
                )

                await client.create_optimization_session(request)

                # Verify session was created with headers
                mock_aiohttp.ClientSession.assert_called_once()
                call_kwargs = mock_aiohttp.ClientSession.call_args[1]
                assert "headers" in call_kwargs
                headers = call_kwargs["headers"]
                assert "Authorization" in headers

    @pytest.mark.asyncio
    async def test_context_manager_creates_session_with_headers(
        self, valid_api_key, mock_aiohttp_session
    ):
        """Test context manager creates session with authentication headers."""
        with patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs:
            mock_session = Mock()
            mock_session.close = AsyncMock()  # Mock close method as async
            mock_cs.return_value = mock_session

            with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                client = TraigentCloudClient(api_key=valid_api_key)

                async with client:
                    # Verify session was created with headers
                    mock_cs.assert_called_once()
                    call_kwargs = mock_cs.call_args[1]
                    assert "headers" in call_kwargs
                    headers = call_kwargs["headers"]
                    assert "Authorization" in headers

    @pytest.mark.asyncio
    async def test_execute_agent_includes_headers(
        self, valid_api_key, mock_aiohttp_session
    ):
        """Test execute_agent method includes authentication headers."""
        client = TraigentCloudClient(api_key=valid_api_key)

        # Create agent execution request
        from traigent.cloud.models import AgentExecutionRequest, AgentSpecification

        agent_spec = AgentSpecification(
            name="test_agent",
            agent_platform="langchain",
            model_parameters={"model": "gpt-4"},
        )

        request = AgentExecutionRequest(
            agent_spec=agent_spec,
            input_data={"query": "test query"},
            config_overrides={"temperature": 0.5},
        )

        # Execute agent
        await client.execute_agent(request)

        # Verify headers were included in the POST request
        mock_aiohttp_session.post.assert_called()
        call_kwargs = mock_aiohttp_session.post.call_args[1]
        assert "headers" in call_kwargs
        assert "Authorization" in call_kwargs["headers"]
        assert call_kwargs["headers"]["Authorization"].startswith("Bearer ")
        assert valid_api_key in call_kwargs["headers"]["Authorization"]

    @pytest.mark.asyncio
    async def test_get_agent_optimization_status_includes_headers(
        self, valid_api_key, mock_aiohttp_session
    ):
        """Test get_agent_optimization_status includes authentication headers."""
        client = TraigentCloudClient(api_key=valid_api_key)

        # Get optimization status
        optimization_id = "opt-test-123"
        await client.get_agent_optimization_status(optimization_id)

        # Verify headers were included in the GET request
        mock_aiohttp_session.get.assert_called()
        call_kwargs = mock_aiohttp_session.get.call_args[1]
        assert "headers" in call_kwargs
        assert "Authorization" in call_kwargs["headers"]
        assert valid_api_key in call_kwargs["headers"]["Authorization"]

    @pytest.mark.asyncio
    async def test_cancel_agent_optimization_includes_headers(
        self, valid_api_key, mock_aiohttp_session
    ):
        """Test cancel_agent_optimization includes authentication headers."""
        client = TraigentCloudClient(api_key=valid_api_key)

        # Cancel optimization
        optimization_id = "opt-test-456"
        await client.cancel_agent_optimization(optimization_id)

        # Verify headers were included in the POST request
        mock_aiohttp_session.post.assert_called()
        call_kwargs = mock_aiohttp_session.post.call_args[1]
        assert "headers" in call_kwargs
        assert "Authorization" in call_kwargs["headers"]
        assert valid_api_key in call_kwargs["headers"]["Authorization"]

    @pytest.mark.asyncio
    async def test_optimize_agent_includes_headers(
        self, valid_api_key, mock_aiohttp_session
    ):
        """Test optimize_agent (alias for agent_optimize) includes headers."""
        client = TraigentCloudClient(api_key=valid_api_key)

        from traigent.cloud.models import AgentSpecification

        agent_spec = AgentSpecification(
            name="optimizer_test",
            agent_platform="openai",
            model_parameters={"model": "gpt-3.5-turbo"},
        )

        # Call optimize_agent
        from traigent.evaluators.base import Dataset, EvaluationExample

        examples = [
            EvaluationExample(input_data={"input": "test"}, expected_output="output")
        ]
        dataset = Dataset(examples)

        await client.optimize_agent(
            agent_spec=agent_spec,
            dataset=dataset,
            configuration_space={"temperature": [0.1, 0.9]},
            objectives=["accuracy", "cost"],
        )

        # Verify headers in request
        mock_aiohttp_session.post.assert_called()
        call_kwargs = mock_aiohttp_session.post.call_args[1]
        assert "headers" in call_kwargs
        assert "Authorization" in call_kwargs["headers"]

    @pytest.mark.asyncio
    async def test_all_http_methods_use_ensure_session(self, valid_api_key):
        """Verify all HTTP methods call _ensure_session before making requests."""
        client = TraigentCloudClient(api_key=valid_api_key)

        # Track _ensure_session calls
        ensure_session_calls = []

        async def mock_ensure():
            ensure_session_calls.append(True)
            client._session = Mock()  # Provide mock session

            # Setup mock response
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={})
            mock_response.text = AsyncMock(return_value="")

            mock_context = Mock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)

            client._session.post = Mock(return_value=mock_context)
            client._session.get = Mock(return_value=mock_context)

            return client._session

        client._ensure_session = mock_ensure

        # Test various methods
        methods_to_test = [
            ("check_service_status", []),
            ("get_next_trial", ["session-123"]),
            (
                "submit_trial_result",
                ["session-123", "trial-456", {"accuracy": 0.9}, 1.0],
            ),
            ("finalize_optimization", ["session-123"]),
        ]

        for method_name, args in methods_to_test:
            ensure_session_calls.clear()
            method = getattr(client, method_name)
            try:
                await method(*args)
            except Exception:
                pass  # We only care about _ensure_session being called

            assert (
                len(ensure_session_calls) > 0
            ), f"{method_name} did not call _ensure_session"


class TestBackendIntegratedClientCore:
    """Core tests for BackendIntegratedClient HTTP methods."""

    @pytest.mark.asyncio
    async def test_backend_client_all_methods_include_headers(
        self, mock_aiohttp_session
    ):
        """Test all BackendIntegratedClient HTTP methods include headers."""
        config = BackendClientConfig(backend_base_url="http://test.backend.com")

        # Mock auth manager - BackendIntegratedClient uses auth_manager.auth.get_headers()
        with patch("traigent.cloud.backend_client.BackendAuthManager") as mock_auth:
            mock_auth_instance = Mock()
            # The actual path is auth_manager.auth.get_headers()
            mock_auth_instance.auth = Mock()
            mock_auth_instance.auth.get_headers = AsyncMock(
                return_value={
                    "Authorization": "Bearer test-backend-token",
                    "X-API-Key": "backend-key",
                }
            )
            mock_auth.return_value = mock_auth_instance

            client = BackendIntegratedClient(config)
            client.auth_manager = mock_auth_instance

            # Create mock response for create_hybrid_session
            mock_create_response = Mock()
            mock_create_response.status = 201
            mock_create_response.json = AsyncMock(
                return_value={
                    "session_id": "hybrid-session-123",
                    "token": "token-456",
                    "optimizer_endpoint": "http://optimizer.test.com",
                    "status": "active",
                }
            )

            mock_create_context = Mock()
            mock_create_context.__aenter__ = AsyncMock(
                return_value=mock_create_response
            )
            mock_create_context.__aexit__ = AsyncMock(return_value=None)

            # Override the specific mock for create_hybrid_session
            mock_aiohttp_session.post = Mock(return_value=mock_create_context)

            # Test create_hybrid_session
            await client.create_hybrid_session(
                problem_statement="Test problem",
                search_space={"param": [1, 2, 3]},
                optimization_config={"max_trials": 10},
            )

            # Verify headers in create_hybrid_session
            assert mock_aiohttp_session.post.called
            post_kwargs = mock_aiohttp_session.post.call_args[1]
            assert "headers" in post_kwargs
            assert "Authorization" in post_kwargs["headers"]

            # Reset mock
            mock_aiohttp_session.post.reset_mock()
            mock_aiohttp_session.get.reset_mock()

            # Test get_hybrid_session_status
            await client.get_hybrid_session_status("session-789")

            # Verify headers in get status
            assert mock_aiohttp_session.get.called
            get_kwargs = mock_aiohttp_session.get.call_args[1]
            assert "headers" in get_kwargs

            # Reset mock and setup for finalize
            mock_aiohttp_session.post.reset_mock()

            # Create mock response for finalize_hybrid_session
            mock_finalize_response = Mock()
            mock_finalize_response.status = 200
            mock_finalize_response.json = AsyncMock(
                return_value={
                    "session_id": "session-789",
                    "status": "completed",
                    "best_config": {"param": 2},
                    "best_metrics": {"accuracy": 0.95},
                }
            )
            mock_finalize_response.text = AsyncMock(return_value="OK")

            mock_finalize_context = Mock()
            mock_finalize_context.__aenter__ = AsyncMock(
                return_value=mock_finalize_response
            )
            mock_finalize_context.__aexit__ = AsyncMock(return_value=None)

            mock_aiohttp_session.post = Mock(return_value=mock_finalize_context)

            # Test finalize_hybrid_session
            await client.finalize_hybrid_session("session-789")

            # Verify headers in finalize
            assert mock_aiohttp_session.post.called
            post_kwargs = mock_aiohttp_session.post.call_args[1]
            assert "headers" in post_kwargs

    @pytest.mark.asyncio
    async def test_backend_client_ensure_session_with_fallback(self):
        """Test BackendIntegratedClient _ensure_session handles auth failures gracefully."""
        config = BackendClientConfig(
            backend_base_url="http://test.backend.com",
        )

        with patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True):
            with patch(
                "traigent.cloud.backend_client.aiohttp.ClientSession"
            ) as mock_session_class:
                mock_session = Mock()
                mock_session_class.return_value = mock_session

                # Mock auth manager that fails - BackendIntegratedClient uses BackendAuthManager
                with patch(
                    "traigent.cloud.backend_client.BackendAuthManager"
                ) as mock_auth:
                    mock_auth_instance = Mock()
                    mock_auth_instance.get_headers = AsyncMock(
                        side_effect=Exception("Auth failed")
                    )
                    mock_auth.return_value = mock_auth_instance

                    client = BackendIntegratedClient(config)
                    # Properly mock the nested structure: auth_manager.auth
                    client.auth_manager.auth = mock_auth_instance

                    # Call _ensure_session - should fallback to api_key
                    await client._ensure_session()

                    # Verify session was created with fallback headers
                    mock_session_class.assert_called_once()
                    call_kwargs = mock_session_class.call_args[1]
                    assert "headers" in call_kwargs
                    assert call_kwargs["headers"]["Content-Type"] == "application/json"


class TestHTTPMethodCoverage:
    """Verify complete coverage of all HTTP methods."""

    @pytest.mark.asyncio
    async def test_all_client_http_methods_covered(
        self, valid_api_key, mock_aiohttp_session
    ):
        """Ensure every HTTP method in TraigentCloudClient includes headers."""
        client = TraigentCloudClient(api_key=valid_api_key)

        # List of all methods that make HTTP calls
        http_methods = [
            # GET methods
            ("check_service_status", "get", []),
            ("get_agent_optimization_status", "get", ["opt-123"]),
            # POST methods for sessions
            (
                "create_optimization_session",
                "post",
                ["test_func", {"param": [1]}, ["accuracy"]],
            ),
            ("get_next_trial", "post", ["session-123"]),
            (
                "submit_trial_result",
                "post",
                ["session-123", "trial-456", {"acc": 0.9}, 1.0],
            ),
            ("finalize_optimization", "post", ["session-123"]),
            # POST methods for agents
            ("cancel_agent_optimization", "post", ["opt-456"]),
        ]

        for method_name, http_verb, args in http_methods:
            # Reset mock
            mock_aiohttp_session.post.reset_mock()
            mock_aiohttp_session.get.reset_mock()

            # Call method
            method = getattr(client, method_name)
            try:
                await method(*args)
            except Exception:
                pass  # Some methods might fail due to response parsing

            # Verify correct HTTP method was called with headers
            if http_verb == "get":
                assert mock_aiohttp_session.get.called, f"{method_name} didn't call GET"
                call_kwargs = mock_aiohttp_session.get.call_args[1]
            else:
                assert (
                    mock_aiohttp_session.post.called
                ), f"{method_name} didn't call POST"
                call_kwargs = mock_aiohttp_session.post.call_args[1]

            assert "headers" in call_kwargs, f"{method_name} missing headers"
            assert (
                "Authorization" in call_kwargs["headers"]
            ), f"{method_name} missing Authorization header"

    @pytest.mark.asyncio
    async def test_agent_specification_methods(
        self, valid_api_key, mock_aiohttp_session
    ):
        """Test methods that use AgentSpecification objects."""
        client = TraigentCloudClient(api_key=valid_api_key)

        from traigent.cloud.models import AgentExecutionRequest, AgentSpecification

        agent_spec = AgentSpecification(
            name="comprehensive_test",
            agent_platform="custom",
            model_parameters={"model": "test-model", "params": {"key": "value"}},
            metadata={"version": "1.0"},
        )

        # Test optimize_agent (full optimization)
        from traigent.evaluators.base import Dataset, EvaluationExample

        examples = [EvaluationExample(input_data={"input": "q1"}, expected_output="a1")]
        dataset = Dataset(examples)

        await client.optimize_agent(
            agent_spec=agent_spec,
            dataset=dataset,
            configuration_space={"temperature": [0.1, 1.0]},
            objectives=["accuracy", "latency"],
            max_trials=5,
        )

        # Verify POST with headers
        assert mock_aiohttp_session.post.called
        call_kwargs = mock_aiohttp_session.post.call_args[1]
        assert "headers" in call_kwargs
        assert "Authorization" in call_kwargs["headers"]

        # Reset
        mock_aiohttp_session.post.reset_mock()

        # Test execute_agent with AgentExecutionRequest
        exec_request = AgentExecutionRequest(
            agent_spec=agent_spec,
            input_data={"prompt": "test prompt"},
            config_overrides={"max_tokens": 100},
            execution_context={"user": "test-user"},
        )

        await client.execute_agent(exec_request)

        # Verify POST with headers
        assert mock_aiohttp_session.post.called
        call_kwargs = mock_aiohttp_session.post.call_args[1]
        assert "headers" in call_kwargs
        assert "Authorization" in call_kwargs["headers"]


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error scenarios with authentication headers."""

    @pytest.mark.asyncio
    async def test_headers_preserved_on_retry(self, valid_api_key):
        """Test that headers are preserved when requests are retried."""
        client = TraigentCloudClient(api_key=valid_api_key)

        call_count = 0
        headers_captured = []

        # Mock session that fails first then succeeds
        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            headers_captured.append(kwargs.get("headers", {}))

            mock_response = Mock()
            if call_count == 1:
                # First call fails with retryable error
                mock_response.status = 429  # Rate limited
                mock_response.headers = {"Retry-After": "0.1"}
                mock_response.text = AsyncMock(return_value="Rate limited")
            else:
                # Second call succeeds
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={"status": "healthy"})

            mock_context = Mock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            return mock_context

        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
            with patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs:
                mock_session = Mock()
                mock_session.get = Mock(side_effect=mock_get)
                mock_cs.return_value = mock_session

                # Mock auth.get_headers
                client.auth.get_headers = AsyncMock(
                    return_value={
                        "Authorization": f"Bearer {valid_api_key}",
                        "X-Traigent-Client": "test",
                    }
                )

                with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                    # This should retry and preserve headers
                    await client.check_service_status()

        # Verify both attempts included headers
        assert call_count >= 1
        for headers in headers_captured:
            assert "Authorization" in headers
            assert headers["Authorization"].startswith("Bearer ")

    @pytest.mark.asyncio
    async def test_headers_not_leaked_on_error(self, valid_api_key, caplog):
        """Test that authentication headers are not leaked in error messages or logs."""
        client = TraigentCloudClient(api_key=valid_api_key)

        # Mock session that raises an error
        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
            with patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs:
                mock_session = Mock()
                mock_session.post = Mock(side_effect=Exception("Connection failed"))
                mock_cs.return_value = mock_session

                with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                    try:
                        await client.get_next_trial("session-123")
                    except Exception as e:
                        error_message = str(e)

        # Verify API key not in error message
        assert valid_api_key not in error_message

        # Verify API key not in logs
        for record in caplog.records:
            assert valid_api_key not in record.getMessage()
            if hasattr(record, "args"):
                assert valid_api_key not in str(record.args)

    @pytest.mark.asyncio
    async def test_different_auth_header_formats(self):
        """Test that different authentication header formats are handled correctly."""
        test_cases = [
            ("Bearer token123", "Bearer token123"),
            ("ApiKey key456", "ApiKey key456"),
            ("Basic dXNlcjpwYXNz", "Basic dXNlcjpwYXNz"),
        ]

        for input_header, expected_header in test_cases:
            with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
                with patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs:
                    mock_session = Mock()
                    mock_cs.return_value = mock_session

                    # Mock auth manager to return custom header
                    with patch("traigent.cloud.auth.AuthManager") as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.get_headers = AsyncMock(
                            return_value={
                                "Authorization": input_header,
                                "X-Traigent-Client": "test",
                            }
                        )
                        mock_auth_instance.is_authenticated = AsyncMock(
                            return_value=True
                        )
                        mock_auth.return_value = mock_auth_instance

                        with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                            client = TraigentCloudClient()
                            client.auth = mock_auth_instance

                            await client._ensure_session()

                            # Verify session created with correct header format
                            mock_cs.assert_called_once()
                            call_kwargs = mock_cs.call_args[1]
                            assert (
                                call_kwargs["headers"]["Authorization"]
                                == expected_header
                            )

    @pytest.mark.asyncio
    async def test_no_aiohttp_raises_error(self, valid_api_key):
        """Test that missing aiohttp raises appropriate error."""
        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", False):
            client = TraigentCloudClient(api_key=valid_api_key)

            with pytest.raises(CloudServiceError) as exc_info:
                await client._ensure_session()

            assert "aiohttp not available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_api_key_error(self):
        """Test that missing API key raises appropriate error."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_credentials",
                return_value={},
            ),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_api_key",
                return_value=None,
            ),
        ):
            client = TraigentCloudClient()  # No API key

            with pytest.raises(Exception) as exc_info:
                await client._ensure_session()

        # Should fail when trying to get headers without API key
        assert "Not authenticated" in str(exc_info.value) or "API key" in str(
            exc_info.value
        )


class TestIntegrationScenarios:
    """Integration tests for real-world authentication scenarios."""

    @pytest.mark.asyncio
    async def test_execution_mode_edge_analytics_with_auth(
        self, valid_api_key, mock_aiohttp_session
    ):
        """Test execution_mode='edge_analytics' includes authentication headers for full flow."""
        # This is the unique integration test from test_authentication_headers.py
        # Setup mock responses for the full flow
        mock_create_response = Mock()
        mock_create_response.status = 201
        mock_create_response.json = AsyncMock(
            return_value={"session_id": "test-session", "status": "created"}
        )

        mock_trial_response = Mock()
        mock_trial_response.status = 200
        mock_trial_response.json = AsyncMock(
            return_value={
                "trial_id": "test-trial",
                "configuration": {"param": 1},
                "should_continue": True,
                "status": "pending",
            }
        )

        mock_submit_response = Mock()
        mock_submit_response.status = 200
        mock_submit_response.json = AsyncMock(return_value={})

        mock_finalize_response = Mock()
        mock_finalize_response.status = 200
        mock_finalize_response.json = AsyncMock(
            return_value={
                "session_id": "test-session",
                "best_config": {"param": 1},
                "best_metrics": {"accuracy": 0.95},
                "total_trials": 10,
                "successful_trials": 10,
                "total_duration": 100.0,
                "cost_savings": 0.5,
                "status": "completed",
            }
        )

        # Setup different responses for different calls
        responses = [
            mock_create_response,
            mock_trial_response,
            mock_submit_response,
            mock_finalize_response,
        ]

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            response = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            mock_context = Mock()
            mock_context.__aenter__ = AsyncMock(return_value=response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            return mock_context

        mock_aiohttp_session.post = Mock(side_effect=mock_post)

        client = TraigentCloudClient(api_key=valid_api_key)

        # Simulate a full optimization flow
        await client.create_optimization_session(
            "test_function",  # request_or_function_name as first positional arg
            configuration_space={"param": [1, 2, 3]},
            objectives=["accuracy"],
        )

        await client.get_next_trial(session_id="test-session")

        await client.submit_trial_result(
            session_id="test-session",
            trial_id="test-trial",
            metrics={"accuracy": 0.95},
            duration=1.5,
        )

        await client.finalize_optimization(session_id="test-session")

        # Verify all calls included headers
        assert mock_aiohttp_session.post.call_count == 4
        for call in mock_aiohttp_session.post.call_args_list:
            call_kwargs = call[1]
            assert "headers" in call_kwargs
            assert "Authorization" in call_kwargs["headers"]

    @pytest.mark.asyncio
    async def test_manual_session_creation_deprecated(self, valid_api_key):
        """Test that manually setting _session still works but with headers."""
        # This test verifies backward compatibility while ensuring headers are included
        client = TraigentCloudClient(api_key=valid_api_key)

        # Manually create a mock session (simulating old test pattern)
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "ok"})
        mock_session.get.return_value.__aenter__.return_value = mock_response

        # Set session manually (deprecated pattern)
        client._session = mock_session

        # Even with manual session, methods should add headers
        await client.check_service_status()

        # Verify headers were still added
        mock_session.get.assert_called_once()
        call_kwargs = mock_session.get.call_args[1]
        assert "headers" in call_kwargs
