"""Integration tests for authentication across complete workflows.

This module tests authentication behavior in complete, real-world workflows
to ensure the authentication header fixes work correctly end-to-end.
"""

import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud.backend_client import BackendClientConfig, BackendIntegratedClient
from traigent.cloud.client import CloudServiceError, TraigentCloudClient
from traigent.cloud.models import AgentExecutionRequest, AgentSpecification


class TestFullOptimizationWorkflows:
    """Test complete optimization workflows with authentication."""

    @pytest.mark.asyncio
    async def test_complete_local_optimization_workflow(self):
        """Test complete local optimization workflow with authentication headers."""
        api_key = "tg_integration_" + "a" * 50  # pragma: allowlist secret

        # Track all HTTP requests and their headers
        http_requests = []

        def track_http_request(method):
            def track(*args, **kwargs):
                http_requests.append(
                    {
                        "method": method,
                        "headers": kwargs.get("headers", {}).copy(),
                        "url": args[0] if args else kwargs.get("url", ""),
                        "data": kwargs.get("json"),
                    }
                )

                # Create appropriate mock response based on the URL
                mock_response = Mock()
                url = args[0] if args else kwargs.get("url", "")

                if (
                    "/v1/sessions" in str(url)
                    and "next-trial" not in str(url)
                    and "results" not in str(url)
                    and "finalize" not in str(url)
                ):
                    # create_optimization_session
                    mock_response.status = 201
                    mock_response.json = AsyncMock(
                        return_value={
                            "session_id": "session-integration-123",
                            "status": "created",
                        }
                    )
                    mock_response.text = AsyncMock(return_value="Created")
                elif "/next-trial" in str(url):
                    # get_next_trial
                    mock_response.status = 200
                    mock_response.json = AsyncMock(
                        return_value={
                            "trial_id": "trial-456",
                            "configuration": {"param": 0.5},
                            "status": "pending",
                            "should_continue": True,
                        }
                    )
                    mock_response.text = AsyncMock(return_value="OK")
                elif "/results" in str(url):
                    # submit_trial_result
                    mock_response.status = 200
                    mock_response.json = AsyncMock(
                        return_value={
                            "status": "accepted",
                            "next_trial_available": True,
                        }
                    )
                    mock_response.text = AsyncMock(return_value="OK")
                elif "/finalize" in str(url):
                    # finalize_optimization
                    mock_response.status = 200
                    mock_response.json = AsyncMock(
                        return_value={
                            "session_id": "session-integration-123",
                            "best_config": {"param": 0.7},
                            "best_metrics": {"accuracy": 0.95},
                            "status": "completed",
                            "total_trials": 10,
                            "successful_trials": 10,
                            "total_duration": 100.0,
                            "cost_savings": 0.5,
                        }
                    )
                    mock_response.text = AsyncMock(return_value="OK")
                else:
                    # Default response
                    mock_response.status = 200
                    mock_response.json = AsyncMock(return_value={"status": "ok"})
                    mock_response.text = AsyncMock(return_value="OK")

                mock_context = Mock()
                mock_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_context.__aexit__ = AsyncMock(return_value=None)
                return mock_context

            return track

        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
            with patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs:
                with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                    # Mock the auth manager
                    with patch("traigent.cloud.client.AuthManager") as mock_auth_mgr:
                        mock_auth_instance = Mock()
                        mock_auth_instance.get_headers = AsyncMock(
                            return_value={
                                "Authorization": f"Bearer {api_key}",
                                "X-Traigent-Client": "integration-test",
                                "Content-Type": "application/json",
                            }
                        )
                        mock_auth_instance.is_authenticated = AsyncMock(
                            return_value=True
                        )
                        mock_auth_mgr.return_value = mock_auth_instance

                        mock_session = Mock()
                        mock_session.post = Mock(side_effect=track_http_request("POST"))
                        mock_session.get = Mock(side_effect=track_http_request("GET"))
                        mock_cs.return_value = mock_session

                        client = TraigentCloudClient(api_key=api_key)
                        client.auth = mock_auth_instance

                        # Execute complete optimization workflow

                        # 1. Create optimization session
                        await client.create_optimization_session(
                            "test_function",
                            configuration_space={
                                "param": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                            },
                            objectives=["accuracy", "latency"],
                        )

                        # 2. Get trial
                        await client.get_next_trial("session-integration-123")

                        # 3. Submit results
                        await client.submit_trial_result(
                            session_id="session-integration-123",
                            trial_id="trial-456",
                            metrics={"accuracy": 0.85, "latency": 0.12},
                            duration=2.5,
                        )

                        # 4. Finalize optimization
                        await client.finalize_optimization("session-integration-123")

                        # Verify all requests included proper authentication headers
                        assert len(http_requests) == 4

                        for i, request in enumerate(http_requests):
                            headers = request["headers"]

                            # Every request must have authentication header
                            assert (
                                "Authorization" in headers
                            ), f"Request {i} missing Authorization header"
                            auth_header = headers["Authorization"]
                            assert auth_header.startswith(
                                "Bearer "
                            ), f"Request {i} malformed auth header"
                            assert (
                                api_key in auth_header
                            ), f"Request {i} missing API key"

                            # Consistent headers across requests
                            if i > 0:
                                assert (
                                    headers["Authorization"]
                                    == http_requests[0]["headers"]["Authorization"]
                                )
                                assert headers.get(
                                    "X-Traigent-Client"
                                ) == http_requests[0]["headers"].get(
                                    "X-Traigent-Client"
                                )

    @pytest.mark.asyncio
    async def test_optimization_workflow_with_errors_and_retry(self):
        """Test optimization workflow with error recovery maintains authentication."""
        api_key = "tg_retry_test_" + "b" * 50  # pragma: allowlist secret

        request_count = 0
        headers_in_retries = []

        def mock_request_with_retry(*args, **kwargs):
            nonlocal request_count
            request_count += 1

            # Capture headers from each attempt
            headers_in_retries.append(kwargs.get("headers", {}).copy())

            mock_response = Mock()

            # First two requests fail, third succeeds
            if request_count <= 2:
                mock_response.status = 500
                mock_response.json = AsyncMock(
                    return_value={"error": "Service temporarily unavailable"}
                )
                mock_response.text = AsyncMock(
                    return_value="Service temporarily unavailable"
                )
            else:
                mock_response.status = 201  # Correct status for session creation
                mock_response.json = AsyncMock(
                    return_value={
                        "session_id": "retry-session-789",
                        "status": "created",
                    }
                )
                mock_response.text = AsyncMock(return_value="Created")

            mock_context = Mock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            return mock_context

        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
            with patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs:
                with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                    # Mock the auth manager
                    with patch("traigent.cloud.client.AuthManager") as mock_auth_mgr:
                        mock_auth_instance = Mock()
                        mock_auth_instance.get_headers = AsyncMock(
                            return_value={
                                "Authorization": f"Bearer {api_key}",
                                "X-Traigent-Client": "test-client",
                                "Content-Type": "application/json",
                            }
                        )
                        mock_auth_instance.is_authenticated = AsyncMock(
                            return_value=True
                        )
                        mock_auth_mgr.return_value = mock_auth_instance

                        mock_session = Mock()
                        mock_session.post = Mock(side_effect=mock_request_with_retry)
                        mock_cs.return_value = mock_session

                        client = TraigentCloudClient(api_key=api_key)
                        client.auth = mock_auth_instance

                        # Simulate retries by calling create_optimization_session multiple times
                        # In real implementation, retry logic would be automatic

                        # First attempt (should fail) - suppress expected error
                        with suppress(CloudServiceError, Exception):
                            await client.create_optimization_session(
                                "test_function",
                                configuration_space={"param": [0.1, 0.5, 0.9]},
                                objectives=["accuracy"],
                            )

                        # Force new session for retry
                        client._session = None

                        # Second attempt (should also fail) - suppress expected error
                        with suppress(CloudServiceError, Exception):
                            await client.create_optimization_session(
                                "test_function",
                                configuration_space={"param": [0.1, 0.5, 0.9]},
                                objectives=["accuracy"],
                            )

                        # Force new session for final attempt
                        client._session = None

                        # Third attempt (should succeed)
                        await client.create_optimization_session(
                            "test_function",
                            configuration_space={"param": [0.1, 0.5, 0.9]},
                            objectives=["accuracy"],
                        )

                        # Verify authentication headers were consistent across all retries
                        assert len(headers_in_retries) == 3

                        for i, headers in enumerate(headers_in_retries):
                            assert (
                                "Authorization" in headers
                            ), f"Retry {i} missing auth header"
                            auth_header = headers["Authorization"]
                            assert api_key in auth_header, f"Retry {i} missing API key"

                            # All retries should use same authentication
                            if i > 0:
                                assert (
                                    headers["Authorization"]
                                    == headers_in_retries[0]["Authorization"]
                                )


class TestAgentWorkflows:
    """Test complete agent workflows with authentication."""

    @pytest.mark.asyncio
    async def test_complete_agent_optimization_workflow(self):
        """Test complete agent optimization workflow with authentication headers."""
        api_key = "tg_agent_workflow_" + "c" * 45  # pragma: allowlist secret

        # Track agent-specific requests
        agent_requests = []

        def track_agent_request(method):
            def track(*args, **kwargs):
                agent_requests.append(
                    {
                        "method": method,
                        "headers": kwargs.get("headers", {}).copy(),
                        "url": args[0] if args else kwargs.get("url", ""),
                        "data": kwargs.get("json"),
                    }
                )

                mock_response = Mock()
                url = args[0] if args else kwargs.get("url", "")

                # Simulate different agent workflow responses
                if (
                    method == "POST"
                    and "/agent/optimize" in str(url)
                    and "/cancel" not in str(url)
                ):
                    mock_response.status = 200
                    mock_response.json = AsyncMock(
                        return_value={
                            "session_id": "session-opt-123",
                            "optimization_id": "agent-opt-123",
                            "status": "started",
                            "estimated_duration": 300,
                            "estimated_cost": 1.50,
                            "next_steps": [
                                "Initializing optimization",
                                "Preparing dataset",
                            ],
                        }
                    )
                    mock_response.text = AsyncMock(return_value="OK")
                elif (
                    method == "GET"
                    and "/agent/optimize/" in str(url)
                    and "/status" in str(url)
                ):
                    mock_response.status = 200
                    mock_response.json = AsyncMock(
                        return_value={
                            "optimization_id": "agent-opt-123",
                            "status": "completed",
                            "progress": 1.0,
                            "completed_trials": 10,
                            "total_trials": 10,
                            "current_best": {
                                "config": {"temperature": 0.3},
                                "metrics": {"accuracy": 0.92},
                            },
                            "estimated_time_remaining": 0,
                        }
                    )
                    mock_response.text = AsyncMock(return_value="OK")
                elif method == "POST" and "/agent/execute" in str(url):
                    mock_response.status = 200
                    mock_response.json = AsyncMock(
                        return_value={
                            "output": "Agent executed successfully",
                            "duration": 0.8,
                            "tokens_used": 150,
                            "cost": 0.01,
                            "metadata": {"execution_id": "exec-456"},
                        }
                    )
                    mock_response.text = AsyncMock(return_value="OK")
                else:
                    mock_response.status = 200
                    mock_response.json = AsyncMock(return_value={"status": "ok"})
                    mock_response.text = AsyncMock(return_value="OK")

                mock_context = Mock()
                mock_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_context.__aexit__ = AsyncMock(return_value=None)
                return mock_context

            return track

        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
            with patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs:
                with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                    # Mock the auth manager
                    with patch("traigent.cloud.client.AuthManager") as mock_auth_mgr:
                        mock_auth_instance = Mock()
                        mock_auth_instance.get_headers = AsyncMock(
                            return_value={
                                "Authorization": f"Bearer {api_key}",
                                "X-Traigent-Client": "agent-test",
                                "Content-Type": "application/json",
                            }
                        )
                        mock_auth_instance.is_authenticated = AsyncMock(
                            return_value=True
                        )
                        mock_auth_mgr.return_value = mock_auth_instance

                        mock_session = Mock()
                        mock_session.post = Mock(
                            side_effect=track_agent_request("POST")
                        )
                        mock_session.get = Mock(side_effect=track_agent_request("GET"))
                        mock_cs.return_value = mock_session

                        client = TraigentCloudClient(api_key=api_key)
                        client.auth = mock_auth_instance

                        # Execute complete agent workflow

                        # 1. Create agent specification
                        agent_spec = AgentSpecification(
                            name="integration_test_agent",
                            agent_platform="langchain",
                            model_parameters={
                                "model": "gpt-4",
                                "temperature": [0.1, 0.5, 0.9],
                                "max_tokens": [100, 500, 1000],
                            },
                            metadata={"purpose": "integration_testing"},
                        )

                        # 2. Start agent optimization
                        from traigent.evaluators.base import Dataset, EvaluationExample

                        examples = [
                            EvaluationExample(
                                input_data={"input": "What is AI?"},
                                expected_output="Artificial Intelligence explanation",
                            ),
                            EvaluationExample(
                                input_data={"input": "Explain ML"},
                                expected_output="Machine Learning explanation",
                            ),
                        ]
                        dataset = Dataset(examples)

                        await client.optimize_agent(
                            agent_spec=agent_spec,
                            dataset=dataset,
                            configuration_space={
                                "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
                                "max_tokens": [100, 300, 500],
                            },
                            objectives=["accuracy", "cost", "latency"],
                            max_trials=10,
                        )

                        # 3. Check optimization status
                        await client.get_agent_optimization_status("agent-opt-123")

                        # 4. Execute optimized agent
                        execution_request = AgentExecutionRequest(
                            agent_spec=agent_spec,
                            input_data={"query": "Test query for optimized agent"},
                            config_overrides={"temperature": 0.3},
                            execution_context={"user_id": "test_user"},
                        )

                        await client.execute_agent(execution_request)

                        # Verify all agent requests included proper authentication
                        assert len(agent_requests) == 3

                        for i, request in enumerate(agent_requests):
                            headers = request["headers"]

                            # Every agent request must have authentication
                            assert (
                                "Authorization" in headers
                            ), f"Agent request {i} missing Authorization"
                            auth_header = headers["Authorization"]
                            assert auth_header.startswith(
                                "Bearer "
                            ), f"Agent request {i} malformed auth"
                            assert (
                                api_key in auth_header
                            ), f"Agent request {i} missing API key"

                            # Check for additional agent-specific headers
                            assert (
                                "X-Traigent-Client" in headers
                            ), f"Agent request {i} missing client header"

    @pytest.mark.asyncio
    async def test_concurrent_agent_executions_auth_isolation(self):
        """Test concurrent agent executions maintain authentication isolation."""
        # Different API keys for different agent executions
        api_keys = [
            "tg_concurrent_1_" + "x" * 50,
            "tg_concurrent_2_" + "y" * 50,
            "tg_concurrent_3_" + "z" * 50,
        ]

        execution_headers = {}

        def track_execution_headers(client_id):
            def track(*args, **kwargs):
                execution_headers[client_id] = kwargs.get("headers", {}).copy()

                mock_response = Mock()
                mock_response.status = 200
                mock_response.json = AsyncMock(
                    return_value={
                        "output": f"Success for client {client_id}",
                        "duration": 0.5,
                        "tokens_used": 100,
                        "cost": 0.01,
                        "metadata": {"execution_id": f"exec-{client_id}"},
                    }
                )
                mock_response.text = AsyncMock(return_value="OK")

                mock_context = Mock()
                mock_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_context.__aexit__ = AsyncMock(return_value=None)
                return mock_context

            return track

        async def execute_agent_with_client(client_id, api_key):
            with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
                with patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs:
                    with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                        # Mock the auth manager
                        with patch(
                            "traigent.cloud.client.AuthManager"
                        ) as mock_auth_mgr:
                            mock_auth_instance = Mock()
                            mock_auth_instance.get_headers = AsyncMock(
                                return_value={
                                    "Authorization": f"Bearer {api_key}",
                                    "X-Traigent-Client": f"client-{client_id}",
                                    "Content-Type": "application/json",
                                }
                            )
                            mock_auth_instance.is_authenticated = AsyncMock(
                                return_value=True
                            )
                            mock_auth_mgr.return_value = mock_auth_instance

                            mock_session = Mock()
                            mock_session.post = Mock(
                                side_effect=track_execution_headers(client_id)
                            )
                            mock_cs.return_value = mock_session

                            client = TraigentCloudClient(api_key=api_key)
                            client.auth = mock_auth_instance

                            agent_spec = AgentSpecification(
                                name=f"concurrent_agent_{client_id}",
                                agent_platform="openai",
                                model_parameters={"model": "gpt-3.5-turbo"},
                            )

                            execution_request = AgentExecutionRequest(
                                agent_spec=agent_spec,
                                input_data={"query": f"Test query {client_id}"},
                                execution_context={"client_id": client_id},
                            )

                            return await client.execute_agent(execution_request)

        # Execute agents concurrently with different authentication
        tasks = [
            execute_agent_with_client(i, api_keys[i]) for i in range(len(api_keys))
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

        # Verify each execution used correct authentication
        assert len(execution_headers) == len(api_keys)

        for client_id in range(len(api_keys)):
            headers = execution_headers[client_id]
            expected_key = api_keys[client_id]

            # Should have correct API key
            auth_header = headers.get("Authorization", "")
            assert (
                expected_key in auth_header
            ), f"Client {client_id} missing correct API key"

            # Should not have other clients' API keys
            for other_id in range(len(api_keys)):
                if other_id != client_id:
                    other_key = api_keys[other_id]
                    assert (
                        other_key not in auth_header
                    ), f"Client {client_id} contaminated with key from {other_id}"


class TestBackendIntegrationWorkflows:
    """Test BackendIntegratedClient workflows with authentication."""

    @pytest.mark.asyncio
    async def test_complete_hybrid_session_workflow(self):
        """Test complete hybrid session workflow with BackendIntegratedClient."""

        # Track backend integration requests
        backend_requests = []

        def track_backend_request(method):
            def track(*args, **kwargs):
                backend_requests.append(
                    {
                        "method": method,
                        "headers": kwargs.get("headers", {}).copy(),
                        "url": args[0] if args else kwargs.get("url", ""),
                        "data": kwargs.get("json"),
                    }
                )

                mock_response = Mock()
                url = args[0] if args else kwargs.get("url", "")

                if (
                    method == "POST"
                    and "/api/v1/hybrid/sessions" in str(url)
                    and "/finalize" not in str(url)
                ):
                    mock_response.status = 201
                    mock_response.json = AsyncMock(
                        return_value={
                            "session_id": "hybrid-session-abc123",
                            "token": "session-token-xyz789",
                            "optimizer_endpoint": "http://optimizer.test.com/api/v1",
                            "status": "initialized",
                        }
                    )
                    mock_response.text = AsyncMock(return_value="Created")
                elif (
                    method == "GET"
                    and "/api/v1/hybrid/sessions/" in str(url)
                    and "/status" in str(url)
                ):
                    mock_response.status = 200
                    mock_response.json = AsyncMock(
                        return_value={
                            "session_id": "hybrid-session-abc123",
                            "status": "running",
                            "progress": 0.65,
                            "current_trial": 13,
                            "best_metrics": {"accuracy": 0.87},
                        }
                    )
                    mock_response.text = AsyncMock(return_value="OK")
                elif (
                    method == "POST"
                    and "/api/v1/hybrid/sessions/" in str(url)
                    and "/finalize" in str(url)
                ):
                    mock_response.status = 200
                    mock_response.json = AsyncMock(
                        return_value={
                            "session_id": "hybrid-session-abc123",
                            "status": "completed",
                            "final_results": {
                                "best_configuration": {"lr": 0.001, "batch_size": 64},
                                "best_metrics": {"accuracy": 0.92, "f1": 0.89},
                                "total_trials": 25,
                            },
                        }
                    )
                    mock_response.text = AsyncMock(return_value="OK")
                else:
                    mock_response.status = 200
                    mock_response.json = AsyncMock(return_value={"status": "ok"})
                    mock_response.text = AsyncMock(return_value="OK")

                mock_context = Mock()
                mock_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_context.__aexit__ = AsyncMock(return_value=None)
                return mock_context

            return track

        config = BackendClientConfig(backend_base_url="http://backend.test.com")

        with patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True):
            with patch(
                "traigent.cloud.backend_client.aiohttp.ClientSession"
            ) as mock_cs:
                with patch("traigent.cloud.backend_client.aiohttp.ClientTimeout"):
                    # Mock AuthManager for backend client
                    with patch(
                        "traigent.cloud.backend_client.BackendAuthManager"
                    ) as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.get_headers = AsyncMock(
                            return_value={
                                "Authorization": "Bearer backend-jwt-token-xyz",
                                "X-API-Key": "backend-api-key-12345",
                                "X-Client-Type": "BackendIntegrated",
                            }
                        )
                        mock_auth.return_value = mock_auth_instance

                        mock_session = Mock()
                        mock_session.post = Mock(
                            side_effect=track_backend_request("POST")
                        )
                        mock_session.get = Mock(
                            side_effect=track_backend_request("GET")
                        )
                        mock_cs.return_value = mock_session

                        client = BackendIntegratedClient(config)
                        client.auth_manager = Mock()
                        client.auth_manager.auth = mock_auth_instance

                        # Execute complete hybrid session workflow

                        # 1. Create hybrid session
                        await client.create_hybrid_session(
                            problem_statement="Optimize machine learning model hyperparameters",
                            search_space={
                                "learning_rate": [0.001, 0.01, 0.1],
                                "batch_size": [16, 32, 64, 128],
                                "epochs": [10, 50, 100],
                            },
                            optimization_config={
                                "max_trials": 25,
                                "timeout": 3600,
                                "objectives": ["accuracy", "training_time"],
                            },
                        )

                        # 2. Monitor session status
                        await client.get_hybrid_session_status("hybrid-session-abc123")

                        # 3. Finalize session
                        await client.finalize_hybrid_session("hybrid-session-abc123")

                        # Verify all backend requests included proper authentication
                        assert len(backend_requests) == 3

                        for i, request in enumerate(backend_requests):
                            headers = request["headers"]

                            # Backend requests should have appropriate auth headers
                            assert (
                                "Authorization" in headers or "X-API-Key" in headers
                            ), f"Backend request {i} missing authentication headers"

                            # Should have backend-specific headers
                            if "Authorization" in headers:
                                auth_header = headers["Authorization"]
                                assert (
                                    "Bearer" in auth_header
                                ), f"Backend request {i} malformed Bearer token"

                            if "X-API-Key" in headers:
                                api_key_header = headers["X-API-Key"]
                                assert (
                                    "backend-api-key-12345" in api_key_header
                                ), f"Backend request {i} wrong API key"

    @pytest.mark.asyncio
    async def test_backend_auth_fallback_during_workflow(self):
        """Test backend authentication fallback during complete workflow."""

        auth_attempts = []
        backend_requests = []

        async def mock_get_headers_with_fallback():
            auth_attempts.append("primary_auth_attempt")

            if len(auth_attempts) <= 2:
                # First two attempts fail (simulate auth service down)
                raise Exception("Primary auth service unavailable")

            # Later attempts succeed
            return {
                "Authorization": "Bearer recovered-auth-token",
                "X-Recovery-Mode": "true",
            }

        def track_fallback_request(method):
            def track(*args, **kwargs):
                backend_requests.append(
                    {
                        "method": method,
                        "headers": kwargs.get("headers", {}).copy(),
                        "auth_attempt": len(auth_attempts),
                    }
                )

                mock_response = Mock()
                mock_response.status = (
                    201
                    if "/hybrid" in str(args[0] if args else "") and method == "POST"
                    else 200
                )
                mock_response.json = AsyncMock(
                    return_value={
                        "session_id": "fallback-session-123",
                        "status": "created_with_fallback",
                        "token": "fallback-token",
                        "optimizer_endpoint": "http://fallback.test.com",
                    }
                )
                mock_response.text = AsyncMock(return_value="OK")

                mock_context = Mock()
                mock_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_context.__aexit__ = AsyncMock(return_value=None)
                return mock_context

            return track

        config = BackendClientConfig(backend_base_url="http://backend.test.com")

        with patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True):
            with patch(
                "traigent.cloud.backend_client.aiohttp.ClientSession"
            ) as mock_cs:
                with patch("traigent.cloud.backend_client.aiohttp.ClientTimeout"):
                    # Mock AuthManager with failing primary auth
                    with patch(
                        "traigent.cloud.backend_client.BackendAuthManager"
                    ) as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.get_headers = mock_get_headers_with_fallback
                        mock_auth_instance.api_key = (
                            "fallback-key-67890"  # Set fallback key
                        )
                        mock_auth.return_value = mock_auth_instance

                        mock_session = Mock()
                        mock_session.post = Mock(
                            side_effect=track_fallback_request("POST")
                        )
                        mock_cs.return_value = mock_session

                        client = BackendIntegratedClient(config)
                        client.auth_manager = Mock()
                        client.auth_manager.auth = mock_auth_instance
                        client.auth_manager.api_key = (
                            "fallback-key-67890"  # Set fallback key
                        )

                        # First request - should use fallback auth - suppress expected error
                        with suppress(CloudServiceError, Exception):
                            await client.create_hybrid_session("test", {}, {})

                        # Reset session for second attempt
                        client._session = None

                        # Second request - should still use fallback - suppress expected error
                        with suppress(CloudServiceError, Exception):
                            await client.create_hybrid_session("test2", {}, {})

                        # Reset session for third attempt
                        client._session = None

                        # Third request - should now use primary auth (recovered)
                        await client.create_hybrid_session("test3", {}, {})

                        # Verify authentication progression
                        assert len(backend_requests) >= 1
                        assert len(auth_attempts) >= 3

                        # Check that eventual auth recovery works
                        last_headers = backend_requests[-1]["headers"]
                        assert (
                            "Authorization" in last_headers
                            or "X-API-Key" in last_headers
                        )


class TestMultiClientWorkflows:
    """Test workflows with multiple clients and authentication states."""

    @pytest.mark.asyncio
    async def test_multiple_clients_different_auth_states(self):
        """Test multiple clients with different authentication states working together."""

        # Different client configurations
        client_configs = [
            {
                "api_key": "tg_client_alpha_" + "a" * 50,  # pragma: allowlist secret
                "name": "alpha",
            },
            {
                "api_key": "tg_client_beta_" + "b" * 50,  # pragma: allowlist secret
                "name": "beta",
            },
            {
                "api_key": "tg_client_gamma_" + "c" * 50,  # pragma: allowlist secret
                "name": "gamma",
            },
        ]

        client_requests = {}

        def track_client_requests(client_name):
            def track(*args, **kwargs):
                if client_name not in client_requests:
                    client_requests[client_name] = []

                client_requests[client_name].append(
                    {
                        "headers": kwargs.get("headers", {}).copy(),
                        "timestamp": asyncio.get_event_loop().time(),
                    }
                )

                mock_response = Mock()
                mock_response.status = 201  # Correct status for session creation
                mock_response.json = AsyncMock(
                    return_value={
                        "session_id": f"session-{client_name}-001",
                        "status": "created",
                    }
                )
                mock_response.text = AsyncMock(return_value="OK")

                mock_context = Mock()
                mock_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_context.__aexit__ = AsyncMock(return_value=None)
                return mock_context

            return track

        async def client_workflow(config):
            client_name = config["name"]
            api_key = config["api_key"]

            with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
                with patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs:
                    with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                        # Mock the auth manager
                        with patch(
                            "traigent.cloud.client.AuthManager"
                        ) as mock_auth_mgr:
                            mock_auth_instance = Mock()
                            mock_auth_instance.get_headers = AsyncMock(
                                return_value={
                                    "Authorization": f"Bearer {api_key}",
                                    "X-Traigent-Client": f"client-{client_name}",
                                    "Content-Type": "application/json",
                                }
                            )
                            mock_auth_instance.is_authenticated = AsyncMock(
                                return_value=True
                            )
                            mock_auth_mgr.return_value = mock_auth_instance

                            mock_session = Mock()
                            mock_session.post = Mock(
                                side_effect=track_client_requests(client_name)
                            )
                            mock_cs.return_value = mock_session

                            client = TraigentCloudClient(api_key=api_key)
                            client.auth = mock_auth_instance

                            # Each client performs its workflow
                            await client.create_optimization_session(
                                f"function_{client_name}",
                                configuration_space={
                                    f"param_{client_name}": [0.1, 0.5, 0.9]
                                },
                                objectives=[f"metric_{client_name}"],
                            )

                            return client_name

        # Run all client workflows concurrently
        tasks = [client_workflow(config) for config in client_configs]
        await asyncio.gather(*tasks)

        # Verify each client maintained its own authentication
        assert len(client_requests) == len(client_configs)

        for config in client_configs:
            client_name = config["name"]
            expected_key = config["api_key"]

            requests = client_requests[client_name]
            assert len(requests) == 1

            headers = requests[0]["headers"]
            auth_header = headers.get("Authorization", "")

            # Should have correct API key
            assert (
                expected_key in auth_header
            ), f"Client {client_name} missing correct API key"

            # Should not have other clients' keys
            for other_config in client_configs:
                if other_config["name"] != client_name:
                    other_key = other_config["api_key"]
                    assert (
                        other_key not in auth_header
                    ), f"Client {client_name} contaminated with {other_config['name']} key"

    @pytest.mark.asyncio
    async def test_client_authentication_state_transitions(self):
        """Test authentication state transitions during workflow execution."""

        api_key_1 = "tg_state_initial_" + "x" * 50  # pragma: allowlist secret
        api_key_2 = "tg_state_updated_" + "y" * 50  # pragma: allowlist secret

        state_transitions = []

        def track_state_transition(*args, **kwargs):
            headers = kwargs.get("headers", {})
            auth_header = headers.get("Authorization", "")

            state_transitions.append(
                {
                    "auth_header": auth_header,
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )

            mock_response = Mock()
            mock_response.status = 201  # Correct status for session creation
            mock_response.json = AsyncMock(
                return_value={
                    "session_id": "state-tracked-session",
                    "status": "created",
                }
            )
            mock_response.text = AsyncMock(return_value="OK")

            mock_context = Mock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            return mock_context

        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
            with patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs:
                with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                    mock_session = Mock()
                    mock_session.post = Mock(side_effect=track_state_transition)
                    mock_cs.return_value = mock_session

                    # Start with first API key
                    with patch("traigent.cloud.client.AuthManager") as mock_auth_mgr1:
                        mock_auth_instance1 = Mock()
                        mock_auth_instance1.get_headers = AsyncMock(
                            return_value={
                                "Authorization": f"Bearer {api_key_1}",
                                "X-Traigent-Client": "test",
                                "Content-Type": "application/json",
                            }
                        )
                        mock_auth_instance1.is_authenticated = AsyncMock(
                            return_value=True
                        )
                        mock_auth_mgr1.return_value = mock_auth_instance1

                        client = TraigentCloudClient(api_key=api_key_1)
                        client.auth = mock_auth_instance1

                        # Make initial request
                        await client.create_optimization_session(
                            "initial_function",
                            configuration_space={"param": [1, 2, 3]},
                            objectives=["metric1"],
                        )

                    # Update to new API key (simulating key rotation)
                    with patch("traigent.cloud.client.AuthManager") as mock_auth_mgr2:
                        mock_auth_instance2 = Mock()
                        mock_auth_instance2.get_headers = AsyncMock(
                            return_value={
                                "Authorization": f"Bearer {api_key_2}",
                                "X-Traigent-Client": "test",
                                "Content-Type": "application/json",
                            }
                        )
                        mock_auth_instance2.is_authenticated = AsyncMock(
                            return_value=True
                        )
                        mock_auth_mgr2.return_value = mock_auth_instance2

                        client = TraigentCloudClient(api_key=api_key_2)
                        client.auth = mock_auth_instance2

                        # Make request with new key
                        await client.create_optimization_session(
                            "updated_function",
                            configuration_space={"param": [4, 5, 6]},
                            objectives=["metric2"],
                        )

                    # Verify state transitions
                    assert len(state_transitions) == 2

                    # First request used initial API key
                    assert api_key_1 in state_transitions[0]["auth_header"]
                    assert api_key_2 not in state_transitions[0]["auth_header"]

                    # Second request used updated API key
                    assert api_key_2 in state_transitions[1]["auth_header"]
                    assert api_key_1 not in state_transitions[1]["auth_header"]

                    # Verify clean state transition (no mixing)
                    for transition in state_transitions:
                        auth_header = transition["auth_header"]
                        # Should not contain both keys
                        has_key_1 = api_key_1 in auth_header
                        has_key_2 = api_key_2 in auth_header
                        assert not (
                            has_key_1 and has_key_2
                        ), "Authentication state mixing detected"
