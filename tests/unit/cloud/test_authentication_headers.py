"""Unified test coverage for authentication headers in all HTTP methods.

This module ensures 100% coverage of all HTTP methods in both TraigentCloudClient
and BackendIntegratedClient, verifying authentication headers are always included.
Combines tests from multiple authentication test files into a single comprehensive suite.
"""

import re
import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider as _SdkTracerProvider

from traigent.cloud.auth import (
    AuthManager,
    _inject_trace_context,
    _strip_trace_context_headers,
)
from traigent.cloud.backend_client import BackendClientConfig, BackendIntegratedClient
from traigent.cloud.client import CloudServiceError, TraigentCloudClient


async def _stub_validate(self, api_key):  # noqa: ARG001
    """Bypass backend API key validation for header-only tests."""
    return None


def _patch_backend_validate():
    """Convenience patcher to keep auth offline in header tests."""
    return patch.object(
        AuthManager, "_validate_api_key_with_backend", new=_stub_validate
    )


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
        with (
            _patch_backend_validate(),
            patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs,
        ):
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
                # API key can be in Authorization header or X-API-Key
                assert "X-API-Key" in headers or "Authorization" in headers
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
                assert "X-API-Key" in headers or "Authorization" in headers

    @pytest.mark.asyncio
    async def test_context_manager_creates_session_with_headers(
        self, valid_api_key, mock_aiohttp_session
    ):
        """Test context manager creates session with authentication headers."""
        with (
            _patch_backend_validate(),
            patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs,
        ):
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
                    assert "X-API-Key" in headers or "Authorization" in headers

    @pytest.mark.asyncio
    async def test_execute_agent_includes_headers(
        self, valid_api_key, mock_aiohttp_session
    ):
        """Test execute_agent method includes authentication headers."""
        with _patch_backend_validate():
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
        headers = call_kwargs["headers"]
        assert "X-API-Key" in headers or "Authorization" in headers

    @pytest.mark.asyncio
    async def test_get_agent_optimization_status_includes_headers(
        self, valid_api_key, mock_aiohttp_session
    ):
        """Test get_agent_optimization_status includes authentication headers."""
        with _patch_backend_validate():
            client = TraigentCloudClient(api_key=valid_api_key)

            # Get optimization status
            optimization_id = "opt-test-123"
            await client.get_agent_optimization_status(optimization_id)

        # Verify headers were included in the GET request
        mock_aiohttp_session.get.assert_called()
        call_kwargs = mock_aiohttp_session.get.call_args[1]
        assert "headers" in call_kwargs
        headers = call_kwargs["headers"]
        assert "X-API-Key" in headers or "Authorization" in headers

    @pytest.mark.asyncio
    async def test_cancel_agent_optimization_includes_headers(
        self, valid_api_key, mock_aiohttp_session
    ):
        """Test cancel_agent_optimization includes authentication headers."""
        with _patch_backend_validate():
            client = TraigentCloudClient(api_key=valid_api_key)

            # Cancel optimization
            optimization_id = "opt-test-456"
            await client.cancel_agent_optimization(optimization_id)

        # Verify headers were included in the POST request
        mock_aiohttp_session.post.assert_called()
        call_kwargs = mock_aiohttp_session.post.call_args[1]
        assert "headers" in call_kwargs
        headers = call_kwargs["headers"]
        assert "X-API-Key" in headers or "Authorization" in headers

    @pytest.mark.asyncio
    async def test_optimize_agent_includes_headers(
        self, valid_api_key, mock_aiohttp_session
    ):
        """Test optimize_agent (alias for agent_optimize) includes headers."""
        with _patch_backend_validate():
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
                EvaluationExample(
                    input_data={"input": "test"}, expected_output="output"
                )
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
        headers = call_kwargs["headers"]
        assert "X-API-Key" in headers or "Authorization" in headers

    @pytest.mark.asyncio
    async def test_all_http_methods_use_ensure_session(self, valid_api_key):
        """Verify all HTTP methods call _ensure_session before making requests."""
        # Bypass backend API-key validation in unit tests; without it,
        # _get_headers raises InvalidCredentialsError before the per-method
        # path runs, which defeats the test's premise.
        with _patch_backend_validate():
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

            # Exceptions raised from inside response parsing are tolerable
            # (the mocked HTTP response is intentionally empty), but ONLY
            # after ``_ensure_session`` has been called. Anything else ARE
            # regressions and must surface, not be silently swallowed.
            _PARSING_EXC_TYPES = (KeyError, TypeError, AttributeError, ValueError)

            for method_name, args in methods_to_test:
                ensure_session_calls.clear()
                method = getattr(client, method_name)
                method_exc: Exception | None = None
                try:
                    await method(*args)
                except _PARSING_EXC_TYPES as exc:
                    method_exc = exc

                assert len(ensure_session_calls) > 0, (
                    f"{method_name} did not call _ensure_session "
                    f"(method_exc={method_exc!r})"
                )


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
            headers = post_kwargs["headers"]
            assert "X-API-Key" in headers or "Authorization" in headers
            post_url = mock_aiohttp_session.post.call_args[0][0]
            assert post_url == "http://test.backend.com/api/v1/hybrid/sessions"

            # Reset mock
            mock_aiohttp_session.post.reset_mock()
            mock_aiohttp_session.get.reset_mock()

            # Test get_hybrid_session_status
            await client.get_hybrid_session_status("session-789")

            # Verify headers in get status
            assert mock_aiohttp_session.get.called
            get_kwargs = mock_aiohttp_session.get.call_args[1]
            assert "headers" in get_kwargs
            get_url = mock_aiohttp_session.get.call_args[0][0]
            assert (
                get_url
                == "http://test.backend.com/api/v1/hybrid/sessions/session-789/status"
            )

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
            post_url = mock_aiohttp_session.post.call_args[0][0]
            assert (
                post_url
                == "http://test.backend.com/api/v1/hybrid/sessions/session-789/finalize"
            )

    @pytest.mark.asyncio
    async def test_backend_client_ensure_session_fails_closed_on_auth_error(self):
        """B4 ROUND 4: ``_ensure_session`` must fail closed on auth errors.

        Previously this test asserted that ``_ensure_session`` silently
        rebuilt headers (Content-Type only, no auth) when ``get_headers``
        raised -- and worse, when auth raised it would also splice in
        raw-key headers via ``_build_session_fallback_headers``. After
        round 4, that fail-open path is removed: any unexpected exception
        from ``get_headers`` is surfaced as ``CloudServiceError`` (and
        ``AuthenticationError`` propagates unchanged).
        """
        from traigent.cloud.client import CloudServiceError

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

                    # Generic exceptions become CloudServiceError; no
                    # session is built with fallback headers.
                    with pytest.raises(CloudServiceError):
                        await client._ensure_session()

                    mock_session_class.assert_not_called()


class TestHTTPMethodCoverage:
    """Verify complete coverage of all HTTP methods."""

    @pytest.mark.asyncio
    async def test_all_client_http_methods_covered(
        self, valid_api_key, mock_aiohttp_session
    ):
        """Ensure every HTTP method in TraigentCloudClient includes headers."""
        with _patch_backend_validate():
            await self._run_all_client_http_methods_covered(
                valid_api_key, mock_aiohttp_session
            )

    async def _run_all_client_http_methods_covered(
        self, valid_api_key, mock_aiohttp_session
    ):
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

        # Same rule as above: tolerate parsing-shape exceptions from the empty
        # mocked response, but require any other exception to surface — and
        # in all cases require the underlying HTTP call to have happened with
        # the correct authentication headers. CloudServiceError is allowed
        # because the mocked HTTP response intentionally has empty/invalid
        # JSON, which the client wraps into CloudServiceError AFTER the HTTP
        # call (and its headers) have already been issued.
        _PARSING_EXC_TYPES = (
            KeyError,
            TypeError,
            AttributeError,
            ValueError,
            CloudServiceError,
        )

        for method_name, http_verb, args in http_methods:
            mock_aiohttp_session.post.reset_mock()
            mock_aiohttp_session.get.reset_mock()

            method = getattr(client, method_name)
            method_exc: Exception | None = None
            try:
                await method(*args)
            except _PARSING_EXC_TYPES as exc:
                method_exc = exc

            if http_verb == "get":
                assert mock_aiohttp_session.get.called, (
                    f"{method_name} didn't call GET (method_exc={method_exc!r})"
                )
                call_kwargs = mock_aiohttp_session.get.call_args[1]
            else:
                assert mock_aiohttp_session.post.called, (
                    f"{method_name} didn't call POST (method_exc={method_exc!r})"
                )
                call_kwargs = mock_aiohttp_session.post.call_args[1]

            assert "headers" in call_kwargs, (
                f"{method_name} missing headers (method_exc={method_exc!r})"
            )
            headers = call_kwargs["headers"]
            assert "X-API-Key" in headers or "Authorization" in headers, (
                f"{method_name} missing authentication header "
                f"(method_exc={method_exc!r})"
            )

    @pytest.mark.asyncio
    async def test_agent_specification_methods(
        self, valid_api_key, mock_aiohttp_session
    ):
        """Test methods that use AgentSpecification objects."""
        with _patch_backend_validate():
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

            examples = [
                EvaluationExample(input_data={"input": "q1"}, expected_output="a1")
            ]
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
            headers = call_kwargs["headers"]
            assert "X-API-Key" in headers or "Authorization" in headers

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
        headers = call_kwargs["headers"]
        assert "X-API-Key" in headers or "Authorization" in headers


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
        with (
            _patch_backend_validate(),
            patch("traigent.cloud.client.AIOHTTP_AVAILABLE", False),
        ):
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

        with _patch_backend_validate():
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
            headers = call_kwargs["headers"]
            assert "X-API-Key" in headers or "Authorization" in headers

    @pytest.mark.asyncio
    async def test_manual_session_creation_deprecated(self, valid_api_key):
        """Test that manually setting _session still works but with headers."""
        # This test verifies backward compatibility while ensuring headers are included
        with _patch_backend_validate():
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


def test_api_key_headers_use_x_api_key_only_no_bearer(valid_api_key):
    """API-key auth must send X-API-Key only, never Authorization: Bearer.

    Regression for the SDK<->backend interop bug: emitting the API key as an
    ``Authorization: Bearer`` token makes a backend that tries JWT auth first
    parse the key as a JWT, fail, and reject the request before the valid
    ``X-API-Key`` credential is considered.
    """
    manager = AuthManager(api_key=valid_api_key)

    headers = manager._get_api_key_headers()

    assert headers.get("X-API-Key") == valid_api_key
    assert "Authorization" not in headers


# --- SDK #1882: W3C traceparent injection on SDK->backend HTTP calls ---------

_TRACEPARENT_RE = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-0[0-9a-f]$")


@contextmanager
def _recording_span():
    """Attach a real, recording OTel span to the current context.

    Uses a *local* SDK ``TracerProvider`` (never the global one) so the test
    does not pollute global tracing state. ``start_as_current_span`` binds the
    span to the current context regardless of provider, which is exactly what
    the W3C propagator reads when injecting.
    """
    provider = _SdkTracerProvider()
    tracer = provider.get_tracer("test-sdk-1882")
    with tracer.start_as_current_span("test-span") as span:
        yield span


@contextmanager
def _recording_span_with_tracestate(vendor_state="vendor=1"):
    """Attach a recording span whose context INHERITS a ``tracestate``.

    Builds a remote parent ``SpanContext`` carrying a vendor ``TraceState``
    (as if extracted from an upstream/user-controlled trace context), then
    starts a child recording span under it. The child inherits the parent's
    ``tracestate``, so the W3C propagator -- left to itself -- would inject
    BOTH ``traceparent`` and ``tracestate``. This is exactly the metadata-egress
    surface SDK #1893 must not forward to the backend.
    """
    from opentelemetry.trace import (
        NonRecordingSpan,
        SpanContext,
        TraceFlags,
        TraceState,
        set_span_in_context,
    )

    trace_state = TraceState(
        [tuple(pair.split("=", 1)) for pair in vendor_state.split(",")]
    )
    parent_ctx = SpanContext(
        trace_id=0x0AF7651916CD43DD8448EB211C80319C,
        span_id=0x00F067AA0BA902B7,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
        trace_state=trace_state,
    )
    parent_context = set_span_in_context(NonRecordingSpan(parent_ctx))
    provider = _SdkTracerProvider()
    tracer = provider.get_tracer("test-sdk-1893")
    with tracer.start_as_current_span("child-span", context=parent_context) as span:
        yield span


class TestTraceparentInjection:
    """SDK #1882: outbound backend/hybrid headers must carry W3C traceparent."""

    def test_add_common_headers_injects_traceparent_with_active_span(self):
        """(a) With a recording span, _add_common_headers adds a valid
        traceparent whose trace-id matches the active trace."""
        manager = AuthManager(api_key="tg_" + "a" * 61)
        headers: dict[str, str] = {}
        with _recording_span() as span:
            manager._add_common_headers(headers, target="backend")
            expected_trace_id = format(span.get_span_context().trace_id, "032x")

        assert "traceparent" in headers
        assert _TRACEPARENT_RE.match(headers["traceparent"]), headers["traceparent"]
        # trace-id segment of "00-<trace-id>-<span-id>-<flags>" matches the span
        assert headers["traceparent"].split("-")[1] == expected_trace_id

    def test_helper_injects_traceparent_with_active_span(self):
        """The shared helper injects a valid traceparent when a span is active."""
        headers: dict[str, str] = {}
        with _recording_span() as span:
            _inject_trace_context(headers)
            expected_trace_id = format(span.get_span_context().trace_id, "032x")

        assert headers["traceparent"].split("-")[1] == expected_trace_id

    def test_no_traceparent_without_active_span(self):
        """(b) With no active span, headers are byte-identical to today's."""
        manager = AuthManager(api_key="tg_" + "a" * 61)
        headers: dict[str, str] = {}
        manager._add_common_headers(headers, target="backend")

        assert "traceparent" not in headers
        assert "tracestate" not in headers
        # Exactly the headers set before this change.
        assert headers == {
            "X-Client-Version": "0.1.0",
            "X-Integration-Mode": "unified",
            "Content-Type": "application/json",
            "X-Traigent-Service": "sdk",
            "X-Backend-Integration": "true",
        }

    def test_helper_is_noop_without_active_span(self):
        """The shared helper adds nothing when no span is active."""
        headers: dict[str, str] = {"X-Existing": "1"}
        _inject_trace_context(headers)
        assert headers == {"X-Existing": "1"}

    def test_caller_supplied_traceparent_never_overridden(self):
        """(c) A pre-existing caller traceparent is never overwritten, even
        when an unrelated span is active."""
        caller_tp = "00-" + "b" * 32 + "-" + "c" * 16 + "-01"
        headers = {"traceparent": caller_tp}
        with _recording_span():
            _inject_trace_context(headers)
        assert headers["traceparent"] == caller_tp

    def test_no_traceparent_when_opentelemetry_unavailable(self):
        """(b, degrade path) With opentelemetry not importable, the helper is a
        silent no-op even while (hypothetically) a span would be active."""
        headers: dict[str, str] = {"X-Existing": "1"}
        # Force the lazy ``from opentelemetry.trace.propagation.tracecontext
        # import ...`` to raise ImportError, as it would when the ``tracing``
        # extra is not installed (patch the full module path: the import
        # system resolves the deepest name in sys.modules first).
        with patch.dict(
            sys.modules,
            {
                "opentelemetry": None,
                "opentelemetry.trace.propagation.tracecontext": None,
            },
        ):
            _inject_trace_context(headers)
        assert headers == {"X-Existing": "1"}

    def test_caller_supplied_traceparent_case_insensitive_not_duplicated(self):
        """(c, case-insensitivity) HTTP header names are case-insensitive: a
        caller-supplied ``Traceparent`` must be preserved with no duplicate
        lowercase ``traceparent`` injected next to it."""
        caller_tp = "00-" + "b" * 32 + "-" + "c" * 16 + "-01"
        headers = {"Traceparent": caller_tp}
        with _recording_span():
            _inject_trace_context(headers)
        assert headers == {"Traceparent": caller_tp}

    def test_w3c_traceparent_even_when_global_propagator_is_not_w3c(self):
        """A host that configured its GLOBAL propagator to B3-only must still
        get a W3C traceparent from our helper (explicit W3C propagator, not
        ``propagate.inject``) -- and none of the host propagator's headers."""
        from opentelemetry import propagate
        from opentelemetry.context import Context
        from opentelemetry.propagators import textmap

        class _B3OnlyStub(textmap.TextMapPropagator):
            """Stand-in for a host-configured non-W3C (e.g. B3) propagator."""

            def inject(
                self,
                carrier,
                context=None,
                setter=textmap.default_setter,
            ) -> None:
                setter.set(carrier, "x-b3-stub", "1")

            def extract(
                self,
                carrier,
                context=None,
                getter=textmap.default_getter,
            ) -> Context:
                return context or Context()

            @property
            def fields(self):
                return {"x-b3-stub"}

        original = propagate.get_global_textmap()
        propagate.set_global_textmap(_B3OnlyStub())
        try:
            headers: dict[str, str] = {}
            with _recording_span() as span:
                _inject_trace_context(headers)
                expected_trace_id = format(span.get_span_context().trace_id, "032x")
        finally:
            propagate.set_global_textmap(original)

        assert headers["traceparent"].split("-")[1] == expected_trace_id
        assert "x-b3-stub" not in headers

    def test_strip_trace_context_headers_case_insensitive_copy(self):
        """The session-default strip removes traceparent/tracestate in any
        casing, keeps everything else, and never mutates its input."""
        original = {
            "Traceparent": "00-" + "b" * 32 + "-" + "c" * 16 + "-01",
            "tracestate": "vendor=1",
            "X-API-Key": "k",
        }
        snapshot = dict(original)

        stripped = _strip_trace_context_headers(original)

        assert stripped == {"X-API-Key": "k"}
        assert original == snapshot

    @pytest.mark.asyncio
    async def test_per_request_headers_carry_fresh_traceparent_per_span(self):
        """(freshness) Two spans on one auth manager produce their OWN
        traceparents -- header construction is per-request, never cached."""
        with _patch_backend_validate():
            manager = AuthManager(api_key="tg_" + "a" * 61)

            with _recording_span() as span_a:
                headers_a = await manager.get_auth_headers("backend")
                trace_a = format(span_a.get_span_context().trace_id, "032x")
            with _recording_span() as span_b:
                headers_b = await manager.get_auth_headers("backend")
                trace_b = format(span_b.get_span_context().trace_id, "032x")

        assert headers_a["traceparent"].split("-")[1] == trace_a
        assert headers_b["traceparent"].split("-")[1] == trace_b
        assert trace_a != trace_b
        assert headers_a["traceparent"] != headers_b["traceparent"]

    @pytest.mark.asyncio
    async def test_session_default_headers_never_carry_traceparent(self):
        """(staleness guard) Long-lived aiohttp session DEFAULT headers must
        never freeze a traceparent, even when a span is active at session
        creation -- trace context travels only on per-request headers."""
        valid_api_key = "tg_" + "a" * 61
        with (
            _patch_backend_validate(),
            patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True),
            patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs,
            patch("traigent.cloud.client.aiohttp.ClientTimeout"),
        ):
            mock_session = Mock()
            mock_cs.return_value = mock_session

            client = TraigentCloudClient(api_key=valid_api_key)
            with _recording_span():
                await client._ensure_session()

            mock_cs.assert_called_once()
            session_headers = mock_cs.call_args[1]["headers"]
            # Auth headers survive; trace context does not.
            assert "X-API-Key" in session_headers or "Authorization" in session_headers
            assert not any(
                name.lower() in ("traceparent", "tracestate")
                for name in session_headers
            )

    def test_backend_client_per_request_sync_headers_carry_traceparent(self):
        """Per-request injection: ``_get_sync_auth_headers`` builds a FRESH
        dict per call (its result goes straight into a per-call
        ``requests.post(headers=...)``, never into session defaults), so it
        must carry the traceparent of the currently-active span. Covers the
        anonymous-Edge path (auth_manager is None)."""
        config = BackendClientConfig(backend_base_url="http://test.backend.com")
        client = BackendIntegratedClient(config)
        client.auth_manager = None  # anonymous Edge mode short-circuit

        with _recording_span() as span:
            headers = client._get_sync_auth_headers(target="backend")
            expected_trace_id = format(span.get_span_context().trace_id, "032x")

        assert _TRACEPARENT_RE.match(headers.get("traceparent", "")), headers
        assert headers["traceparent"].split("-")[1] == expected_trace_id

    @pytest.mark.asyncio
    async def test_backend_client_session_default_headers_never_carry_traceparent(
        self,
    ):
        """(staleness guard, backend client) ``_ensure_session`` must strip
        trace-context headers (any casing) out of the long-lived
        ``aiohttp.ClientSession`` DEFAULT headers -- even when the auth layer
        handed back a traceparent because a span was active at session
        creation."""
        config = BackendClientConfig(backend_base_url="http://test.backend.com")
        with (
            patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True),
            patch("traigent.cloud.backend_client.aiohttp.ClientSession") as mock_cs,
            patch("traigent.cloud.backend_client.aiohttp.ClientTimeout"),
        ):
            mock_session = Mock()
            mock_cs.return_value = mock_session

            client = BackendIntegratedClient(config)
            mock_auth = Mock()
            mock_auth.get_headers = AsyncMock(
                return_value={
                    "X-API-Key": "backend-key",
                    "traceparent": "00-" + "b" * 32 + "-" + "c" * 16 + "-01",
                    "Tracestate": "vendor=1",
                }
            )
            client.auth_manager = Mock()
            client.auth_manager.auth = mock_auth

            await client._ensure_session()

            mock_cs.assert_called_once()
            session_headers = mock_cs.call_args[1]["headers"]
            assert "X-API-Key" in session_headers
            assert not any(
                name.lower() in ("traceparent", "tracestate")
                for name in session_headers
            )

    @pytest.mark.asyncio
    async def test_backend_client_aenter_session_defaults_never_carry_traceparent(
        self,
    ):
        """Same staleness guard for the ``__aenter__`` session-creation site."""
        config = BackendClientConfig(backend_base_url="http://test.backend.com")
        with (
            patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True),
            patch("traigent.cloud.backend_client.aiohttp.ClientSession") as mock_cs,
            patch("traigent.cloud.backend_client.aiohttp.ClientTimeout"),
        ):
            mock_session = Mock()
            mock_cs.return_value = mock_session

            client = BackendIntegratedClient(config)
            mock_auth = Mock()
            mock_auth.get_headers = AsyncMock(
                return_value={
                    "X-API-Key": "backend-key",
                    "traceparent": "00-" + "b" * 32 + "-" + "c" * 16 + "-01",
                    "tracestate": "vendor=1",
                }
            )
            client.auth_manager = Mock()
            client.auth_manager.auth = mock_auth

            await client.__aenter__()

            mock_cs.assert_called_once()
            session_headers = mock_cs.call_args[1]["headers"]
            assert "X-API-Key" in session_headers
            assert not any(
                name.lower() in ("traceparent", "tracestate")
                for name in session_headers
            )

    def test_caller_supplied_tracestate_skips_injection_entirely(self):
        """A caller-supplied ``Tracestate`` (any casing, even without a
        traceparent) means the caller owns the trace context: injection is
        skipped entirely -- no traceparent added, no duplicate lowercase
        ``tracestate`` next to the caller's."""
        headers = {"Tracestate": "vendor=1"}
        with _recording_span():
            _inject_trace_context(headers)
        assert headers == {"Tracestate": "vendor=1"}

    def test_active_span_tracestate_never_forwarded_to_backend(self):
        """SDK #1893 (privacy): when the ACTIVE span's context inherited a
        vendor ``tracestate`` from an upstream/user-controlled trace context,
        the helper must still emit a well-formed ``traceparent`` but must NOT
        forward ``tracestate`` (any casing) to the backend -- that is inherited
        vendor/user metadata and defaults to redaction. Guards the metadata
        egress gap: the caller-supplied-tracestate skip does not cover a
        tracestate that rides in on the active span itself."""
        headers: dict[str, str] = {}
        with _recording_span_with_tracestate("vendor=1,acme=xyz") as span:
            _inject_trace_context(headers)
            expected_trace_id = format(span.get_span_context().trace_id, "032x")

        # traceparent is present, well-formed, and on the active trace.
        assert _TRACEPARENT_RE.match(headers.get("traceparent", "")), headers
        assert headers["traceparent"].split("-")[1] == expected_trace_id
        # tracestate must NOT be forwarded, in ANY header-name casing.
        assert not any(name.lower() == "tracestate" for name in headers), headers
