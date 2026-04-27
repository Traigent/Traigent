"""Test concurrent access and race conditions in authentication handling.

This module tests for race conditions and concurrency issues in the _ensure_session
method to ensure thread-safe session creation with proper authentication headers.
"""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud.backend_client import BackendClientConfig, BackendIntegratedClient
from traigent.cloud.client import TraigentCloudClient


class TestConcurrentSessionCreation:
    """Test concurrent calls to _ensure_session don't create race conditions."""

    @pytest.mark.asyncio
    async def test_concurrent_ensure_session_creates_single_session(self):
        """Test that concurrent calls to _ensure_session only create one session."""
        api_key = "tg_" + "a" * 61  # pragma: allowlist secret

        # Track ClientSession creation calls
        session_creation_count = 0
        session_creation_lock = threading.Lock()

        def mock_session_constructor(*args, **kwargs):
            nonlocal session_creation_count
            with session_creation_lock:
                session_creation_count += 1
                # Simulate some delay in session creation
                time.sleep(0.01)
            session = Mock()
            session.close = AsyncMock()
            return session

        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
            with patch(
                "traigent.cloud.client.aiohttp.ClientSession",
                side_effect=mock_session_constructor,
            ):
                with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                    client = TraigentCloudClient(api_key=api_key)

                    # Mock the auth.get_headers to return valid headers
                    client.auth.get_headers = AsyncMock(
                        return_value={
                            "Authorization": f"Bearer {api_key}",
                            "X-Traigent-Client": "test",
                        }
                    )

                    # Launch multiple concurrent calls to _ensure_session
                    tasks = [client._ensure_session() for _ in range(10)]
                    await asyncio.gather(*tasks)

                    # Verify only one session was created
                    assert (
                        session_creation_count == 1
                    ), f"Expected 1 session, but {session_creation_count} were created"

    @pytest.mark.asyncio
    async def test_concurrent_http_methods_share_session(self):
        """Test that concurrent HTTP method calls share the same session."""
        api_key = "tg_" + "a" * 61  # pragma: allowlist secret

        # Track unique sessions created
        sessions_created = set()

        def mock_session_constructor(*args, **kwargs):
            session = Mock()
            session.id = id(session)  # Unique identifier
            sessions_created.add(session.id)

            # Setup mock responses with all required fields
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "status": "ok",
                    "trial_id": "test-trial",
                    "configuration": {"param": 0.5},
                    "should_continue": True,
                    "session_id": "test-session",
                }
            )
            mock_response.text = AsyncMock(return_value="OK")

            mock_context = Mock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)

            session.get = Mock(return_value=mock_context)
            session.post = Mock(return_value=mock_context)
            session.close = AsyncMock()

            return session

        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
            with patch(
                "traigent.cloud.client.aiohttp.ClientSession",
                side_effect=mock_session_constructor,
            ):
                with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                    client = TraigentCloudClient(api_key=api_key)

                    # Mock the auth.get_headers to return valid headers
                    client.auth.get_headers = AsyncMock(
                        return_value={
                            "Authorization": f"Bearer {api_key}",
                            "X-Traigent-Client": "test",
                        }
                    )

                    # Launch multiple concurrent HTTP method calls
                    tasks = []
                    for i in range(5):
                        tasks.append(client.check_service_status())
                        tasks.append(client.get_next_trial(f"session-{i}"))
                        tasks.append(
                            client.submit_trial_result(
                                f"session-{i}", f"trial-{i}", {"acc": 0.9}, 1.0
                            )
                        )

                    # Execute all tasks concurrently
                    await asyncio.gather(*tasks, return_exceptions=True)

                    # Verify only one session was created
                    assert (
                        len(sessions_created) == 1
                    ), f"Expected 1 unique session, but {len(sessions_created)} were created"

    @pytest.mark.asyncio
    async def test_race_condition_in_session_initialization(self):
        """Test that session initialization is thread-safe under race conditions.

        This test was previously marked as xfail but the race condition has been
        fixed with proper async locking in _ensure_session.
        """
        api_key = "tg_" + "a" * 61  # pragma: allowlist secret

        # Track the order of operations
        operations = []
        operations_lock = threading.Lock()

        async def slow_get_headers():
            """Simulate slow header retrieval to increase race condition likelihood."""
            with operations_lock:
                operations.append("get_headers_start")
            await asyncio.sleep(0.05)  # Simulate network delay
            with operations_lock:
                operations.append("get_headers_end")
            return {"Authorization": f"Bearer {api_key}", "X-Traigent-Client": "test"}

        def mock_session_constructor(*args, **kwargs):
            with operations_lock:
                operations.append("session_create")
            session = Mock()
            session.close = AsyncMock()
            return session

        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
            with patch(
                "traigent.cloud.client.aiohttp.ClientSession",
                side_effect=mock_session_constructor,
            ):
                with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                    client = TraigentCloudClient(api_key=api_key)

                    # Mock the auth.get_headers to be slow
                    client.auth.get_headers = slow_get_headers

                    # Launch multiple concurrent calls
                    tasks = [client._ensure_session() for _ in range(5)]
                    await asyncio.gather(*tasks)

                    # Verify proper ordering - only one session creation
                    session_creates = [
                        op for op in operations if op == "session_create"
                    ]
                    assert (
                        len(session_creates) == 1
                    ), f"Session created {len(session_creates)} times"

                    # Verify get_headers was called appropriately
                    get_headers_starts = [
                        op for op in operations if op == "get_headers_start"
                    ]
                    # Due to our implementation, get_headers should only be called once
                    assert (
                        len(get_headers_starts) <= 1
                    ), "get_headers called multiple times unnecessarily"

    @pytest.mark.asyncio
    async def test_session_recreation_after_failure(self):
        """Test that session is properly recreated after a failure."""
        api_key = "tg_" + "a" * 61  # pragma: allowlist secret

        call_count = 0

        def mock_session_constructor(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First creation fails
                raise Exception("Network error during session creation")

            # Subsequent creations succeed
            session = Mock()
            session.close = AsyncMock()
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "ok"})

            mock_context = Mock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)

            session.get = Mock(return_value=mock_context)
            return session

        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
            with patch(
                "traigent.cloud.client.aiohttp.ClientSession",
                side_effect=mock_session_constructor,
            ):
                with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                    client = TraigentCloudClient(api_key=api_key)

                    # Mock the auth.get_headers to return valid headers
                    client.auth.get_headers = AsyncMock(
                        return_value={
                            "Authorization": f"Bearer {api_key}",
                            "X-Traigent-Client": "test",
                        }
                    )

                    # First call should fail
                    with pytest.raises(Exception) as exc_info:
                        await client._ensure_session()
                    assert "Network error" in str(exc_info.value)

                    # Second call should succeed and create new session
                    session = await client._ensure_session()
                    assert session is not None

                    # Verify session was created twice (once failed, once succeeded)
                    assert call_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_auth_header_updates(self):
        """Test that concurrent requests use the most recent auth headers."""
        api_key_1 = "tg_" + "a" * 61  # pragma: allowlist secret
        api_key_2 = "tg_" + "b" * 61  # pragma: allowlist secret

        # Track headers used in requests
        request_headers = []

        def track_headers(*args, **kwargs):
            if "headers" in kwargs:
                request_headers.append(kwargs["headers"].get("Authorization"))

            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "ok"})

            mock_context = Mock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            return mock_context

        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
            with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                # Start with first API key
                client = TraigentCloudClient(api_key=api_key_1)

                # Mock the auth.get_headers to return valid headers for first key
                client.auth.get_headers = AsyncMock(
                    return_value={
                        "Authorization": f"Bearer {api_key_1}",
                        "X-Traigent-Client": "test",
                    }
                )

                # Make initial request
                with patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs:
                    mock_session = Mock()
                    mock_session.close = AsyncMock()
                    mock_session.get = Mock(side_effect=track_headers)
                    mock_cs.return_value = mock_session

                    await client.check_service_status()

                    # Verify first API key was used
                    assert len(request_headers) == 1
                    assert api_key_1 in request_headers[0]

                # Clear tracked headers
                request_headers.clear()

                # Update API key (simulating key rotation)
                client = TraigentCloudClient(api_key=api_key_2)

                # Mock the auth.get_headers to return valid headers for second key
                client.auth.get_headers = AsyncMock(
                    return_value={
                        "Authorization": f"Bearer {api_key_2}",
                        "X-Traigent-Client": "test",
                    }
                )

                # Make request with new key
                with patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs:
                    mock_session = Mock()
                    mock_session.close = AsyncMock()
                    mock_session.get = Mock(side_effect=track_headers)
                    mock_cs.return_value = mock_session

                    await client.check_service_status()

                    # Verify new API key was used
                    assert len(request_headers) == 1
                    assert api_key_2 in request_headers[0]


class TestBackendClientConcurrency:
    """Test concurrency in BackendIntegratedClient."""

    @pytest.mark.asyncio
    async def test_backend_client_concurrent_session_creation(self):
        """Test BackendIntegratedClient handles concurrent session creation properly."""

        session_creation_count = 0

        def mock_session_constructor(*args, **kwargs):
            nonlocal session_creation_count
            session_creation_count += 1
            time.sleep(0.01)  # Simulate delay
            session = Mock()
            session.close = AsyncMock()
            return session

        config = BackendClientConfig(backend_base_url="http://test.com")

        with patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True):
            with patch(
                "traigent.cloud.backend_client.aiohttp.ClientSession",
                side_effect=mock_session_constructor,
            ):
                with patch("traigent.cloud.backend_client.aiohttp.ClientTimeout"):
                    # Mock AuthManager (from backend_components where it's used)
                    with patch(
                        "traigent.cloud.backend_components.AuthManager"
                    ) as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.get_headers = AsyncMock(
                            return_value={
                                "Authorization": "Bearer test-token",
                                "X-Traigent-Client": "test",
                            }
                        )
                        mock_auth.return_value = mock_auth_instance

                        client = BackendIntegratedClient(
                            api_key="test-key",  # pragma: allowlist secret
                            backend_config=config,
                        )
                        client.auth = mock_auth_instance

                        # Launch concurrent calls
                        tasks = [client._ensure_session() for _ in range(10)]
                        await asyncio.gather(*tasks)

                        # Verify only one session was created
                        assert session_creation_count == 1

    @pytest.mark.asyncio
    async def test_backend_client_auth_fallback_race_condition(self):
        """B4 ROUND 4: Concurrent auth-failure callers all fail closed.

        Pre-round-4, auth failures during ``_ensure_session`` silently
        rebuilt headers from the raw stored API key. This regression
        test originally asserted that behavior. After round 4, every
        concurrent caller must observe the auth failure (as
        ``CloudServiceError`` for generic exceptions) and no session
        with raw-key headers may be created.
        """
        from traigent.cloud.client import CloudServiceError

        config = BackendClientConfig(backend_base_url="http://test.com")

        # Track operations
        operations = []

        async def failing_get_headers():
            operations.append("auth_attempt")
            await asyncio.sleep(0.02)
            raise Exception("Auth service unavailable")

        session_creations = []

        def mock_session_constructor(*args, **kwargs):
            session_creations.append(kwargs.get("headers", {}))
            session = Mock()
            session.close = AsyncMock()
            return session

        with patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True):
            with patch(
                "traigent.cloud.backend_client.aiohttp.ClientSession",
                side_effect=mock_session_constructor,
            ):
                with patch("traigent.cloud.backend_client.aiohttp.ClientTimeout"):
                    # Mock failing AuthManager
                    with patch(
                        "traigent.cloud.backend_components.AuthManager"
                    ) as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.get_headers = failing_get_headers
                        mock_auth.return_value = mock_auth_instance

                        client = BackendIntegratedClient(
                            api_key="fallback-api-key",  # pragma: allowlist secret
                            backend_config=config,
                        )
                        client.auth = mock_auth_instance

                        # Launch concurrent calls - every caller must see
                        # the auth failure surfaced as CloudServiceError.
                        tasks = [client._ensure_session() for _ in range(5)]
                        results = await asyncio.gather(
                            *tasks, return_exceptions=True
                        )

                        # All concurrent attempts must have failed closed.
                        assert len(results) == 5
                        for r in results:
                            assert isinstance(r, CloudServiceError), r

                        # No session should have been created with the raw
                        # fallback key as auth headers.
                        for headers in session_creations:
                            assert "X-API-Key" not in headers
                            assert "Authorization" not in headers


class TestSessionLifecycleConcurrency:
    """Test session lifecycle under concurrent access."""

    @pytest.mark.asyncio
    async def test_session_cleanup_during_concurrent_requests(self):
        """Test that session cleanup doesn't interfere with ongoing requests."""
        api_key = "tg_" + "a" * 61  # pragma: allowlist secret

        # Track ongoing requests
        active_requests = 0
        request_lock = threading.Lock()

        def mock_request(*args, **kwargs):
            async def do_request():
                nonlocal active_requests
                with request_lock:
                    active_requests += 1

                # Simulate request processing
                await asyncio.sleep(0.05)

                with request_lock:
                    active_requests -= 1

                mock_response = Mock()
                mock_response.status = 200
                mock_response.json = AsyncMock(
                    return_value={
                        "status": "ok",
                        "trial_id": "test-trial",
                        "configuration": {"param": 0.5},
                        "should_continue": True,
                    }
                )
                return mock_response

            mock_context = Mock()
            mock_context.__aenter__ = AsyncMock(side_effect=do_request)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            return mock_context

        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
            with patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs:
                with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                    mock_session = Mock()
                    mock_session.get = Mock(side_effect=mock_request)
                    mock_session.post = Mock(side_effect=mock_request)
                    mock_session.close = AsyncMock()
                    mock_cs.return_value = mock_session

                    client = TraigentCloudClient(api_key=api_key)

                    # Mock the auth.get_headers to return valid headers
                    client.auth.get_headers = AsyncMock(
                        return_value={
                            "Authorization": f"Bearer {api_key}",
                            "X-Traigent-Client": "test",
                        }
                    )

                    # Start multiple concurrent requests
                    tasks = []
                    for i in range(10):
                        tasks.append(client.check_service_status())
                        tasks.append(client.get_next_trial(f"session-{i}"))

                    # Start requests
                    request_futures = asyncio.gather(*tasks, return_exceptions=True)

                    # Wait a bit for requests to start
                    await asyncio.sleep(0.01)

                    # Attempt to close session while requests are active
                    # This should be handled gracefully
                    if client._session:
                        asyncio.create_task(client._session.close())

                    # Wait for all requests to complete
                    results = await request_futures

                    # Verify no exceptions occurred
                    exceptions = [r for r in results if isinstance(r, Exception)]
                    assert len(exceptions) == 0, f"Unexpected exceptions: {exceptions}"

    @pytest.mark.asyncio
    async def test_concurrent_context_manager_usage(self):
        """Test concurrent usage of client as context manager."""
        api_key = "tg_" + "a" * 61  # pragma: allowlist secret

        # Track context manager operations
        enter_count = 0
        exit_count = 0
        operation_lock = threading.Lock()

        async def concurrent_operation(client, operation_id):
            """Perform operations within context manager."""
            async with client:
                nonlocal enter_count
                with operation_lock:
                    enter_count += 1

                # Perform some operations
                await client.check_service_status()
                await asyncio.sleep(0.01)  # Simulate work

                with operation_lock:
                    nonlocal exit_count
                    exit_count += 1

        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
            with patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs:
                with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                    # Setup mock session
                    mock_session = Mock()
                    mock_response = Mock()
                    mock_response.status = 200
                    mock_response.json = AsyncMock(return_value={"status": "ok"})

                    mock_context = Mock()
                    mock_context.__aenter__ = AsyncMock(return_value=mock_response)
                    mock_context.__aexit__ = AsyncMock(return_value=None)

                    mock_session.get = Mock(return_value=mock_context)
                    mock_session.close = AsyncMock()
                    mock_cs.return_value = mock_session

                    # Create multiple clients for concurrent context manager usage
                    clients = []
                    for _ in range(5):
                        client = TraigentCloudClient(api_key=api_key)
                        # Mock the auth.get_headers to return valid headers
                        client.auth.get_headers = AsyncMock(
                            return_value={
                                "Authorization": f"Bearer {api_key}",
                                "X-Traigent-Client": "test",
                            }
                        )
                        clients.append(client)

                    # Use all clients concurrently as context managers
                    tasks = [
                        concurrent_operation(client, i)
                        for i, client in enumerate(clients)
                    ]
                    await asyncio.gather(*tasks)

                    # Verify all context managers were properly entered and exited
                    assert enter_count == 5
                    assert exit_count == 5


class TestHeaderConsistencyUnderLoad:
    """Test authentication headers remain consistent under load."""

    @pytest.mark.asyncio
    async def test_headers_consistent_under_high_concurrency(self):
        """Test that headers remain consistent across many concurrent requests."""
        api_key = "tg_" + "a" * 61  # pragma: allowlist secret
        expected_header = f"Bearer {api_key}"

        # Track all headers used
        headers_used = []
        headers_lock = threading.Lock()

        def track_request(*args, **kwargs):
            if "headers" in kwargs:
                with headers_lock:
                    headers_used.append(kwargs["headers"].get("Authorization"))

            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "result": "ok",
                    "trial_id": "test-trial",
                    "configuration": {"param": 0.5},
                    "should_continue": True,
                }
            )

            mock_context = Mock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            return mock_context

        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
            with patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs:
                with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                    mock_session = Mock()
                    mock_session.close = AsyncMock()
                    mock_session.get = Mock(side_effect=track_request)
                    mock_session.post = Mock(side_effect=track_request)
                    mock_cs.return_value = mock_session

                    client = TraigentCloudClient(api_key=api_key)

                    # Mock the auth.get_headers to return valid headers
                    client.auth.get_headers = AsyncMock(
                        return_value={
                            "Authorization": expected_header,
                            "X-Traigent-Client": "test",
                        }
                    )

                    # Create many concurrent requests
                    tasks = []
                    for i in range(100):
                        if i % 2 == 0:
                            tasks.append(client.check_service_status())
                        else:
                            tasks.append(client.get_next_trial(f"session-{i}"))

                    # Execute all concurrently
                    await asyncio.gather(*tasks, return_exceptions=True)

                    # Verify all requests used the same correct header
                    assert len(headers_used) == 100
                    for header in headers_used:
                        assert (
                            header == expected_header
                        ), f"Inconsistent header: {header}"

    @pytest.mark.asyncio
    async def test_no_header_mutation_during_concurrent_access(self):
        """Test that headers are not mutated during concurrent access."""
        api_key = "tg_" + "a" * 61  # pragma: allowlist secret

        # Use a mutable header dict to test for mutations
        original_headers = {
            "Authorization": f"Bearer {api_key}",
            "X-Traigent-Client": "test",
            "Content-Type": "application/json",
        }

        # Track any mutations
        mutations_detected = []

        async def get_headers_with_check():
            headers = original_headers.copy()
            # Check if headers were mutated
            if headers != original_headers:
                mutations_detected.append(headers)
            return headers

        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
            with patch("traigent.cloud.client.aiohttp.ClientSession") as mock_cs:
                with patch("traigent.cloud.client.aiohttp.ClientTimeout"):
                    mock_session = Mock()
                    mock_session.close = AsyncMock()

                    # Setup responses
                    mock_response = Mock()
                    mock_response.status = 200
                    mock_response.json = AsyncMock(return_value={"status": "ok"})

                    mock_context = Mock()
                    mock_context.__aenter__ = AsyncMock(return_value=mock_response)
                    mock_context.__aexit__ = AsyncMock(return_value=None)

                    mock_session.get = Mock(return_value=mock_context)
                    mock_session.post = Mock(return_value=mock_context)
                    mock_cs.return_value = mock_session

                    client = TraigentCloudClient(api_key=api_key)
                    client.auth.get_headers = get_headers_with_check

                    # Run concurrent requests
                    tasks = [client.check_service_status() for _ in range(50)]
                    await asyncio.gather(*tasks, return_exceptions=True)

                    # Verify no mutations were detected
                    assert (
                        len(mutations_detected) == 0
                    ), f"Header mutations detected: {mutations_detected}"
