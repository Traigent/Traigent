"""Live Redis coverage for the security session manager."""

from __future__ import annotations

import os
import time

import pytest

from traigent.security.session_manager import SessionManager


REDIS_URL = os.getenv("TRAIGENT_TEST_REDIS_URL")


@pytest.mark.integration
@pytest.mark.skipif(
    not REDIS_URL,
    reason="TRAIGENT_TEST_REDIS_URL is required for the live Redis test",
)
def test_session_manager_connects_and_expires_sessions_in_redis() -> None:
    """Exercise the real redis-py connection and Redis key expiration."""
    manager = SessionManager(
        redis_url=REDIS_URL,
        session_ttl=1,
        enable_rate_limiting=False,
    )

    try:
        assert manager.use_redis
        assert manager.redis_client is not None

        session_id, auth_value = manager.create_session(
            "live-redis-test-user",
            session_id="live-redis-session",
        )
        key = f"session:{session_id}"

        assert manager.redis_client.exists(key) == 1
        assert 0 < manager.redis_client.ttl(key) <= 1
        assert manager.validate_session(session_id, auth_value) is not None

        expiry_deadline = time.monotonic() + 3
        while manager.redis_client.exists(key) and time.monotonic() < expiry_deadline:
            time.sleep(0.05)
        assert manager.redis_client.exists(key) == 0
        assert manager.validate_session(session_id, auth_value) is None
    finally:
        if manager.redis_client is not None:
            manager.redis_client.delete(
                "session:live-redis-session",
                "user_sessions:live-redis-test-user",
            )
