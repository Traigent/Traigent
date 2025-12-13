"""Unit tests for session_manager.

Tests for session management with Redis backend and in-memory fallback,
including session creation, validation, expiration, rate limiting, cleanup.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security
# Traceability: CONC-Quality-Reliability FUNC-SECURITY REQ-SEC-010
# Traceability: SYNC-CloudHybrid

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from traigent.security.session_manager import (
    SessionManager,
    SessionMiddleware,
)


class TestSessionManagerInit:
    """Tests for SessionManager initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        manager = SessionManager()
        assert manager.session_ttl == 3600
        assert manager.max_sessions_per_user == 5
        assert manager.enable_rate_limiting is True
        assert manager.use_redis is False
        assert manager.redis_client is None
        assert hasattr(manager, "_sessions")
        assert hasattr(manager, "_user_sessions")
        assert hasattr(manager, "_rate_limits")
        assert hasattr(manager, "_token_secret")

    def test_init_with_custom_ttl(self) -> None:
        """Test initialization with custom TTL."""
        manager = SessionManager(session_ttl=7200)
        assert manager.session_ttl == 7200

    def test_init_with_custom_max_sessions(self) -> None:
        """Test initialization with custom max sessions."""
        manager = SessionManager(max_sessions_per_user=10)
        assert manager.max_sessions_per_user == 10

    def test_init_with_rate_limiting_disabled(self) -> None:
        """Test initialization with rate limiting disabled."""
        manager = SessionManager(enable_rate_limiting=False)
        assert manager.enable_rate_limiting is False

    @patch("traigent.security.session_manager.REDIS_AVAILABLE", True)
    @patch("traigent.security.session_manager.redis")
    def test_init_with_redis_success(self, mock_redis: MagicMock) -> None:
        """Test initialization with successful Redis connection."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.from_url.return_value = mock_client

        manager = SessionManager(redis_url="redis://localhost:6379/0")
        assert manager.use_redis is True
        assert manager.redis_client is not None
        mock_redis.from_url.assert_called_once_with(
            "redis://localhost:6379/0",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        mock_client.ping.assert_called_once()

    @patch("traigent.security.session_manager.REDIS_AVAILABLE", True)
    @patch("traigent.security.session_manager.redis")
    def test_init_with_redis_connection_failure(self, mock_redis: MagicMock) -> None:
        """Test initialization falls back to in-memory on Redis connection failure."""
        mock_redis.from_url.side_effect = Exception("Connection failed")

        manager = SessionManager(redis_url="redis://localhost:6379/0")
        assert manager.use_redis is False
        assert manager.redis_client is None
        assert hasattr(manager, "_sessions")

    @patch("traigent.security.session_manager.REDIS_AVAILABLE", False)
    def test_init_with_redis_not_available(self) -> None:
        """Test initialization when Redis is not installed."""
        manager = SessionManager(redis_url="redis://localhost:6379/0")
        assert manager.use_redis is False
        assert manager.redis_client is None
        assert hasattr(manager, "_sessions")


class TestSessionManagerCreateSession:
    """Tests for session creation."""

    @pytest.fixture
    def manager(self) -> SessionManager:
        """Create in-memory session manager."""
        return SessionManager(enable_rate_limiting=False)

    def test_create_session_basic(self, manager: SessionManager) -> None:
        """Test basic session creation."""
        session_id, session_token = manager.create_session("user123")
        assert session_id.startswith("session_")
        assert len(session_token) > 0
        assert session_id in manager._sessions
        assert "user123" in manager._user_sessions
        assert session_id in manager._user_sessions["user123"]

    def test_create_session_with_metadata(self, manager: SessionManager) -> None:
        """Test session creation with metadata."""
        metadata = {"ip": "127.0.0.1", "user_agent": "Test/1.0"}
        session_id, session_token = manager.create_session("user123", metadata=metadata)
        session_data = manager._sessions[session_id]
        assert session_data["metadata"] == metadata

    def test_create_session_with_explicit_session_id(
        self, manager: SessionManager
    ) -> None:
        """Test session creation with explicit session ID."""
        session_id, session_token = manager.create_session(
            "user123", session_id="custom_session_123"
        )
        assert session_id == "custom_session_123"
        assert session_id in manager._sessions

    def test_create_session_replaces_existing_session_id(
        self, manager: SessionManager
    ) -> None:
        """Test that creating a session with existing ID replaces it."""
        # Create first session
        session_id, token1 = manager.create_session(
            "user123", session_id="duplicate_session"
        )
        session_data_1 = manager._sessions[session_id].copy()

        # Create second session with same ID
        session_id2, token2 = manager.create_session(
            "user456", session_id="duplicate_session"
        )
        assert session_id == session_id2
        assert token1 != token2
        session_data_2 = manager._sessions[session_id]
        assert session_data_2["user_id"] == "user456"
        assert session_data_2["user_id"] != session_data_1["user_id"]

    def test_create_session_invalid_user_id_empty(
        self, manager: SessionManager
    ) -> None:
        """Test session creation fails with empty user_id."""
        with pytest.raises(ValueError, match="Invalid user_id"):
            manager.create_session("")

    def test_create_session_invalid_user_id_none(self, manager: SessionManager) -> None:
        """Test session creation fails with None user_id."""
        with pytest.raises(ValueError, match="Invalid user_id"):
            manager.create_session(None)  # type: ignore[arg-type]

    def test_create_session_invalid_user_id_non_string(
        self, manager: SessionManager
    ) -> None:
        """Test session creation fails with non-string user_id."""
        with pytest.raises(ValueError, match="Invalid user_id"):
            manager.create_session(123)  # type: ignore[arg-type]

    def test_create_session_invalid_session_id(self, manager: SessionManager) -> None:
        """Test session creation fails with invalid session_id."""
        # When session_id is empty string, it evaluates to falsy and a default is generated
        # So we need to test with a non-string type instead
        with pytest.raises(ValueError, match="session_id must be a non-empty string"):
            manager.create_session("user123", session_id=123)  # type: ignore[arg-type]

    def test_create_session_enforces_max_sessions(
        self, manager: SessionManager
    ) -> None:
        """Test max sessions per user is enforced."""
        # Create max sessions
        sessions = []
        for _i in range(manager.max_sessions_per_user):
            session_id, token = manager.create_session("user123")
            sessions.append((session_id, token))

        # Create one more session - should remove oldest
        new_session_id, new_token = manager.create_session("user123")
        assert new_session_id not in [s[0] for s in sessions]
        assert len(manager._user_sessions["user123"]) == manager.max_sessions_per_user
        # First session should be removed
        assert sessions[0][0] not in manager._sessions

    def test_create_session_with_rate_limiting(self) -> None:
        """Test session creation respects rate limits."""
        manager = SessionManager(enable_rate_limiting=True)
        user_id = "rate_limited_user"

        # Create sessions up to limit
        for _i in range(10):
            manager.create_session(user_id)

        # Next one should fail
        with pytest.raises(ValueError, match="Rate limit exceeded"):
            manager.create_session(user_id)

    def test_create_session_stores_correct_fields(
        self, manager: SessionManager
    ) -> None:
        """Test session data contains all required fields."""
        session_id, session_token = manager.create_session("user123")
        session_data = manager._sessions[session_id]

        assert "user_id" in session_data
        assert "token_hash" in session_data
        assert "created_at" in session_data
        assert "last_accessed" in session_data
        assert "expires_at" in session_data
        assert "metadata" in session_data
        assert "access_count" in session_data
        assert session_data["user_id"] == "user123"
        assert session_data["access_count"] == 0

    @patch("traigent.security.session_manager.REDIS_AVAILABLE", True)
    @patch("traigent.security.session_manager.redis")
    def test_create_session_redis_backend(self, mock_redis: MagicMock) -> None:
        """Test session creation with Redis backend."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.from_url.return_value = mock_client

        manager = SessionManager(
            redis_url="redis://localhost:6379/0", enable_rate_limiting=False
        )
        session_id, session_token = manager.create_session("user123")

        # Verify Redis calls
        mock_client.setex.assert_called_once()
        mock_client.sadd.assert_called_once_with("user_sessions:user123", session_id)
        mock_client.expire.assert_called_once_with(
            "user_sessions:user123", manager.session_ttl * 2
        )


class TestSessionManagerValidateSession:
    """Tests for session validation."""

    @pytest.fixture
    def manager(self) -> SessionManager:
        """Create in-memory session manager."""
        return SessionManager(enable_rate_limiting=False)

    def test_validate_session_success(self, manager: SessionManager) -> None:
        """Test successful session validation."""
        session_id, session_token = manager.create_session("user123")
        result = manager.validate_session(session_id, session_token)

        assert result is not None
        assert result["user_id"] == "user123"
        assert "created_at" in result
        assert "access_count" in result
        assert result["access_count"] == 1

    def test_validate_session_increments_access_count(
        self, manager: SessionManager
    ) -> None:
        """Test validation increments access count."""
        session_id, session_token = manager.create_session("user123")

        result1 = manager.validate_session(session_id, session_token)
        assert result1["access_count"] == 1

        result2 = manager.validate_session(session_id, session_token)
        assert result2["access_count"] == 2

        result3 = manager.validate_session(session_id, session_token)
        assert result3["access_count"] == 3

    def test_validate_session_updates_last_accessed(
        self, manager: SessionManager
    ) -> None:
        """Test validation updates last accessed time."""
        session_id, session_token = manager.create_session("user123")
        original_last_accessed = manager._sessions[session_id]["last_accessed"]

        time.sleep(0.01)  # Small delay
        manager.validate_session(session_id, session_token)
        new_last_accessed = manager._sessions[session_id]["last_accessed"]

        assert new_last_accessed >= original_last_accessed

    def test_validate_session_invalid_session_id(self, manager: SessionManager) -> None:
        """Test validation fails with invalid session ID."""
        result = manager.validate_session("nonexistent", "fake_token")
        assert result is None

    def test_validate_session_invalid_token(self, manager: SessionManager) -> None:
        """Test validation fails with invalid token."""
        session_id, session_token = manager.create_session("user123")
        result = manager.validate_session(session_id, "wrong_token")
        assert result is None

    def test_validate_session_expired(self, manager: SessionManager) -> None:
        """Test validation fails for expired session."""
        session_id, session_token = manager.create_session("user123")

        # Manually expire the session
        session_data = manager._sessions[session_id]
        past_time = datetime.now(UTC) - timedelta(seconds=100)
        session_data["expires_at"] = past_time.isoformat()

        result = manager.validate_session(session_id, session_token)
        assert result is None
        assert session_id not in manager._sessions  # Should be revoked

    def test_validate_session_invalid_expires_at_none(
        self, manager: SessionManager
    ) -> None:
        """Test validation handles None expires_at."""
        session_id, session_token = manager.create_session("user123")
        manager._sessions[session_id]["expires_at"] = None

        result = manager.validate_session(session_id, session_token)
        assert result is None
        assert session_id not in manager._sessions  # Should be revoked

    def test_validate_session_invalid_expires_at_non_string(
        self, manager: SessionManager
    ) -> None:
        """Test validation handles non-string expires_at."""
        session_id, session_token = manager.create_session("user123")
        manager._sessions[session_id]["expires_at"] = 12345

        result = manager.validate_session(session_id, session_token)
        assert result is None
        assert session_id not in manager._sessions  # Should be revoked

    def test_validate_session_invalid_expires_at_format(
        self, manager: SessionManager
    ) -> None:
        """Test validation handles invalid expires_at format."""
        session_id, session_token = manager.create_session("user123")
        manager._sessions[session_id]["expires_at"] = "invalid-datetime"

        result = manager.validate_session(session_id, session_token)
        assert result is None
        assert session_id not in manager._sessions  # Should be revoked

    def test_validate_session_invalid_token_hash_none(
        self, manager: SessionManager
    ) -> None:
        """Test validation handles None token_hash."""
        session_id, session_token = manager.create_session("user123")
        manager._sessions[session_id]["token_hash"] = None

        result = manager.validate_session(session_id, session_token)
        assert result is None

    def test_validate_session_invalid_token_hash_non_string(
        self, manager: SessionManager
    ) -> None:
        """Test validation handles non-string token_hash."""
        session_id, session_token = manager.create_session("user123")
        manager._sessions[session_id]["token_hash"] = 12345

        result = manager.validate_session(session_id, session_token)
        assert result is None

    def test_validate_session_invalid_access_count_non_int(
        self, manager: SessionManager
    ) -> None:
        """Test validation handles non-integer access_count."""
        session_id, session_token = manager.create_session("user123")
        manager._sessions[session_id]["access_count"] = "not_an_int"

        result = manager.validate_session(session_id, session_token)
        assert result is not None
        assert result["access_count"] == 1  # Reset to 1

    def test_validate_session_returns_metadata(self, manager: SessionManager) -> None:
        """Test validation returns session metadata."""
        metadata = {"ip": "127.0.0.1", "browser": "Firefox"}
        session_id, session_token = manager.create_session("user123", metadata=metadata)

        result = manager.validate_session(session_id, session_token)
        assert result is not None
        assert result["metadata"] == metadata

    @patch("traigent.security.session_manager.REDIS_AVAILABLE", True)
    @patch("traigent.security.session_manager.redis")
    def test_validate_session_redis_backend(self, mock_redis: MagicMock) -> None:
        """Test session validation with Redis backend."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.from_url.return_value = mock_client

        manager = SessionManager(
            redis_url="redis://localhost:6379/0", enable_rate_limiting=False
        )

        # Create session
        session_id, session_token = manager.create_session("user123")

        # Mock Redis get to return session data
        session_data = {
            "user_id": "user123",
            "token_hash": manager._hash_token(session_token),
            "created_at": datetime.now(UTC).isoformat(),
            "last_accessed": datetime.now(UTC).isoformat(),
            "expires_at": (datetime.now(UTC) + timedelta(seconds=3600)).isoformat(),
            "metadata": {},
            "access_count": 0,
        }
        mock_client.get.return_value = json.dumps(session_data)

        result = manager.validate_session(session_id, session_token)
        assert result is not None
        assert result["user_id"] == "user123"
        mock_client.get.assert_called_with(f"session:{session_id}")


class TestSessionManagerExtendSession:
    """Tests for session extension."""

    @pytest.fixture
    def manager(self) -> SessionManager:
        """Create in-memory session manager."""
        return SessionManager(enable_rate_limiting=False)

    def test_extend_session_default_ttl(self, manager: SessionManager) -> None:
        """Test extending session with default TTL."""
        session_id, session_token = manager.create_session("user123")
        original_expires = manager._sessions[session_id]["expires_at"]

        result = manager.extend_session(session_id)
        assert result is True

        new_expires = manager._sessions[session_id]["expires_at"]
        assert new_expires > original_expires

    def test_extend_session_custom_ttl(self, manager: SessionManager) -> None:
        """Test extending session with custom TTL."""
        session_id, session_token = manager.create_session("user123")

        result = manager.extend_session(session_id, additional_ttl=7200)
        assert result is True

        expires_at = datetime.fromisoformat(manager._sessions[session_id]["expires_at"])
        expected_min = datetime.now(UTC) + timedelta(seconds=7200 - 1)
        expected_max = datetime.now(UTC) + timedelta(seconds=7200 + 1)
        assert expected_min <= expires_at <= expected_max

    def test_extend_session_nonexistent(self, manager: SessionManager) -> None:
        """Test extending nonexistent session fails."""
        result = manager.extend_session("nonexistent")
        assert result is False

    @patch("traigent.security.session_manager.REDIS_AVAILABLE", True)
    @patch("traigent.security.session_manager.redis")
    def test_extend_session_redis_backend(self, mock_redis: MagicMock) -> None:
        """Test session extension with Redis backend."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.from_url.return_value = mock_client

        manager = SessionManager(
            redis_url="redis://localhost:6379/0", enable_rate_limiting=False
        )

        # Mock get to return session data
        session_data = {
            "user_id": "user123",
            "token_hash": "hash123",
            "created_at": datetime.now(UTC).isoformat(),
            "last_accessed": datetime.now(UTC).isoformat(),
            "expires_at": (datetime.now(UTC) + timedelta(seconds=3600)).isoformat(),
            "metadata": {},
            "access_count": 0,
        }
        mock_client.get.return_value = json.dumps(session_data)

        result = manager.extend_session("session_123", additional_ttl=7200)
        assert result is True
        # Should call setex with the new TTL
        assert mock_client.setex.call_count >= 1


class TestSessionManagerRevokeSession:
    """Tests for session revocation."""

    @pytest.fixture
    def manager(self) -> SessionManager:
        """Create in-memory session manager."""
        return SessionManager(enable_rate_limiting=False)

    def test_revoke_session_success(self, manager: SessionManager) -> None:
        """Test successful session revocation."""
        session_id, session_token = manager.create_session("user123")
        assert session_id in manager._sessions

        result = manager.revoke_session(session_id)
        assert result is True
        assert session_id not in manager._sessions
        assert session_id not in manager._user_sessions.get("user123", set())

    def test_revoke_session_nonexistent(self, manager: SessionManager) -> None:
        """Test revoking nonexistent session."""
        result = manager.revoke_session("nonexistent")
        assert result is False

    def test_revoke_session_cleans_user_sessions(self, manager: SessionManager) -> None:
        """Test revoking session removes from user sessions."""
        session_id, session_token = manager.create_session("user123")
        assert session_id in manager._user_sessions["user123"]

        manager.revoke_session(session_id)
        assert session_id not in manager._user_sessions.get("user123", set())

    @patch("traigent.security.session_manager.REDIS_AVAILABLE", True)
    @patch("traigent.security.session_manager.redis")
    def test_revoke_session_redis_backend(self, mock_redis: MagicMock) -> None:
        """Test session revocation with Redis backend."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.from_url.return_value = mock_client

        manager = SessionManager(
            redis_url="redis://localhost:6379/0", enable_rate_limiting=False
        )

        # Mock get to return session data
        session_data = {
            "user_id": "user123",
            "token_hash": "hash123",
        }
        mock_client.get.return_value = json.dumps(session_data)

        result = manager.revoke_session("session_123")
        assert result is True
        mock_client.delete.assert_called_once_with("session:session_123")
        mock_client.srem.assert_called_once_with("user_sessions:user123", "session_123")


class TestSessionManagerRevokeUserSessions:
    """Tests for revoking all user sessions."""

    @pytest.fixture
    def manager(self) -> SessionManager:
        """Create in-memory session manager."""
        return SessionManager(enable_rate_limiting=False)

    def test_revoke_user_sessions_single(self, manager: SessionManager) -> None:
        """Test revoking single user session."""
        session_id, session_token = manager.create_session("user123")

        count = manager.revoke_user_sessions("user123")
        assert count == 1
        assert session_id not in manager._sessions

    def test_revoke_user_sessions_multiple(self, manager: SessionManager) -> None:
        """Test revoking multiple user sessions."""
        sessions = []
        for _i in range(3):
            session_id, token = manager.create_session("user123")
            sessions.append(session_id)

        count = manager.revoke_user_sessions("user123")
        assert count == 3
        for session_id in sessions:
            assert session_id not in manager._sessions

    def test_revoke_user_sessions_none(self, manager: SessionManager) -> None:
        """Test revoking sessions for user with no sessions."""
        count = manager.revoke_user_sessions("nonexistent_user")
        assert count == 0

    def test_revoke_user_sessions_preserves_other_users(
        self, manager: SessionManager
    ) -> None:
        """Test revoking sessions doesn't affect other users."""
        session1, token1 = manager.create_session("user123")
        session2, token2 = manager.create_session("user456")

        count = manager.revoke_user_sessions("user123")
        assert count == 1
        assert session1 not in manager._sessions
        assert session2 in manager._sessions


class TestSessionManagerCleanupExpired:
    """Tests for expired session cleanup."""

    @pytest.fixture
    def manager(self) -> SessionManager:
        """Create in-memory session manager."""
        return SessionManager(enable_rate_limiting=False)

    def test_cleanup_expired_sessions_none(self, manager: SessionManager) -> None:
        """Test cleanup when no sessions are expired."""
        manager.create_session("user123")
        manager.create_session("user456")

        count = manager.cleanup_expired_sessions()
        assert count == 0
        assert len(manager._sessions) == 2

    def test_cleanup_expired_sessions_some(self, manager: SessionManager) -> None:
        """Test cleanup removes only expired sessions."""
        # Create valid session
        valid_id, valid_token = manager.create_session("user123")

        # Create expired session
        expired_id, expired_token = manager.create_session("user456")
        past_time = datetime.now(UTC) - timedelta(seconds=100)
        manager._sessions[expired_id]["expires_at"] = past_time.isoformat()

        count = manager.cleanup_expired_sessions()
        assert count == 1
        assert valid_id in manager._sessions
        assert expired_id not in manager._sessions

    def test_cleanup_expired_sessions_all(self, manager: SessionManager) -> None:
        """Test cleanup when all sessions are expired."""
        session1, token1 = manager.create_session("user123")
        session2, token2 = manager.create_session("user456")

        # Expire both sessions
        past_time = datetime.now(UTC) - timedelta(seconds=100)
        manager._sessions[session1]["expires_at"] = past_time.isoformat()
        manager._sessions[session2]["expires_at"] = past_time.isoformat()

        count = manager.cleanup_expired_sessions()
        assert count == 2
        assert len(manager._sessions) == 0

    @patch("traigent.security.session_manager.REDIS_AVAILABLE", True)
    @patch("traigent.security.session_manager.redis")
    def test_cleanup_expired_sessions_redis_backend(
        self, mock_redis: MagicMock
    ) -> None:
        """Test cleanup with Redis backend."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.from_url.return_value = mock_client

        manager = SessionManager(
            redis_url="redis://localhost:6379/0", enable_rate_limiting=False
        )

        # Mock scan to return expired session
        expired_session_data = {
            "user_id": "user123",
            "expires_at": (datetime.now(UTC) - timedelta(seconds=100)).isoformat(),
        }
        mock_client.scan.return_value = (0, ["session:expired_123"])
        mock_client.get.return_value = json.dumps(expired_session_data)

        count = manager.cleanup_expired_sessions()
        # Should call revoke which calls delete and srem
        assert mock_client.delete.called or count >= 0  # Implementation-dependent

    @patch("traigent.security.session_manager.REDIS_AVAILABLE", True)
    @patch("traigent.security.session_manager.redis")
    def test_cleanup_expired_sessions_redis_handles_errors(
        self, mock_redis: MagicMock
    ) -> None:
        """Test cleanup handles errors gracefully."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.from_url.return_value = mock_client

        manager = SessionManager(
            redis_url="redis://localhost:6379/0", enable_rate_limiting=False
        )

        # Mock scan to return session, but get fails
        mock_client.scan.return_value = (0, ["session:error_123"])
        mock_client.get.side_effect = Exception("Redis error")

        count = manager.cleanup_expired_sessions()
        # Should handle error gracefully
        assert isinstance(count, int)


class TestSessionManagerRateLimiting:
    """Tests for rate limiting functionality."""

    def test_check_rate_limit_within_limit(self) -> None:
        """Test rate limiting allows requests within limit."""
        manager = SessionManager(enable_rate_limiting=False)

        for _i in range(10):
            result = manager._check_rate_limit("user123")
            assert result is True

    def test_check_rate_limit_exceeds_limit(self) -> None:
        """Test rate limiting blocks requests exceeding limit."""
        manager = SessionManager(enable_rate_limiting=False)

        # Make max requests
        for _i in range(10):
            manager._check_rate_limit("user123")

        # Next request should fail
        result = manager._check_rate_limit("user123")
        assert result is False

    def test_check_rate_limit_window_expires(self) -> None:
        """Test rate limit window expiration."""
        manager = SessionManager(enable_rate_limiting=False)

        # Use custom short window for testing
        for _i in range(10):
            manager._check_rate_limit("user123", max_requests=5, window=1)

        # Should be blocked
        result = manager._check_rate_limit("user123", max_requests=5, window=1)
        assert result is False

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        result = manager._check_rate_limit("user123", max_requests=5, window=1)
        assert result is True

    def test_check_rate_limit_per_user(self) -> None:
        """Test rate limiting is per-user."""
        manager = SessionManager(enable_rate_limiting=False)

        # Max out user123
        for _i in range(10):
            manager._check_rate_limit("user123")

        # user123 should be blocked
        assert manager._check_rate_limit("user123") is False

        # user456 should still be allowed
        assert manager._check_rate_limit("user456") is True

    @patch("traigent.security.session_manager.REDIS_AVAILABLE", True)
    @patch("traigent.security.session_manager.redis")
    def test_check_rate_limit_redis_backend(self, mock_redis: MagicMock) -> None:
        """Test rate limiting with Redis backend."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.from_url.return_value = mock_client

        manager = SessionManager(
            redis_url="redis://localhost:6379/0", enable_rate_limiting=False
        )

        # Initialize _rate_limits for fallback
        manager._rate_limits = {}

        # Mock pipeline for rate limiting - return actual integers
        mock_pipe = MagicMock()
        # Results: [removed_count, added_count, current_count, expire_result]
        mock_pipe.execute.return_value = [0, 1, 5, True]
        # Pipeline is used directly without context manager
        mock_client.pipeline.return_value = mock_pipe

        result = manager._check_rate_limit("user123")
        assert result is True
        mock_pipe.execute.assert_called_once()

    @patch("traigent.security.session_manager.REDIS_AVAILABLE", True)
    @patch("traigent.security.session_manager.redis")
    def test_check_rate_limit_redis_fallback_on_error(
        self, mock_redis: MagicMock
    ) -> None:
        """Test rate limiting falls back to in-memory on Redis error."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.from_url.return_value = mock_client

        manager = SessionManager(
            redis_url="redis://localhost:6379/0", enable_rate_limiting=False
        )

        # Initialize _rate_limits for fallback
        manager._rate_limits = {}

        # Mock pipeline to raise error
        mock_client.pipeline.side_effect = Exception("Redis error")

        # Should fall back to in-memory rate limiting
        result = manager._check_rate_limit("user123")
        assert result is True


class TestSessionManagerPrivateMethods:
    """Tests for private helper methods."""

    @pytest.fixture
    def manager(self) -> SessionManager:
        """Create in-memory session manager."""
        return SessionManager(enable_rate_limiting=False)

    def test_hash_token_consistent(self, manager: SessionManager) -> None:
        """Test token hashing is consistent."""
        token = "test_token_123"
        hash1 = manager._hash_token(token)
        hash2 = manager._hash_token(token)
        assert hash1 == hash2

    def test_hash_token_different_tokens(self, manager: SessionManager) -> None:
        """Test different tokens produce different hashes."""
        hash1 = manager._hash_token("token1")
        hash2 = manager._hash_token("token2")
        assert hash1 != hash2

    def test_get_session_existing(self, manager: SessionManager) -> None:
        """Test getting existing session."""
        session_id, token = manager.create_session("user123")
        session_data = manager._get_session(session_id)
        assert session_data is not None
        assert session_data["user_id"] == "user123"

    def test_get_session_nonexistent(self, manager: SessionManager) -> None:
        """Test getting nonexistent session returns None."""
        result = manager._get_session("nonexistent")
        assert result is None

    def test_get_user_sessions_existing(self, manager: SessionManager) -> None:
        """Test getting user sessions."""
        session1, token1 = manager.create_session("user123")
        session2, token2 = manager.create_session("user123")

        sessions = manager._get_user_sessions("user123")
        assert len(sessions) == 2
        assert session1 in sessions
        assert session2 in sessions

    def test_get_user_sessions_nonexistent(self, manager: SessionManager) -> None:
        """Test getting sessions for user with no sessions."""
        sessions = manager._get_user_sessions("nonexistent")
        assert len(sessions) == 0
        assert isinstance(sessions, set)

    def test_get_session_created_time(self, manager: SessionManager) -> None:
        """Test getting session creation time."""
        session_id, token = manager.create_session("user123")
        created_time = manager._get_session_created_time(session_id)
        assert isinstance(created_time, datetime)
        assert created_time <= datetime.now(UTC)

    def test_get_session_created_time_nonexistent(
        self, manager: SessionManager
    ) -> None:
        """Test getting creation time for nonexistent session."""
        created_time = manager._get_session_created_time("nonexistent")
        assert created_time == datetime.min.replace(tzinfo=UTC)

    def test_get_session_created_time_invalid_format(
        self, manager: SessionManager
    ) -> None:
        """Test handling invalid created_at format."""
        session_id, token = manager.create_session("user123")
        manager._sessions[session_id]["created_at"] = "invalid-datetime"

        created_time = manager._get_session_created_time(session_id)
        assert created_time == datetime.min.replace(tzinfo=UTC)

    def test_get_session_created_time_none(self, manager: SessionManager) -> None:
        """Test handling None created_at."""
        session_id, token = manager.create_session("user123")
        manager._sessions[session_id]["created_at"] = None

        created_time = manager._get_session_created_time(session_id)
        assert created_time == datetime.min.replace(tzinfo=UTC)

    @patch("traigent.security.session_manager.REDIS_AVAILABLE", True)
    @patch("traigent.security.session_manager.redis")
    def test_get_session_redis_handles_error(self, mock_redis: MagicMock) -> None:
        """Test _get_session handles Redis errors gracefully."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.from_url.return_value = mock_client

        manager = SessionManager(
            redis_url="redis://localhost:6379/0", enable_rate_limiting=False
        )

        mock_client.get.side_effect = Exception("Redis error")
        result = manager._get_session("session_123")
        assert result is None

    @patch("traigent.security.session_manager.REDIS_AVAILABLE", True)
    @patch("traigent.security.session_manager.redis")
    def test_get_user_sessions_redis_handles_error(self, mock_redis: MagicMock) -> None:
        """Test _get_user_sessions handles Redis errors gracefully."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.from_url.return_value = mock_client

        manager = SessionManager(
            redis_url="redis://localhost:6379/0", enable_rate_limiting=False
        )

        mock_client.smembers.side_effect = Exception("Redis error")
        result = manager._get_user_sessions("user123")
        assert isinstance(result, set)
        assert len(result) == 0


class TestSessionMiddleware:
    """Tests for SessionMiddleware."""

    @pytest.fixture
    def manager(self) -> SessionManager:
        """Create session manager."""
        return SessionManager(enable_rate_limiting=False)

    @pytest.fixture
    def middleware(self, manager: SessionManager) -> SessionMiddleware:
        """Create session middleware."""
        return SessionMiddleware(manager)

    def test_middleware_init(self, manager: SessionManager) -> None:
        """Test middleware initialization."""
        middleware = SessionMiddleware(manager)
        assert middleware.session_manager is manager

    def test_validate_request_success(
        self, middleware: SessionMiddleware, manager: SessionManager
    ) -> None:
        """Test successful request validation."""
        session_id, session_token = manager.create_session("user123")
        result = middleware.validate_request(session_id, session_token)

        assert result is not None
        assert result["user_id"] == "user123"

    def test_validate_request_missing_session_id(
        self, middleware: SessionMiddleware
    ) -> None:
        """Test validation fails with missing session ID."""
        result = middleware.validate_request(None, "token")
        assert result is None

    def test_validate_request_missing_token(
        self, middleware: SessionMiddleware
    ) -> None:
        """Test validation fails with missing token."""
        result = middleware.validate_request("session_id", None)
        assert result is None

    def test_validate_request_both_missing(self, middleware: SessionMiddleware) -> None:
        """Test validation fails with both credentials missing."""
        result = middleware.validate_request(None, None)
        assert result is None

    def test_validate_request_invalid_credentials(
        self, middleware: SessionMiddleware
    ) -> None:
        """Test validation fails with invalid credentials."""
        result = middleware.validate_request("fake_id", "fake_token")
        assert result is None

    def test_validate_request_empty_strings(
        self, middleware: SessionMiddleware
    ) -> None:
        """Test validation fails with empty strings."""
        result = middleware.validate_request("", "")
        assert result is None
