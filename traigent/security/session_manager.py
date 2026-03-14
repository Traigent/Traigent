"""Session management with Redis backend for production use.

Provides secure session handling with token rotation, rate limiting,
and persistent storage using Redis or fallback to in-memory cache.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security CONC-Quality-Reliability FUNC-SECURITY REQ-SEC-010 SYNC-CloudHybrid

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import Redis, fallback to in-memory if not available
try:
    import redis
    from redis import Redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    if TYPE_CHECKING:
        from typing import Any as Redis
    else:
        Redis = None
    redis = None


class SessionManager:
    """Production-ready session manager with Redis backend."""

    def __init__(
        self,
        redis_url: str | None = None,
        session_ttl: int = 3600,  # 1 hour default
        max_sessions_per_user: int = 5,
        enable_rate_limiting: bool = True,
    ) -> None:
        """Initialize session manager.

        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379/0)
            session_ttl: Session time-to-live in seconds
            max_sessions_per_user: Maximum concurrent sessions per user
            enable_rate_limiting: Enable built-in rate limiting
        """
        self.session_ttl = session_ttl
        self.max_sessions_per_user = max_sessions_per_user
        self.enable_rate_limiting = enable_rate_limiting

        # Initialize storage backend
        self.redis_client: Redis | None = None
        if redis_url and REDIS_AVAILABLE and redis:
            try:
                self.redis_client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
                # Test connection
                self.redis_client.ping()
                self.use_redis = True
                logger.info("Session manager initialized with Redis backend")
            except Exception as e:
                logger.warning(
                    f"Failed to connect to Redis: {e}. Using in-memory storage."
                )
                self.redis_client = None
                self.use_redis = False
        else:
            if redis_url and not REDIS_AVAILABLE:
                logger.warning(
                    "Redis requested but not installed. Using in-memory storage."
                )
            self.redis_client = None
            self.use_redis = False

        # In-memory fallback storage
        if not self.use_redis:
            self._sessions: dict[str, dict[str, Any]] = {}
            self._user_sessions: dict[str, set[str]] = {}
            self._rate_limits: dict[str, list[float]] = {}
            logger.info("Session manager initialized with in-memory backend")

        # Tokens are already high-entropy random values. Use a stable hash so
        # persisted sessions remain valid across restarts and worker boundaries.
        self._token_secret: bytes | None = None

    def create_session(
        self,
        user_id: str,
        metadata: dict[str, Any | None] = None,
        *,
        session_id: str | None = None,
    ) -> tuple[str, str]:
        """Create a new session for a user.

        Args:
            user_id: User identifier
            metadata: Optional session metadata
            session_id: Optional explicit session identifier to use

        Returns:
            Tuple of (session_id, session_token)
        """
        # Validate user_id
        if not user_id or not isinstance(user_id, str):
            raise ValueError("Invalid user_id") from None

        # Check rate limiting
        if self.enable_rate_limiting and not self._check_rate_limit(user_id):
            raise ValueError("Rate limit exceeded for session creation")

        # Check max sessions per user
        existing_sessions = self._get_user_sessions(user_id)
        if len(existing_sessions) >= self.max_sessions_per_user:
            # Remove oldest session
            oldest = min(
                existing_sessions,
                key=lambda sid: self._get_session_created_time(sid),
            )
            self.revoke_session(oldest)

        # Generate secure session ID and token
        session_id = session_id or f"session_{secrets.token_urlsafe(32)}"
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")

        # If a session with this ID already exists, revoke it to avoid collisions
        if self._get_session(session_id):
            self.revoke_session(session_id)

        session_token = secrets.token_urlsafe(48)

        # Hash token for storage
        token_hash = self._hash_token(session_token)

        # Create session data
        now = datetime.now(UTC)
        session_data = {
            "user_id": user_id,
            "token_hash": token_hash,
            "created_at": now.isoformat(),
            "last_accessed": now.isoformat(),
            "expires_at": (now + timedelta(seconds=self.session_ttl)).isoformat(),
            "metadata": metadata or {},
            "access_count": 0,
        }

        # Store session
        if self.use_redis and self.redis_client:
            # Store session data
            self.redis_client.setex(
                f"session:{session_id}",
                self.session_ttl,
                json.dumps(session_data),
            )
            # Track user sessions
            self.redis_client.sadd(f"user_sessions:{user_id}", session_id)
            self.redis_client.expire(f"user_sessions:{user_id}", self.session_ttl * 2)
        else:
            # In-memory storage
            self._sessions[session_id] = session_data
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = set()
            self._user_sessions[user_id].add(session_id)

        logger.info(f"Created session for user {user_id}")
        return session_id, session_token

    def validate_session(
        self, session_id: str, session_token: str
    ) -> dict[str, Any] | None:
        """Validate a session and return session data if valid.

        Args:
            session_id: Session identifier
            session_token: Session token

        Returns:
            Session data if valid, None otherwise
        """
        # Get session data
        session_data = self._get_session(session_id)
        if not session_data:
            return None

        # Check expiration - handle potential None values
        expires_at_str = session_data.get("expires_at")
        if not expires_at_str or not isinstance(expires_at_str, str):
            logger.warning(f"Invalid expires_at for session {session_id}")
            self.revoke_session(session_id)
            return None

        try:
            expires_at = datetime.fromisoformat(expires_at_str)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse expires_at for session {session_id}: {e}")
            self.revoke_session(session_id)
            return None

        if datetime.now(UTC) > expires_at:
            self.revoke_session(session_id)
            return None

        # Verify token - handle potential None values
        stored_token_hash = session_data.get("token_hash")
        if not stored_token_hash or not isinstance(stored_token_hash, str):
            logger.warning(f"Invalid token_hash for session {session_id}")
            return None

        token_hash = self._hash_token(session_token)
        if not hmac.compare_digest(stored_token_hash, token_hash):
            logger.warning(f"Invalid token for session {session_id}")
            return None

        # Update last accessed time and access count - handle potential None values
        current_access_count = session_data.get("access_count", 0)
        if not isinstance(current_access_count, int):
            current_access_count = 0

        session_data["last_accessed"] = datetime.now(UTC).isoformat()
        session_data["access_count"] = current_access_count + 1

        # Update storage
        if self.use_redis and self.redis_client:
            self.redis_client.setex(
                f"session:{session_id}",
                self.session_ttl,
                json.dumps(session_data),
            )
        else:
            self._sessions[session_id] = session_data

        return {
            "user_id": session_data["user_id"],
            "metadata": session_data["metadata"],
            "created_at": session_data["created_at"],
            "access_count": session_data["access_count"],
        }

    def extend_session(
        self, session_id: str, additional_ttl: int | None = None
    ) -> bool:
        """Extend a session's expiration time.

        Args:
            session_id: Session identifier
            additional_ttl: Additional time in seconds (default: session_ttl)

        Returns:
            True if session extended, False otherwise
        """
        session_data = self._get_session(session_id)
        if not session_data:
            return False

        additional_ttl = additional_ttl or self.session_ttl
        new_expires = datetime.now(UTC) + timedelta(seconds=additional_ttl)
        session_data["expires_at"] = new_expires.isoformat()

        if self.use_redis and self.redis_client:
            self.redis_client.setex(
                f"session:{session_id}",
                additional_ttl,
                json.dumps(session_data),
            )
        else:
            self._sessions[session_id] = session_data

        logger.info(f"Extended session {session_id} by {additional_ttl} seconds")
        return True

    def revoke_session(self, session_id: str) -> bool:
        """Revoke a session.

        Args:
            session_id: Session identifier

        Returns:
            True if session revoked, False if not found
        """
        session_data = self._get_session(session_id)
        if not session_data:
            return False

        user_id = session_data["user_id"]

        if self.use_redis and self.redis_client:
            self.redis_client.delete(f"session:{session_id}")
            self.redis_client.srem(f"user_sessions:{user_id}", session_id)
        else:
            del self._sessions[session_id]
            if user_id in self._user_sessions:
                self._user_sessions[user_id].discard(session_id)

        logger.info(f"Revoked session {session_id}")
        return True

    def revoke_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of sessions revoked
        """
        sessions = self._get_user_sessions(user_id)
        count = 0
        for session_id in sessions:
            if self.revoke_session(session_id):
                count += 1

        logger.info(f"Revoked {count} sessions for user {user_id}")
        return count

    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions from storage.

        Returns:
            Number of sessions cleaned up
        """
        count = 0
        now = datetime.now(UTC)

        if self.use_redis and self.redis_client:
            # Redis handles expiration automatically
            # This is just for manual cleanup if needed
            cursor = 0
            while True:
                cursor, keys = self.redis_client.scan(
                    cursor, match="session:*", count=100
                )
                for key in keys:
                    try:
                        data = self.redis_client.get(key)
                        if data:
                            session_data = json.loads(data)
                            expires_at_str = session_data.get("expires_at")
                            if expires_at_str and isinstance(expires_at_str, str):
                                expires_at = datetime.fromisoformat(expires_at_str)
                                if now > expires_at:
                                    session_id = key.replace("session:", "")
                                    self.revoke_session(session_id)
                                    count += 1
                    except Exception as e:
                        logger.warning(f"Error cleaning up session {key}: {e}")

                if cursor == 0:
                    break
        else:
            # In-memory cleanup
            expired = []
            for session_id, session_data in self._sessions.items():
                expires_at = datetime.fromisoformat(session_data["expires_at"])
                if now > expires_at:
                    expired.append(session_id)

            for session_id in expired:
                self.revoke_session(session_id)
                count += 1

        if count > 0:
            logger.info(f"Cleaned up {count} expired sessions")
        return count

    # Private helper methods

    def _hash_token(self, token: str) -> str:
        """Hash a session token for secure storage."""
        token_bytes = token.encode("utf-8")
        if self._token_secret:
            return hmac.new(
                self._token_secret,
                token_bytes,
                hashlib.sha256,
            ).hexdigest()
        return hashlib.sha256(token_bytes).hexdigest()

    def _get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get session data from storage."""
        if self.use_redis and self.redis_client:
            try:
                data = self.redis_client.get(f"session:{session_id}")
                if data:
                    return cast(dict[str, Any], json.loads(data))
            except Exception as e:
                logger.warning(f"Failed to get session {session_id} from Redis: {e}")
                return None
        else:
            return self._sessions.get(session_id)
        return None

    def _get_user_sessions(self, user_id: str) -> set[str]:
        """Get all session IDs for a user."""
        if self.use_redis and self.redis_client:
            try:
                members = self.redis_client.smembers(f"user_sessions:{user_id}")
                return members or set()
            except Exception as e:
                logger.warning(f"Failed to get user sessions for {user_id}: {e}")
                return set()
        else:
            return self._user_sessions.get(user_id, set()).copy()

    def _get_session_created_time(self, session_id: str) -> datetime:
        """Get session creation time."""
        session_data = self._get_session(session_id)
        if session_data:
            created_at_str = session_data.get("created_at")
            if created_at_str and isinstance(created_at_str, str):
                try:
                    return datetime.fromisoformat(created_at_str)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse created_at for session {session_id}: {e}"
                    )
        return datetime.min.replace(tzinfo=UTC)

    def _check_rate_limit(
        self, user_id: str, max_requests: int = 10, window: int = 60
    ) -> bool:
        """Check if user is within rate limits.

        Args:
            user_id: User identifier
            max_requests: Maximum requests in window
            window: Time window in seconds

        Returns:
            True if within limits, False otherwise
        """
        now = time.time()
        cutoff = now - window

        if self.use_redis and self.redis_client:
            # Use Redis for rate limiting
            try:
                key = f"rate_limit:{user_id}"
                pipe = self.redis_client.pipeline()
                pipe.zremrangebyscore(key, 0, cutoff)
                pipe.zadd(key, {str(now): now})
                pipe.zcard(key)
                pipe.expire(key, window)
                results = pipe.execute()
                count = results[2]
                return cast(bool, count <= max_requests)
            except Exception as e:
                logger.warning(f"Redis rate limiting failed for {user_id}: {e}")
                # Fall back to in-memory rate limiting
                # Continue to in-memory logic below

        # In-memory rate limiting (used when Redis is not available or fails)
        if user_id not in self._rate_limits:
            self._rate_limits[user_id] = []

        # Remove old entries
        self._rate_limits[user_id] = [
            t for t in self._rate_limits[user_id] if t > cutoff
        ]

        # Check count
        if len(self._rate_limits[user_id]) >= max_requests:
            return False

        # Add new entry
        self._rate_limits[user_id].append(now)
        return True


class SessionMiddleware:
    """Middleware for session validation in web applications."""

    def __init__(self, session_manager: SessionManager) -> None:
        """Initialize middleware with a session manager."""
        self.session_manager = session_manager

    def validate_request(
        self, session_id: str | None, session_token: str | None
    ) -> dict[str, Any] | None:
        """Validate a request's session credentials.

        Args:
            session_id: Session ID from request
            session_token: Session token from request

        Returns:
            User session data if valid, None otherwise
        """
        if not session_id or not session_token:
            return None

        return self.session_manager.validate_session(session_id, session_token)
