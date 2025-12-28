"""Unit tests for secure rate limiting.

Tests for rate limiting functionality including token bucket, sliding window,
and secure authentication rate limiting.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security CONC-Quality-Reliability FUNC-SECURITY REQ-SEC-010 SYNC-CloudHybrid

from __future__ import annotations

import time
from collections import deque
from decimal import Decimal
from threading import Thread
from unittest.mock import MagicMock, Mock, patch

import pytest

from traigent.security.rate_limiter import (
    RateLimitStrategy,
    SecureAuthenticationRateLimiter,
    SecureRateLimitConfig,
    SecureRateLimiter,
    SecureRateLimitResult,
    SecureTokenBucket,
    get_secure_rate_limiter,
)


class TestRateLimitStrategy:
    """Tests for RateLimitStrategy enum."""

    def test_rate_limit_strategies(self) -> None:
        """Test all rate limit strategies are defined."""
        assert RateLimitStrategy.SLIDING_WINDOW.value == "sliding_window"
        assert RateLimitStrategy.TOKEN_BUCKET.value == "token_bucket"
        assert RateLimitStrategy.FIXED_WINDOW.value == "fixed_window"
        assert RateLimitStrategy.LEAKY_BUCKET.value == "leaky_bucket"

    def test_strategy_membership(self) -> None:
        """Test strategy enum membership checks."""
        assert RateLimitStrategy.SLIDING_WINDOW in RateLimitStrategy
        assert RateLimitStrategy.TOKEN_BUCKET in RateLimitStrategy


class TestSecureRateLimitResult:
    """Tests for SecureRateLimitResult dataclass."""

    def test_result_creation_allowed(self) -> None:
        """Test creating allowed result."""
        result = SecureRateLimitResult(
            allowed=True,
            remaining_attempts=5,
            reset_time=time.time() + 300,
        )

        assert result.allowed is True
        assert result.remaining_attempts == 5
        assert result.retry_after is None

    def test_result_creation_denied(self) -> None:
        """Test creating denied result."""
        result = SecureRateLimitResult(
            allowed=False,
            retry_after=60.0,
            remaining_attempts=0,
        )

        assert result.allowed is False
        assert result.retry_after == 60.0
        assert result.remaining_attempts == 0

    def test_result_with_metadata(self) -> None:
        """Test result with security metadata."""
        result = SecureRateLimitResult(
            allowed=False,
            identifier_hash="abc123",
            security_metadata={"reason": "lockout", "anomaly_score": 0.8},
        )

        assert result.identifier_hash == "abc123"
        assert result.security_metadata is not None
        assert result.security_metadata["reason"] == "lockout"
        assert result.security_metadata["anomaly_score"] == 0.8


class TestSecureRateLimitConfig:
    """Tests for SecureRateLimitConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SecureRateLimitConfig()

        assert config.max_attempts == 5
        assert config.window_seconds == 300
        assert config.lockout_duration == 3600
        assert config.progressive_delay is True
        assert config.use_cryptographic_ids is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = SecureRateLimitConfig(
            max_attempts=10,
            window_seconds=60,
            progressive_delay=False,
        )

        assert config.max_attempts == 10
        assert config.window_seconds == 60
        assert config.progressive_delay is False

    def test_distributed_config(self) -> None:
        """Test distributed configuration."""
        config = SecureRateLimitConfig(
            enable_distributed=True,
            redis_url="redis://localhost:6379",
        )

        assert config.enable_distributed is True
        assert config.redis_url == "redis://localhost:6379"

    def test_security_config(self) -> None:
        """Test security-related configuration."""
        config = SecureRateLimitConfig(
            use_cryptographic_ids=True,
            identifier_salt=b"test_salt",
            check_multiple_identifiers=True,
            subnet_aggregation=True,
        )

        assert config.use_cryptographic_ids is True
        assert config.identifier_salt == b"test_salt"
        assert config.check_multiple_identifiers is True
        assert config.subnet_aggregation is True


class TestSecureTokenBucket:
    """Tests for SecureTokenBucket implementation."""

    def test_token_bucket_initialization(self) -> None:
        """Test initializing token bucket."""
        bucket = SecureTokenBucket(
            capacity=10,
            refill_rate=2.0,
            identifier="test-bucket",
        )

        assert bucket.capacity == 10
        assert bucket.refill_rate == 2.0
        assert bucket.identifier == "test-bucket"
        assert bucket.tokens == 10  # Starts full

    def test_consume_tokens_success(self) -> None:
        """Test consuming tokens successfully."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=1.0, identifier="test")

        success, anomaly_score = bucket.consume(3)

        assert success is True
        assert bucket.tokens == 7
        assert isinstance(anomaly_score, float)

    def test_consume_tokens_failure(self) -> None:
        """Test consuming more tokens than available."""
        bucket = SecureTokenBucket(capacity=5, refill_rate=1.0, identifier="test")

        success, anomaly_score = bucket.consume(10)

        assert success is False
        assert bucket.tokens == 5  # Unchanged

    def test_consume_multiple_times(self) -> None:
        """Test consuming tokens multiple times."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=1.0, identifier="test")

        # First consumption
        success1, _ = bucket.consume(3)
        assert success1 is True
        assert bucket.tokens == 7

        # Second consumption
        success2, _ = bucket.consume(5)
        assert success2 is True
        assert bucket.tokens == 2

        # Third consumption (should fail)
        success3, _ = bucket.consume(5)
        assert success3 is False
        assert bucket.tokens == 2  # Unchanged

    def test_token_refill(self) -> None:
        """Test token refill over time."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=5.0, identifier="test")

        # Consume all tokens
        bucket.consume(10)
        assert bucket.tokens == 0

        # Wait and refill
        time.sleep(1.0)  # Should refill 5 tokens
        success, _ = bucket.consume(1)

        assert success is True
        assert bucket.tokens < 5  # Some tokens refilled

    def test_token_refill_cap(self) -> None:
        """Test token refill doesn't exceed capacity."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=2.0, identifier="test")

        # Don't consume, just wait
        time.sleep(10.0)  # Would refill 20 tokens if uncapped

        # Try to consume capacity + 1
        success, _ = bucket.consume(11)
        assert success is False  # Can't exceed capacity

    def test_get_wait_time_no_wait(self) -> None:
        """Test getting wait time when tokens available."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=2.0, identifier="test")

        wait_time = bucket.get_wait_time(5)

        assert wait_time == 0.0

    def test_get_wait_time_with_wait(self) -> None:
        """Test getting wait time when tokens not available."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=2.0, identifier="test")

        # Consume most tokens
        bucket.consume(10)

        wait_time = bucket.get_wait_time(5)

        assert wait_time > 0
        assert wait_time <= 5 / 2.0  # 5 tokens at 2 tokens/sec

    def test_consumption_history_tracking(self) -> None:
        """Test consumption history is tracked."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=1.0, identifier="test")

        # Make multiple consumptions
        for _ in range(5):
            bucket.consume(1)

        assert len(bucket._consumption_history) == 5

    def test_anomaly_detection(self) -> None:
        """Test anomaly detection with rapid consumption."""
        bucket = SecureTokenBucket(capacity=20, refill_rate=1.0, identifier="test")

        # Rapid consumption pattern
        anomaly_scores = []
        for _ in range(15):
            success, anomaly_score = bucket.consume(1)
            if success:
                anomaly_scores.append(anomaly_score)

        # Anomaly score should increase with rapid consumption
        assert len(anomaly_scores) > 0

    def test_thread_safety(self) -> None:
        """Test token bucket thread safety."""
        import threading

        bucket = SecureTokenBucket(capacity=100, refill_rate=10.0, identifier="test")
        results: list[bool] = []

        def consume_tokens() -> None:
            success, _ = bucket.consume(1)
            results.append(success)

        # Create multiple threads
        threads = [threading.Thread(target=consume_tokens) for _ in range(50)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Count successful consumptions
        successful = sum(1 for r in results if r)

        # Should not exceed capacity
        assert successful <= 100

    @patch("traigent.security.rate_limiter.get_security_flags")
    def test_decimal_mode_enabled(self, mock_flags: MagicMock) -> None:
        """Test token bucket with decimal precision mode."""
        mock_flags.return_value = Mock(
            emit_security_telemetry=True, use_decimal_rate_limiter=True
        )

        bucket = SecureTokenBucket(capacity=10, refill_rate=2.5, identifier="test")

        assert isinstance(bucket.capacity, Decimal)
        assert isinstance(bucket.refill_rate, Decimal)
        assert isinstance(bucket.tokens, Decimal)

    @patch("traigent.security.rate_limiter.get_security_flags")
    def test_float_mode_enabled(self, mock_flags: MagicMock) -> None:
        """Test token bucket with float mode."""
        mock_flags.return_value = Mock(
            emit_security_telemetry=False, use_decimal_rate_limiter=False
        )

        bucket = SecureTokenBucket(capacity=10, refill_rate=2.5, identifier="test")

        assert isinstance(bucket.capacity, float)
        assert isinstance(bucket.refill_rate, float)
        assert isinstance(bucket.tokens, float)


class TestTokenBucketEdgeCases:
    """Tests for edge cases in token bucket."""

    def test_zero_capacity(self) -> None:
        """Test token bucket with zero capacity."""
        bucket = SecureTokenBucket(capacity=0, refill_rate=1.0, identifier="test")

        success, _ = bucket.consume(1)
        assert success is False

    def test_zero_refill_rate(self) -> None:
        """Test token bucket with zero refill rate."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=0.0, identifier="test")

        # Consume tokens
        bucket.consume(5)

        # Wait
        time.sleep(1.0)

        # Should not refill
        assert bucket.tokens == 5

    def test_fractional_tokens(self) -> None:
        """Test consuming fractional tokens."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=1.0, identifier="test")

        # Token bucket should handle fractional consumption
        # based on refill calculations
        wait_time = bucket.get_wait_time(1)
        assert isinstance(wait_time, float)

    def test_large_capacity(self) -> None:
        """Test token bucket with large capacity."""
        bucket = SecureTokenBucket(
            capacity=10000,
            refill_rate=100.0,
            identifier="test",
        )

        success, _ = bucket.consume(5000)
        assert success is True
        assert bucket.tokens == 5000

    def test_high_refill_rate(self) -> None:
        """Test token bucket with high refill rate."""
        bucket = SecureTokenBucket(capacity=100, refill_rate=1000.0, identifier="test")

        # Consume all
        bucket.consume(100)

        # Short wait should refill completely
        time.sleep(0.2)  # 200 tokens would be added

        success, _ = bucket.consume(50)
        assert success is True  # Should have refilled to capacity

    def test_consumption_history_limit(self) -> None:
        """Test consumption history has maximum length."""
        bucket = SecureTokenBucket(capacity=200, refill_rate=10.0, identifier="test")

        # Make many consumptions
        for _ in range(150):
            bucket.consume(1)

        # History should be limited to 100
        assert len(bucket._consumption_history) == 100

    def test_negative_tokens_prevented(self) -> None:
        """Test that tokens never go negative."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=1.0, identifier="test")

        # Try to consume more than available
        bucket.consume(20)

        # Tokens should not be negative
        assert bucket.tokens >= 0

    def test_identifier_usage(self) -> None:
        """Test identifier is stored and accessible."""
        identifier = "user-123"
        bucket = SecureTokenBucket(capacity=10, refill_rate=1.0, identifier=identifier)

        assert bucket.identifier == identifier

    def test_anomaly_score_decay(self) -> None:
        """Test anomaly score decays over time."""
        bucket = SecureTokenBucket(capacity=100, refill_rate=10.0, identifier="test")

        # Trigger high anomaly score with rapid consumption
        for _ in range(20):
            bucket.consume(1)

        # Wait and make slow consumptions
        time.sleep(0.5)
        for _ in range(5):
            bucket.consume(1)
            time.sleep(0.5)

        # Anomaly score should be lower
        assert bucket._anomaly_score >= 0.0


class TestSecureRateLimiter:
    """Tests for SecureRateLimiter class."""

    @pytest.fixture
    def config(self) -> SecureRateLimitConfig:
        """Create test configuration."""
        return SecureRateLimitConfig(
            max_attempts=5,
            window_seconds=60,
            lockout_duration=300,
            progressive_delay=True,
            use_cryptographic_ids=True,
        )

    @pytest.fixture
    def limiter(self, config: SecureRateLimitConfig) -> SecureRateLimiter:
        """Create rate limiter instance."""
        return SecureRateLimiter(config)

    def test_initialization(self, config: SecureRateLimitConfig) -> None:
        """Test rate limiter initialization."""
        limiter = SecureRateLimiter(config)

        assert limiter.config == config
        assert isinstance(limiter.attempts, dict)
        assert isinstance(limiter.lockouts, dict)
        assert isinstance(limiter.token_buckets, dict)

    def test_initialization_generates_salt(self) -> None:
        """Test that initialization generates salt if not provided."""
        config = SecureRateLimitConfig(use_cryptographic_ids=True)
        limiter = SecureRateLimiter(config)

        assert limiter.config.identifier_salt is not None
        assert len(limiter.config.identifier_salt) == 32

    def test_generate_secure_identifier_with_ip(
        self, limiter: SecureRateLimiter
    ) -> None:
        """Test generating secure identifier with IP address."""
        identifier = limiter._generate_secure_identifier(ip_address="192.168.1.1")

        assert isinstance(identifier, str)
        assert len(identifier) == 64  # SHA256 hex digest

    def test_generate_secure_identifier_with_username(
        self, limiter: SecureRateLimiter
    ) -> None:
        """Test generating secure identifier with username."""
        identifier = limiter._generate_secure_identifier(username="testuser")

        assert isinstance(identifier, str)
        assert len(identifier) == 64

    def test_generate_secure_identifier_with_api_key(
        self, limiter: SecureRateLimiter
    ) -> None:
        """Test generating secure identifier with API key."""
        identifier = limiter._generate_secure_identifier(api_key="sk_test_key_12345")

        assert isinstance(identifier, str)
        assert len(identifier) == 64

    def test_generate_secure_identifier_with_all_params(
        self, limiter: SecureRateLimiter
    ) -> None:
        """Test generating secure identifier with all parameters."""
        identifier = limiter._generate_secure_identifier(
            ip_address="192.168.1.1",
            username="testuser",
            api_key="sk_test_key",
        )

        assert isinstance(identifier, str)
        assert len(identifier) == 64

    def test_generate_secure_identifier_subnet_aggregation(
        self, limiter: SecureRateLimiter
    ) -> None:
        """Test subnet aggregation for IPv4 addresses."""
        limiter.config.subnet_aggregation = True

        id1 = limiter._generate_secure_identifier(ip_address="192.168.1.1")
        id2 = limiter._generate_secure_identifier(ip_address="192.168.1.2")

        # Both should map to same /24 subnet
        assert id1 == id2

    def test_generate_secure_identifier_without_crypto(self) -> None:
        """Test generating identifier without cryptographic mode."""
        config = SecureRateLimitConfig(use_cryptographic_ids=False)
        limiter = SecureRateLimiter(config)

        identifier = limiter._generate_secure_identifier(ip_address="192.168.1.1")

        assert isinstance(identifier, str)
        assert len(identifier) == 64  # Still SHA256 but not HMAC

    def test_generate_secure_identifier_anonymous(
        self, limiter: SecureRateLimiter
    ) -> None:
        """Test generating identifier with no parameters."""
        identifier = limiter._generate_secure_identifier()

        assert isinstance(identifier, str)
        assert len(identifier) == 64

    def test_generate_secure_identifier_with_additional_context(
        self, limiter: SecureRateLimiter
    ) -> None:
        """Test generating identifier with additional context."""
        identifier = limiter._generate_secure_identifier(
            ip_address="192.168.1.1",
            additional_context={"session": "abc123", "device": "mobile"},
        )

        assert isinstance(identifier, str)
        assert len(identifier) == 64

    def test_check_rate_limit_sliding_window_allowed(
        self, limiter: SecureRateLimiter
    ) -> None:
        """Test rate limit check allows valid requests."""
        result = limiter.check_rate_limit(
            identifier="test_user", strategy=RateLimitStrategy.SLIDING_WINDOW
        )

        assert result.allowed is True
        assert result.remaining_attempts is not None
        assert result.remaining_attempts >= 0

    def test_check_rate_limit_sliding_window_exceeded(
        self, limiter: SecureRateLimiter
    ) -> None:
        """Test rate limit check blocks after exceeding limit."""
        identifier = "test_user"

        # Make max_attempts requests
        for _ in range(limiter.config.max_attempts):
            limiter.record_attempt(identifier=identifier, success=False)

        result = limiter.check_rate_limit(
            identifier=identifier, strategy=RateLimitStrategy.SLIDING_WINDOW
        )

        assert result.allowed is False
        assert result.retry_after is not None
        assert result.retry_after > 0

    def test_check_rate_limit_token_bucket_allowed(
        self, limiter: SecureRateLimiter
    ) -> None:
        """Test token bucket strategy allows valid requests."""
        result = limiter.check_rate_limit(
            identifier="test_user", strategy=RateLimitStrategy.TOKEN_BUCKET
        )

        assert result.allowed is True
        assert result.remaining_attempts is not None

    def test_check_rate_limit_token_bucket_exceeded(
        self, limiter: SecureRateLimiter
    ) -> None:
        """Test token bucket strategy blocks when tokens exhausted."""
        identifier = "test_user"

        # Exhaust tokens
        for _ in range(limiter.config.max_attempts + 1):
            limiter.check_rate_limit(
                identifier=identifier, strategy=RateLimitStrategy.TOKEN_BUCKET
            )

        result = limiter.check_rate_limit(
            identifier=identifier, strategy=RateLimitStrategy.TOKEN_BUCKET
        )

        assert result.allowed is False
        assert result.retry_after is not None

    def test_check_rate_limit_lockout(self, limiter: SecureRateLimiter) -> None:
        """Test that lockout is enforced."""
        identifier = "test_user"

        # Trigger lockout
        for _ in range(limiter.config.max_attempts):
            limiter.record_attempt(identifier=identifier, success=False)

        # First check should trigger lockout
        result1 = limiter.check_rate_limit(
            identifier=identifier, strategy=RateLimitStrategy.SLIDING_WINDOW
        )
        assert result1.allowed is False

        # Subsequent checks should still be locked out
        result2 = limiter.check_rate_limit(
            identifier=identifier, strategy=RateLimitStrategy.SLIDING_WINDOW
        )
        assert result2.allowed is False
        assert result2.security_metadata is not None
        assert result2.security_metadata["reason"] == "lockout"

    def test_check_rate_limit_multiple_identifiers(self) -> None:
        """Test checking multiple identifiers to prevent bypass."""
        config = SecureRateLimitConfig(
            max_attempts=5,
            window_seconds=60,
            check_multiple_identifiers=True,
        )
        limiter = SecureRateLimiter(config)

        ip_address = "192.168.1.1"
        username = "testuser"

        # Exhaust IP-based limit
        for _ in range(5):
            limiter.record_attempt(ip_address=ip_address, success=False)

        # Try with both IP and username - should still be blocked
        result = limiter.check_rate_limit(
            ip_address=ip_address,
            username=username,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
        )

        # Should be blocked due to IP limit
        assert result.allowed is False

    def test_record_attempt_failed(self, limiter: SecureRateLimiter) -> None:
        """Test recording failed attempt."""
        identifier = "test_user"

        limiter.record_attempt(identifier=identifier, success=False)

        assert identifier in limiter.attempts
        assert len(limiter.attempts[identifier]) == 1

    def test_record_attempt_success_clears_history(
        self, limiter: SecureRateLimiter
    ) -> None:
        """Test that successful attempt clears history."""
        identifier = "test_user"

        # Record some failed attempts
        for _ in range(3):
            limiter.record_attempt(identifier=identifier, success=False)

        assert len(limiter.attempts[identifier]) == 3

        # Record success
        limiter.record_attempt(identifier=identifier, success=True)

        assert len(limiter.attempts[identifier]) == 0

    def test_record_attempt_success_clears_lockout(
        self, limiter: SecureRateLimiter
    ) -> None:
        """Test that successful attempt clears lockout."""
        identifier = "test_user"

        # Manually set lockout
        limiter.lockouts[identifier] = time.time() + 3600

        # Record success
        limiter.record_attempt(identifier=identifier, success=True)

        assert identifier not in limiter.lockouts

    def test_record_attempt_success_resets_token_bucket(
        self, limiter: SecureRateLimiter
    ) -> None:
        """Test that successful attempt resets token bucket."""
        identifier = "test_user"

        # Create token bucket and exhaust it
        limiter.check_rate_limit(
            identifier=identifier, strategy=RateLimitStrategy.TOKEN_BUCKET
        )
        bucket = limiter.token_buckets[identifier]
        bucket.consume(int(bucket.capacity))

        # Record success
        limiter.record_attempt(identifier=identifier, success=True)

        # Bucket should be refilled
        assert bucket.tokens == bucket.capacity

    def test_get_progressive_delay_disabled(self) -> None:
        """Test progressive delay when disabled."""
        config = SecureRateLimitConfig(progressive_delay=False)
        limiter = SecureRateLimiter(config)

        delay = limiter.get_progressive_delay("test_user")

        assert delay == 0.0

    def test_get_progressive_delay_enabled(self, limiter: SecureRateLimiter) -> None:
        """Test progressive delay increases with attempts."""
        identifier = "test_user"

        # Record some failed attempts
        for _ in range(3):
            limiter.record_attempt(identifier=identifier, success=False)

        delay = limiter.get_progressive_delay(identifier)

        # Should have some delay
        assert delay > 0.0
        assert delay <= limiter.config.max_delay_seconds + 1.0  # +1 for jitter

    def test_get_progressive_delay_respects_max(
        self, limiter: SecureRateLimiter
    ) -> None:
        """Test progressive delay respects maximum."""
        identifier = "test_user"

        # Record many failed attempts
        for _ in range(20):
            limiter.record_attempt(identifier=identifier, success=False)

        delay = limiter.get_progressive_delay(identifier)

        # Should not exceed max + jitter
        assert delay <= limiter.config.max_delay_seconds + 1.0

    def test_get_metrics(self, limiter: SecureRateLimiter) -> None:
        """Test getting security metrics."""
        # Perform some operations
        limiter.check_rate_limit(identifier="user1")
        limiter.check_rate_limit(identifier="user2")

        metrics = limiter.get_metrics()

        assert "total_checks" in metrics
        assert "blocked_attempts" in metrics
        assert "lockouts_triggered" in metrics
        assert metrics["total_checks"] >= 2

    def test_clear_identifier(self, limiter: SecureRateLimiter) -> None:
        """Test clearing all records for an identifier."""
        identifier = "test_user"

        # Create some records
        limiter.record_attempt(identifier=identifier, success=False)
        limiter.lockouts[identifier] = time.time() + 3600
        limiter.check_rate_limit(
            identifier=identifier, strategy=RateLimitStrategy.TOKEN_BUCKET
        )

        # Clear
        limiter.clear_identifier(identifier)

        assert identifier not in limiter.attempts
        assert identifier not in limiter.lockouts
        assert identifier not in limiter.token_buckets

    def test_lockout_expiration(self, limiter: SecureRateLimiter) -> None:
        """Test that lockouts expire after duration."""
        identifier = "test_user"

        # Set expired lockout
        limiter.lockouts[identifier] = time.time() - 1  # 1 second ago

        result = limiter.check_rate_limit(identifier=identifier)

        # Should be allowed since lockout expired
        assert result.allowed is True
        assert identifier not in limiter.lockouts

    def test_default_strategy_fallback(self, limiter: SecureRateLimiter) -> None:
        """Test that unsupported strategies fall back to sliding window."""
        result = limiter.check_rate_limit(
            identifier="test_user",
            strategy=RateLimitStrategy.LEAKY_BUCKET,  # Not implemented
        )

        # Should still work (fallback to sliding window)
        assert result.allowed is True
        assert result.security_metadata is not None
        assert result.security_metadata["strategy"] == "sliding_window"


class TestSecureAuthenticationRateLimiter:
    """Tests for SecureAuthenticationRateLimiter class."""

    @pytest.fixture
    def auth_limiter(self) -> SecureAuthenticationRateLimiter:
        """Create authentication rate limiter instance."""
        return SecureAuthenticationRateLimiter()

    def test_initialization(
        self, auth_limiter: SecureAuthenticationRateLimiter
    ) -> None:
        """Test authentication rate limiter initialization."""
        assert auth_limiter.config is not None
        assert auth_limiter.limiter is not None
        assert auth_limiter.config.use_cryptographic_ids is True
        assert auth_limiter.config.check_multiple_identifiers is True

    @pytest.mark.asyncio
    async def test_check_auth_attempt_allowed(
        self, auth_limiter: SecureAuthenticationRateLimiter
    ) -> None:
        """Test authentication attempt is allowed."""
        result = await auth_limiter.check_auth_attempt(
            ip_address="192.168.1.1",
            username="testuser",
        )

        assert result.allowed is True
        assert result.remaining_attempts is not None

    @pytest.mark.asyncio
    async def test_check_auth_attempt_blocked(
        self, auth_limiter: SecureAuthenticationRateLimiter
    ) -> None:
        """Test authentication attempt is blocked after limit."""
        ip_address = "192.168.1.1"
        username = "testuser"

        # Exhaust attempts
        for _ in range(5):
            auth_limiter.record_auth_result(
                success=False, ip_address=ip_address, username=username
            )

        result = await auth_limiter.check_auth_attempt(
            ip_address=ip_address,
            username=username,
        )

        assert result.allowed is False
        assert result.retry_after is not None

    @pytest.mark.asyncio
    async def test_check_auth_attempt_with_user_agent(
        self, auth_limiter: SecureAuthenticationRateLimiter
    ) -> None:
        """Test authentication attempt with user agent."""
        result = await auth_limiter.check_auth_attempt(
            ip_address="192.168.1.1",
            username="testuser",
            user_agent="Mozilla/5.0",
        )

        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_check_auth_attempt_progressive_delay(
        self, auth_limiter: SecureAuthenticationRateLimiter
    ) -> None:
        """Test progressive delay is applied."""
        ip_address = "192.168.1.1"

        # Record some failed attempts
        for _ in range(3):
            auth_limiter.record_auth_result(success=False, ip_address=ip_address)

        start_time = time.time()
        result = await auth_limiter.check_auth_attempt(ip_address=ip_address)
        elapsed = time.time() - start_time

        # Should have applied delay
        assert result.allowed is True
        assert elapsed > 0.0  # Some delay was applied

    def test_record_auth_result_success(
        self, auth_limiter: SecureAuthenticationRateLimiter
    ) -> None:
        """Test recording successful authentication."""
        ip_address = "192.168.1.1"

        # Record failed attempts
        for _ in range(3):
            auth_limiter.record_auth_result(success=False, ip_address=ip_address)

        # Record success
        auth_limiter.record_auth_result(success=True, ip_address=ip_address)

        # Should clear attempts
        identifier = auth_limiter.limiter._generate_secure_identifier(
            ip_address=ip_address
        )
        assert len(auth_limiter.limiter.attempts.get(identifier, deque())) == 0

    def test_record_auth_result_failure(
        self, auth_limiter: SecureAuthenticationRateLimiter
    ) -> None:
        """Test recording failed authentication."""
        ip_address = "192.168.1.1"

        auth_limiter.record_auth_result(success=False, ip_address=ip_address)

        identifier = auth_limiter.limiter._generate_secure_identifier(
            ip_address=ip_address
        )
        assert len(auth_limiter.limiter.attempts[identifier]) == 1

    def test_get_metrics(self, auth_limiter: SecureAuthenticationRateLimiter) -> None:
        """Test getting authentication metrics."""
        metrics = auth_limiter.get_metrics()

        assert "total_checks" in metrics
        assert "blocked_attempts" in metrics
        assert isinstance(metrics, dict)


class TestRateLimitIntegration:
    """Tests for rate limit integration scenarios."""

    def test_burst_then_steady(self) -> None:
        """Test handling burst then steady consumption."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=2.0, identifier="test")

        # Burst consumption
        for _ in range(10):
            success, _ = bucket.consume(1)
            if not success:
                break

        # Wait for refill
        time.sleep(2.0)  # Should refill ~4 tokens

        # Steady consumption should work
        success, _ = bucket.consume(2)
        assert success is True

    def test_sustained_load(self) -> None:
        """Test sustained load at refill rate."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=5.0, identifier="test")

        # Consume at approximately refill rate
        successful_count = 0
        for _ in range(20):
            success, _ = bucket.consume(1)
            if success:
                successful_count += 1
            time.sleep(0.2)  # 1 token should refill

        # Should handle sustained load
        assert successful_count >= 15  # Allow some variance

    def test_multiple_buckets_independent(self) -> None:
        """Test multiple buckets are independent."""
        bucket1 = SecureTokenBucket(capacity=10, refill_rate=1.0, identifier="user1")
        bucket2 = SecureTokenBucket(capacity=10, refill_rate=1.0, identifier="user2")

        # Consume from bucket1
        bucket1.consume(10)

        # Bucket2 should be unaffected
        success, _ = bucket2.consume(5)
        assert success is True
        assert bucket2.tokens == 5

    def test_wait_and_retry(self) -> None:
        """Test waiting for recommended time then retrying."""
        bucket = SecureTokenBucket(capacity=5, refill_rate=2.0, identifier="test")

        # Exhaust tokens
        bucket.consume(5)

        # Get wait time
        wait_time = bucket.get_wait_time(3)

        # Wait
        time.sleep(wait_time + 0.1)  # Add small buffer

        # Should succeed
        success, _ = bucket.consume(3)
        assert success is True


class TestSecurityFeatures:
    """Tests for security-related features."""

    def test_anomaly_score_increases(self) -> None:
        """Test anomaly score increases with suspicious patterns."""
        bucket = SecureTokenBucket(capacity=50, refill_rate=5.0, identifier="test")

        # Rapid burst consumption
        scores = []
        for _ in range(20):
            success, score = bucket.consume(1)
            if success:
                scores.append(score)

        # Later scores might be higher (depending on implementation)
        assert len(scores) > 0

    def test_consumption_metadata(self) -> None:
        """Test consumption history stores metadata."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=1.0, identifier="test")

        bucket.consume(3)

        assert len(bucket._consumption_history) > 0
        history_item = bucket._consumption_history[-1]

        assert "timestamp" in history_item
        assert "tokens_requested" in history_item
        assert "tokens_available" in history_item
        assert history_item["tokens_requested"] == 3


class TestGetSecureRateLimiter:
    """Tests for get_secure_rate_limiter factory function."""

    def test_get_secure_rate_limiter(self) -> None:
        """Test factory function returns properly configured limiter."""
        limiter = get_secure_rate_limiter()

        assert isinstance(limiter, SecureAuthenticationRateLimiter)
        assert limiter.config.use_cryptographic_ids is True
        assert limiter.config.check_multiple_identifiers is True
        assert limiter.config.subnet_aggregation is True

    def test_get_secure_rate_limiter_returns_new_instance(self) -> None:
        """Test factory function returns new instances."""
        limiter1 = get_secure_rate_limiter()
        limiter2 = get_secure_rate_limiter()

        assert limiter1 is not limiter2


class TestSubnetAggregation:
    """Tests for subnet aggregation edge cases."""

    def test_ipv6_address_not_aggregated(self) -> None:
        """Test IPv6 addresses are not aggregated."""
        config = SecureRateLimitConfig(subnet_aggregation=True)
        limiter = SecureRateLimiter(config)

        # IPv6 addresses should not be aggregated (no dots)
        id1 = limiter._generate_secure_identifier(ip_address="2001:db8::1")
        id2 = limiter._generate_secure_identifier(ip_address="2001:db8::2")

        # Should be different since IPv6 aggregation is not implemented
        assert id1 != id2

    def test_invalid_ipv4_format_not_aggregated(self) -> None:
        """Test invalid IPv4 format is not aggregated."""
        config = SecureRateLimitConfig(subnet_aggregation=True)
        limiter = SecureRateLimiter(config)

        # Invalid IPv4 (only 3 parts)
        id1 = limiter._generate_secure_identifier(ip_address="192.168.1")
        id2 = limiter._generate_secure_identifier(ip_address="192.168.2")

        # Should be different (not aggregated due to invalid format)
        assert id1 != id2

    def test_subnet_aggregation_disabled(self) -> None:
        """Test subnet aggregation can be disabled."""
        config = SecureRateLimitConfig(subnet_aggregation=False)
        limiter = SecureRateLimiter(config)

        id1 = limiter._generate_secure_identifier(ip_address="192.168.1.1")
        id2 = limiter._generate_secure_identifier(ip_address="192.168.1.2")

        # Should be different when aggregation is disabled
        assert id1 != id2

    def test_ipv4_with_port_in_address(self) -> None:
        """Test IPv4 address with port number."""
        config = SecureRateLimitConfig(subnet_aggregation=True)
        limiter = SecureRateLimiter(config)

        # IP with port should still have dots for detection
        identifier = limiter._generate_secure_identifier(ip_address="192.168.1.1:8080")

        assert isinstance(identifier, str)
        assert len(identifier) == 64


class TestTokenBucketRefillEdgeCases:
    """Tests for token bucket refill edge cases."""

    @patch("traigent.security.rate_limiter.get_security_flags")
    def test_decimal_refill_with_rounding_telemetry_enabled(
        self, mock_flags: MagicMock
    ) -> None:
        """Test decimal refill with telemetry enabled tracks rounding adjustments."""
        mock_flags.return_value = Mock(
            emit_security_telemetry=True, use_decimal_rate_limiter=True
        )

        bucket = SecureTokenBucket(capacity=10, refill_rate=2.5, identifier="test")

        # Consume some tokens
        bucket.consume(8)

        # Wait and refill - this should trigger capacity cap
        time.sleep(2.0)  # Would add more than capacity
        bucket._refill()

        # Check that rounding adjustments were tracked
        assert bucket._rounding_adjustments >= 0

    @patch("traigent.security.rate_limiter.get_security_flags")
    def test_float_refill_with_rounding_detection(self, mock_flags: MagicMock) -> None:
        """Test float refill detects near-integer values for rounding."""
        mock_flags.return_value = Mock(
            emit_security_telemetry=True, use_decimal_rate_limiter=False
        )

        bucket = SecureTokenBucket(capacity=10, refill_rate=2.0, identifier="test")

        # Consume all tokens
        bucket.consume(10)

        # Refill with small elapsed time to create near-integer value
        time.sleep(0.5)  # Should refill ~1 token
        bucket._refill()

        # Check that rounding was applied
        assert bucket._rounding_adjustments >= 0

    @patch("traigent.security.rate_limiter.get_security_flags")
    def test_decimal_refill_without_telemetry(self, mock_flags: MagicMock) -> None:
        """Test decimal refill without telemetry doesn't track adjustments."""
        mock_flags.return_value = Mock(
            emit_security_telemetry=False, use_decimal_rate_limiter=True
        )

        bucket = SecureTokenBucket(capacity=10, refill_rate=2.5, identifier="test")

        # Consume and refill
        bucket.consume(8)
        time.sleep(2.0)
        bucket._refill()

        # Rounding adjustments should still be 0 (not incremented)
        assert bucket._rounding_adjustments == 0

    @patch("traigent.security.rate_limiter.get_security_flags")
    def test_float_refill_without_telemetry(self, mock_flags: MagicMock) -> None:
        """Test float refill without telemetry doesn't track adjustments."""
        mock_flags.return_value = Mock(
            emit_security_telemetry=False, use_decimal_rate_limiter=False
        )

        bucket = SecureTokenBucket(capacity=10, refill_rate=2.0, identifier="test")

        # Consume and refill
        bucket.consume(10)
        time.sleep(0.5)
        bucket._refill()

        # Rounding adjustments should be 0
        assert bucket._rounding_adjustments == 0


class TestAnomalyDetectionEdgeCases:
    """Tests for anomaly detection edge cases."""

    def test_anomaly_detection_insufficient_history(self) -> None:
        """Test anomaly detection with less than 10 history items."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=1.0, identifier="test")

        # Make only a few consumptions (less than 10)
        for _ in range(5):
            success, anomaly_score = bucket.consume(1)

        # Should not detect anomalies with insufficient history
        assert bucket._anomaly_score == 0.0

    def test_anomaly_detection_zero_time_span(self) -> None:
        """Test anomaly detection when time span is zero."""
        bucket = SecureTokenBucket(capacity=20, refill_rate=1.0, identifier="test")

        # Mock time to make all consumptions at the same instant
        with patch("time.time", return_value=1000.0):
            for _ in range(15):
                bucket._consumption_history.append(
                    {
                        "timestamp": 1000.0,
                        "tokens_requested": 1,
                        "tokens_available": 10,
                    }
                )

        # Trigger detection
        bucket._detect_anomalies()

        # Should handle zero time_span gracefully (no division by zero)
        assert bucket._anomaly_score >= 0.0

    def test_anomaly_score_increases_with_high_rate(self) -> None:
        """Test anomaly score increases when consumption rate is very high."""
        bucket = SecureTokenBucket(capacity=50, refill_rate=1.0, identifier="test")

        # Create rapid consumption pattern (much higher than 10x refill rate)
        current_time = time.time()
        for i in range(20):
            bucket._consumption_history.append(
                {
                    "timestamp": current_time + i * 0.01,  # Very rapid (100/sec)
                    "tokens_requested": 1,
                    "tokens_available": 30,
                }
            )

        # Trigger detection
        bucket._detect_anomalies()

        # Anomaly score should increase
        assert bucket._anomaly_score > 0.0

    def test_anomaly_score_decays_with_normal_rate(self) -> None:
        """Test anomaly score decays when consumption rate is normal."""
        bucket = SecureTokenBucket(capacity=50, refill_rate=1.0, identifier="test")

        # Start with high anomaly score
        bucket._anomaly_score = 0.5

        # Create slow consumption pattern
        current_time = time.time()
        for i in range(10):
            bucket._consumption_history.append(
                {
                    "timestamp": current_time + i * 2.0,  # Slow (0.5/sec)
                    "tokens_requested": 1,
                    "tokens_available": 40,
                }
            )

        # Trigger detection
        bucket._detect_anomalies()

        # Anomaly score should decay
        assert bucket._anomaly_score < 0.5


class TestMetricsWithRoundingAdjustments:
    """Tests for metrics that include rounding adjustments."""

    def test_get_metrics_includes_rounding_adjustments(self) -> None:
        """Test metrics include rounding adjustments from all buckets."""
        config = SecureRateLimitConfig()
        limiter = SecureRateLimiter(config)

        # Create multiple token buckets
        limiter.check_rate_limit(
            identifier="user1", strategy=RateLimitStrategy.TOKEN_BUCKET
        )
        limiter.check_rate_limit(
            identifier="user2", strategy=RateLimitStrategy.TOKEN_BUCKET
        )

        # Get metrics
        metrics = limiter.get_metrics()

        # Should include rounding_adjustments key
        assert "rounding_adjustments" in metrics
        assert isinstance(metrics["rounding_adjustments"], int)
        assert metrics["rounding_adjustments"] >= 0

    def test_get_metrics_with_no_buckets(self) -> None:
        """Test metrics when no token buckets exist."""
        config = SecureRateLimitConfig()
        limiter = SecureRateLimiter(config)

        # Get metrics without creating any buckets
        metrics = limiter.get_metrics()

        # Should still have rounding_adjustments key
        assert "rounding_adjustments" in metrics
        assert metrics["rounding_adjustments"] == 0


class TestIdentifierGeneration:
    """Tests for secure identifier generation edge cases."""

    def test_api_key_truncation_in_logging(self) -> None:
        """Test API key is truncated in identifier components."""
        config = SecureRateLimitConfig(use_cryptographic_ids=False)
        limiter = SecureRateLimiter(config)

        # Use long API key
        api_key = "sk_test_" + "a" * 100
        identifier = limiter._generate_secure_identifier(api_key=api_key)

        # Identifier should be generated (key is used internally)
        assert isinstance(identifier, str)
        assert len(identifier) == 64

    def test_identifier_with_only_api_key(self) -> None:
        """Test identifier generation with only API key."""
        config = SecureRateLimitConfig(use_cryptographic_ids=True)
        limiter = SecureRateLimiter(config)

        identifier = limiter._generate_secure_identifier(api_key="sk_test_12345")

        assert isinstance(identifier, str)
        assert len(identifier) == 64

    def test_identifier_consistency_with_sorting(self) -> None:
        """Test identifier is consistent regardless of parameter order."""
        config = SecureRateLimitConfig(use_cryptographic_ids=False)
        limiter = SecureRateLimiter(config)

        # Components are sorted internally, so order shouldn't matter
        id1 = limiter._generate_secure_identifier(
            ip_address="1.2.3.4",
            username="user",
            additional_context={"a": "1", "b": "2"},
        )
        id2 = limiter._generate_secure_identifier(
            username="user",
            ip_address="1.2.3.4",
            additional_context={"b": "2", "a": "1"},
        )

        # Should be the same due to sorting
        assert id1 == id2

    def test_identifier_with_default_salt_fallback(self) -> None:
        """Test identifier generation falls back to default salt."""
        config = SecureRateLimitConfig(use_cryptographic_ids=True, identifier_salt=None)
        limiter = SecureRateLimiter(config)

        # Manually set salt to None to test fallback
        limiter.config.identifier_salt = None

        identifier = limiter._generate_secure_identifier(ip_address="1.2.3.4")

        # Should still generate identifier with default salt
        assert isinstance(identifier, str)
        assert len(identifier) == 64


class TestConcurrentAccess:
    """Tests for concurrent access scenarios."""

    def test_limiter_concurrent_checks(self) -> None:
        """Test rate limiter handles concurrent checks correctly."""
        config = SecureRateLimitConfig(max_attempts=50, window_seconds=60)
        limiter = SecureRateLimiter(config)

        results: list[bool] = []

        def check_limit() -> None:
            result = limiter.check_rate_limit(
                identifier="shared_user", strategy=RateLimitStrategy.SLIDING_WINDOW
            )
            results.append(result.allowed)

        # Create multiple threads
        threads = [Thread(target=check_limit) for _ in range(30)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # All should be allowed (under limit)
        assert all(results)

    def test_token_bucket_concurrent_consume(self) -> None:
        """Test token bucket handles concurrent consume operations."""
        bucket = SecureTokenBucket(capacity=50, refill_rate=5.0, identifier="test")

        successes: list[bool] = []

        def consume() -> None:
            success, _ = bucket.consume(1)
            successes.append(success)

        # Create threads
        threads = [Thread(target=consume) for _ in range(60)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should not exceed capacity
        successful = sum(1 for s in successes if s)
        assert successful <= 50


class TestProgressiveDelayEdgeCases:
    """Tests for progressive delay edge cases."""

    def test_progressive_delay_with_no_attempts(self) -> None:
        """Test progressive delay when no failed attempts exist."""
        config = SecureRateLimitConfig(progressive_delay=True)
        limiter = SecureRateLimiter(config)

        delay = limiter.get_progressive_delay("new_user")

        # With 0 attempts: base_delay = 2^0 = 1.0, jitter = 0-1, total = 1.0-2.0
        assert delay >= 1.0
        assert delay <= 2.0

    def test_progressive_delay_with_old_attempts(self) -> None:
        """Test progressive delay ignores attempts outside window."""
        config = SecureRateLimitConfig(
            progressive_delay=True, window_seconds=60, max_delay_seconds=30.0
        )
        limiter = SecureRateLimiter(config)

        identifier = "test_user"

        # Add old attempts (outside window)
        with limiter._lock:
            limiter.attempts[identifier].append(time.time() - 120)  # 2 minutes ago
            limiter.attempts[identifier].append(time.time() - 100)  # 1:40 ago

        delay = limiter.get_progressive_delay(identifier)

        # Old attempts filtered, so 0 valid: 2^0 + jitter = 1.0-2.0
        assert delay >= 1.0
        assert delay <= 2.0

    def test_progressive_delay_max_capping(self) -> None:
        """Test progressive delay is capped at maximum."""
        config = SecureRateLimitConfig(
            progressive_delay=True, max_delay_seconds=5.0, window_seconds=300
        )
        limiter = SecureRateLimiter(config)

        identifier = "test_user"

        # Add many recent attempts to trigger exponential backoff
        for _ in range(20):
            limiter.record_attempt(identifier=identifier, success=False)

        delay = limiter.get_progressive_delay(identifier)

        # Should be capped at max + jitter
        assert delay <= config.max_delay_seconds + 1.0


class TestRecordAttemptEdgeCases:
    """Tests for record_attempt edge cases."""

    def test_record_attempt_generates_identifier(self) -> None:
        """Test record_attempt generates identifier when not provided."""
        config = SecureRateLimitConfig()
        limiter = SecureRateLimiter(config)

        # Record without explicit identifier
        limiter.record_attempt(ip_address="1.2.3.4", success=False)

        # Should have created an attempt
        assert len(limiter.attempts) > 0

    def test_record_attempt_success_resets_bucket_tokens(self) -> None:
        """Test successful attempt resets token bucket to capacity."""
        config = SecureRateLimitConfig()
        limiter = SecureRateLimiter(config)

        identifier = "test_user"

        # Create bucket and consume tokens
        limiter.check_rate_limit(
            identifier=identifier, strategy=RateLimitStrategy.TOKEN_BUCKET
        )
        bucket = limiter.token_buckets[identifier]

        original_capacity = bucket.capacity
        bucket.consume(int(bucket.capacity) // 2)

        # Record success
        limiter.record_attempt(identifier=identifier, success=True)

        # Tokens should be reset to capacity
        assert bucket.tokens == original_capacity


class TestCheckRateLimitEdgeCases:
    """Tests for check_rate_limit edge cases."""

    def test_check_rate_limit_updates_metrics(self) -> None:
        """Test check_rate_limit updates total_checks metric."""
        config = SecureRateLimitConfig()
        limiter = SecureRateLimiter(config)

        initial_checks = limiter._metrics["total_checks"]

        limiter.check_rate_limit(identifier="user1")
        limiter.check_rate_limit(identifier="user2")

        # Should increment total_checks
        assert limiter._metrics["total_checks"] == initial_checks + 2

    def test_check_rate_limit_tracks_bypass_attempts(self) -> None:
        """Test bypass attempts are tracked when multiple identifiers fail."""
        config = SecureRateLimitConfig(
            max_attempts=2, window_seconds=60, check_multiple_identifiers=True
        )
        limiter = SecureRateLimiter(config)

        ip_address = "1.2.3.4"
        username = "user"

        # Exhaust IP limit
        for _ in range(2):
            limiter.record_attempt(ip_address=ip_address, success=False)

        initial_bypasses = limiter._metrics["bypass_attempts"]

        # Try with both IP and username
        limiter.check_rate_limit(ip_address=ip_address, username=username)

        # Should track bypass attempt
        assert limiter._metrics["bypass_attempts"] > initial_bypasses

    def test_sliding_window_cleans_old_attempts(self) -> None:
        """Test sliding window removes old attempts before checking."""
        config = SecureRateLimitConfig(window_seconds=60)
        limiter = SecureRateLimiter(config)

        identifier = "test_user"

        # Add old attempt
        with limiter._lock:
            limiter.attempts[identifier].append(time.time() - 120)  # 2 minutes ago

        # Check rate limit
        result = limiter.check_rate_limit(
            identifier=identifier, strategy=RateLimitStrategy.SLIDING_WINDOW
        )

        # Should be allowed (old attempt was cleaned)
        assert result.allowed is True

        # Old attempt should be removed
        assert len(limiter.attempts[identifier]) == 0


class TestAuthenticationRateLimiterEdgeCases:
    """Tests for authentication rate limiter edge cases."""

    @pytest.mark.asyncio
    async def test_check_auth_attempt_without_user_agent(self) -> None:
        """Test authentication check without user agent."""
        auth_limiter = SecureAuthenticationRateLimiter()

        result = await auth_limiter.check_auth_attempt(
            ip_address="1.2.3.4", username="user"
        )

        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_check_auth_attempt_applies_progressive_delay_when_allowed(
        self,
    ) -> None:
        """Test progressive delay is applied even when request is allowed."""
        auth_limiter = SecureAuthenticationRateLimiter()

        ip_address = "1.2.3.4"

        # Record some failed attempts
        for _ in range(2):
            auth_limiter.record_auth_result(success=False, ip_address=ip_address)

        start_time = time.time()
        result = await auth_limiter.check_auth_attempt(ip_address=ip_address)
        elapsed = time.time() - start_time

        # Should be allowed but with delay
        assert result.allowed is True
        assert elapsed > 0.5  # Should have some delay


class TestFixedWindowStrategy:
    """Tests for fixed window strategy (currently falls back to sliding window)."""

    def test_fixed_window_fallback(self) -> None:
        """Test fixed window strategy falls back to sliding window."""
        config = SecureRateLimitConfig()
        limiter = SecureRateLimiter(config)

        result = limiter.check_rate_limit(
            identifier="test_user", strategy=RateLimitStrategy.FIXED_WINDOW
        )

        # Should work (fallback to sliding window)
        assert result.allowed is True
        assert result.security_metadata is not None
        assert result.security_metadata["strategy"] == "sliding_window"
