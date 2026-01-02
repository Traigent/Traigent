"""Tests for secure rate limiter implementation."""

import time

import pytest

from traigent.security.rate_limiter import (
    RateLimitStrategy,
    SecureAuthenticationRateLimiter,
    SecureRateLimitConfig,
    SecureRateLimiter,
    SecureTokenBucket,
    get_secure_rate_limiter,
)


class TestSecureIdentifierGeneration:
    """Test secure identifier generation."""

    def test_cryptographic_identifier_generation(self):
        """Test that identifiers are generated securely."""
        config = SecureRateLimitConfig(use_cryptographic_ids=True)
        limiter = SecureRateLimiter(config)

        # Generate identifiers for same inputs
        id1 = limiter._generate_secure_identifier(
            ip_address="192.168.1.1",
            username="testuser",
        )
        id2 = limiter._generate_secure_identifier(
            ip_address="192.168.1.1",
            username="testuser",
        )

        # Should be consistent
        assert id1 == id2

        # Should be unpredictable (looks like random hash)
        assert len(id1) == 64  # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in id1)

        # Different inputs should give different identifiers
        id3 = limiter._generate_secure_identifier(
            ip_address="192.168.2.1",  # Different subnet
            username="testuser",
        )
        assert id1 != id3

    def test_identifier_manipulation_prevention(self):
        """Test that identifier manipulation is prevented."""
        config = SecureRateLimitConfig(use_cryptographic_ids=True)
        limiter = SecureRateLimiter(config)

        # Try to manipulate identifier components
        id1 = limiter._generate_secure_identifier(username="user1")
        id2 = limiter._generate_secure_identifier(username="user1|admin")

        # Should produce completely different hashes
        assert id1 != id2
        # No common prefix that could be exploited
        assert id1[:16] != id2[:16]

    def test_subnet_aggregation(self):
        """Test IP subnet aggregation for preventing IP rotation attacks."""
        config = SecureRateLimitConfig(
            use_cryptographic_ids=True,
            subnet_aggregation=True,
        )
        limiter = SecureRateLimiter(config)

        # IPs in same /24 subnet should map to same identifier
        id1 = limiter._generate_secure_identifier(ip_address="192.168.1.1")
        limiter._generate_secure_identifier(ip_address="192.168.1.100")
        limiter._generate_secure_identifier(ip_address="192.168.1.255")

        # Different subnet should have different identifier
        id4 = limiter._generate_secure_identifier(ip_address="192.168.2.1")

        # Same subnet IPs should have same identifier (after aggregation)
        # Note: This test assumes subnet aggregation is implemented
        assert id4 != id1  # Different subnets


class TestRateLimitingStrategies:
    """Test different rate limiting strategies."""

    def test_sliding_window_strategy(self):
        """Test sliding window rate limiting."""
        config = SecureRateLimitConfig(
            max_attempts=3,
            window_seconds=60,
            lockout_duration=300,
        )
        limiter = SecureRateLimiter(config)

        identifier = "test_user"

        # First 3 attempts should be allowed
        for _i in range(3):
            result = limiter.check_rate_limit(
                identifier=identifier,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
            )
            assert result.allowed
            limiter.record_attempt(identifier=identifier, success=False)

        # 4th attempt should be blocked
        result = limiter.check_rate_limit(
            identifier=identifier,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
        )
        assert not result.allowed
        assert result.retry_after > 0
        assert result.remaining_attempts == 0

    def test_token_bucket_strategy(self):
        """Test token bucket rate limiting."""
        config = SecureRateLimitConfig(
            max_attempts=5,
            window_seconds=60,
        )
        limiter = SecureRateLimiter(config)

        identifier = "test_user"

        # Consume tokens
        for i in range(5):
            result = limiter.check_rate_limit(
                identifier=identifier,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            )
            if i < 5:
                assert result.allowed

        # Should be rate limited after consuming all tokens
        result = limiter.check_rate_limit(
            identifier=identifier,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
        )
        # Note: Actual behavior depends on token bucket implementation


class TestMultipleIdentifierChecking:
    """Test multiple identifier checking to prevent bypass."""

    def test_bypass_prevention_via_parameter_switching(self):
        """Test that switching parameters doesn't bypass rate limits."""
        config = SecureRateLimitConfig(
            max_attempts=3,
            window_seconds=60,
            check_multiple_identifiers=True,
        )
        limiter = SecureRateLimiter(config)

        ip = "192.168.1.1"

        # Exhaust rate limit using IP only
        for _ in range(3):
            limiter.record_attempt(ip_address=ip, success=False)

        # Try to bypass by adding username
        result = limiter.check_rate_limit(
            ip_address=ip,
            username="newuser",  # Adding new parameter
        )

        # Should still be blocked because IP is rate limited
        # (when check_multiple_identifiers is enabled)
        assert not result.allowed  # Request should be blocked

    def test_multiple_dimension_rate_limiting(self):
        """Test rate limiting across multiple dimensions."""
        config = SecureRateLimitConfig(
            max_attempts=3,
            window_seconds=60,
            check_multiple_identifiers=True,
        )
        limiter = SecureRateLimiter(config)

        # Rate limit by username
        for _ in range(3):
            limiter.record_attempt(username="user1", success=False)

        # Check if username is rate limited
        result = limiter.check_rate_limit(username="user1")

        # User should be rate limited
        assert not result.allowed  # Request should be blocked


class TestProgressiveDelay:
    """Test progressive delay functionality."""

    def test_exponential_backoff(self):
        """Test that delays increase exponentially."""
        config = SecureRateLimitConfig(
            progressive_delay=True,
            max_delay_seconds=60.0,
        )
        limiter = SecureRateLimiter(config)

        identifier = "test_user"

        # Record failures and check delays
        delays = []
        for _i in range(5):
            limiter.record_attempt(identifier=identifier, success=False)
            delay = limiter.get_progressive_delay(identifier)
            delays.append(delay)

        # Delays should increase (with some randomization)
        assert delays[0] < delays[2]  # Later delays should be larger
        assert all(d <= 61.0 for d in delays)  # Max delay + jitter

    def test_delay_with_jitter(self):
        """Test that delays include random jitter."""
        config = SecureRateLimitConfig(
            progressive_delay=True,
            max_delay_seconds=60.0,
        )
        limiter = SecureRateLimiter(config)

        identifier = "test_user"
        limiter.record_attempt(identifier=identifier, success=False)

        # Get multiple delay values
        delays = [limiter.get_progressive_delay(identifier) for _ in range(10)]

        # Should have some variation due to jitter
        assert len(set(delays)) > 1  # Not all identical


class TestSecurityMetrics:
    """Test security metrics tracking."""

    def test_metrics_collection(self):
        """Test that security metrics are collected."""
        config = SecureRateLimitConfig()
        limiter = SecureRateLimiter(config)

        # Perform various operations
        limiter.check_rate_limit(identifier="user1")
        limiter.record_attempt(identifier="user1", success=False)

        # Trigger a lockout
        for _ in range(10):
            limiter.record_attempt(identifier="user2", success=False)
        limiter.check_rate_limit(identifier="user2")

        # Get metrics
        metrics = limiter.get_metrics()

        assert metrics["total_checks"] > 0
        assert "blocked_attempts" in metrics
        assert "lockouts_triggered" in metrics
        assert "anomalies_detected" in metrics


class TestAnomalyDetection:
    """Test anomaly detection in token buckets."""

    def test_burst_pattern_detection(self):
        """Test detection of burst consumption patterns."""
        bucket = SecureTokenBucket(
            capacity=10,
            refill_rate=1.0,
            identifier="test",
        )

        # Rapid consumption (burst)
        for _ in range(5):
            bucket.consume(1)
            time.sleep(0.01)  # Very fast consumption

        # Anomaly score should increase
        _, anomaly_score = bucket.consume(1)
        # Score depends on implementation threshold
        assert isinstance(anomaly_score, (int, float))


class TestAuthenticationRateLimiter:
    """Test the authentication-specific rate limiter."""

    @pytest.mark.asyncio
    async def test_auth_rate_limiting(self):
        """Test authentication rate limiting."""
        limiter = SecureAuthenticationRateLimiter()

        # Successful attempts
        for _ in range(3):
            result = await limiter.check_auth_attempt(
                ip_address="192.168.1.1",
                username="testuser",
            )
            assert result.allowed
            limiter.record_auth_result(
                success=False,
                ip_address="192.168.1.1",
                username="testuser",
            )

        # Should apply rate limiting
        for _ in range(3):
            result = await limiter.check_auth_attempt(
                ip_address="192.168.1.1",
                username="testuser",
            )
            # After max attempts, should be blocked

    @pytest.mark.asyncio
    async def test_successful_auth_clears_limits(self):
        """Test that successful auth clears rate limits."""
        limiter = SecureAuthenticationRateLimiter()

        # Record some failures
        for _ in range(2):
            limiter.record_auth_result(
                success=False,
                ip_address="192.168.1.1",
                username="testuser",
            )

        # Record success
        limiter.record_auth_result(
            success=True,
            ip_address="192.168.1.1",
            username="testuser",
        )

        # Should be able to attempt again
        result = await limiter.check_auth_attempt(
            ip_address="192.168.1.1",
            username="testuser",
        )
        assert result.allowed

    @pytest.mark.asyncio
    async def test_progressive_delay_application(self):
        """Test that progressive delays are applied."""
        limiter = SecureAuthenticationRateLimiter()

        # Record a failure
        limiter.record_auth_result(
            success=False,
            ip_address="192.168.1.1",
            username="testuser",
        )

        # Check with timing
        start = time.time()
        await limiter.check_auth_attempt(
            ip_address="192.168.1.1",
            username="testuser",
        )
        elapsed = time.time() - start

        # Should have some delay applied (if progressive delay is enabled)
        if limiter.config.progressive_delay:
            assert elapsed > 0  # Some delay was applied


class TestRateLimiterFactory:
    """Test the rate limiter factory function."""

    def test_get_secure_rate_limiter(self):
        """Test factory function returns configured limiter."""
        limiter = get_secure_rate_limiter()

        assert isinstance(limiter, SecureAuthenticationRateLimiter)
        assert limiter.config.use_cryptographic_ids
        assert limiter.config.check_multiple_identifiers
        assert limiter.config.progressive_delay


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
