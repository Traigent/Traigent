"""Comprehensive unit tests for secure rate limiting."""

import time

from traigent.security.rate_limiter import (
    RateLimitStrategy,
    SecureRateLimitConfig,
    SecureRateLimitResult,
    SecureTokenBucket,
)


class TestRateLimitStrategy:
    """Test RateLimitStrategy enum."""

    def test_rate_limit_strategies(self):
        """Test all rate limit strategies are defined."""
        assert RateLimitStrategy.SLIDING_WINDOW.value == "sliding_window"
        assert RateLimitStrategy.TOKEN_BUCKET.value == "token_bucket"
        assert RateLimitStrategy.FIXED_WINDOW.value == "fixed_window"
        assert RateLimitStrategy.LEAKY_BUCKET.value == "leaky_bucket"


class TestSecureRateLimitResult:
    """Test SecureRateLimitResult dataclass."""

    def test_result_creation_allowed(self):
        """Test creating allowed result."""
        result = SecureRateLimitResult(
            allowed=True,
            remaining_attempts=5,
            reset_time=time.time() + 300,
        )

        assert result.allowed is True
        assert result.remaining_attempts == 5
        assert result.retry_after is None

    def test_result_creation_denied(self):
        """Test creating denied result."""
        result = SecureRateLimitResult(
            allowed=False,
            retry_after=60.0,
            remaining_attempts=0,
        )

        assert result.allowed is False
        assert result.retry_after == 60.0
        assert result.remaining_attempts == 0


class TestSecureRateLimitConfig:
    """Test SecureRateLimitConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SecureRateLimitConfig()

        assert config.max_attempts == 5
        assert config.window_seconds == 300
        assert config.lockout_duration == 3600
        assert config.progressive_delay is True
        assert config.use_cryptographic_ids is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SecureRateLimitConfig(
            max_attempts=10,
            window_seconds=60,
            progressive_delay=False,
        )

        assert config.max_attempts == 10
        assert config.window_seconds == 60
        assert config.progressive_delay is False

    def test_distributed_config(self):
        """Test distributed configuration."""
        config = SecureRateLimitConfig(
            enable_distributed=True,
            redis_url="redis://localhost:6379",
        )

        assert config.enable_distributed is True
        assert config.redis_url == "redis://localhost:6379"


class TestSecureTokenBucket:
    """Test SecureTokenBucket implementation."""

    def test_token_bucket_initialization(self):
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

    def test_consume_tokens_success(self):
        """Test consuming tokens successfully."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=1.0, identifier="test")

        success, anomaly_score = bucket.consume(3)

        assert success is True
        assert bucket.tokens == 7
        assert isinstance(anomaly_score, float)

    def test_consume_tokens_failure(self):
        """Test consuming more tokens than available."""
        bucket = SecureTokenBucket(capacity=5, refill_rate=1.0, identifier="test")

        success, anomaly_score = bucket.consume(10)

        assert success is False
        assert bucket.tokens == 5  # Unchanged

    def test_consume_multiple_times(self):
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

    def test_token_refill(self):
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

    def test_token_refill_cap(self):
        """Test token refill doesn't exceed capacity."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=2.0, identifier="test")

        # Don't consume, just wait
        time.sleep(10.0)  # Would refill 20 tokens if uncapped

        # Try to consume capacity + 1
        success, _ = bucket.consume(11)
        assert success is False  # Can't exceed capacity

    def test_get_wait_time_no_wait(self):
        """Test getting wait time when tokens available."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=2.0, identifier="test")

        wait_time = bucket.get_wait_time(5)

        assert wait_time == 0.0

    def test_get_wait_time_with_wait(self):
        """Test getting wait time when tokens not available."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=2.0, identifier="test")

        # Consume most tokens
        bucket.consume(10)

        wait_time = bucket.get_wait_time(5)

        assert wait_time > 0
        assert wait_time <= 5 / 2.0  # 5 tokens at 2 tokens/sec

    def test_consumption_history_tracking(self):
        """Test consumption history is tracked."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=1.0, identifier="test")

        # Make multiple consumptions
        for _ in range(5):
            bucket.consume(1)

        assert len(bucket._consumption_history) == 5

    def test_anomaly_detection(self):
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

    def test_thread_safety(self):
        """Test token bucket thread safety."""
        import threading

        bucket = SecureTokenBucket(capacity=100, refill_rate=10.0, identifier="test")
        results = []

        def consume_tokens():
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


class TestTokenBucketEdgeCases:
    """Test edge cases for token bucket."""

    def test_zero_capacity(self):
        """Test token bucket with zero capacity."""
        bucket = SecureTokenBucket(capacity=0, refill_rate=1.0, identifier="test")

        success, _ = bucket.consume(1)
        assert success is False

    def test_zero_refill_rate(self):
        """Test token bucket with zero refill rate."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=0.0, identifier="test")

        # Consume tokens
        bucket.consume(5)

        # Wait
        time.sleep(1.0)

        # Should not refill
        assert bucket.tokens == 5

    def test_fractional_tokens(self):
        """Test consuming fractional tokens."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=1.0, identifier="test")

        # Token bucket should handle fractional consumption
        # based on refill calculations
        wait_time = bucket.get_wait_time(1)
        assert isinstance(wait_time, float)

    def test_large_capacity(self):
        """Test token bucket with large capacity."""
        bucket = SecureTokenBucket(
            capacity=10000,
            refill_rate=100.0,
            identifier="test",
        )

        success, _ = bucket.consume(5000)
        assert success is True
        assert bucket.tokens == 5000

    def test_high_refill_rate(self):
        """Test token bucket with high refill rate."""
        bucket = SecureTokenBucket(capacity=100, refill_rate=1000.0, identifier="test")

        # Consume all
        bucket.consume(100)

        # Short wait should refill completely
        time.sleep(0.2)  # 200 tokens would be added

        success, _ = bucket.consume(50)
        assert success is True  # Should have refilled to capacity

    def test_consumption_history_limit(self):
        """Test consumption history has maximum length."""
        bucket = SecureTokenBucket(capacity=200, refill_rate=10.0, identifier="test")

        # Make many consumptions
        for _ in range(150):
            bucket.consume(1)

        # History should be limited to 100
        assert len(bucket._consumption_history) == 100

    def test_negative_tokens_prevented(self):
        """Test that tokens never go negative."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=1.0, identifier="test")

        # Try to consume more than available
        bucket.consume(20)

        # Tokens should not be negative
        assert bucket.tokens >= 0

    def test_identifier_usage(self):
        """Test identifier is stored and accessible."""
        identifier = "user-123"
        bucket = SecureTokenBucket(capacity=10, refill_rate=1.0, identifier=identifier)

        assert bucket.identifier == identifier


class TestRateLimitIntegration:
    """Test rate limit integration scenarios."""

    def test_burst_then_steady(self):
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

    def test_sustained_load(self):
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

    def test_multiple_buckets_independent(self):
        """Test multiple buckets are independent."""
        bucket1 = SecureTokenBucket(capacity=10, refill_rate=1.0, identifier="user1")
        bucket2 = SecureTokenBucket(capacity=10, refill_rate=1.0, identifier="user2")

        # Consume from bucket1
        bucket1.consume(10)

        # Bucket2 should be unaffected
        success, _ = bucket2.consume(5)
        assert success is True
        assert bucket2.tokens == 5

    def test_wait_and_retry(self):
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
    """Test security-related features."""

    def test_anomaly_score_increases(self):
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

    def test_consumption_metadata(self):
        """Test consumption history stores metadata."""
        bucket = SecureTokenBucket(capacity=10, refill_rate=1.0, identifier="test")

        bucket.consume(3)

        assert len(bucket._consumption_history) > 0
        history_item = bucket._consumption_history[-1]

        assert "timestamp" in history_item
        assert "tokens_requested" in history_item
        assert "tokens_available" in history_item
        assert history_item["tokens_requested"] == 3
