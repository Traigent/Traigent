"""Secure rate limiting implementation for Traigent SDK.

This module provides enhanced rate limiting with cryptographic identifier
generation and distributed coordination support.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security CONC-Quality-Reliability FUNC-SECURITY REQ-SEC-010 SYNC-CloudHybrid

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal, getcontext
from enum import Enum
from threading import Lock
from typing import Any, cast

from cryptography.hazmat.primitives import hashes, hmac

from traigent.security.config import get_security_flags

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""

    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class SecureRateLimitResult:
    """Result of secure rate limit check with additional metadata."""

    allowed: bool
    retry_after: float | None = None
    remaining_attempts: int | None = None
    reset_time: float | None = None
    identifier_hash: str | None = None  # For audit logging
    security_metadata: dict[str, Any] | None = None


@dataclass
class SecureRateLimitConfig:
    """Enhanced configuration for secure rate limiting."""

    # Basic limits
    max_attempts: int = 5
    window_seconds: int = 300  # 5 minutes
    lockout_duration: int = 3600  # 1 hour after max attempts

    # Security enhancements
    progressive_delay: bool = True
    max_delay_seconds: float = 60.0
    use_cryptographic_ids: bool = True
    identifier_salt: bytes | None = None  # For keyed identifier derivation

    # Distributed coordination
    enable_distributed: bool = False
    redis_url: str | None = None

    # Anti-bypass features
    check_multiple_identifiers: bool = True  # Check IP, user, key separately
    global_rate_limit: int = 1000  # Global limit across all identifiers
    subnet_aggregation: bool = True  # Aggregate IPs by subnet


class SecureTokenBucket:
    """Enhanced token bucket with security features."""

    capacity: Decimal | float
    refill_rate: Decimal | float
    tokens: Decimal | float

    def __init__(self, capacity: int, refill_rate: float, identifier: str) -> None:
        """Initialize secure token bucket.

        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens per second refill rate
            identifier: Unique identifier for this bucket
        """
        flags = get_security_flags()
        self._telemetry_enabled = flags.emit_security_telemetry
        self._use_decimal = flags.use_decimal_rate_limiter
        self.identifier = identifier
        if self._use_decimal:
            getcontext().prec = 28
            self.capacity = Decimal(capacity)
            self.refill_rate = Decimal(str(refill_rate))
            self.tokens = Decimal(capacity)
        else:
            self.capacity = float(capacity)
            self.refill_rate = float(refill_rate)
            self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = Lock()

        # Security tracking
        self._consumption_history: deque[dict[str, Any]] = deque(maxlen=100)
        self._anomaly_score = 0.0
        self._rounding_adjustments = 0

    def consume(self, tokens: int = 1) -> tuple[bool, float]:
        """Try to consume tokens from bucket with anomaly detection.

        Returns:
            Tuple of (success, anomaly_score)
        """
        with self._lock:
            self._refill()

            # Record consumption attempt
            self._consumption_history.append(
                {
                    "timestamp": time.time(),
                    "tokens_requested": tokens,
                    "tokens_available": self.tokens,
                }
            )

            # Detect anomalies (rapid consumption patterns)
            self._detect_anomalies()

            token_request = Decimal(tokens) if self._use_decimal else tokens
            if self.tokens >= token_request:
                if self._use_decimal:
                    self.tokens = cast(Decimal, self.tokens) - cast(
                        Decimal, token_request
                    )
                else:
                    self.tokens = cast(float, self.tokens) - cast(float, token_request)
                return True, self._anomaly_score

            return False, self._anomaly_score

    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait until tokens are available."""
        with self._lock:
            self._refill()

            token_request = Decimal(tokens) if self._use_decimal else tokens
            if self.tokens >= token_request:
                return 0.0

            tokens_needed: Decimal | float
            wait: Decimal | float
            if self._use_decimal:
                tokens_needed = cast(Decimal, token_request) - cast(
                    Decimal, self.tokens
                )
                wait = tokens_needed / cast(Decimal, self.refill_rate)
            else:
                tokens_needed = cast(float, token_request) - cast(float, self.tokens)
                wait = tokens_needed / cast(float, self.refill_rate)
            return float(wait)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        tokens_to_add: Decimal | float
        new_total: Decimal | float
        if self._use_decimal:
            elapsed_decimal = Decimal(str(elapsed))
            tokens_to_add = elapsed_decimal * cast(Decimal, self.refill_rate)
            new_total = min(
                cast(Decimal, self.capacity), cast(Decimal, self.tokens) + tokens_to_add
            )
            if (
                new_total != cast(Decimal, self.tokens) + tokens_to_add
                and self._telemetry_enabled
            ):
                self._rounding_adjustments += 1
            self.tokens = new_total
        else:
            tokens_to_add = elapsed * cast(float, self.refill_rate)
            new_total = min(
                cast(float, self.capacity), cast(float, self.tokens) + tokens_to_add
            )
            if abs(new_total - round(new_total)) < 1e-4:
                if self._telemetry_enabled:
                    self._rounding_adjustments += 1
                new_total = float(round(new_total))
            self.tokens = new_total

        self.last_refill = now

    def _detect_anomalies(self) -> None:
        """Detect anomalous consumption patterns."""
        if len(self._consumption_history) < 10:
            return

        # Check for burst patterns
        recent = list(self._consumption_history)[-10:]
        time_span = recent[-1]["timestamp"] - recent[0]["timestamp"]

        if time_span > 0:
            rate = len(recent) / time_span
            # If rate is unusually high, increase anomaly score
            if rate > self.refill_rate * 10:  # 10x normal rate
                self._anomaly_score = min(1.0, self._anomaly_score + 0.1)
            else:
                # Decay anomaly score
                self._anomaly_score = max(0.0, self._anomaly_score - 0.01)


class SecureRateLimiter:
    """Production-hardened rate limiter with enhanced security."""

    def __init__(self, config: SecureRateLimitConfig) -> None:
        """Initialize secure rate limiter."""
        self.config = config
        self.attempts: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=1000))
        self.lockouts: dict[str, float] = {}
        self.token_buckets: dict[str, SecureTokenBucket] = {}
        self._lock = Lock()

        if self.config.enable_distributed:
            raise NotImplementedError(
                "Distributed rate limiting is not implemented in the SDK. "
                "Do not set enable_distributed=True until a Redis-backed "
                "counter implementation is available."
            )

        # Initialize cryptographic salt if not provided
        if self.config.use_cryptographic_ids and not self.config.identifier_salt:
            env_salt = os.getenv("TRAIGENT_RATE_LIMIT_IDENTIFIER_SALT")
            if env_salt:
                self.config.identifier_salt = env_salt.encode("utf-8")
            else:
                self.config.identifier_salt = secrets.token_bytes(32)
                logger.warning(
                    "Generated process-local identifier salt for rate limiting. "
                    "Set TRAIGENT_RATE_LIMIT_IDENTIFIER_SALT for stable "
                    "deployment-wide buckets."
                )

        # Security metrics
        self._metrics: dict[str, Any] = {
            "total_checks": 0,
            "blocked_attempts": 0,
            "lockouts_triggered": 0,
            "anomalies_detected": 0,
            "bypass_attempts": 0,
        }

    @classmethod
    def _keyed_identifier(cls, secret: bytes | None, value: str) -> str:
        if not secret:
            raise ValueError("identifier_salt is required for keyed identifiers")
        key = secret
        mac = hmac.HMAC(key, hashes.SHA256())
        mac.update(value.encode("utf-8"))
        return cast(bytes, mac.finalize()).hex()

    @staticmethod
    def _plain_identifier(value: str) -> str:
        return hashlib.blake2b(value.encode("utf-8"), digest_size=32).hexdigest()

    def _generate_secure_identifier(
        self,
        ip_address: str | None = None,
        username: str | None = None,
        api_key: str | None = None,
        additional_context: dict[str, str] | None = None,
    ) -> str:
        """Generate cryptographically secure identifier.

        This prevents identifier manipulation attacks by using keyed derivation.
        """
        components = []
        identifier_secret = self.config.identifier_salt

        # Add components with type prefixes
        if ip_address:
            # Apply subnet aggregation if enabled
            if self.config.subnet_aggregation:
                # For IPv4, aggregate to /24
                if "." in ip_address:
                    parts = ip_address.split(".")
                    if len(parts) == 4:
                        ip_address = f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
            components.append(f"ip:{ip_address}")

        if username:
            components.append(f"user:{username}")

        if api_key:
            key_fingerprint = self._keyed_identifier(
                identifier_secret,
                f"api_key:{api_key}",
            )[:16]
            components.append(f"key:{key_fingerprint}")

        if additional_context:
            for key, value in additional_context.items():
                components.append(f"{key}:{value}")

        if not components:
            components.append("anonymous")

        # Create base identifier
        base_identifier = "|".join(sorted(components))  # Sort for consistency

        if self.config.use_cryptographic_ids:
            return self._keyed_identifier(identifier_secret, base_identifier)
        else:
            return self._plain_identifier(base_identifier)

    def check_rate_limit(
        self,
        identifier: str | None = None,
        ip_address: str | None = None,
        username: str | None = None,
        api_key: str | None = None,
        strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW,
    ) -> SecureRateLimitResult:
        """Check if request is within rate limits with multiple strategies."""
        self._metrics["total_checks"] += 1

        # Generate secure identifier if not provided
        if not identifier:
            identifier = self._generate_secure_identifier(ip_address, username, api_key)

        # Check multiple identifiers if enabled (prevents bypass via parameter switching)
        if self.config.check_multiple_identifiers:
            identifiers_to_check = []

            if ip_address:
                identifiers_to_check.append(
                    self._generate_secure_identifier(ip_address=ip_address)
                )
            if username:
                identifiers_to_check.append(
                    self._generate_secure_identifier(username=username)
                )
            if api_key:
                identifiers_to_check.append(
                    self._generate_secure_identifier(api_key=api_key)
                )

            # Check each identifier
            for check_id in identifiers_to_check:
                result = self._check_single_identifier(check_id, strategy)
                if not result.allowed:
                    self._metrics["bypass_attempts"] += 1
                    logger.warning(
                        f"Rate limit bypass attempt detected for {check_id[:16]}..."
                    )
                    return result

        # Check the combined identifier
        return self._check_single_identifier(identifier, strategy)

    def _check_single_identifier(
        self,
        identifier: str,
        strategy: RateLimitStrategy,
    ) -> SecureRateLimitResult:
        """Check rate limit for a single identifier."""
        with self._lock:
            now = time.time()

            # Check if currently locked out
            if identifier in self.lockouts:
                lockout_until = self.lockouts[identifier]
                if now < lockout_until:
                    self._metrics["blocked_attempts"] += 1
                    return SecureRateLimitResult(
                        allowed=False,
                        retry_after=lockout_until - now,
                        remaining_attempts=0,
                        reset_time=lockout_until,
                        identifier_hash=identifier[:16],  # Log only prefix
                        security_metadata={
                            "reason": "lockout",
                            "lockout_until": lockout_until,
                        },
                    )
                else:
                    # Lockout expired
                    del self.lockouts[identifier]

            if strategy == RateLimitStrategy.SLIDING_WINDOW:
                return self._check_sliding_window(identifier, now)
            elif strategy == RateLimitStrategy.TOKEN_BUCKET:
                return self._check_token_bucket(identifier, now)
            else:
                # Default to sliding window
                return self._check_sliding_window(identifier, now)

    def _check_sliding_window(
        self, identifier: str, now: float
    ) -> SecureRateLimitResult:
        """Check using sliding window algorithm."""
        attempts = self.attempts[identifier]
        window_start = now - self.config.window_seconds

        # Clean old attempts
        while attempts and attempts[0] < window_start:
            attempts.popleft()

        # Check if within limits
        if len(attempts) < self.config.max_attempts:
            return SecureRateLimitResult(
                allowed=True,
                remaining_attempts=self.config.max_attempts - len(attempts) - 1,
                reset_time=now + self.config.window_seconds,
                identifier_hash=identifier[:16],
                security_metadata={"strategy": "sliding_window"},
            )

        # Rate limit exceeded - trigger lockout
        lockout_until = now + self.config.lockout_duration
        self.lockouts[identifier] = lockout_until
        self._metrics["lockouts_triggered"] += 1
        self._metrics["blocked_attempts"] += 1

        logger.warning(f"Rate limit lockout triggered for {identifier[:16]}...")

        return SecureRateLimitResult(
            allowed=False,
            retry_after=self.config.lockout_duration,
            remaining_attempts=0,
            reset_time=lockout_until,
            identifier_hash=identifier[:16],
            security_metadata={
                "strategy": "sliding_window",
                "reason": "rate_limit_exceeded",
            },
        )

    def _check_token_bucket(self, identifier: str, now: float) -> SecureRateLimitResult:
        """Check using token bucket algorithm."""
        # Get or create token bucket for identifier
        if identifier not in self.token_buckets:
            self.token_buckets[identifier] = SecureTokenBucket(
                capacity=self.config.max_attempts,
                refill_rate=self.config.max_attempts / self.config.window_seconds,
                identifier=identifier,
            )

        bucket = self.token_buckets[identifier]
        allowed, anomaly_score = bucket.consume()

        # Check for anomalies
        if anomaly_score > 0.5:
            self._metrics["anomalies_detected"] += 1
            logger.warning(
                f"Anomalous rate pattern detected for {identifier[:16]}... (score: {anomaly_score:.2f})"
            )

        if not allowed:
            self._metrics["blocked_attempts"] += 1
            wait_time = bucket.get_wait_time()

            return SecureRateLimitResult(
                allowed=False,
                retry_after=wait_time,
                remaining_attempts=int(bucket.tokens),
                reset_time=now + wait_time,
                identifier_hash=identifier[:16],
                security_metadata={
                    "strategy": "token_bucket",
                    "anomaly_score": anomaly_score,
                },
            )

        return SecureRateLimitResult(
            allowed=True,
            remaining_attempts=int(bucket.tokens),
            reset_time=now + self.config.window_seconds,
            identifier_hash=identifier[:16],
            security_metadata={
                "strategy": "token_bucket",
                "anomaly_score": anomaly_score,
            },
        )

    def record_attempt(
        self,
        identifier: str | None = None,
        ip_address: str | None = None,
        username: str | None = None,
        api_key: str | None = None,
        success: bool = False,
    ) -> None:
        """Record an attempt with proper security tracking."""
        if not identifier:
            identifier = self._generate_secure_identifier(ip_address, username, api_key)

        with self._lock:
            now = time.time()

            if success:
                # Clear failed attempts on success
                if identifier in self.attempts:
                    self.attempts[identifier].clear()
                if identifier in self.lockouts:
                    del self.lockouts[identifier]
                # Reset token bucket
                if identifier in self.token_buckets:
                    bucket = self.token_buckets[identifier]
                    bucket.tokens = bucket.capacity
            else:
                # Record failed attempt
                self.attempts[identifier].append(now)

    def get_progressive_delay(self, identifier: str) -> float:
        """Get progressive delay with security limits."""
        if not self.config.progressive_delay:
            return 0.0

        with self._lock:
            attempts = self.attempts.get(identifier, deque())

            # Clean old attempts
            now = time.time()
            window_start = now - self.config.window_seconds
            valid_attempts = [a for a in attempts if a >= window_start]

            # Calculate delay with exponential backoff
            # Add randomization to prevent timing attacks
            base_delay = min(self.config.max_delay_seconds, 2.0 ** len(valid_attempts))
            jitter = secrets.randbelow(1000) / 1000.0  # 0-1 second random jitter

            return base_delay + jitter

    def get_metrics(self) -> dict[str, Any]:
        """Get security metrics for monitoring."""
        metrics = self._metrics.copy()
        metrics["rounding_adjustments"] = sum(
            getattr(bucket, "_rounding_adjustments", 0)
            for bucket in self.token_buckets.values()
        )
        return metrics

    def clear_identifier(self, identifier: str) -> None:
        """Clear all records for an identifier."""
        with self._lock:
            if identifier in self.attempts:
                del self.attempts[identifier]
            if identifier in self.lockouts:
                del self.lockouts[identifier]
            if identifier in self.token_buckets:
                del self.token_buckets[identifier]


class SecureAuthenticationRateLimiter:
    """Enhanced authentication rate limiter with security features."""

    def __init__(self) -> None:
        """Initialize with production-ready secure defaults."""
        self.config = SecureRateLimitConfig(
            max_attempts=5,
            window_seconds=300,
            lockout_duration=3600,
            progressive_delay=True,
            max_delay_seconds=60.0,
            use_cryptographic_ids=True,
            check_multiple_identifiers=True,
            subnet_aggregation=True,
        )
        self.limiter = SecureRateLimiter(self.config)

    async def check_auth_attempt(
        self,
        ip_address: str | None = None,
        username: str | None = None,
        api_key: str | None = None,
        user_agent: str | None = None,
    ) -> SecureRateLimitResult:
        """Check if authentication attempt is allowed with enhanced security."""
        # Add user agent to context for better fingerprinting
        additional_context = {}
        if user_agent:
            # Hash user agent for privacy
            ua_hash = hashlib.sha256(user_agent.encode()).hexdigest()[:8]
            additional_context["ua"] = ua_hash

        # Generate secure identifier
        identifier = self.limiter._generate_secure_identifier(
            ip_address, username, api_key, additional_context
        )

        # Check rate limit with multiple strategies
        result = self.limiter.check_rate_limit(
            identifier=identifier,
            ip_address=ip_address,
            username=username,
            api_key=api_key,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
        )

        if not result.allowed:
            logger.warning(
                f"Authentication rate limit exceeded. "
                f"Retry after: {result.retry_after}s"
            )

        # Apply progressive delay even if allowed
        if result.allowed and self.config.progressive_delay:
            delay = self.limiter.get_progressive_delay(identifier)
            if delay > 0:
                logger.info(f"Applying progressive delay: {delay:.2f}s")
                import asyncio

                await asyncio.sleep(delay)

        return result

    def record_auth_result(
        self,
        success: bool,
        ip_address: str | None = None,
        username: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Record authentication result."""
        self.limiter.record_attempt(
            ip_address=ip_address,
            username=username,
            api_key=api_key,
            success=success,
        )

        if not success:
            logger.info("Failed authentication attempt recorded")
        else:
            logger.info("Successful authentication, clearing rate limit")

    def get_metrics(self) -> dict[str, Any]:
        """Get rate limiting metrics."""
        return self.limiter.get_metrics()


def get_secure_rate_limiter() -> SecureAuthenticationRateLimiter:
    """Get production-ready secure rate limiter."""
    return SecureAuthenticationRateLimiter()
