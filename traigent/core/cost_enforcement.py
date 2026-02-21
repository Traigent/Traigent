"""Cost enforcement module for Traigent SDK.

Provides thread-safe real-time cost tracking with user handshake approval.
This is the single source of truth for cost tracking - shared by orchestrator
and stop conditions to avoid double counting.

Adaptive Cost Estimation:
    The module uses adaptive cost estimation based on observed trial costs:
    - Initial estimates use configured estimated_cost_per_trial
    - After trials complete, estimates update via exponential moving average (EMA)
    - Confidence levels indicate reliability of estimates (0.0-1.0)
    - Warnings log when estimates diverge significantly from reality

Environment Variables:
    TRAIGENT_RUN_COST_LIMIT: Maximum USD spending per optimization run (default: 2.0)
    TRAIGENT_COST_APPROVED: Skip handshake if "true" (default: false)
    TRAIGENT_COST_WARNING_THRESHOLD: Warn at this fraction of limit (default: 0.5)
    TRAIGENT_MOCK_LLM: Bypass all cost tracking when "true" (no real LLM costs)
    TRAIGENT_REQUIRE_COST_TRACKING: Raise exception if cost extraction fails (default: false)
    TRAIGENT_STRICT_COST_ACCOUNTING: Fail fast for unknown/missing runtime costs
        (default: false)
    TRAIGENT_COST_DIVERGENCE_THRESHOLD: Log warning if actual/estimated ratio exceeds
        this value (default: 2.0, meaning 2x divergence triggers warning)
"""

from __future__ import annotations

import logging
import math
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from traigent.utils.exceptions import CostLimitExceeded

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CostTrackingRequiredError(Exception):
    """Raised when cost tracking fails but strict mode is enabled."""

    pass


@dataclass
class Permit:
    """Represents a cost permit with single-use semantics.

    A Permit is acquired before a trial executes and must be released
    exactly once via either track_cost() or release_permit(). The
    mark_released() method ensures single-release semantics.

    Attributes:
        id: Monotonic permit ID (unique per CostEnforcer instance).
            Special values: 0 for mock mode, -1 for denied permit.
        amount: The reserved amount in USD.
        active: Whether the permit is still active (not yet released).
    """

    id: int
    amount: float
    active: bool = True

    def mark_released(self) -> bool:
        """Mark permit as released. Returns False if already released.

        This method provides single-release semantics - it can only
        return True once per Permit instance.

        Returns:
            True if this is the first release call, False if already released.
        """
        if not self.active:
            return False
        self.active = False
        return True

    @property
    def is_granted(self) -> bool:
        """Check if this permit was granted (not denied)."""
        return self.amount > 0.0 and self.id >= 0


class OptimizationAborted(Exception):
    """Raised when user declines cost approval."""

    pass


@dataclass
class CostEnforcerConfig:
    """Configuration for cost enforcement.

    Attributes:
        limit: Maximum USD spending per optimization run.
        approved: Whether to skip the approval prompt.
        warning_threshold: Fraction of limit at which to emit warnings.
        fallback_trial_limit: If cost cannot be determined, limit by trial count.
        estimated_cost_per_trial: Initial estimate for per-trial cost reservation.
            Used for in-flight budget reservation in parallel execution.
    """

    limit: float = 2.0
    approved: bool = False
    warning_threshold: float = 0.5
    fallback_trial_limit: int = 10
    estimated_cost_per_trial: float = 0.05  # $0.05 default estimate


@dataclass
class CostEstimate:
    """Adaptive cost estimation result.

    Provides estimated remaining cost with confidence level based on
    observed trial cost variance.

    Attributes:
        estimated_remaining_cost: Estimated cost for remaining trials in USD
        confidence: Confidence level (0.0-1.0) based on observed variance
            - 0.0-0.3: Low confidence (few samples or high variance)
            - 0.3-0.7: Medium confidence (moderate samples and variance)
            - 0.7-1.0: High confidence (many samples, low variance)
        samples_used: Number of cost samples used for estimation
        safety_margin: Safety margin multiplier applied (default 1.2)
    """

    estimated_remaining_cost: float
    confidence: float
    samples_used: int
    safety_margin: float = 1.2


@dataclass
class CostStatus:
    """Current cost tracking status for logging/auditing."""

    accumulated_cost_usd: float
    trial_count: int
    limit_usd: float
    unknown_cost_mode: bool
    limit_reached: bool
    warning_threshold_reached: bool = field(default=False)
    in_flight_count: int = field(default=0)
    reserved_cost_usd: float = field(default=0.0)
    estimated_cost_per_trial: float = field(default=0.05)
    cost_confidence: float = field(default=0.5)


class CostEnforcer:
    """Thread-safe real-time cost tracking with user handshake.

    This class provides:
    - Pre-optimization cost estimation and user approval
    - Per-request permit checking for parallel execution
    - Thread-safe and async-safe cost accumulation
    - Fallback to trial count limit when cost is unknown
    - XDG-compliant approval token support for CI/CD

    Usage:
        enforcer = CostEnforcer()

        # Pre-optimization check
        if not enforcer.check_and_approve(estimated_cost=5.0):
            raise OptimizationAborted("User declined")

        # During optimization - acquire permit before each LLM call
        if enforcer.acquire_permit():
            result = await make_llm_call()
            enforcer.track_cost(result.cost)
        else:
            # Cancel this request - limit reached
            pass

    Attributes:
        config: The enforcement configuration.
    """

    def __init__(self, config: CostEnforcerConfig | None = None) -> None:
        """Initialize the cost enforcer.

        Args:
            config: Optional configuration. If None, loads from environment.
        """
        self.config = config or self._load_config()
        self._accumulated_cost: float = 0.0
        self._trial_count: int = 0
        # Single RLock for both sync and async methods (Gemini recommendation).
        # IMPORTANT: Critical sections must remain fast (arithmetic only). Do NOT add:
        # - await statements inside locked blocks (will deadlock)
        # - logging or I/O operations (will cause event-loop stalls in async)
        # The lock is safe to use in async because all critical sections complete
        # without yielding to the event loop.
        self._lock = RLock()
        self._unknown_cost_mode: bool = False
        self._warning_emitted: bool = False
        self._approval_token_path = self._get_approval_token_path()
        # In-flight reservation tracking for parallel execution
        self._in_flight_count: int = 0
        self._reserved_cost: float = 0.0
        self._cost_samples: list[float] = []  # Track actual costs for estimation
        self._estimated_cost: float = self.config.estimated_cost_per_trial

        # Permit tracking for single-release semantics (Phase 1)
        self._permit_counter: int = 0  # Monotonic counter for permit IDs
        self._active_permits: dict[int, Permit] = {}  # Track active permits by ID

        # Cache mock LLM mode at init to prevent mid-run env var changes (Phase 2.2).
        # NOTE: Changing TRAIGENT_MOCK_LLM after CostEnforcer is initialized will
        # have NO EFFECT. This is intentional - toggling mock mode mid-run could
        # cause stranded permits (acquired with tracking, released without tracking).
        # If you need to change mock mode, create a new CostEnforcer instance.
        self._mock_mode_cached: bool = self._check_mock_mode()
        # Cache strict-cost-tracking mode at init for consistent run semantics.
        # NOTE: Changing TRAIGENT_REQUIRE_COST_TRACKING or
        # TRAIGENT_STRICT_COST_ACCOUNTING after CostEnforcer initialization has
        # NO EFFECT. This mirrors mock-mode latching and avoids env reads in hot paths.
        self._require_cost_tracking_cached: bool = (
            self._check_require_cost_tracking_mode()
        )

        # Sync/async usage tracking for mixing detection (Phase 3.2)
        self._sync_used: bool = False
        self._async_used: bool = False

    def _check_mock_mode(self) -> bool:
        """Check if mock LLM mode is enabled from environment.

        When TRAIGENT_MOCK_LLM=true, cost tracking is bypassed because
        there are no real LLM API costs to track.
        """
        return os.getenv("TRAIGENT_MOCK_LLM", "false").lower() == "true"

    @staticmethod
    def _check_require_cost_tracking_mode() -> bool:
        """Read strict cost-tracking mode from environment."""
        require_tracking = (
            os.environ.get("TRAIGENT_REQUIRE_COST_TRACKING", "").lower() == "true"
        )
        strict_accounting = (
            os.environ.get("TRAIGENT_STRICT_COST_ACCOUNTING", "").lower() == "true"
        )
        return require_tracking or strict_accounting

    def _require_cost_tracking(self) -> bool:
        """Return latched strict cost-tracking mode for this instance."""
        return self._require_cost_tracking_cached

    def _check_mixing(self, is_async: bool) -> None:
        """Log when switching between sync and async method usage."""
        if is_async:
            if self._sync_used and not self._async_used:
                logger.info("CostEnforcer: switching from sync to async methods")
            self._async_used = True
        else:
            if self._async_used and not self._sync_used:
                logger.info("CostEnforcer: switching from async to sync methods")
            self._sync_used = True

    def update_limit(self, new_limit: float) -> None:
        """Update the cost limit with synchronization."""
        with self._lock:
            self.config.limit = new_limit

    def _load_config(self) -> CostEnforcerConfig:
        """Load configuration from environment variables with safe parsing."""

        def safe_float(key: str, default: float) -> float:
            val = os.getenv(key)
            if val is None:
                return default
            try:
                parsed = float(val)
                if parsed < 0:
                    logger.warning(
                        f"Negative value for {key}='{val}', using default {default}",
                    )
                    return default
                return parsed
            except ValueError:
                logger.warning(
                    f"Invalid {key}='{val}', using default {default}",
                )
                return default

        def safe_int(key: str, default: int) -> int:
            val = os.getenv(key)
            if val is None:
                return default
            try:
                parsed = int(val)
                if parsed < 1:
                    logger.warning(
                        f"Value too low for {key}='{val}', using default {default}",
                    )
                    return default
                return parsed
            except ValueError:
                logger.warning(
                    f"Invalid {key}='{val}', using default {default}",
                )
                return default

        return CostEnforcerConfig(
            limit=safe_float("TRAIGENT_RUN_COST_LIMIT", 2.0),
            approved=os.getenv("TRAIGENT_COST_APPROVED", "false").lower() == "true",
            warning_threshold=safe_float("TRAIGENT_COST_WARNING_THRESHOLD", 0.5),
            fallback_trial_limit=safe_int("TRAIGENT_FALLBACK_TRIAL_LIMIT", 10),
        )

    def _get_approval_token_path(self) -> Path:
        """Get XDG-compliant token path."""
        xdg_config = os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        return Path(xdg_config) / "traigent" / "cost_approval.token"

    @property
    def accumulated_cost(self) -> float:
        """Get the total accumulated cost (thread-safe)."""
        with self._lock:
            return self._accumulated_cost

    @property
    def trial_count(self) -> int:
        """Get the total trial count (thread-safe)."""
        with self._lock:
            return self._trial_count

    @property
    def is_limit_reached(self) -> bool:
        """Check if the cost or trial limit has been reached (thread-safe)."""
        with self._lock:
            if self._unknown_cost_mode:
                return self._trial_count >= self.config.fallback_trial_limit
            return self._accumulated_cost >= self.config.limit

    @property
    def is_mock_mode(self) -> bool:
        """Check if mock mode is enabled (bypasses all cost tracking).

        Uses cached value from init to prevent mid-run env var changes
        from causing stranded permits.
        """
        return self._mock_mode_cached

    def check_and_approve(self, estimated_cost: float) -> bool:
        """Pre-optimization handshake. Returns True if approved.

        This should be called before starting optimization to get user
        approval if the estimated cost exceeds the configured limit.

        For non-interactive shells (CI/CD), this will fail-safe by aborting
        rather than blocking indefinitely.

        Args:
            estimated_cost: Estimated total cost in USD for the optimization run.

        Returns:
            True if approved to proceed, False if user declined or non-interactive abort.
        """
        if self.is_mock_mode:
            logger.debug("Mock mode enabled, skipping cost approval")
            return True

        if self.config.approved:
            logger.info(
                f"Cost pre-approved via TRAIGENT_COST_APPROVED "
                f"(limit: ${self.config.limit:.2f})"
            )
            return True

        if self._check_approval_token():
            logger.info(
                f"Cost approved via token file " f"(limit: ${self.config.limit:.2f})"
            )
            return True

        if estimated_cost <= self.config.limit:
            logger.debug(
                f"Estimated cost ${estimated_cost:.2f} within limit ${self.config.limit:.2f}"
            )
            return True

        return self._request_user_approval(estimated_cost)

    def _check_approval_token(self) -> bool:
        """Check for XDG approval token file.

        Token format:
            - "approved" - approve with default limit
            - "approved:10.0" - approve with custom limit

        Returns:
            True if valid approval token found.
        """
        if not self._approval_token_path.exists():
            return False

        try:
            content = self._approval_token_path.read_text().strip()
            if content.startswith("approved"):
                if ":" in content:
                    try:
                        token_limit = float(content.split(":")[1])
                        if token_limit > 0:
                            self.config.limit = max(self.config.limit, token_limit)
                            logger.debug(f"Token limit: ${token_limit:.2f}")
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid token format: {content}")
                return True
        except OSError as e:
            logger.warning(f"Failed to read approval token: {e}")

        return False

    def _request_user_approval(self, estimated: float) -> bool:
        """Interactive approval prompt. Fail-safe: abort if non-interactive.

        Args:
            estimated: Estimated cost in USD.

        Returns:
            True if user approved, False otherwise.
        """
        # Check if stdin is a TTY (interactive terminal)
        if not sys.stdin.isatty():
            print(
                f"\nTraigent: Estimated cost ${estimated:.2f} exceeds limit "
                f"${self.config.limit:.2f}.\n"
                f"Set TRAIGENT_COST_APPROVED=true or increase TRAIGENT_RUN_COST_LIMIT.\n",
                file=sys.stderr,
            )
            logger.warning(
                f"Non-interactive mode: aborting (estimated ${estimated:.2f} > "
                f"limit ${self.config.limit:.2f})"
            )
            return False  # Fail-safe: abort in non-interactive mode

        suggested_limit = estimated * 1.5

        print(
            f"""
================================================================================
Traigent Cost Warning
================================================================================

Estimated optimization cost: ${estimated:.2f} USD
Your current cost limit:     ${self.config.limit:.2f} USD

NOTE: This is an ESTIMATE based on maximum context. Actual billing is
      determined solely by your LLM provider.

Options:
  [y] Approve and continue with current limit
  [n] Abort optimization
  [r] Raise limit to ${suggested_limit:.2f} and continue

================================================================================
""",
            file=sys.stderr,
        )

        try:
            choice = input("Enter choice [y/n/r]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.", file=sys.stderr)
            logger.info("User aborted via EOF/interrupt")
            return False

        if choice == "y":
            logger.info(f"User approved estimated cost ${estimated:.2f}")
            return True
        elif choice == "r":
            self.config.limit = suggested_limit
            logger.info(f"User raised limit to ${suggested_limit:.2f}")
            return True
        else:
            logger.info("User declined cost approval")
            return False

    def acquire_permit(self) -> Permit:
        """Token bucket: acquire permit before each LLM request.

        Called immediately before network request, even inside parallel batches.
        This is the main mechanism to prevent cost overruns in parallel execution.

        When a permit is acquired, budget is reserved for the in-flight trial
        to prevent multiple parallel trials from all starting when there's only
        budget for one. The reservation is released when track_cost is called.

        Returns:
            Permit object with is_granted=True if permitted, or is_granted=False
            if limit reached (amount=0.0, id=-1).
        """
        self._check_mixing(is_async=False)

        if self.is_mock_mode:
            # Mock mode: return valid permit with estimated cost, id=0
            return Permit(id=0, amount=self._estimated_cost, active=True)

        with self._lock:
            if self._unknown_cost_mode:
                # In unknown cost mode, use trial count including in-flight
                total_trials = self._trial_count + self._in_flight_count
                if total_trials >= self.config.fallback_trial_limit:
                    return Permit(id=-1, amount=0.0, active=False)  # Denied

                self._in_flight_count += 1
                self._permit_counter += 1
                permit = Permit(
                    id=self._permit_counter,
                    amount=self._estimated_cost,
                    active=True,
                )
                self._active_permits[permit.id] = permit
                return permit

            # Check if there's budget available considering reserved in-flight cost
            reserved_amount = self._estimated_cost
            total_committed = self._accumulated_cost + self._reserved_cost
            if total_committed + reserved_amount > self.config.limit:
                return Permit(id=-1, amount=0.0, active=False)  # Denied

            # Reserve budget for this trial
            self._in_flight_count += 1
            self._reserved_cost += reserved_amount
            self._permit_counter += 1
            permit = Permit(
                id=self._permit_counter,
                amount=reserved_amount,
                active=True,
            )
            self._active_permits[permit.id] = permit
            return permit

    def release_permit(self, permit: Permit) -> bool:
        """Release a permit without tracking cost.

        Use this when a trial is cancelled before it executes (e.g., due to
        errors in setup, not cost limits). This releases the in-flight
        reservation without incrementing the trial count.

        Note: Do NOT call this for trials cancelled due to cost limits -
        those never acquired a permit in the first place.

        This method has single-release semantics: calling it twice on the same
        permit will log a warning and return False on the second call.

        Args:
            permit: The Permit object returned by acquire_permit.

        Returns:
            True if permit was released, False if already released (double-release).
        """
        self._check_mixing(is_async=False)

        if self.is_mock_mode:
            permit.mark_released()  # Mark even in mock mode for consistency
            return True

        with self._lock:
            # Check single-release semantics under lock to prevent race condition
            # where two concurrent calls both see active=True
            if not permit.mark_released():
                logger.warning(
                    "Double-release attempt for permit %d - ignoring",
                    permit.id,
                )
                return False

            # Validate permit exists in active registry
            if permit.id not in self._active_permits:
                logger.warning(
                    "Foreign or already-removed permit %d passed to release_permit",
                    permit.id,
                )
                # Still return True since permit was marked released, but don't
                # adjust counters for unknown permits

            # Remove from active permits registry
            removed = self._active_permits.pop(permit.id, None)

            if removed is not None and self._in_flight_count > 0:
                self._in_flight_count -= 1
                self._reserved_cost = max(0.0, self._reserved_cost - permit.amount)
                logger.debug(
                    "Released permit %d without cost tracking "
                    "(in_flight: %d, released: $%.4f)",
                    permit.id,
                    self._in_flight_count,
                    permit.amount,
                )
            elif removed is None:
                # Already warned above about foreign/removed permit
                pass
            else:
                logger.warning(
                    "release_permit called with no in-flight permits (permit %d)",
                    permit.id,
                )

        return True

    async def release_permit_async(self, permit: Permit) -> bool:
        """Async version of release_permit.

        This method has single-release semantics: calling it twice on the same
        permit will log a warning and return False on the second call.

        Args:
            permit: The Permit object returned by acquire_permit_async.

        Returns:
            True if permit was released, False if already released (double-release).
        """
        self._check_mixing(is_async=True)

        if self.is_mock_mode:
            permit.mark_released()  # Mark even in mock mode for consistency
            return True

        # Use single RLock for both sync and async (Gemini recommendation)
        # Critical section is fast, no I/O
        with self._lock:
            # Check single-release semantics under lock to prevent race condition
            # where two concurrent calls both see active=True
            if not permit.mark_released():
                logger.warning(
                    "Double-release attempt for permit %d - ignoring",
                    permit.id,
                )
                return False

            # Validate permit exists in active registry
            if permit.id not in self._active_permits:
                logger.warning(
                    "Foreign or already-removed permit %d passed to release_permit_async",
                    permit.id,
                )
                # Still return True since permit was marked released, but don't
                # adjust counters for unknown permits

            # Remove from active permits registry
            removed = self._active_permits.pop(permit.id, None)

            if removed is not None and self._in_flight_count > 0:
                self._in_flight_count -= 1
                self._reserved_cost = max(0.0, self._reserved_cost - permit.amount)
                logger.debug(
                    "Released permit %d without cost tracking "
                    "(in_flight: %d, released: $%.4f)",
                    permit.id,
                    self._in_flight_count,
                    permit.amount,
                )
            elif removed is None:
                # Already warned above about foreign/removed permit
                pass
            else:
                logger.warning(
                    "release_permit_async called with no in-flight permits (permit %d)",
                    permit.id,
                )

        return True

    async def acquire_permit_async(self) -> Permit:
        """Async version of acquire_permit for use in async contexts.

        When a permit is acquired, budget is reserved for the in-flight trial
        to prevent multiple parallel trials from all starting when there's only
        budget for one. The reservation is released when track_cost_async is called.

        Returns:
            Permit object with is_granted=True if permitted, or is_granted=False
            if limit reached (amount=0.0, id=-1).
        """
        self._check_mixing(is_async=True)

        if self.is_mock_mode:
            # Mock mode: return valid permit with estimated cost, id=0
            return Permit(id=0, amount=self._estimated_cost, active=True)

        # Use single RLock for both sync and async (Gemini recommendation)
        # Critical section is fast, no I/O
        with self._lock:
            if self._unknown_cost_mode:
                # In unknown cost mode, use trial count including in-flight
                total_trials = self._trial_count + self._in_flight_count
                if total_trials >= self.config.fallback_trial_limit:
                    return Permit(id=-1, amount=0.0, active=False)  # Denied

                self._in_flight_count += 1
                self._permit_counter += 1
                permit = Permit(
                    id=self._permit_counter,
                    amount=self._estimated_cost,
                    active=True,
                )
                self._active_permits[permit.id] = permit
                return permit

            # Check if there's budget available considering reserved in-flight cost
            reserved_amount = self._estimated_cost
            total_committed = self._accumulated_cost + self._reserved_cost
            if total_committed + reserved_amount > self.config.limit:
                return Permit(id=-1, amount=0.0, active=False)  # Denied

            # Reserve budget for this trial
            self._in_flight_count += 1
            self._reserved_cost += reserved_amount
            self._permit_counter += 1
            permit = Permit(
                id=self._permit_counter,
                amount=reserved_amount,
                active=True,
            )
            self._active_permits[permit.id] = permit
            return permit

    def track_cost(
        self,
        cost: float | None,
        *,
        permit: Permit,
        trial_failed: bool = False,
        trial_id: str | None = None,
    ) -> None:
        """Track actual cost after each trial (thread-safe).

        This method releases the in-flight reservation made by acquire_permit
        and records the actual cost. It also updates the cost estimate for
        future reservations based on observed costs.

        This method has single-release semantics via the Permit object.
        If the permit was already released (e.g., via exception handling),
        the cost is still tracked but no reservation is released.

        Args:
            cost: Cost in USD, or None if unknown (triggers fallback mode).
            permit: The Permit object returned by acquire_permit.
            trial_failed: Whether the trial failed/was cancelled (still counts).
            trial_id: Optional trial identifier for logging.

        Raises:
            CostTrackingRequiredError: If cost is None and
                TRAIGENT_REQUIRE_COST_TRACKING=true or
                TRAIGENT_STRICT_COST_ACCOUNTING=true.
            ValueError: If a negative cost is provided.
        """
        self._check_mixing(is_async=False)

        if cost is not None and cost < 0:
            raise ValueError(f"Cost must be non-negative, got {cost}")

        if self.is_mock_mode:
            permit.mark_released()  # Mark for consistency
            return

        # Read environment-driven strictness once per call.
        # Keep this outside the lock to minimize lock hold time.
        require_cost_tracking = self._require_cost_tracking()

        with self._lock:
            # Check if permit was already released (e.g., via exception path)
            # Done under lock to prevent race condition where two concurrent
            # calls both see active=True
            permit_was_active = permit.mark_released()
            if not permit_was_active:
                logger.debug(
                    "Permit %d already released before track_cost - "
                    "tracking cost but not releasing reservation",
                    permit.id,
                )

            # Validate permit exists in active registry (unless already released)
            if permit_was_active and permit.id not in self._active_permits:
                logger.warning(
                    "Foreign or already-removed permit %d passed to track_cost",
                    permit.id,
                )

            self._trial_count += 1
            trial_desc = f"trial {trial_id or self._trial_count}"

            # Release in-flight reservation only if permit was active
            if permit_was_active:
                # Remove from active permits registry
                removed = self._active_permits.pop(permit.id, None)

                if removed is not None and self._in_flight_count > 0:
                    self._in_flight_count -= 1
                    self._reserved_cost = max(0.0, self._reserved_cost - permit.amount)

            # Handle unknown cost with optional strict mode
            if cost is None:
                if require_cost_tracking:
                    raise CostTrackingRequiredError(
                        f"Cost extraction failed for {trial_desc} but "
                        "TRAIGENT_REQUIRE_COST_TRACKING=true or "
                        "TRAIGENT_STRICT_COST_ACCOUNTING=true. "
                        "Set to 'false' or fix cost extraction."
                    )
                if not self._unknown_cost_mode:
                    logger.warning(
                        "Cost unknown for %s. Falling back to trial count limit (%d).",
                        trial_desc,
                        self.config.fallback_trial_limit,
                    )
                    self._unknown_cost_mode = True
            else:
                self._accumulated_cost += cost
                status = "failed" if trial_failed else "completed"
                logger.debug(
                    "%s %s: +$%.4f (total: $%.4f)",
                    trial_desc,
                    status,
                    cost,
                    self._accumulated_cost,
                )

                # Update cost estimate using exponential moving average
                self._update_cost_estimate(cost)

                # Emit warnings at thresholds
                self._check_thresholds()

    def _update_cost_estimate(self, actual_cost: float) -> None:
        """Update the cost estimate using exponential moving average.

        Called with lock held. Uses EMA with alpha=0.3 to balance responsiveness
        with stability. Also maintains a sample buffer for more accurate estimates.
        Logs warning if actual cost diverges significantly from estimate.

        Args:
            actual_cost: The actual cost observed for a trial.
        """
        # Check for divergence before updating estimate
        self._check_cost_divergence(actual_cost)

        # Keep last 10 samples for reference
        self._cost_samples.append(actual_cost)
        if len(self._cost_samples) > 10:
            self._cost_samples.pop(0)

        # Use exponential moving average for estimation
        alpha = 0.3  # Weight for new observations
        self._estimated_cost = alpha * actual_cost + (1 - alpha) * self._estimated_cost

    def _check_cost_divergence(self, actual_cost: float) -> None:
        """Check if actual cost diverges significantly from estimate.

        Called with lock held. Logs warning if ratio exceeds threshold.
        Configured via TRAIGENT_COST_DIVERGENCE_THRESHOLD (default 2.0).

        Args:
            actual_cost: The actual cost observed for a trial.
        """
        if self._estimated_cost <= 0 or actual_cost <= 0:
            return

        threshold_str = os.getenv("TRAIGENT_COST_DIVERGENCE_THRESHOLD", "2.0")
        try:
            threshold = float(threshold_str)
        except ValueError:
            threshold = 2.0

        ratio = actual_cost / self._estimated_cost

        if ratio > threshold:
            logger.warning(
                "Cost divergence detected: actual $%.4f is %.1fx higher than "
                "estimated $%.4f. Estimates may be inaccurate.",
                actual_cost,
                ratio,
                self._estimated_cost,
            )
        elif ratio < 1 / threshold:
            logger.info(
                "Cost below estimate: actual $%.4f is %.1fx lower than "
                "estimated $%.4f. Adjusting estimates.",
                actual_cost,
                1 / ratio,
                self._estimated_cost,
            )

    def get_cost_confidence(self) -> float:
        """Calculate confidence level for cost estimates.

        Confidence is based on the coefficient of variation (CV) of observed
        cost samples. Lower variance relative to mean = higher confidence.

        Returns:
            Confidence level between 0.0 and 1.0:
            - 0.0-0.3: Low confidence (few samples or high variance)
            - 0.3-0.7: Medium confidence (moderate samples/variance)
            - 0.7-1.0: High confidence (many samples, low variance)
        """
        with self._lock:
            if len(self._cost_samples) < 3:
                # Too few samples for reliable confidence
                return 0.5 if len(self._cost_samples) > 0 else 0.3

            try:
                mean = statistics.mean(self._cost_samples)
                if mean <= 0:
                    return 0.3

                variance = statistics.variance(self._cost_samples)
                # Coefficient of variation (CV) = std_dev / mean
                cv = math.sqrt(variance) / mean

                # Map CV to confidence: CV=0 -> conf=1.0, CV=1 -> conf=0.1
                # Using exponential decay for smooth mapping
                confidence = math.exp(-cv * 2)

                # Adjust based on sample count (more samples = higher confidence)
                sample_factor = min(len(self._cost_samples) / 10.0, 1.0)
                confidence = confidence * (0.5 + 0.5 * sample_factor)

                return max(0.1, min(1.0, confidence))

            except (ValueError, ZeroDivisionError):
                return 0.3

    def estimate_remaining_cost(
        self, remaining_trials: int, *, safety_margin: float = 1.2
    ) -> CostEstimate:
        """Estimate cost for remaining trials with confidence.

        Uses adaptive estimation based on observed trial costs:
        - Before 3 trials: Uses initial estimate
        - After 3+ trials: Uses rolling average with confidence-based margin

        Args:
            remaining_trials: Number of trials still to execute
            safety_margin: Multiplier to add safety buffer (default 1.2 = 20%)

        Returns:
            CostEstimate with estimated remaining cost and confidence level
        """
        with self._lock:
            confidence = self.get_cost_confidence()
            samples_used = len(self._cost_samples)

            if samples_used < 3:
                # Use initial estimate with low confidence
                estimated = self._estimated_cost * remaining_trials * safety_margin
                return CostEstimate(
                    estimated_remaining_cost=estimated,
                    confidence=confidence,
                    samples_used=samples_used,
                    safety_margin=safety_margin,
                )

            # Use rolling average of recent samples (more stable than EMA)
            recent_avg = statistics.mean(self._cost_samples[-10:])

            # Adjust safety margin based on confidence (lower confidence = higher margin)
            adjusted_margin = safety_margin + (1 - confidence) * 0.5

            estimated = recent_avg * remaining_trials * adjusted_margin

            return CostEstimate(
                estimated_remaining_cost=estimated,
                confidence=confidence,
                samples_used=samples_used,
                safety_margin=adjusted_margin,
            )

    def _check_thresholds(self) -> None:
        """Check and log warnings at cost thresholds (called with lock held)."""
        if self.config.limit <= 0:
            return

        pct = self._accumulated_cost / self.config.limit

        if pct >= 1.0:
            logger.warning(
                f"COST LIMIT REACHED: ${self._accumulated_cost:.2f} "
                f"(limit: ${self.config.limit:.2f})"
            )
        elif pct >= self.config.warning_threshold and not self._warning_emitted:
            logger.info(
                f"Cost at {pct*100:.0f}% of limit: "
                f"${self._accumulated_cost:.2f} / ${self.config.limit:.2f}"
            )
            self._warning_emitted = True

    async def track_cost_async(
        self,
        cost: float | None,
        *,
        permit: Permit,
        trial_failed: bool = False,
        trial_id: str | None = None,
    ) -> None:
        """Async version of track_cost.

        This method releases the in-flight reservation made by acquire_permit_async
        and records the actual cost. It also updates the cost estimate for
        future reservations based on observed costs.

        This method has single-release semantics via the Permit object.
        If the permit was already released (e.g., via exception handling),
        the cost is still tracked but no reservation is released.

        Args:
            cost: Cost in USD, or None if unknown.
            permit: The Permit object returned by acquire_permit_async.
            trial_failed: Whether the trial failed/was cancelled.
            trial_id: Optional trial identifier for logging.

        Raises:
            CostTrackingRequiredError: If cost is None and
                TRAIGENT_REQUIRE_COST_TRACKING=true or
                TRAIGENT_STRICT_COST_ACCOUNTING=true.
            ValueError: If a negative cost is provided.
        """
        self._check_mixing(is_async=True)

        if cost is not None and cost < 0:
            raise ValueError(f"Cost must be non-negative, got {cost}")

        if self.is_mock_mode:
            permit.mark_released()  # Mark for consistency
            return

        # Read environment-driven strictness once per call.
        # Keep this outside the lock to minimize lock hold time.
        require_cost_tracking = self._require_cost_tracking()

        # Use single RLock for both sync and async (Gemini recommendation)
        # Critical section is fast, no I/O
        with self._lock:
            # Check if permit was already released (e.g., via exception path)
            # Done under lock to prevent race condition where two concurrent
            # calls both see active=True
            permit_was_active = permit.mark_released()
            if not permit_was_active:
                logger.debug(
                    "Permit %d already released before track_cost_async - "
                    "tracking cost but not releasing reservation",
                    permit.id,
                )

            # Validate permit exists in active registry (unless already released)
            if permit_was_active and permit.id not in self._active_permits:
                logger.warning(
                    "Foreign or already-removed permit %d passed to track_cost_async",
                    permit.id,
                )

            self._trial_count += 1
            trial_desc = f"trial {trial_id or self._trial_count}"

            # Release in-flight reservation only if permit was active
            if permit_was_active:
                # Remove from active permits registry
                removed = self._active_permits.pop(permit.id, None)

                if removed is not None and self._in_flight_count > 0:
                    self._in_flight_count -= 1
                    self._reserved_cost = max(0.0, self._reserved_cost - permit.amount)

            # Handle unknown cost with optional strict mode
            if cost is None:
                if require_cost_tracking:
                    raise CostTrackingRequiredError(
                        f"Cost extraction failed for {trial_desc} but "
                        "TRAIGENT_REQUIRE_COST_TRACKING=true or "
                        "TRAIGENT_STRICT_COST_ACCOUNTING=true. "
                        "Set to 'false' or fix cost extraction."
                    )
                if not self._unknown_cost_mode:
                    logger.warning(
                        "Cost unknown for %s, falling back to trial limit",
                        trial_desc,
                    )
                    self._unknown_cost_mode = True
            else:
                self._accumulated_cost += cost
                self._update_cost_estimate(cost)
                self._check_thresholds()

    def get_status(self) -> CostStatus:
        """Get current cost tracking status for logging/auditing.

        Returns:
            CostStatus with current tracking state including cost confidence.
        """
        with self._lock:
            pct = (
                self._accumulated_cost / self.config.limit
                if self.config.limit > 0
                else 0.0
            )
            return CostStatus(
                accumulated_cost_usd=self._accumulated_cost,
                trial_count=self._trial_count,
                limit_usd=self.config.limit,
                unknown_cost_mode=self._unknown_cost_mode,
                limit_reached=self.is_limit_reached,
                warning_threshold_reached=pct >= self.config.warning_threshold,
                in_flight_count=self._in_flight_count,
                reserved_cost_usd=self._reserved_cost,
                estimated_cost_per_trial=self._estimated_cost,
                cost_confidence=self.get_cost_confidence(),
            )

    def reset(self) -> None:
        """Reset the enforcer state for a new optimization run.

        Note: This does NOT reset the configuration, only the accumulated state.
        """
        with self._lock:
            self._accumulated_cost = 0.0
            self._trial_count = 0
            self._unknown_cost_mode = False
            self._warning_emitted = False
            self._in_flight_count = 0
            self._reserved_cost = 0.0
            self._cost_samples = []
            self._estimated_cost = self.config.estimated_cost_per_trial
            # Clear active permits registry (Phase 4 fix)
            self._active_permits.clear()
            # NOTE: Do NOT reset _permit_counter - IDs must remain unique across
            # resets to prevent collisions when orphaned permits are passed to
            # track_cost/release_permit. This was identified by Hypothesis testing.
            self._sync_used = False
            self._async_used = False
            logger.debug("CostEnforcer state reset")

    def _verify_invariants(self) -> list[str]:
        """Runtime invariant verification for debugging.

        Verifies the 8 critical invariants defined in the cost enforcement
        concept specification (docs/traceability/concepts/cost_enforcement.yml).

        This method is designed to be called after state mutations during
        development and testing. Enable via TRAIGENT_DEBUG_INVARIANTS=true.

        Returns:
            List of invariant violation descriptions. Empty list if all hold.

        Invariants checked:
            I1: in_flight_count >= 0
            I2: reserved_cost >= 0
            I3: len(active_permits) == in_flight_count
            I4: accumulated_cost + reserved_cost <= limit + ε
            I5: Released permits have active=False (structural - verified via Permit design)
            I6: Permit IDs monotonically increasing (structural - verified via counter)
            I7: Denied permits: id=-1, amount=0 (structural - verified via construction)
            I8: Sum of active permit amounts equals reserved_cost
        """
        violations: list[str] = []

        with self._lock:
            # I1: in_flight_count >= 0
            if self._in_flight_count < 0:
                violations.append(
                    f"I1 violated: in_flight_count = {self._in_flight_count} < 0"
                )

            # I2: reserved_cost >= 0
            if self._reserved_cost < 0:
                violations.append(
                    f"I2 violated: reserved_cost = {self._reserved_cost:.4f} < 0"
                )

            # I3: len(active_permits) == in_flight_count
            active_count = len(self._active_permits)
            if active_count != self._in_flight_count:
                violations.append(
                    f"I3 violated: len(active_permits) = {active_count} != "
                    f"in_flight_count = {self._in_flight_count}"
                )

            # I4: accumulated + reserved <= limit + ε
            total = self._accumulated_cost + self._reserved_cost
            epsilon = 0.0001  # Floating point tolerance
            if total > self.config.limit + epsilon:
                violations.append(
                    f"I4 violated: accumulated ({self._accumulated_cost:.4f}) + "
                    f"reserved ({self._reserved_cost:.4f}) = {total:.4f} > "
                    f"limit ({self.config.limit:.4f}) + ε"
                )

            # I5, I6, I7 are structural invariants enforced by code design:
            # - I5: Permit.mark_released() sets active=False atomically
            # - I6: _permit_counter only increments, never decreases
            # - I7: Denied permits are constructed with id=-1, amount=0.0, active=False

            # I8: Sum of active permit amounts equals reserved_cost
            permit_sum = sum(p.amount for p in self._active_permits.values())
            if abs(permit_sum - self._reserved_cost) > epsilon:
                violations.append(
                    f"I8 violated: sum(permit.amount) = {permit_sum:.4f} != "
                    f"reserved_cost = {self._reserved_cost:.4f}"
                )

        if violations:
            for v in violations:
                logger.error("Invariant violation: %s", v)

        return violations

    def assert_invariants(self) -> None:
        """Assert all invariants hold. Raises AssertionError if any fail.

        Use this method in tests or enable via TRAIGENT_DEBUG_INVARIANTS=true
        environment variable for automatic checking after each mutation.

        Raises:
            AssertionError: If any invariant is violated.
        """
        violations = self._verify_invariants()
        if violations:
            raise AssertionError(
                "CostEnforcer invariant violations:\n  - " + "\n  - ".join(violations)
            )

    def __repr__(self) -> str:
        status = self.get_status()
        return (
            f"CostEnforcer("
            f"accumulated=${status.accumulated_cost_usd:.2f}, "
            f"limit=${status.limit_usd:.2f}, "
            f"trials={status.trial_count}, "
            f"limit_reached={status.limit_reached})"
        )


__all__ = [
    "CostEnforcer",
    "CostEnforcerConfig",
    "CostEstimate",
    "CostLimitExceeded",
    "CostStatus",
    "CostTrackingRequiredError",
    "OptimizationAborted",
    "Permit",
]
