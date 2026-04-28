"""Unit tests for cost enforcement module.

Tests CostEnforcer, CostLimitStopCondition, and integration with orchestrator.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Generator
from datetime import UTC
from pathlib import Path
from unittest.mock import patch

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.cost_enforcement import (
    DEFAULT_COST_LIMIT_USD,
    EMA_COLD_START_WARNING_TRIALS,
    CostEnforcer,
    CostEnforcerConfig,
    CostEstimate,
    CostLimitExceeded,
    CostStatus,
    CostTrackingRequiredError,
    OptimizationAborted,
    Permit,
)
from traigent.core.stop_conditions import CostLimitStopCondition


def _create_mock_permit(amount: float = 0.0) -> Permit:
    """Create a mock permit for tests that don't need real permit tracking."""
    return Permit(id=0, amount=amount, active=True)


UNKNOWN_COST_PAIRWISE_CASES = [
    pytest.param(
        False,
        False,
        False,
        id="require-off_strict-off_permit-active",
    ),
    pytest.param(
        False,
        True,
        True,
        id="require-off_strict-on_permit-released",
    ),
    pytest.param(
        True,
        False,
        True,
        id="require-on_strict-off_permit-released",
    ),
    pytest.param(
        True,
        True,
        False,
        id="require-on_strict-on_permit-active",
    ),
]


class TestCostEnforcerConfig:
    """Tests for CostEnforcerConfig defaults and validation."""

    def test_default_values(self) -> None:
        """Default config has sensible values."""
        config = CostEnforcerConfig()
        assert config.limit == DEFAULT_COST_LIMIT_USD
        assert config.approved is False
        assert config.warning_threshold == 0.5
        assert config.fallback_trial_limit == 10

    def test_custom_values(self) -> None:
        """Custom config values are preserved."""
        config = CostEnforcerConfig(
            limit=10.0,
            approved=True,
            warning_threshold=0.8,
            fallback_trial_limit=20,
        )
        assert config.limit == 10.0
        assert config.approved is True
        assert config.warning_threshold == 0.8
        assert config.fallback_trial_limit == 20


class TestCostEnforcerBasic:
    """Basic CostEnforcer functionality tests."""

    @pytest.fixture(autouse=True)
    def clear_mock_mode(self) -> Generator[None, None, None]:
        """Ensure mock mode is disabled for tests that need real tracking."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "false"}):
            yield

    def test_initialization_default(self) -> None:
        """CostEnforcer initializes with defaults."""
        enforcer = CostEnforcer()
        assert enforcer.accumulated_cost == 0.0
        assert enforcer.trial_count == 0
        assert not enforcer.is_limit_reached

    def test_initialization_custom_config(self) -> None:
        """CostEnforcer respects custom config."""
        config = CostEnforcerConfig(limit=5.0)
        enforcer = CostEnforcer(config=config)
        assert enforcer.config.limit == 5.0

    def test_track_cost_accumulates(self) -> None:
        """track_cost properly accumulates costs."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=10.0))
        # Use mock permits for simple cost tracking tests
        enforcer.track_cost(1.0, permit=_create_mock_permit())
        enforcer.track_cost(2.5, permit=_create_mock_permit())
        enforcer.track_cost(0.5, permit=_create_mock_permit())

        assert enforcer.accumulated_cost == 4.0
        assert enforcer.trial_count == 3

    def test_track_cost_rejects_negative(self) -> None:
        """track_cost rejects negative costs."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=10.0))
        with pytest.raises(ValueError, match="non-negative"):
            enforcer.track_cost(-0.5, permit=_create_mock_permit())

    def test_limit_reached_detection(self) -> None:
        """is_limit_reached returns True when limit exceeded."""
        config = CostEnforcerConfig(limit=2.0)
        enforcer = CostEnforcer(config=config)

        enforcer.track_cost(1.0, permit=_create_mock_permit())
        assert not enforcer.is_limit_reached

        enforcer.track_cost(1.5, permit=_create_mock_permit())
        assert enforcer.is_limit_reached

    def test_session_isolation(self) -> None:
        """Verify CostEnforcer state doesn't leak between sessions."""
        enforcer1 = CostEnforcer(config=CostEnforcerConfig(limit=10.0))
        enforcer1.track_cost(0.5, permit=_create_mock_permit())

        enforcer2 = CostEnforcer(config=CostEnforcerConfig(limit=10.0))
        assert enforcer2.accumulated_cost == 0.0
        assert enforcer2.trial_count == 0

        # Verify enforcer1 is still intact
        assert enforcer1.accumulated_cost == 0.5

    def test_reset_clears_state(self) -> None:
        """reset() clears accumulated state including permits."""
        enforcer = CostEnforcer()
        enforcer.track_cost(1.0, permit=_create_mock_permit())

        # Acquire a permit but don't release it to test permit clearing
        permit = enforcer.acquire_permit()
        assert permit is not None
        assert enforcer.get_status().in_flight_count == 1

        enforcer.reset()

        assert enforcer.accumulated_cost == 0.0
        assert enforcer.trial_count == 0
        assert enforcer.get_status().in_flight_count == 0
        assert enforcer.get_status().reserved_cost_usd == 0.0
        # Verify internal state
        assert len(enforcer._active_permits) == 0

    def test_acquire_permit_before_limit(self) -> None:
        """acquire_permit returns granted Permit before limit."""
        config = CostEnforcerConfig(limit=5.0)
        enforcer = CostEnforcer(config=config)
        permit = enforcer.acquire_permit()
        enforcer.track_cost(2.0, permit=permit)

        permit = enforcer.acquire_permit()
        assert permit.is_granted, "Permit should be granted"
        assert permit.amount > 0, "Permit should have positive reserved amount"

    def test_acquire_permit_at_limit(self) -> None:
        """acquire_permit returns denied Permit at limit."""
        config = CostEnforcerConfig(limit=2.0)
        enforcer = CostEnforcer(config=config)
        permit = enforcer.acquire_permit()
        enforcer.track_cost(2.0, permit=permit)

        permit = enforcer.acquire_permit()
        assert not permit.is_granted, "Permit should be denied"
        assert permit.amount <= 0, "Permit should have zero reserved amount"


class TestCostEnforcerMockMode:
    """Tests pinning that TRAIGENT_MOCK_LLM is NOT honored by CostEnforcer.

    S2-B Round 3 removed the mock-mode bypass: cost approval, permit
    acquisition, and cost tracking always run regardless of the env flag.
    """

    def test_mock_mode_does_not_bypass_tracking(self) -> None:
        """track_cost must update accumulated_cost even with TRAIGENT_MOCK_LLM=true."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "true"}):
            enforcer = CostEnforcer()
            permit = enforcer.acquire_permit()
            enforcer.track_cost(0.05, permit=permit)
            assert enforcer.accumulated_cost == pytest.approx(0.05)

    def test_mock_mode_does_not_force_permits(self) -> None:
        """Permit denial under tight limits must still occur with TRAIGENT_MOCK_LLM=true."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "true"}):
            config = CostEnforcerConfig(
                limit=0.01, estimated_cost_per_trial=1.0
            )  # Very low limit, high estimate
            enforcer = CostEnforcer(config=config)

            permit = enforcer.acquire_permit()
            # First permit reserves budget; subsequent acquisitions must be
            # denied because the in-flight reservation already exceeds limit.
            second = enforcer.acquire_permit()
            assert not second.is_granted, (
                "Mock mode must not override real permit limits"
            )
            enforcer.release_permit(permit)

    def test_mock_mode_does_not_auto_approve(self) -> None:
        """check_and_approve must NOT short-circuit when TRAIGENT_MOCK_LLM=true.

        With a non-interactive shell and an estimated cost above the limit,
        approval should fail-safe to False (no auto-approval via mock mode).
        """
        with patch.dict(
            os.environ,
            {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_COST_APPROVED": "false"},
        ):
            enforcer = CostEnforcer(CostEnforcerConfig(limit=1.0, approved=False))
            # Far above limit — without auto-approval and with no interactive
            # shell available, this must NOT return True solely due to mock mode.
            with patch(
                "traigent.core.cost_enforcement.CostEnforcer._request_user_approval",
                return_value=False,
            ):
                assert enforcer.check_and_approve(1000.0) is False


class TestCostEnforcerUnknownCost:
    """Tests for unknown cost fallback mode."""

    @pytest.fixture(autouse=True)
    def clear_mock_mode(self) -> Generator[None, None, None]:
        """Ensure mock mode is disabled for tests that need real tracking."""
        with patch.dict(
            os.environ,
            {
                "TRAIGENT_MOCK_LLM": "false",
                "TRAIGENT_REQUIRE_COST_TRACKING": "false",
                "TRAIGENT_STRICT_COST_ACCOUNTING": "false",
            },
        ):
            yield

    def test_unknown_cost_triggers_fallback(self) -> None:
        """None cost triggers fallback to trial count mode."""
        config = CostEnforcerConfig(fallback_trial_limit=5)
        enforcer = CostEnforcer(config=config)

        enforcer.track_cost(None, permit=_create_mock_permit())

        status = enforcer.get_status()
        assert status.unknown_cost_mode is True

    def test_fallback_uses_trial_limit(self) -> None:
        """Fallback mode uses trial count limit."""
        config = CostEnforcerConfig(fallback_trial_limit=3)
        enforcer = CostEnforcer(config=config)

        enforcer.track_cost(None, permit=_create_mock_permit())  # Trigger fallback
        enforcer.track_cost(None, permit=_create_mock_permit())
        assert not enforcer.is_limit_reached

        enforcer.track_cost(None, permit=_create_mock_permit())
        assert enforcer.is_limit_reached

    def test_fallback_permit_check(self) -> None:
        """Permit check works in fallback mode."""
        config = CostEnforcerConfig(fallback_trial_limit=2)
        enforcer = CostEnforcer(config=config)

        permit = enforcer.acquire_permit()
        enforcer.track_cost(None, permit=permit)
        permit = enforcer.acquire_permit()
        assert permit.is_granted, "First permit in fallback mode should be granted"

        enforcer.track_cost(None, permit=permit)
        permit = enforcer.acquire_permit()
        assert not permit.is_granted, "Third permit should be denied (limit=2)"

    def test_unknown_cost_raises_when_strict_accounting_enabled(self) -> None:
        """Strict accounting mode raises instead of entering fallback mode."""
        with patch.dict(
            os.environ, {"TRAIGENT_STRICT_COST_ACCOUNTING": "true"}, clear=False
        ):
            config = CostEnforcerConfig(fallback_trial_limit=5)
            enforcer = CostEnforcer(config=config)
            with pytest.raises(CostTrackingRequiredError) as exc_info:
                enforcer.track_cost(None, permit=_create_mock_permit())

        assert "TRAIGENT_STRICT_COST_ACCOUNTING=true" in str(exc_info.value)
        assert enforcer.get_status().unknown_cost_mode is False

    def test_strict_mode_is_latched_at_init(self) -> None:
        """Strict mode env changes after init do not affect existing enforcer."""
        with patch.dict(
            os.environ, {"TRAIGENT_STRICT_COST_ACCOUNTING": "false"}, clear=False
        ):
            enforcer = CostEnforcer(CostEnforcerConfig(fallback_trial_limit=5))

        with patch.dict(
            os.environ, {"TRAIGENT_STRICT_COST_ACCOUNTING": "true"}, clear=False
        ):
            enforcer.track_cost(None, permit=_create_mock_permit())

        assert enforcer.get_status().unknown_cost_mode is True

    def test_require_cost_tracking_is_latched_at_init(self) -> None:
        """Require-cost-tracking env changes after init do not affect instance."""
        with patch.dict(
            os.environ, {"TRAIGENT_REQUIRE_COST_TRACKING": "true"}, clear=False
        ):
            enforcer = CostEnforcer(CostEnforcerConfig(fallback_trial_limit=5))

        with patch.dict(
            os.environ, {"TRAIGENT_REQUIRE_COST_TRACKING": "false"}, clear=False
        ):
            with pytest.raises(CostTrackingRequiredError):
                enforcer.track_cost(None, permit=_create_mock_permit())


class TestCostEnforcerUnknownCostCTD:
    """Pairwise coverage of strict-mode unknown-cost behavior."""

    @pytest.mark.parametrize(
        ("require_tracking", "strict_accounting", "permit_pre_released"),
        UNKNOWN_COST_PAIRWISE_CASES,
    )
    def test_unknown_cost_sync_pairwise(
        self,
        require_tracking: bool,
        strict_accounting: bool,
        permit_pre_released: bool,
    ) -> None:
        """Pairwise matrix: env flags x permit state for sync track_cost."""
        with patch.dict(
            os.environ,
            {
                "TRAIGENT_MOCK_LLM": "false",
                "TRAIGENT_REQUIRE_COST_TRACKING": str(require_tracking).lower(),
                "TRAIGENT_STRICT_COST_ACCOUNTING": str(strict_accounting).lower(),
            },
            clear=False,
        ):
            enforcer = CostEnforcer(
                CostEnforcerConfig(
                    limit=1.0,
                    estimated_cost_per_trial=0.2,
                    fallback_trial_limit=5,
                )
            )
            permit = enforcer.acquire_permit()
            assert permit.is_granted
            assert enforcer.get_status().in_flight_count == 1
            if permit_pre_released:
                assert enforcer.release_permit(permit) is True
                assert enforcer.get_status().in_flight_count == 0
            else:
                assert enforcer.get_status().in_flight_count == 1

            should_raise = require_tracking or strict_accounting
            if should_raise:
                with pytest.raises(CostTrackingRequiredError):
                    enforcer.track_cost(None, permit=permit)
                status = enforcer.get_status()
                assert status.unknown_cost_mode is False
            else:
                enforcer.track_cost(None, permit=permit)
                status = enforcer.get_status()
                assert status.unknown_cost_mode is True

            assert status.trial_count == 1
            assert status.in_flight_count == 0
            assert status.reserved_cost_usd == pytest.approx(0.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("require_tracking", "strict_accounting", "permit_pre_released"),
        UNKNOWN_COST_PAIRWISE_CASES,
    )
    async def test_unknown_cost_async_pairwise(
        self,
        require_tracking: bool,
        strict_accounting: bool,
        permit_pre_released: bool,
    ) -> None:
        """Pairwise matrix: env flags x permit state for async track_cost."""
        with patch.dict(
            os.environ,
            {
                "TRAIGENT_MOCK_LLM": "false",
                "TRAIGENT_REQUIRE_COST_TRACKING": str(require_tracking).lower(),
                "TRAIGENT_STRICT_COST_ACCOUNTING": str(strict_accounting).lower(),
            },
            clear=False,
        ):
            enforcer = CostEnforcer(
                CostEnforcerConfig(
                    limit=1.0,
                    estimated_cost_per_trial=0.2,
                    fallback_trial_limit=5,
                )
            )
            permit = await enforcer.acquire_permit_async()
            assert permit.is_granted
            assert enforcer.get_status().in_flight_count == 1
            if permit_pre_released:
                assert await enforcer.release_permit_async(permit) is True
                assert enforcer.get_status().in_flight_count == 0
            else:
                assert enforcer.get_status().in_flight_count == 1

            should_raise = require_tracking or strict_accounting
            if should_raise:
                with pytest.raises(CostTrackingRequiredError):
                    await enforcer.track_cost_async(None, permit=permit)
                status = enforcer.get_status()
                assert status.unknown_cost_mode is False
            else:
                await enforcer.track_cost_async(None, permit=permit)
                status = enforcer.get_status()
                assert status.unknown_cost_mode is True

            assert status.trial_count == 1
            assert status.in_flight_count == 0
            assert status.reserved_cost_usd == pytest.approx(0.0)


class TestCostEnforcerThreadSafety:
    """Thread safety tests for CostEnforcer."""

    @pytest.fixture(autouse=True)
    def clear_mock_mode(self) -> Generator[None, None, None]:
        """Ensure mock mode is disabled for tests that need real tracking."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "false"}):
            yield

    def test_concurrent_track_cost(self) -> None:
        """Concurrent track_cost calls are thread-safe."""
        config = CostEnforcerConfig(limit=1000.0)
        enforcer = CostEnforcer(config=config)
        num_threads = 10
        costs_per_thread = 100

        def track_costs() -> None:
            for _ in range(costs_per_thread):
                enforcer.track_cost(0.01, permit=_create_mock_permit())

        threads = [threading.Thread(target=track_costs) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_cost = num_threads * costs_per_thread * 0.01
        # Allow for small floating point variance
        assert abs(enforcer.accumulated_cost - expected_cost) < 0.001
        assert enforcer.trial_count == num_threads * costs_per_thread


class TestCostEnforcerAsync:
    """Async functionality tests."""

    @pytest.fixture(autouse=True)
    def clear_mock_mode(self) -> Generator[None, None, None]:
        """Ensure mock mode is disabled for tests that need real tracking."""
        with patch.dict(
            os.environ,
            {
                "TRAIGENT_MOCK_LLM": "false",
                "TRAIGENT_REQUIRE_COST_TRACKING": "false",
                "TRAIGENT_STRICT_COST_ACCOUNTING": "false",
            },
        ):
            yield

    @pytest.mark.asyncio
    async def test_acquire_permit_async(self) -> None:
        """acquire_permit_async works correctly (returns Permit with is_granted status)."""
        config = CostEnforcerConfig(limit=5.0)
        enforcer = CostEnforcer(config=config)

        permit = await enforcer.acquire_permit_async()
        assert permit.is_granted, "First permit should be granted"

        await enforcer.track_cost_async(5.0, permit=permit)
        permit = await enforcer.acquire_permit_async()
        assert not permit.is_granted, "Permit should be denied after limit reached"

    @pytest.mark.asyncio
    async def test_track_cost_async(self) -> None:
        """track_cost_async accumulates correctly."""
        enforcer = CostEnforcer()

        await enforcer.track_cost_async(1.0, permit=_create_mock_permit())
        await enforcer.track_cost_async(2.0, permit=_create_mock_permit())

        assert enforcer.accumulated_cost == 3.0
        assert enforcer.trial_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_async_tracking(self) -> None:
        """Concurrent async track_cost calls are safe."""
        config = CostEnforcerConfig(limit=1000.0)
        enforcer = CostEnforcer(config=config)

        async def track_many() -> None:
            for _ in range(50):
                await enforcer.track_cost_async(0.01, permit=_create_mock_permit())

        await asyncio.gather(*[track_many() for _ in range(10)])

        expected = 10 * 50 * 0.01
        assert abs(enforcer.accumulated_cost - expected) < 0.001

    @pytest.mark.asyncio
    async def test_unknown_cost_async_raises_when_strict_accounting_enabled(
        self,
    ) -> None:
        """Strict accounting mode raises on async unknown cost."""
        with patch.dict(
            os.environ, {"TRAIGENT_STRICT_COST_ACCOUNTING": "true"}, clear=False
        ):
            enforcer = CostEnforcer(config=CostEnforcerConfig())
            permit = _create_mock_permit()
            with pytest.raises(CostTrackingRequiredError) as exc_info:
                await enforcer.track_cost_async(None, permit=permit)

        assert "TRAIGENT_STRICT_COST_ACCOUNTING=true" in str(exc_info.value)


class TestCostEnforcerApproval:
    """Tests for cost approval flow."""

    @pytest.fixture(autouse=True)
    def clear_mock_mode(self) -> Generator[None, None, None]:
        """Ensure mock mode is disabled for tests that need real tracking."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "false"}):
            yield

    def test_pre_approved_via_config(self) -> None:
        """Pre-approved config skips handshake."""
        config = CostEnforcerConfig(approved=True)
        enforcer = CostEnforcer(config=config)

        # High estimated cost should still be approved
        assert enforcer.check_and_approve(100.0) is True

    def test_within_limit_auto_approved(self) -> None:
        """Costs within limit are auto-approved."""
        config = CostEnforcerConfig(limit=10.0, approved=False)
        enforcer = CostEnforcer(config=config)

        assert enforcer.check_and_approve(5.0) is True

    def test_non_interactive_aborts(self) -> None:
        """Non-interactive mode aborts when approval needed."""
        config = CostEnforcerConfig(limit=1.0, approved=False)
        enforcer = CostEnforcer(config=config)

        with patch("sys.stdin.isatty", return_value=False):
            # Cost exceeds limit, not approved, non-interactive
            assert enforcer.check_and_approve(10.0) is False


class TestCostStatus:
    """Tests for CostStatus reporting."""

    @pytest.fixture(autouse=True)
    def clear_mock_mode(self) -> Generator[None, None, None]:
        """Ensure mock mode is disabled for tests that need real tracking."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "false"}):
            yield

    def test_get_status_structure(self) -> None:
        """get_status returns complete status."""
        config = CostEnforcerConfig(limit=10.0)
        enforcer = CostEnforcer(config=config)
        enforcer.track_cost(3.0, permit=_create_mock_permit())

        status = enforcer.get_status()

        assert isinstance(status, CostStatus)
        assert status.accumulated_cost_usd == 3.0
        assert status.trial_count == 1
        assert status.limit_usd == 10.0
        assert status.unknown_cost_mode is False
        assert status.limit_reached is False

    def test_status_warning_threshold(self) -> None:
        """Status tracks warning threshold."""
        config = CostEnforcerConfig(limit=10.0, warning_threshold=0.5)
        enforcer = CostEnforcer(config=config)

        enforcer.track_cost(4.0, permit=_create_mock_permit())
        status = enforcer.get_status()
        assert status.warning_threshold_reached is False

        enforcer.track_cost(2.0, permit=_create_mock_permit())  # Now at 60%
        status = enforcer.get_status()
        assert status.warning_threshold_reached is True


class TestCostLimitStopCondition:
    """Tests for CostLimitStopCondition."""

    @pytest.fixture(autouse=True)
    def clear_mock_mode(self) -> Generator[None, None, None]:
        """Ensure mock mode is disabled for tests that need real tracking."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "false"}):
            yield

    def _make_trial(self, trial_id: str = "test") -> TrialResult:
        """Create a minimal trial result."""
        from datetime import datetime

        return TrialResult(
            trial_id=trial_id,
            config={},
            metrics={},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(UTC),
        )

    def test_stop_when_limit_reached(self) -> None:
        """Stop condition triggers when limit reached."""
        config = CostEnforcerConfig(limit=1.0)
        enforcer = CostEnforcer(config=config)
        condition = CostLimitStopCondition(enforcer)

        trials = [self._make_trial()]

        # Before limit
        assert condition.should_stop(trials) is False

        # After limit
        enforcer.track_cost(2.0, permit=_create_mock_permit())
        assert condition.should_stop(trials) is True

    def test_reason_attribute(self) -> None:
        """Condition has proper reason attribute."""
        enforcer = CostEnforcer()
        condition = CostLimitStopCondition(enforcer)

        assert condition.reason == "cost_limit"

    def test_get_reason_message(self) -> None:
        """get_reason provides descriptive message."""
        config = CostEnforcerConfig(limit=5.0)
        enforcer = CostEnforcer(config=config)
        condition = CostLimitStopCondition(enforcer)

        enforcer.track_cost(5.0, permit=_create_mock_permit())
        reason = condition.get_reason()

        assert "Cost limit reached" in reason
        assert "$5.00" in reason

    def test_reset_not_affect_enforcer(self) -> None:
        """Reset on condition doesn't affect shared enforcer."""
        config = CostEnforcerConfig(limit=10.0)
        enforcer = CostEnforcer(config=config)
        condition = CostLimitStopCondition(enforcer)

        enforcer.track_cost(5.0, permit=_create_mock_permit())
        condition.reset()

        # Enforcer state should be unchanged
        assert enforcer.accumulated_cost == 5.0


class TestCostEnforcerEnvironmentVariables:
    """Tests for environment variable configuration."""

    def test_load_limit_from_env(self) -> None:
        """TRAIGENT_RUN_COST_LIMIT is respected."""
        with patch.dict(os.environ, {"TRAIGENT_RUN_COST_LIMIT": "15.0"}):
            enforcer = CostEnforcer()
            assert enforcer.config.limit == 15.0

    def test_load_approved_from_env(self) -> None:
        """TRAIGENT_COST_APPROVED is respected."""
        with patch.dict(os.environ, {"TRAIGENT_COST_APPROVED": "true"}):
            enforcer = CostEnforcer()
            assert enforcer.config.approved is True

    def test_invalid_env_uses_default(self) -> None:
        """Invalid environment values fall back to defaults."""
        with patch.dict(os.environ, {"TRAIGENT_RUN_COST_LIMIT": "not_a_number"}):
            enforcer = CostEnforcer()
            assert enforcer.config.limit == DEFAULT_COST_LIMIT_USD

    def test_negative_env_uses_default(self) -> None:
        """Negative values fall back to defaults."""
        with patch.dict(os.environ, {"TRAIGENT_RUN_COST_LIMIT": "-5.0"}):
            enforcer = CostEnforcer()
            assert enforcer.config.limit == DEFAULT_COST_LIMIT_USD

    def test_invalid_divergence_threshold_raises(self) -> None:
        """Non-numeric divergence threshold should fail fast at initialization."""
        with patch.dict(os.environ, {"TRAIGENT_COST_DIVERGENCE_THRESHOLD": "abc"}):
            with pytest.raises(ValueError, match="TRAIGENT_COST_DIVERGENCE_THRESHOLD"):
                CostEnforcer()


class TestCostEnforcerRepr:
    """Tests for string representation."""

    @pytest.fixture(autouse=True)
    def clear_mock_mode(self) -> Generator[None, None, None]:
        """Ensure mock mode is disabled for tests that need real tracking."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "false"}):
            yield

    def test_repr_format(self) -> None:
        """Repr provides useful information."""
        config = CostEnforcerConfig(limit=10.0)
        enforcer = CostEnforcer(config=config)
        enforcer.track_cost(3.0, permit=_create_mock_permit())

        repr_str = repr(enforcer)

        assert "CostEnforcer" in repr_str
        assert "accumulated=$3.00" in repr_str
        assert "limit=$10.00" in repr_str
        assert "trials=1" in repr_str


class TestCostEnforcerExceptions:
    """Tests for exception classes."""

    def test_cost_limit_exceeded_message(self) -> None:
        """CostLimitExceeded has informative message."""
        exc = CostLimitExceeded(accumulated=5.0, limit=2.0)
        assert "5.00" in str(exc)
        assert "2.00" in str(exc)
        assert exc.accumulated == 5.0
        assert exc.limit == 2.0

    def test_optimization_aborted(self) -> None:
        """OptimizationAborted can be raised."""
        with pytest.raises(OptimizationAborted):
            raise OptimizationAborted("User declined")


class TestCostEnforcerInFlightReservation:
    """Tests for in-flight reservation tracking in parallel execution."""

    @pytest.fixture(autouse=True)
    def clear_mock_mode(self) -> Generator[None, None, None]:
        """Ensure mock mode is disabled for tests that need real tracking."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "false"}):
            yield

    def test_acquire_permit_reserves_budget(self) -> None:
        """acquire_permit reserves estimated cost for in-flight trial."""
        config = CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.2)
        enforcer = CostEnforcer(config=config)

        # First 5 permits should be granted (5 * 0.2 = 1.0 = limit)
        permits: list[Permit] = []
        for i in range(5):
            permit = enforcer.acquire_permit()
            assert permit.is_granted, f"Permit {i} should be granted"
            permits.append(permit)

        # 6th permit should be denied (would exceed limit)
        permit = enforcer.acquire_permit()
        assert not permit.is_granted, "6th permit should be denied"

        # Check in-flight status
        status = enforcer.get_status()
        assert status.in_flight_count == 5
        assert status.reserved_cost_usd == 1.0

    def test_track_cost_releases_reservation(self) -> None:
        """track_cost releases the in-flight reservation."""
        config = CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.2)
        enforcer = CostEnforcer(config=config)

        # Acquire permit
        permit = enforcer.acquire_permit()
        assert enforcer.get_status().in_flight_count == 1

        # Track cost releases reservation
        enforcer.track_cost(0.15, permit=permit)
        status = enforcer.get_status()
        assert status.in_flight_count == 0
        assert status.reserved_cost_usd == 0.0
        assert status.accumulated_cost_usd == 0.15

    def test_actual_cost_overrun_denies_future_permits(self) -> None:
        """Actual cost may exceed estimate, but future permits are denied."""
        config = CostEnforcerConfig(limit=0.10, estimated_cost_per_trial=0.05)
        enforcer = CostEnforcer(config=config)

        permit = enforcer.acquire_permit()
        assert permit.is_granted

        enforcer.track_cost(0.20, permit=permit)
        status = enforcer.get_status()
        assert status.accumulated_cost_usd == pytest.approx(0.20)
        assert status.reserved_cost_usd == 0.0
        assert status.limit_reached is True

        next_permit = enforcer.acquire_permit()
        assert not next_permit.is_granted

    def test_release_permit_without_tracking(self) -> None:
        """release_permit releases reservation without tracking cost."""
        config = CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.2)
        enforcer = CostEnforcer(config=config)

        # Acquire permit
        permit = enforcer.acquire_permit()
        assert enforcer.get_status().in_flight_count == 1

        # Release without tracking
        enforcer.release_permit(permit)
        status = enforcer.get_status()
        assert status.in_flight_count == 0
        assert status.reserved_cost_usd == 0.0
        assert status.accumulated_cost_usd == 0.0  # No cost tracked
        assert status.trial_count == 0  # No trial counted

    def test_parallel_permits_respect_budget(self) -> None:
        """Multiple parallel permits respect total budget including reserved."""
        config = CostEnforcerConfig(limit=0.5, estimated_cost_per_trial=0.1)
        enforcer = CostEnforcer(config=config)

        # Acquire 5 permits (5 * 0.1 = 0.5 = limit)
        permits_granted = 0
        for _ in range(10):  # Try to acquire more than should fit
            permit = enforcer.acquire_permit()
            if permit.is_granted:
                permits_granted += 1

        assert permits_granted == 5  # Only 5 should fit

    def test_cost_estimate_updates_with_actual_costs(self) -> None:
        """Cost estimate is updated based on actual costs."""
        config = CostEnforcerConfig(limit=10.0, estimated_cost_per_trial=0.1)
        enforcer = CostEnforcer(config=config)

        initial_estimate = enforcer._estimated_cost

        # Track some actual costs
        permit = enforcer.acquire_permit()
        enforcer.track_cost(0.5, permit=permit)  # Much higher than initial estimate

        # Estimate should have increased (EMA with alpha=0.3)
        assert enforcer._estimated_cost > initial_estimate
        # Expected: 0.3 * 0.5 + 0.7 * 0.1 = 0.15 + 0.07 = 0.22
        expected = 0.3 * 0.5 + 0.7 * 0.1
        assert abs(enforcer._estimated_cost - expected) < 0.001

    def test_reset_clears_in_flight_state(self) -> None:
        """reset() clears all in-flight tracking state."""
        config = CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.2)
        enforcer = CostEnforcer(config=config)

        # Create some in-flight state
        permit1 = enforcer.acquire_permit()
        enforcer.acquire_permit()
        enforcer.track_cost(0.3, permit=permit1)  # Updates estimate

        # Reset
        enforcer.reset()

        status = enforcer.get_status()
        assert status.in_flight_count == 0
        assert status.reserved_cost_usd == 0.0
        assert status.accumulated_cost_usd == 0.0
        assert enforcer._estimated_cost == 0.2  # Back to config default

    @pytest.mark.asyncio
    async def test_async_permit_reservation(self) -> None:
        """Async permit acquisition reserves budget correctly."""
        config = CostEnforcerConfig(limit=0.5, estimated_cost_per_trial=0.1)
        enforcer = CostEnforcer(config=config)

        # Acquire permits concurrently
        async def try_acquire() -> Permit:
            return await enforcer.acquire_permit_async()

        results = await asyncio.gather(*[try_acquire() for _ in range(10)])
        # Count permits granted
        permits_granted = sum(1 for permit in results if permit.is_granted)

        assert permits_granted == 5  # Only 5 should fit

    @pytest.mark.asyncio
    async def test_async_track_releases_reservation(self) -> None:
        """Async track_cost releases reservation."""
        config = CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.2)
        enforcer = CostEnforcer(config=config)

        permit = await enforcer.acquire_permit_async()
        assert enforcer.get_status().in_flight_count == 1

        await enforcer.track_cost_async(0.15, permit=permit)
        status = enforcer.get_status()
        assert status.in_flight_count == 0
        assert status.reserved_cost_usd == 0.0

    def test_unknown_cost_mode_uses_trial_count_reservation(self) -> None:
        """In unknown cost mode, permits use trial count including in-flight."""
        config = CostEnforcerConfig(limit=10.0, fallback_trial_limit=3)
        enforcer = CostEnforcer(config=config)

        # Trigger unknown cost mode
        permit = enforcer.acquire_permit()
        enforcer.track_cost(None, permit=permit)  # Unknown cost

        # Now in fallback mode, try to acquire more permits
        # We have 1 completed trial, fallback limit is 3
        permit = enforcer.acquire_permit()  # 1 completed + 1 in-flight = 2
        assert permit.is_granted, "Second permit should be granted"
        permit3 = enforcer.acquire_permit()  # 1 completed + 2 in-flight = 3
        assert permit3.is_granted, "Third permit should be granted"
        permit4 = enforcer.acquire_permit()  # Would exceed limit
        assert not permit4.is_granted, "Fourth permit should be denied"

    def test_status_includes_in_flight_info(self) -> None:
        """get_status includes in-flight count and reserved cost."""
        config = CostEnforcerConfig(limit=2.0, estimated_cost_per_trial=0.25)
        enforcer = CostEnforcer(config=config)

        enforcer.acquire_permit()
        enforcer.acquire_permit()

        status = enforcer.get_status()
        assert status.in_flight_count == 2
        assert status.reserved_cost_usd == 0.5  # 2 * 0.25

    def test_per_permit_reservation_with_ema_changes(self) -> None:
        """Issue A fix: Exact reserved amount used even when EMA changes.

        When permit is acquired at one estimate and released later after
        EMA has changed, the exact reserved amount should be released,
        not the current EMA estimate.
        """
        config = CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.10)
        enforcer = CostEnforcer(config=config)

        # Acquire permit at initial estimate (0.10)
        permit1 = enforcer.acquire_permit()
        assert permit1.amount == pytest.approx(0.10)
        assert enforcer.get_status().reserved_cost_usd == pytest.approx(0.10)

        # Track a much higher actual cost - this updates EMA
        # First, acquire and complete a trial with high cost
        permit2 = enforcer.acquire_permit()
        enforcer.track_cost(0.50, permit=permit2)  # High cost updates EMA

        # EMA should now be higher than original estimate
        assert enforcer._estimated_cost > 0.10

        # Now release the first permit using its original reserved amount
        enforcer.release_permit(permit1)

        # Reserved cost should be exactly zero (not negative from over-release)
        status = enforcer.get_status()
        assert status.reserved_cost_usd == pytest.approx(0.0)
        assert status.in_flight_count == 0

    def test_release_permit_with_explicit_reserved_amount(self) -> None:
        """Issue B fix: release_permit uses permit object with exact amount.

        This is critical for exception handling where permit was granted
        but trial failed - we need to release the exact reserved amount.
        """
        config = CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.20)
        enforcer = CostEnforcer(config=config)

        # Acquire permit
        permit = enforcer.acquire_permit()
        assert permit.amount == pytest.approx(0.20)

        # Simulate exception scenario: release permit with exact amount
        enforcer.release_permit(permit)

        status = enforcer.get_status()
        assert status.in_flight_count == 0
        assert status.reserved_cost_usd == pytest.approx(0.0)
        # Cost should NOT be tracked - just reservation released
        assert status.accumulated_cost_usd == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_async_release_permit_with_explicit_amount(self) -> None:
        """Async version of release_permit with permit object."""
        config = CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.25)
        enforcer = CostEnforcer(config=config)

        permit = await enforcer.acquire_permit_async()
        assert permit.amount == pytest.approx(0.25)

        # Simulate exception - release without tracking cost
        await enforcer.release_permit_async(permit)

        status = enforcer.get_status()
        assert status.in_flight_count == 0
        assert status.reserved_cost_usd == pytest.approx(0.0)
        assert status.accumulated_cost_usd == pytest.approx(0.0)

    def test_permit_single_release_semantics(self) -> None:
        """Permit can only be released once (single-release semantics)."""
        config = CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.20)
        enforcer = CostEnforcer(config=config)

        permit = enforcer.acquire_permit()
        assert permit.active, "Permit should be active initially"

        # First release should succeed
        result = enforcer.release_permit(permit)
        assert result is True, "First release should succeed"
        assert not permit.active, "Permit should be inactive after release"

        # Second release should fail (return False)
        result = enforcer.release_permit(permit)
        assert result is False, "Second release should fail"

    def test_permit_double_release_no_double_count(self) -> None:
        """Double-releasing a permit doesn't double-decrement counters."""
        config = CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.20)
        enforcer = CostEnforcer(config=config)

        permit = enforcer.acquire_permit()
        assert enforcer.get_status().in_flight_count == 1

        # Release once
        enforcer.release_permit(permit)
        status = enforcer.get_status()
        assert status.in_flight_count == 0
        assert status.reserved_cost_usd == pytest.approx(0.0)

        # Attempt double release - should have no effect
        enforcer.release_permit(permit)
        status = enforcer.get_status()
        assert status.in_flight_count == 0  # Still 0, not negative
        assert status.reserved_cost_usd == pytest.approx(0.0)  # Still 0

    def test_permit_id_monotonic(self) -> None:
        """Permit IDs are monotonically increasing."""
        config = CostEnforcerConfig(limit=10.0, estimated_cost_per_trial=0.1)
        enforcer = CostEnforcer(config=config)

        permits = [enforcer.acquire_permit() for _ in range(5)]

        for i in range(1, len(permits)):
            assert permits[i].id > permits[i - 1].id, "Permit IDs should be monotonic"

    def test_denied_permit_has_invalid_id(self) -> None:
        """Denied permits have id=-1 and amount=0."""
        config = CostEnforcerConfig(limit=0.1, estimated_cost_per_trial=0.2)
        enforcer = CostEnforcer(config=config)

        # First permit exceeds budget (0.2 > 0.1)
        permit = enforcer.acquire_permit()
        assert not permit.is_granted
        assert permit.id == -1
        assert permit.amount == 0.0
        assert not permit.active


class TestCostEnforcerInternals:
    """Tests for internal state and initialization."""

    @pytest.fixture(autouse=True)
    def clear_mock_mode(self) -> Generator[None, None, None]:
        """Ensure mock mode is disabled for tests that need real tracking."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "false"}):
            yield

    def test_init_state(self) -> None:
        """Verify all internal attributes are initialized correctly."""
        enforcer = CostEnforcer()
        # Check if it's an RLock (either threading._RLock or similar)
        assert type(enforcer._lock).__name__.endswith("RLock")
        assert enforcer._unknown_cost_mode is False
        assert enforcer._warning_emitted is False
        assert isinstance(enforcer._approval_token_path, Path)
        assert enforcer._in_flight_count == 0
        assert enforcer._reserved_cost == 0.0
        assert enforcer._cost_samples == []
        assert enforcer._estimated_cost == enforcer.config.estimated_cost_per_trial
        assert enforcer._permit_counter == 0
        assert enforcer._active_permits == {}
        assert enforcer._sync_used is False
        assert enforcer._async_used is False

    def test_mock_mode_method_and_property_are_removed(self) -> None:
        """S2-B Round 3 removed _check_mock_mode and is_mock_mode from CostEnforcer."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "true"}):
            enforcer = CostEnforcer()
            assert not hasattr(enforcer, "_check_mock_mode")
            assert not hasattr(enforcer, "is_mock_mode")
            assert not hasattr(enforcer, "_mock_mode_cached")

    def test_get_approval_token_path(self) -> None:
        """Verify token path construction."""
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": "/tmp/config"}):
            enforcer = CostEnforcer()
            expected = Path("/tmp/config/traigent/cost_approval.token")
            assert enforcer._get_approval_token_path() == expected

        # Test default fallback
        with patch.dict(os.environ):
            if "XDG_CONFIG_HOME" in os.environ:
                del os.environ["XDG_CONFIG_HOME"]
            enforcer = CostEnforcer()
            expected = Path.home() / ".config" / "traigent" / "cost_approval.token"
            assert enforcer._get_approval_token_path() == expected


class TestCostEnforcerMixing:
    """Tests for sync/async mixing detection."""

    @pytest.fixture(autouse=True)
    def clear_mock_mode(self) -> Generator[None, None, None]:
        """Ensure mock mode is disabled for tests that need real tracking."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "false"}):
            yield

    def test_check_mixing_logging(self) -> None:
        """Verify logging when switching between sync and async."""
        with patch("traigent.core.cost_enforcement.logger") as mock_logger:
            enforcer = CostEnforcer()

            # Initial async usage - no log
            enforcer._check_mixing(is_async=True)
            assert enforcer._async_used is True
            assert enforcer._sync_used is False
            mock_logger.info.assert_not_called()

            # Switch to sync - should log
            enforcer._check_mixing(is_async=False)
            assert enforcer._sync_used is True
            mock_logger.info.assert_called_with(
                "CostEnforcer: switching from async to sync methods"
            )

            # Reset mock
            mock_logger.reset_mock()

            # Test sync -> async transition with a FRESH enforcer
            enforcer2 = CostEnforcer()
            enforcer2._check_mixing(is_async=False)  # Start with sync
            enforcer2._check_mixing(is_async=True)  # Switch to async
            mock_logger.info.assert_called_with(
                "CostEnforcer: switching from sync to async methods"
            )


class TestCostEnforcerConfigLoading:
    """Tests for robust config loading."""

    @pytest.fixture(autouse=True)
    def clear_mock_mode(self) -> Generator[None, None, None]:
        """Ensure mock mode is disabled for tests that need real tracking."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "false"}):
            yield

    @patch("traigent.core.cost_enforcement.logger")
    def test_load_config_invalid_values(self, mock_logger) -> None:
        """Verify safe parsing handles invalid values."""
        env = {
            "TRAIGENT_RUN_COST_LIMIT": "invalid",
            "TRAIGENT_COST_WARNING_THRESHOLD": "-0.5",  # Negative
            "TRAIGENT_FALLBACK_TRIAL_LIMIT": "0",  # Too low
        }
        with patch.dict(os.environ, env):
            enforcer = CostEnforcer()
            # Should use defaults
            assert enforcer.config.limit == 2.0
            assert enforcer.config.warning_threshold == 0.5
            assert enforcer.config.fallback_trial_limit == 10

            # Verify warnings were logged
            assert mock_logger.warning.call_count >= 3


class TestCostEnforcerApprovalToken:
    """Tests for approval logic."""

    @pytest.fixture(autouse=True)
    def clear_mock_mode(self) -> Generator[None, None, None]:
        """Ensure mock mode is disabled for tests that need real tracking."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "false"}):
            yield

    def test_check_approval_token_valid(self, tmp_path) -> None:
        """Verify valid approval token is accepted."""
        token_file = tmp_path / "cost_approval.token"
        token_file.write_text("approved")

        enforcer = CostEnforcer()
        enforcer._approval_token_path = token_file

        assert enforcer._check_approval_token() is True

    def test_check_approval_token_with_limit(self, tmp_path) -> None:
        """Verify approval token with custom limit."""
        token_file = tmp_path / "cost_approval.token"
        token_file.write_text("approved:10.5")

        enforcer = CostEnforcer()
        enforcer._approval_token_path = token_file

        assert enforcer._check_approval_token() is True
        assert enforcer.config.limit == 10.5

    def test_check_approval_token_invalid_format(self, tmp_path) -> None:
        """Verify invalid token format is handled."""
        token_file = tmp_path / "cost_approval.token"
        token_file.write_text("approved:invalid")

        enforcer = CostEnforcer()
        enforcer._approval_token_path = token_file

        # Should return True (approved) but keep default limit
        assert enforcer._check_approval_token() is True
        assert enforcer.config.limit == 2.0

    def test_check_approval_token_missing(self, tmp_path) -> None:
        """Verify missing token returns False."""
        enforcer = CostEnforcer()
        enforcer._approval_token_path = tmp_path / "nonexistent"
        assert enforcer._check_approval_token() is False

    @patch("sys.stdin.isatty", return_value=False)
    def test_check_and_approve_non_interactive(self, mock_isatty) -> None:
        """Verify non-interactive shell aborts if limit exceeded."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=1.0))
        # Estimated cost > limit
        assert enforcer.check_and_approve(2.0) is False

    @patch("sys.stdin.isatty", return_value=True)
    @patch("builtins.input", return_value="y")
    def test_check_and_approve_interactive_yes(self, mock_input, mock_isatty) -> None:
        """Verify interactive approval works."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=1.0))
        assert enforcer.check_and_approve(2.0) is True

    @patch("sys.stdin.isatty", return_value=True)
    @patch("builtins.input", return_value="n")
    def test_check_and_approve_interactive_no(self, mock_input, mock_isatty) -> None:
        """Verify interactive rejection works."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=1.0))
        assert enforcer.check_and_approve(2.0) is False

    def test_reset_full_state(self) -> None:
        """Verify reset clears ALL state."""
        enforcer = CostEnforcer()

        # Dirty the state
        enforcer._accumulated_cost = 10.0
        enforcer._trial_count = 5
        enforcer._unknown_cost_mode = True
        enforcer._warning_emitted = True
        enforcer._in_flight_count = 2
        enforcer._reserved_cost = 1.0
        enforcer._cost_samples = [0.1, 0.2]
        enforcer._estimated_cost = 100.0
        enforcer._permit_counter = 10
        enforcer._active_permits = {1: Permit(1, 0.1)}
        enforcer._sync_used = True
        enforcer._async_used = True

        enforcer.reset()

        assert enforcer._accumulated_cost == 0.0
        assert enforcer._trial_count == 0
        assert enforcer._unknown_cost_mode is False
        assert enforcer._warning_emitted is False
        assert enforcer._in_flight_count == 0
        assert enforcer._reserved_cost == 0.0
        assert enforcer._cost_samples == []
        assert enforcer._estimated_cost == enforcer.config.estimated_cost_per_trial
        # NOTE: _permit_counter should NOT be reset (see CostEnforcer.reset comments)
        assert enforcer._permit_counter == 10
        assert enforcer._active_permits == {}
        assert enforcer._sync_used is False
        assert enforcer._async_used is False

    def test_reset_preserves_latched_require_cost_tracking(self) -> None:
        """Reset should not change latched strict/require tracking mode."""
        with patch.dict(
            os.environ, {"TRAIGENT_REQUIRE_COST_TRACKING": "true"}, clear=False
        ):
            enforcer = CostEnforcer()

        assert enforcer._require_cost_tracking() is True

        # Changing env after init should not affect this instance, including after reset.
        with patch.dict(
            os.environ, {"TRAIGENT_REQUIRE_COST_TRACKING": "false"}, clear=False
        ):
            enforcer.reset()
            assert enforcer._require_cost_tracking() is True


# =============================================================================
# Adaptive Cost Estimation Tests (Issue #66)
# =============================================================================


class TestAdaptiveCostEstimation:
    """Tests for adaptive cost estimation features."""

    @pytest.fixture(autouse=True)
    def clear_mock_mode(self) -> Generator[None, None, None]:
        """Ensure mock mode is disabled for tests that need real tracking."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "false"}):
            yield

    def test_cost_confidence_low_with_few_samples(self) -> None:
        """Confidence is low with few cost samples."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=10.0))

        # No samples
        confidence = enforcer.get_cost_confidence()
        assert confidence <= 0.5

        # Add one sample
        permit = enforcer.acquire_permit()
        enforcer.track_cost(0.05, permit=permit)

        confidence = enforcer.get_cost_confidence()
        assert 0.3 <= confidence <= 0.6

    def test_cost_confidence_increases_with_samples(self) -> None:
        """Confidence increases as more samples are collected."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=100.0))

        # Add consistent cost samples
        for _ in range(5):
            permit = enforcer.acquire_permit()
            enforcer.track_cost(0.05, permit=permit)

        confidence_5 = enforcer.get_cost_confidence()

        # Add more samples
        for _ in range(5):
            permit = enforcer.acquire_permit()
            enforcer.track_cost(0.05, permit=permit)

        confidence_10 = enforcer.get_cost_confidence()

        # Confidence should increase (or stay high) with more consistent samples
        assert confidence_10 >= confidence_5 * 0.9  # Allow small variation

    def test_cost_confidence_low_with_high_variance(self) -> None:
        """Confidence is low when costs vary significantly."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=100.0))

        # Add highly variable cost samples
        costs = [0.01, 0.10, 0.02, 0.15, 0.01, 0.20]
        for cost in costs:
            permit = enforcer.acquire_permit()
            enforcer.track_cost(cost, permit=permit)

        confidence = enforcer.get_cost_confidence()
        # High variance = low confidence
        assert confidence < 0.7

    def test_estimate_remaining_cost_initial(self) -> None:
        """Remaining cost estimate uses initial value with few samples."""
        enforcer = CostEnforcer(
            config=CostEnforcerConfig(limit=100.0, estimated_cost_per_trial=0.05)
        )

        # No samples yet
        estimate = enforcer.estimate_remaining_cost(10)

        assert isinstance(estimate, CostEstimate)
        assert estimate.samples_used == 0
        # Initial estimate: 0.05 * 10 * 1.2 (safety margin) = 0.6
        assert 0.5 <= estimate.estimated_remaining_cost <= 0.7
        assert estimate.confidence <= 0.5

    def test_estimate_remaining_cost_adaptive(self) -> None:
        """Remaining cost estimate adapts to observed costs."""
        enforcer = CostEnforcer(
            config=CostEnforcerConfig(limit=100.0, estimated_cost_per_trial=0.05)
        )

        # Add samples with higher cost than initial estimate
        for _ in range(5):
            permit = enforcer.acquire_permit()
            enforcer.track_cost(0.10, permit=permit)  # 2x initial estimate

        estimate = enforcer.estimate_remaining_cost(10)

        assert estimate.samples_used == 5
        # Should be based on observed 0.10 average, not initial 0.05
        assert estimate.estimated_remaining_cost > 0.10 * 10  # At least 10 * 0.10

    def test_estimate_remaining_cost_safety_margin(self) -> None:
        """Safety margin is applied to remaining cost estimate."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=100.0))

        # Add consistent samples
        for _ in range(5):
            permit = enforcer.acquire_permit()
            enforcer.track_cost(0.10, permit=permit)

        estimate_default = enforcer.estimate_remaining_cost(10)
        estimate_high_margin = enforcer.estimate_remaining_cost(10, safety_margin=2.0)

        # Higher safety margin = higher estimate
        assert (
            estimate_high_margin.estimated_remaining_cost
            > estimate_default.estimated_remaining_cost
        )
        # Safety margin is adjusted based on confidence, so it may be >= requested
        assert estimate_high_margin.safety_margin >= 2.0

    def test_cost_status_includes_confidence(self) -> None:
        """CostStatus includes estimated cost and confidence."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=10.0))

        # Add some samples
        for _ in range(3):
            permit = enforcer.acquire_permit()
            enforcer.track_cost(0.05, permit=permit)

        status = enforcer.get_status()

        assert hasattr(status, "estimated_cost_per_trial")
        assert hasattr(status, "cost_confidence")
        assert status.estimated_cost_per_trial > 0
        assert 0.0 <= status.cost_confidence <= 1.0


class TestCostDivergenceWarning:
    """Tests for cost divergence warning logs."""

    @pytest.fixture(autouse=True)
    def clear_mock_mode(self) -> Generator[None, None, None]:
        """Ensure mock mode is disabled for tests that need real tracking."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "false"}):
            yield

    def test_high_divergence_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning is logged when actual cost diverges high from estimate."""
        enforcer = CostEnforcer(
            config=CostEnforcerConfig(limit=100.0, estimated_cost_per_trial=0.05)
        )

        # Track a cost 3x higher than estimate (default threshold is 2.0)
        permit = enforcer.acquire_permit()
        with caplog.at_level("WARNING"):
            enforcer.track_cost(0.15, permit=permit)  # 3x higher

        assert any("divergence" in record.message.lower() for record in caplog.records)

    def test_low_divergence_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Info is logged when actual cost is much lower than estimate."""
        enforcer = CostEnforcer(
            config=CostEnforcerConfig(limit=100.0, estimated_cost_per_trial=0.10)
        )

        # Track a cost 3x lower than estimate
        permit = enforcer.acquire_permit()
        with caplog.at_level("INFO"):
            enforcer.track_cost(0.03, permit=permit)  # ~3x lower

        assert any(
            "below estimate" in record.message.lower() for record in caplog.records
        )

    def test_normal_cost_no_divergence_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning when cost is within normal range of estimate."""
        enforcer = CostEnforcer(
            config=CostEnforcerConfig(limit=100.0, estimated_cost_per_trial=0.05)
        )

        # Track a cost close to estimate (1.5x)
        permit = enforcer.acquire_permit()
        with caplog.at_level("WARNING"):
            enforcer.track_cost(0.075, permit=permit)  # 1.5x - within threshold

        assert not any(
            "divergence" in record.message.lower() for record in caplog.records
        )

    def test_custom_divergence_threshold(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Custom divergence threshold from environment variable."""
        # Set higher threshold (4.0x)
        with patch.dict(os.environ, {"TRAIGENT_COST_DIVERGENCE_THRESHOLD": "4.0"}):
            enforcer = CostEnforcer(
                config=CostEnforcerConfig(limit=100.0, estimated_cost_per_trial=0.05)
            )
            permit = enforcer.acquire_permit()
            with caplog.at_level("WARNING"):
                enforcer.track_cost(0.15, permit=permit)  # 3x - now within threshold

        # Should NOT log warning since 3x < 4x threshold
        assert not any(
            "divergence" in record.message.lower()
            for record in caplog.records
            if record.levelname == "WARNING"
        )


class TestCostEstimateDataclass:
    """Tests for CostEstimate dataclass."""

    def test_cost_estimate_fields(self) -> None:
        """CostEstimate has expected fields."""
        estimate = CostEstimate(
            estimated_remaining_cost=1.5,
            confidence=0.8,
            samples_used=10,
            safety_margin=1.2,
        )

        assert estimate.estimated_remaining_cost == 1.5
        assert estimate.confidence == 0.8
        assert estimate.samples_used == 10
        assert estimate.safety_margin == 1.2

    def test_cost_estimate_default_margin(self) -> None:
        """CostEstimate has default safety margin."""
        estimate = CostEstimate(
            estimated_remaining_cost=1.0,
            confidence=0.5,
            samples_used=5,
        )

        assert estimate.safety_margin == 1.2  # Default


class TestCostEnforcerInvariants:
    """Tests for invariant verification methods."""

    @pytest.fixture(autouse=True)
    def clear_mock_mode(self) -> Generator[None, None, None]:
        """Ensure mock mode is disabled for tests that need real tracking."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "false"}):
            yield

    def test_verify_invariants_passes_on_valid_state(self) -> None:
        """_verify_invariants returns empty list for valid state."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=10.0))

        # Fresh enforcer should have no violations
        violations = enforcer._verify_invariants()
        assert violations == []

    def test_assert_invariants_passes_on_valid_state(self) -> None:
        """assert_invariants doesn't raise for valid state."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=10.0))

        # Should not raise
        enforcer.assert_invariants()

    def test_verify_invariants_detects_negative_in_flight(self) -> None:
        """_verify_invariants detects negative in_flight_count."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=10.0))

        # Manually corrupt state to test invariant detection
        enforcer._in_flight_count = -1

        violations = enforcer._verify_invariants()
        assert len(violations) >= 1
        assert any("I1 violated" in v for v in violations)

    def test_verify_invariants_detects_mismatched_permits(self) -> None:
        """_verify_invariants detects mismatch between permits and count."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=10.0))

        # Corrupt state: count doesn't match permits dict
        enforcer._in_flight_count = 2
        enforcer._active_permits = {}  # Empty but count says 2

        violations = enforcer._verify_invariants()
        assert any("I3 violated" in v for v in violations)

    def test_assert_invariants_raises_on_violation(self) -> None:
        """assert_invariants raises AssertionError on violation."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=10.0))

        # Corrupt state
        enforcer._reserved_cost = -1.0

        with pytest.raises(AssertionError, match="invariant violations"):
            enforcer.assert_invariants()

    def test_invariants_hold_after_operations(self) -> None:
        """Invariants hold after normal operations."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=10.0))

        # Perform various operations
        permit1 = enforcer.acquire_permit()
        enforcer.assert_invariants()

        permit2 = enforcer.acquire_permit()
        enforcer.assert_invariants()

        enforcer.track_cost(0.1, permit=permit1)
        enforcer.assert_invariants()

        enforcer.release_permit(permit2)
        enforcer.assert_invariants()

    def test_invariants_allow_parallel_actual_overrun_with_active_permit(self) -> None:
        """A completed overrun can coexist with other admitted in-flight work."""
        enforcer = CostEnforcer(
            config=CostEnforcerConfig(limit=0.10, estimated_cost_per_trial=0.05)
        )

        permit1 = enforcer.acquire_permit()
        permit2 = enforcer.acquire_permit()
        assert permit1.is_granted
        assert permit2.is_granted

        enforcer.track_cost(0.20, permit=permit1)
        status = enforcer.get_status()
        assert status.accumulated_cost_usd == pytest.approx(0.20)
        assert status.reserved_cost_usd == pytest.approx(0.05)
        assert status.in_flight_count == 1
        assert status.limit_reached is True

        enforcer.assert_invariants()

        next_permit = enforcer.acquire_permit()
        assert not next_permit.is_granted


class TestCostEnforcerEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture(autouse=True)
    def clear_mock_mode(self) -> Generator[None, None, None]:
        """Ensure mock mode is disabled for tests that need real tracking."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "false"}):
            yield

    def test_check_thresholds_zero_limit(self) -> None:
        """_check_thresholds handles zero limit gracefully."""
        config = CostEnforcerConfig(limit=0.0)
        enforcer = CostEnforcer(config=config)

        # Should not raise or divide by zero
        enforcer._check_thresholds()

    def test_cost_confidence_with_zero_mean(self) -> None:
        """get_cost_confidence handles edge case of zero mean."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=10.0))

        # Add zero-cost samples
        enforcer._cost_samples = [0.0, 0.0, 0.0]

        # Should return low confidence, not crash
        confidence = enforcer.get_cost_confidence()
        assert 0.0 <= confidence <= 1.0

    def test_safe_int_valid_value(self) -> None:
        """safe_int returns valid parsed value."""
        with patch.dict(os.environ, {"TRAIGENT_FALLBACK_TRIAL_LIMIT": "25"}):
            enforcer = CostEnforcer()
            assert enforcer.config.fallback_trial_limit == 25

    def test_safe_int_invalid_string(self) -> None:
        """safe_int handles non-numeric string."""
        with patch.dict(os.environ, {"TRAIGENT_FALLBACK_TRIAL_LIMIT": "not_a_number"}):
            enforcer = CostEnforcer()
            # Should fall back to default
            assert enforcer.config.fallback_trial_limit == 10

    def test_warns_when_ema_stays_at_cold_start_after_unknown_costs(
        self, caplog
    ) -> None:
        """Repeated unknown costs should trigger a one-time cold-start warning."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=10.0))

        with caplog.at_level("WARNING"):
            for idx in range(EMA_COLD_START_WARNING_TRIALS):
                permit = enforcer.acquire_permit()
                enforcer.track_cost(None, permit=permit, trial_id=f"unknown-{idx}")

        cold_start_warnings = [
            record.message
            for record in caplog.records
            if "Cost estimate is still at initial value" in record.message
        ]
        assert len(cold_start_warnings) == 1

    @pytest.mark.asyncio
    async def test_release_permit_async_double_release(self) -> None:
        """Async release_permit handles double-release gracefully."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=10.0))

        permit = await enforcer.acquire_permit_async()
        result1 = await enforcer.release_permit_async(permit)
        assert result1 is True

        result2 = await enforcer.release_permit_async(permit)
        assert result2 is False

    @pytest.mark.asyncio
    async def test_track_cost_async_already_released_permit(self) -> None:
        """track_cost_async handles already-released permit."""
        enforcer = CostEnforcer(config=CostEnforcerConfig(limit=10.0))

        permit = await enforcer.acquire_permit_async()
        await enforcer.release_permit_async(permit)

        # Track cost on already-released permit - should still track cost
        await enforcer.track_cost_async(0.1, permit=permit)
        assert enforcer.accumulated_cost == 0.1
