"""Integration tests for sequential execution with cost enforcement.

Tests trial_lifecycle.py integration with CostEnforcer.
Verifies permit acquisition, tracking, and exception handling.

Key scenarios tested:
- Permit acquisition before trial execution
- Permit denial when limit reached
- Exception path releases permit correctly
- Cost tracking updates accumulated cost
- Unknown cost mode fallback

Reference: /home/nimrodbu/.claude/plans/snazzy-whistling-kettle.md
"""

from __future__ import annotations

import os

import pytest

# Ensure mock mode is disabled for these tests - we want real cost tracking
os.environ["TRAIGENT_MOCK_MODE"] = "false"

from traigent.core.cost_enforcement import (
    CostEnforcer,
    CostEnforcerConfig,
    CostTrackingRequiredError,
    Permit,
)

# Tolerance for floating point comparisons
FLOAT_TOLERANCE = 1e-10


@pytest.fixture(autouse=True)
def disable_mock_mode() -> None:
    """Ensure mock mode is disabled for all tests in this module."""
    os.environ["TRAIGENT_MOCK_MODE"] = "false"
    # Also clear strict mode by default
    if "TRAIGENT_REQUIRE_COST_TRACKING" in os.environ:
        del os.environ["TRAIGENT_REQUIRE_COST_TRACKING"]


class TestSequentialCostEnforcement:
    """Tests for sequential trial execution with cost limits."""

    @pytest.fixture
    def cost_enforcer(self) -> CostEnforcer:
        """Create a cost enforcer with low limit for testing."""
        return CostEnforcer(
            CostEnforcerConfig(
                limit=0.50,
                estimated_cost_per_trial=0.10,
            )
        )

    def test_permit_acquired_before_trial(self, cost_enforcer: CostEnforcer) -> None:
        """Verify permit is acquired and in_flight_count increases."""
        initial_in_flight = cost_enforcer._in_flight_count
        assert initial_in_flight == 0

        # Acquire permit (simulating before trial execution)
        permit = cost_enforcer.acquire_permit()

        assert permit.is_granted
        assert permit.id > 0
        assert abs(permit.amount - 0.10) < FLOAT_TOLERANCE  # estimated cost
        assert permit.active is True
        assert cost_enforcer._in_flight_count == initial_in_flight + 1
        assert abs(cost_enforcer._reserved_cost - 0.10) < FLOAT_TOLERANCE

        # Track cost (simulating trial completion)
        cost_enforcer.track_cost(0.08, permit=permit)

        assert cost_enforcer._in_flight_count == initial_in_flight
        assert abs(cost_enforcer._accumulated_cost - 0.08) < FLOAT_TOLERANCE
        assert cost_enforcer._trial_count == 1

    def test_permit_denied_when_limit_reached(
        self, cost_enforcer: CostEnforcer
    ) -> None:
        """Verify permits are denied when limit is reached."""
        # Exhaust budget with 5 permits (0.10 each = 0.50 total)
        permits: list[Permit] = []
        for _ in range(5):
            permit = cost_enforcer.acquire_permit()
            if permit.is_granted:
                permits.append(permit)

        assert len(permits) == 5

        # Next permit should be denied - budget is fully reserved
        denied_permit = cost_enforcer.acquire_permit()

        assert not denied_permit.is_granted
        assert denied_permit.id == -1
        assert abs(denied_permit.amount) < FLOAT_TOLERANCE  # Should be 0.0
        assert denied_permit.active is False

        # Cleanup
        for p in permits:
            cost_enforcer.release_permit(p)

        assert cost_enforcer._in_flight_count == 0
        assert abs(cost_enforcer._reserved_cost) < FLOAT_TOLERANCE

    def test_exception_releases_permit(self, cost_enforcer: CostEnforcer) -> None:
        """Verify permit is released when exception occurs in trial."""
        permit = cost_enforcer.acquire_permit()
        assert permit.is_granted
        assert cost_enforcer._in_flight_count == 1

        # Simulate exception path with try/finally
        try:
            raise RuntimeError("Simulated trial failure")
        except RuntimeError:
            if permit.active:
                result = cost_enforcer.release_permit(permit)
                assert result is True

        # Permit should be released
        assert cost_enforcer._in_flight_count == 0
        assert not permit.active
        # Trial count not incremented because we used release_permit, not track_cost
        assert cost_enforcer._trial_count == 0

    def test_double_release_returns_false(self, cost_enforcer: CostEnforcer) -> None:
        """Verify double-release of same permit returns False."""
        permit = cost_enforcer.acquire_permit()
        assert permit.is_granted

        # First release should succeed
        result1 = cost_enforcer.release_permit(permit)
        assert result1 is True
        assert cost_enforcer._in_flight_count == 0

        # Second release should fail (double-release)
        result2 = cost_enforcer.release_permit(permit)
        assert result2 is False

        # in_flight_count should not go negative
        assert cost_enforcer._in_flight_count == 0

    def test_track_cost_updates_accumulated(self, cost_enforcer: CostEnforcer) -> None:
        """Verify track_cost accumulates costs correctly."""
        costs = [0.05, 0.08, 0.12, 0.07]
        expected_total = sum(costs)

        for cost in costs:
            permit = cost_enforcer.acquire_permit()
            assert permit.is_granted
            cost_enforcer.track_cost(cost, permit=permit)

        assert abs(cost_enforcer._accumulated_cost - expected_total) < 0.0001
        assert cost_enforcer._trial_count == len(costs)
        assert cost_enforcer._in_flight_count == 0

    def test_unknown_cost_triggers_fallback(self, cost_enforcer: CostEnforcer) -> None:
        """Verify None cost triggers unknown cost mode fallback."""
        permit = cost_enforcer.acquire_permit()
        assert permit.is_granted

        # Track with None cost
        cost_enforcer.track_cost(None, permit=permit)

        assert cost_enforcer._unknown_cost_mode is True
        assert cost_enforcer._trial_count == 1
        assert abs(cost_enforcer._accumulated_cost) < FLOAT_TOLERANCE  # No cost tracked

    def test_require_cost_tracking_raises(self, cost_enforcer: CostEnforcer) -> None:
        """Verify TRAIGENT_REQUIRE_COST_TRACKING=true raises on None cost."""
        os.environ["TRAIGENT_REQUIRE_COST_TRACKING"] = "true"

        try:
            permit = cost_enforcer.acquire_permit()
            assert permit.is_granted

            with pytest.raises(CostTrackingRequiredError) as exc_info:
                cost_enforcer.track_cost(None, permit=permit)

            assert "TRAIGENT_REQUIRE_COST_TRACKING=true" in str(exc_info.value)
        finally:
            del os.environ["TRAIGENT_REQUIRE_COST_TRACKING"]

    def test_sequential_trials_with_varying_costs(
        self, cost_enforcer: CostEnforcer
    ) -> None:
        """Test sequential trials with varying costs updates EMA correctly."""
        # Start with estimated cost of 0.10
        assert abs(cost_enforcer._estimated_cost - 0.10) < FLOAT_TOLERANCE

        # Track some costs that differ from estimate
        costs = [0.05, 0.05, 0.05]  # Actual costs are lower than estimate

        for cost in costs:
            permit = cost_enforcer.acquire_permit()
            assert permit.is_granted
            cost_enforcer.track_cost(cost, permit=permit)

        # EMA should have moved towards actual costs
        assert cost_enforcer._estimated_cost < 0.10
        assert cost_enforcer._trial_count == 3


@pytest.mark.asyncio
class TestSequentialCostEnforcementAsync:
    """Async tests for sequential execution with cost limits."""

    @pytest.fixture
    def cost_enforcer(self) -> CostEnforcer:
        """Create a cost enforcer with low limit for testing."""
        return CostEnforcer(
            CostEnforcerConfig(
                limit=0.50,
                estimated_cost_per_trial=0.10,
            )
        )

    async def test_async_permit_acquired_before_trial(
        self, cost_enforcer: CostEnforcer
    ) -> None:
        """Verify async permit acquisition works correctly."""
        initial_in_flight = cost_enforcer._in_flight_count

        # Acquire permit using async method
        permit = await cost_enforcer.acquire_permit_async()

        assert permit.is_granted
        assert permit.id > 0
        assert cost_enforcer._in_flight_count == initial_in_flight + 1

        # Track cost using async method
        await cost_enforcer.track_cost_async(0.08, permit=permit)

        assert cost_enforcer._in_flight_count == initial_in_flight
        assert abs(cost_enforcer._accumulated_cost - 0.08) < FLOAT_TOLERANCE

    async def test_async_permit_denied_when_limit_reached(
        self, cost_enforcer: CostEnforcer
    ) -> None:
        """Verify async permit denial when limit reached."""
        permits: list[Permit] = []

        # Exhaust budget
        for _ in range(5):
            permit = await cost_enforcer.acquire_permit_async()
            if permit.is_granted:
                permits.append(permit)

        assert len(permits) == 5

        # Next should be denied
        denied = await cost_enforcer.acquire_permit_async()
        assert not denied.is_granted
        assert denied.id == -1

        # Cleanup
        for p in permits:
            await cost_enforcer.release_permit_async(p)

    async def test_async_exception_releases_permit(
        self, cost_enforcer: CostEnforcer
    ) -> None:
        """Verify async permit release on exception."""
        permit = await cost_enforcer.acquire_permit_async()
        assert permit.is_granted
        assert cost_enforcer._in_flight_count == 1

        try:
            raise RuntimeError("Simulated async failure")
        except RuntimeError:
            if permit.active:
                result = await cost_enforcer.release_permit_async(permit)
                assert result is True

        assert cost_enforcer._in_flight_count == 0
        assert not permit.active

    async def test_async_double_release_returns_false(
        self, cost_enforcer: CostEnforcer
    ) -> None:
        """Verify async double-release returns False."""
        permit = await cost_enforcer.acquire_permit_async()
        assert permit.is_granted

        result1 = await cost_enforcer.release_permit_async(permit)
        assert result1 is True

        result2 = await cost_enforcer.release_permit_async(permit)
        assert result2 is False

        assert cost_enforcer._in_flight_count == 0


class TestPermitLifecycle:
    """Tests for permit lifecycle scenarios."""

    @pytest.fixture
    def cost_enforcer(self) -> CostEnforcer:
        """Create a cost enforcer."""
        return CostEnforcer(CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.1))

    def test_permit_ids_monotonically_increasing(
        self, cost_enforcer: CostEnforcer
    ) -> None:
        """Verify permit IDs are strictly monotonically increasing."""
        permit_ids: list[int] = []

        for _ in range(10):
            permit = cost_enforcer.acquire_permit()
            if permit.is_granted:
                permit_ids.append(permit.id)
                cost_enforcer.track_cost(0.05, permit=permit)

        # All IDs should be unique
        assert len(permit_ids) == len(set(permit_ids))

        # All IDs should be strictly increasing
        for i in range(1, len(permit_ids)):
            assert permit_ids[i] > permit_ids[i - 1]

    def test_permit_active_flag_state(self, cost_enforcer: CostEnforcer) -> None:
        """Verify permit active flag transitions correctly."""
        permit = cost_enforcer.acquire_permit()
        assert permit.active is True
        assert permit.is_granted is True

        # After track_cost, permit should be inactive
        cost_enforcer.track_cost(0.05, permit=permit)
        assert permit.active is False

    def test_denied_permit_properties(self, cost_enforcer: CostEnforcer) -> None:
        """Verify denied permits have correct properties."""
        # Exhaust budget
        permits = []
        for _ in range(10):
            p = cost_enforcer.acquire_permit()
            if p.is_granted:
                permits.append(p)

        # Get a denied permit
        denied = cost_enforcer.acquire_permit()

        assert denied.id == -1
        assert abs(denied.amount) < FLOAT_TOLERANCE  # Should be 0.0
        assert denied.active is False
        assert denied.is_granted is False

        # Cleanup
        for p in permits:
            cost_enforcer.release_permit(p)

    def test_release_without_acquire(self, cost_enforcer: CostEnforcer) -> None:
        """Verify releasing a non-existent permit doesn't break state."""
        # Create a fake permit (foreign permit scenario)
        fake_permit = Permit(id=9999, amount=0.1, active=True)

        # Release should succeed (marks permit released) but log warning
        result = cost_enforcer.release_permit(fake_permit)
        assert result is True  # mark_released succeeded

        # in_flight_count should not go negative
        assert cost_enforcer._in_flight_count >= 0
        assert cost_enforcer._reserved_cost >= 0


class TestBudgetScenarios:
    """Tests for budget edge cases."""

    def test_exactly_at_limit(self) -> None:
        """Test budget exactly at limit allows final trial."""
        enforcer = CostEnforcer(
            CostEnforcerConfig(
                limit=0.50,
                estimated_cost_per_trial=0.10,
            )
        )

        # Acquire 4 permits and track exact costs
        for _ in range(4):
            permit = enforcer.acquire_permit()
            assert permit.is_granted
            enforcer.track_cost(0.10, permit=permit)

        assert abs(enforcer._accumulated_cost - 0.40) < FLOAT_TOLERANCE

        # 5th permit should be allowed (0.40 + 0.10 = 0.50 exactly)
        permit5 = enforcer.acquire_permit()
        assert permit5.is_granted

        enforcer.track_cost(0.10, permit=permit5)
        assert abs(enforcer._accumulated_cost - 0.50) < FLOAT_TOLERANCE

        # 6th permit should be denied
        permit6 = enforcer.acquire_permit()
        assert not permit6.is_granted

    def test_actual_cost_higher_than_reserved(self) -> None:
        """Test when actual cost exceeds reserved amount."""
        enforcer = CostEnforcer(
            CostEnforcerConfig(
                limit=1.0,
                estimated_cost_per_trial=0.10,  # Low estimate
            )
        )

        permit = enforcer.acquire_permit()
        assert permit.is_granted
        assert abs(enforcer._reserved_cost - 0.10) < FLOAT_TOLERANCE

        # Actual cost is higher than estimate
        enforcer.track_cost(0.50, permit=permit)

        # Accumulated cost reflects actual, reserved is released
        assert abs(enforcer._accumulated_cost - 0.50) < FLOAT_TOLERANCE
        assert abs(enforcer._reserved_cost) < FLOAT_TOLERANCE
        assert enforcer._in_flight_count == 0

    def test_zero_cost_trial(self) -> None:
        """Test trials with zero cost."""
        enforcer = CostEnforcer(
            CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.10)
        )

        permit = enforcer.acquire_permit()
        assert permit.is_granted

        enforcer.track_cost(0.0, permit=permit)

        assert abs(enforcer._accumulated_cost) < FLOAT_TOLERANCE
        assert enforcer._trial_count == 1
        assert enforcer._in_flight_count == 0


class TestMockModeIntegration:
    """Tests for mock mode integration."""

    def test_mock_mode_bypasses_tracking(self) -> None:
        """Verify mock mode bypasses all cost tracking."""
        os.environ["TRAIGENT_MOCK_MODE"] = "true"

        try:
            enforcer = CostEnforcer(
                CostEnforcerConfig(
                    limit=0.01,  # Very low limit
                    estimated_cost_per_trial=0.10,
                )
            )

            # In mock mode, should always get permit (id=0)
            for _ in range(100):  # Would far exceed limit normally
                permit = enforcer.acquire_permit()
                assert permit.is_granted
                assert permit.id == 0  # Mock permit ID
                enforcer.track_cost(1.0, permit=permit)  # High cost

            # Nothing was actually tracked in mock mode
            assert abs(enforcer._accumulated_cost) < FLOAT_TOLERANCE
            assert enforcer._in_flight_count == 0
        finally:
            os.environ["TRAIGENT_MOCK_MODE"] = "false"

    def test_mock_mode_cached_at_init(self) -> None:
        """Verify mock mode is cached at init time."""
        os.environ["TRAIGENT_MOCK_MODE"] = "false"

        enforcer = CostEnforcer(CostEnforcerConfig(limit=0.10))

        # Change env var after init
        os.environ["TRAIGENT_MOCK_MODE"] = "true"

        # Should still use real tracking (cached at init)
        permit = enforcer.acquire_permit()
        assert permit.id != 0  # Not mock permit ID
        assert enforcer._in_flight_count == 1

        enforcer.track_cost(0.05, permit=permit)
        assert abs(enforcer._accumulated_cost - 0.05) < FLOAT_TOLERANCE

        # Cleanup
        os.environ["TRAIGENT_MOCK_MODE"] = "false"
