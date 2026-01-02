"""Property-based stateful testing for CostEnforcer using Hypothesis.

This module implements a state machine model that generates random sequences
of operations and verifies that invariants hold throughout.

Key invariants tested:
- I1: in_flight_count >= 0
- I2: reserved_cost >= 0
- I3: len(active_permits) == in_flight_count
- I4: shadow model matches enforcer state (model_active_permits == enforcer._active_permits)
- I5: double-release returns False
- I6: denied permits have id=-1, amount=0
- I7: permit IDs are monotonically increasing
- I8: sum(permit.amount for permit in active_permits) == reserved_cost

Note: We do NOT test "accumulated + reserved <= limit" as an invariant because
budget enforcement happens at acquire_permit() time. After track_cost(), the
total CAN exceed the limit if actual_cost > estimated_cost. This is by design.

Reference: /home/nimrodbu/.claude/plans/snazzy-whistling-kettle.md
"""

from __future__ import annotations

import os

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule

# Ensure mock mode is disabled for these tests
os.environ["TRAIGENT_MOCK_MODE"] = "false"

from traigent.core.cost_enforcement import CostEnforcer, CostEnforcerConfig, Permit


class CostEnforcerStateMachine(RuleBasedStateMachine):
    """State machine model for property-based testing of CostEnforcer.

    Tracks a shadow model of expected state to verify against actual state.
    Uses Hypothesis bundles to track Permit objects for later operations.
    """

    # Bundle to store granted permits for later release/track operations
    permits = Bundle("permits")

    def __init__(self) -> None:
        super().__init__()
        # Ensure mock mode is disabled for each test case
        os.environ["TRAIGENT_MOCK_MODE"] = "false"

        # Real enforcer under test - low limit to trigger denials frequently
        self.enforcer = CostEnforcer(
            CostEnforcerConfig(
                limit=1.0,
                estimated_cost_per_trial=0.1,
            )
        )

        # Shadow model for verification
        self.model_active_permits: set[int] = set()  # permit IDs that are still active
        self.model_released_permits: set[int] = (
            set()
        )  # permit IDs that have been released
        self.model_denied_count: int = 0
        # Start at -1 so first permit (id=1) passes monotonicity check
        self.model_last_permit_id: int = -1
        # Track permits orphaned by reset - active Permit objects not in enforcer registry
        self.model_orphaned_permits: set[int] = set()

    # === OPERATIONS ===

    @rule(target=permits)
    def acquire_permit(self) -> Permit:
        """Acquire a permit and add to bundle if granted."""
        permit = self.enforcer.acquire_permit()

        if permit.is_granted:
            # Update model
            self.model_active_permits.add(permit.id)
            assert permit.id > self.model_last_permit_id, "Permit IDs must be monotonic"
            self.model_last_permit_id = permit.id

            # Verify permit properties
            assert permit.active is True
            assert permit.amount > 0
            assert permit.id > 0
        else:
            # Denied permit
            self.model_denied_count += 1
            assert permit.id == -1
            assert permit.amount == 0.0
            assert permit.active is False

        return permit

    @rule(permit=permits)
    def release_permit(self, permit: Permit) -> None:
        """Release a permit (may be already released or orphaned by reset)."""
        was_active = permit.id in self.model_active_permits
        was_orphaned = permit.id in self.model_orphaned_permits
        was_released = permit.id in self.model_released_permits

        result = self.enforcer.release_permit(permit)

        # Denied permits (-1) are no-ops for release/track
        if not permit.is_granted:
            assert result is False
            return

        if was_active:
            # First release of active permit should succeed
            assert result is True, f"First release of permit {permit.id} should succeed"
            self.model_active_permits.discard(permit.id)
            self.model_released_permits.add(permit.id)
        elif was_orphaned:
            # First release of orphaned permit (after reset) should succeed
            # because the Permit object's active flag was not cleared by reset
            assert (
                result is True
            ), f"First release of orphaned permit {permit.id} should succeed"
            self.model_orphaned_permits.discard(permit.id)
            self.model_released_permits.add(permit.id)
        else:
            # Double release (already in released set) should fail
            assert was_released, f"Permit {permit.id} should be tracked somewhere"
            assert result is False, f"Double release of permit {permit.id} should fail"

    @rule(permit=permits, cost=st.floats(min_value=0.0, max_value=0.2))
    def track_cost(self, permit: Permit, cost: float) -> None:
        """Track cost for a permit (may be already released or orphaned).

        Note: If permit is already released, we skip this operation to model
        realistic usage. In practice, track_cost on released permits is only
        used for exception recovery paths, not normal operation.
        """
        was_active = permit.id in self.model_active_permits
        was_orphaned = permit.id in self.model_orphaned_permits
        was_released = permit.id in self.model_released_permits

        # Skip if permit is already released - models realistic usage
        # (double track_cost would add cost twice but only release once)
        if was_released:
            return

        if not permit.is_granted:
            # Nothing to track for denied permits
            return

        # track_cost doesn't return bool, it just processes
        self.enforcer.track_cost(cost, permit=permit)

        if was_active:
            self.model_active_permits.discard(permit.id)
            self.model_released_permits.add(permit.id)
        elif was_orphaned:
            # Orphaned permits can also be tracked (marks the Permit object as released)
            self.model_orphaned_permits.discard(permit.id)
            self.model_released_permits.add(permit.id)

    @rule()
    def reset_enforcer(self) -> None:
        """Reset the enforcer state."""
        self.enforcer.reset()
        # Move active permits to orphaned - they're no longer in enforcer's registry
        # but the Permit objects still have active=True
        self.model_orphaned_permits.update(self.model_active_permits)
        self.model_active_permits.clear()
        # NOTE: Do NOT clear model_released_permits - once a permit is released
        # (Permit.active=False), it stays released even across resets
        # NOTE: Do NOT reset model_last_permit_id - permit counter continues across
        # resets to prevent ID collisions with orphaned permits
        # Note: denied count is not reset as it's for stats

    # === INVARIANTS (checked after every operation) ===

    @invariant()
    def in_flight_count_never_negative(self) -> None:
        """I1: in_flight_count >= 0"""
        assert (
            self.enforcer._in_flight_count >= 0
        ), f"in_flight_count is negative: {self.enforcer._in_flight_count}"

    @invariant()
    def reserved_cost_never_negative(self) -> None:
        """I2: reserved_cost >= 0"""
        assert (
            self.enforcer._reserved_cost >= 0
        ), f"reserved_cost is negative: {self.enforcer._reserved_cost}"

    @invariant()
    def active_permits_equals_in_flight(self) -> None:
        """I3: len(active_permits) == in_flight_count"""
        assert len(self.enforcer._active_permits) == self.enforcer._in_flight_count, (
            f"Mismatch: active_permits={len(self.enforcer._active_permits)}, "
            f"in_flight_count={self.enforcer._in_flight_count}"
        )

    @invariant()
    def model_matches_enforcer(self) -> None:
        """Shadow model should match enforcer state."""
        enforcer_active = set(self.enforcer._active_permits.keys())
        assert self.model_active_permits == enforcer_active, (
            f"Model mismatch: model={self.model_active_permits}, "
            f"enforcer={enforcer_active}"
        )

    @invariant()
    def permit_sum_equals_reserved(self) -> None:
        """I8: Sum of active permit amounts equals reserved_cost.

        This invariant ensures consistency between the individual permit
        reservations and the aggregate reserved_cost counter. A violation
        would indicate a bug in acquire_permit, release_permit, or track_cost.
        """
        permit_sum = sum(p.amount for p in self.enforcer._active_permits.values())
        epsilon = 0.0001  # Floating point tolerance
        assert abs(permit_sum - self.enforcer._reserved_cost) < epsilon, (
            f"I8 violated: sum(permit.amount) = {permit_sum:.4f} != "
            f"reserved_cost = {self.enforcer._reserved_cost:.4f}"
        )

    # NOTE: We do NOT include a budget_bound_respected invariant here because:
    # 1. Budget enforcement happens at acquire_permit() time, not after track_cost()
    # 2. When actual_cost > estimated_cost, accumulated grows faster than reserved shrinks
    # 3. This is by design - we can't "un-do" a trial if it costs more than estimated
    # 4. The real safety guarantee is: at acquire_permit time, we check if
    #    accumulated + reserved + new_reservation <= limit
    # 5. After track_cost, total CAN exceed limit - this is expected behavior
    # See plan file for detailed discussion of this design decision.


# Create test class for pytest
TestCostEnforcerStateMachine = CostEnforcerStateMachine.TestCase

# Configure Hypothesis settings
TestCostEnforcerStateMachine.settings = settings(
    max_examples=100,  # Number of test cases to generate
    stateful_step_count=30,  # Max operations per test
    deadline=None,  # Disable deadline for CI
)


class TestCostEnforcerStateMachineAdditional:
    """Additional property-based tests using @given decorator."""

    @given(
        st.lists(st.integers(min_value=1, max_value=20), min_size=1, max_size=30),
    )
    @settings(max_examples=50, deadline=None)
    def test_random_acquire_release_sequence(self, operations: list[int]) -> None:
        """Test random sequences of acquire/release with counts."""
        os.environ["TRAIGENT_MOCK_MODE"] = "false"
        enforcer = CostEnforcer(CostEnforcerConfig(limit=100.0))
        permits: list[Permit] = []

        for op_count in operations:
            # Acquire op_count % 5 permits (limit to 5 per batch)
            for _ in range(op_count % 5):
                permit = enforcer.acquire_permit()
                if permit.is_granted:
                    permits.append(permit)

            # Release some permits when op_count is even
            if permits and op_count % 2 == 0:
                permit = permits.pop(0)
                enforcer.release_permit(permit)

        # Invariants should hold (I1, I2, I3, I8)
        assert enforcer._in_flight_count >= 0
        assert enforcer._reserved_cost >= 0
        assert len(enforcer._active_permits) == enforcer._in_flight_count
        # I8: Sum of permit amounts equals reserved_cost
        permit_sum = sum(p.amount for p in enforcer._active_permits.values())
        assert abs(permit_sum - enforcer._reserved_cost) < 0.0001

    @given(
        st.lists(
            st.tuples(
                st.sampled_from(["acquire", "release", "track", "reset"]),
                st.floats(min_value=0.01, max_value=0.1),
            ),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_random_operation_sequence(
        self,
        ops: list[tuple[str, float]],
    ) -> None:
        """Test random sequences of all operations."""
        os.environ["TRAIGENT_MOCK_MODE"] = "false"
        enforcer = CostEnforcer(CostEnforcerConfig(limit=10.0))
        active_permits: list[Permit] = []
        released_permits: list[Permit] = []

        for op, cost in ops:
            if op == "acquire":
                permit = enforcer.acquire_permit()
                if permit.is_granted:
                    active_permits.append(permit)
            elif op == "release" and active_permits:
                permit = active_permits.pop(0)
                result = enforcer.release_permit(permit)
                assert result is True
                released_permits.append(permit)
            elif op == "track" and active_permits:
                permit = active_permits.pop(0)
                enforcer.track_cost(cost, permit=permit)
                released_permits.append(permit)
            elif op == "reset":
                enforcer.reset()
                active_permits.clear()
                released_permits.clear()

            # Verify invariants after each operation (I1, I2, I3, I8)
            assert enforcer._in_flight_count >= 0
            assert enforcer._reserved_cost >= 0
            assert len(enforcer._active_permits) == enforcer._in_flight_count
            # I8: Sum of permit amounts equals reserved_cost
            permit_sum = sum(p.amount for p in enforcer._active_permits.values())
            assert abs(permit_sum - enforcer._reserved_cost) < 0.0001

    @given(st.lists(st.floats(min_value=0.01, max_value=0.05), min_size=5, max_size=20))
    @settings(max_examples=30, deadline=None)
    def test_permit_id_monotonicity(self, costs: list[float]) -> None:
        """Test that permit IDs are strictly monotonically increasing."""
        os.environ["TRAIGENT_MOCK_MODE"] = "false"
        enforcer = CostEnforcer(CostEnforcerConfig(limit=100.0))
        permit_ids: list[int] = []

        for cost in costs:
            permit = enforcer.acquire_permit()
            if permit.is_granted:
                permit_ids.append(permit.id)
                enforcer.track_cost(cost, permit=permit)

        # All IDs should be unique
        assert len(permit_ids) == len(set(permit_ids))

        # All IDs should be strictly increasing
        for i in range(1, len(permit_ids)):
            assert (
                permit_ids[i] > permit_ids[i - 1]
            ), f"Permit IDs not monotonic: {permit_ids}"

    @given(st.integers(min_value=5, max_value=15))
    @settings(max_examples=20, deadline=None)
    def test_double_release_always_fails(self, num_permits: int) -> None:
        """Test that double-release always fails regardless of sequence."""
        os.environ["TRAIGENT_MOCK_MODE"] = "false"
        enforcer = CostEnforcer(CostEnforcerConfig(limit=100.0))

        # Acquire permits
        permits = []
        for _ in range(num_permits):
            permit = enforcer.acquire_permit()
            if permit.is_granted:
                permits.append(permit)

        # Release all permits
        for permit in permits:
            result = enforcer.release_permit(permit)
            assert result is True, f"First release of {permit.id} should succeed"

        # Try to release all again - all should fail
        for permit in permits:
            result = enforcer.release_permit(permit)
            assert result is False, f"Double release of {permit.id} should fail"

        # Invariants still hold
        assert enforcer._in_flight_count >= 0
        assert enforcer._reserved_cost >= 0


class TestDeniedPermitProperties:
    """Tests for denied permit behavior."""

    def test_denied_permit_properties(self) -> None:
        """Verify denied permits have correct properties (I7)."""
        os.environ["TRAIGENT_MOCK_MODE"] = "false"
        # Very low limit to force denial
        enforcer = CostEnforcer(
            CostEnforcerConfig(limit=0.01, estimated_cost_per_trial=0.1)
        )

        permit = enforcer.acquire_permit()

        # Denied permit properties
        assert permit.id == -1, "Denied permit should have id=-1"
        assert permit.amount == 0.0, "Denied permit should have amount=0"
        assert permit.active is False, "Denied permit should have active=False"
        assert permit.is_granted is False, "Denied permit should not be granted"

    def test_denied_permit_release_is_safe(self) -> None:
        """Verify releasing a denied permit doesn't affect state."""
        os.environ["TRAIGENT_MOCK_MODE"] = "false"
        enforcer = CostEnforcer(
            CostEnforcerConfig(limit=0.01, estimated_cost_per_trial=0.1)
        )

        initial_in_flight = enforcer._in_flight_count
        initial_reserved = enforcer._reserved_cost

        permit = enforcer.acquire_permit()
        assert permit.is_granted is False

        # Release denied permit - should be no-op
        result = enforcer.release_permit(permit)
        assert result is False, "Releasing denied permit should fail"

        # State unchanged
        assert enforcer._in_flight_count == initial_in_flight
        assert enforcer._reserved_cost == initial_reserved
