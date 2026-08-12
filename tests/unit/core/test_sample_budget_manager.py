import math

import pytest

from traigent.core.sample_budget import BudgetMetrics, LeaseClosure, SampleBudgetManager


def test_remaining_unbounded():
    manager = SampleBudgetManager(total_budget=None)
    lease = manager.create_lease("trial-1")

    assert math.isinf(lease.remaining())
    assert lease.try_take(1)
    assert math.isinf(manager.remaining())
    closure = lease.finalize()
    assert isinstance(closure, LeaseClosure)
    assert math.isinf(closure.global_remaining)


def test_budget_consumption_and_exhaustion():
    manager = SampleBudgetManager(total_budget=3)
    lease = manager.create_lease("trial-1")

    assert lease.try_take(1)
    assert lease.try_take(1)
    assert lease.try_take(1)
    assert manager.remaining() == 0
    assert not lease.try_take(1)

    closure = lease.finalize()
    assert closure.consumed == 3
    assert closure.exhausted
    assert closure.global_remaining == 0


def test_ceiling_enforced_per_lease():
    manager = SampleBudgetManager(total_budget=10)
    lease = manager.create_lease("trial-1", ceiling=2)
    assert lease.try_take(1)
    assert lease.try_take(1)
    assert not lease.try_take(1)
    closure = lease.finalize()
    assert closure.consumed == 2
    assert not closure.exhausted


def test_rollback_returns_budget():
    manager = SampleBudgetManager(total_budget=5)
    lease = manager.create_lease("trial-1")
    assert lease.try_take(3)
    lease.rollback(2)
    assert manager.remaining() == 4
    assert lease.try_take(3)
    assert manager.remaining() == 1
    closure = lease.finalize()
    assert closure.consumed == 4
    assert not closure.exhausted


def test_multiple_leases_share_budget():
    manager = SampleBudgetManager(total_budget=4)
    lease_a = manager.create_lease("trial-a")
    lease_b = manager.create_lease("trial-b")

    assert lease_a.try_take(2)
    assert lease_b.try_take(2)
    assert not lease_a.try_take(1)
    assert not lease_b.try_take(1)

    closure_a = lease_a.finalize()
    closure_b = lease_b.finalize()

    assert closure_a.consumed == 2
    assert closure_b.consumed == 2
    assert closure_a.exhausted
    assert closure_b.exhausted
    assert closure_a.global_remaining == 0
    assert closure_b.global_remaining == 0


def test_finalizing_twice_is_idempotent():
    manager = SampleBudgetManager(total_budget=2)
    lease = manager.create_lease("trial-1")
    assert lease.try_take(1)
    first = lease.finalize()
    second = lease.finalize()
    assert first.consumed == second.consumed == 1
    assert first.exhausted == second.exhausted


def test_efficiency_accounts_for_wasted_samples():
    """Traigent#1965: the original assertion here was ``0 < efficiency < 1``,
    an interval loose enough that it passed on BOTH the pre-fix buggy value
    (0.5, from double-subtracting the rollback) and the correct value
    (2/3) -- it could not have caught the regression it was meant to guard.
    Tightened to the exact, hand-computed value.
    """
    manager = SampleBudgetManager(total_budget=5)
    lease = manager.create_lease("trial-waste")

    assert lease.try_take(1)
    assert lease.try_take(1)
    assert lease.try_take(1)

    lease.rollback(1)

    metrics = manager.snapshot()
    assert metrics.consumed == 2
    assert metrics.wasted == 1
    # Bounded mode: `consumed` (2) is already NET of the 1 rolled-back sample,
    # so the true gross-attempted count is consumed + wasted = 3, of which 2
    # were productive: 2/3, NOT max(2-1,0)/2 = 0.5 (the pre-fix value, which
    # double-subtracted the same rollback that already shrank `consumed`).
    assert metrics.efficiency == pytest.approx(2 / 3)
    assert metrics.efficiency != pytest.approx(0.5)

    closure = lease.finalize()
    assert closure.wasted == 1


def test_efficiency_unbounded_mode_subtracts_wasted_from_gross_consumed():
    """Unbounded mode is the one case where subtracting `wasted` from
    `consumed` IS correct, because `_consumed` stays GROSS across rollbacks
    (never decremented) -- unlike bounded mode above. Same try_take/rollback
    shape as the bounded test, different manager mode, different correct
    answer: this is the asymmetry #1965 is about.
    """
    manager = SampleBudgetManager(total_budget=None)
    lease = manager.create_lease("trial-waste-unbounded")

    assert lease.try_take(1)
    assert lease.try_take(1)
    assert lease.try_take(1)

    lease.rollback(1)

    metrics = manager.snapshot()
    # Unbounded: `consumed` never shrinks on rollback -- it stays the gross
    # count of every successful try_take.
    assert metrics.consumed == 3
    assert metrics.wasted == 1
    assert metrics.efficiency == pytest.approx(2 / 3)


def test_efficiency_bounded_full_rollback_is_zero_not_negative_clamped():
    """Every taken sample is later rolled back: 0 productive out of the 3
    gross-attempted, not a division artifact.
    """
    metrics = BudgetMetrics(total_budget=5, consumed=0, wasted=3)
    assert metrics.efficiency == 0.0


def test_efficiency_bounded_formula_does_not_reduce_to_naive_ratio():
    """Sanity check pinning the exact formula shape: applying the UNBOUNDED
    formula (`max(consumed - wasted, 0) / consumed`) to a bounded snapshot
    must NOT reproduce the correct bounded answer -- otherwise this test
    (and the fix) would be vacuous.
    """
    metrics = BudgetMetrics(total_budget=5, consumed=2, wasted=3)
    naive_unbounded_formula = (
        max(metrics.consumed - metrics.wasted, 0) / metrics.consumed
    )
    assert metrics.efficiency != pytest.approx(naive_unbounded_formula)
    assert naive_unbounded_formula == 0.0
    assert metrics.efficiency == pytest.approx(2 / 5)
