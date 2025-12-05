import math

from traigent.core.sample_budget import LeaseClosure, SampleBudgetManager


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
    manager = SampleBudgetManager(total_budget=5)
    lease = manager.create_lease("trial-waste")

    assert lease.try_take(1)
    assert lease.try_take(1)
    assert lease.try_take(1)

    lease.rollback(1)

    metrics = manager.snapshot()
    assert metrics.consumed == 2
    assert metrics.wasted == 1
    assert 0 < metrics.efficiency < 1

    closure = lease.finalize()
    assert closure.wasted == 1
