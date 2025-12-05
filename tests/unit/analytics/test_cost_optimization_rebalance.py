from __future__ import annotations

from traigent.analytics.cost_optimization import BudgetAllocator, ResourceType


def _seed_allocations(allocator: BudgetAllocator) -> None:
    # Initialize with some allocations
    base_allocs = allocator.allocate_budget(
        total_budget=1000,
        priorities={
            ResourceType.COMPUTE: 0.5,
            ResourceType.STORAGE: 0.3,
            ResourceType.NETWORK: 0.2,
        },
        historical_usage={},
    )
    allocator.allocations = {
        ResourceType.COMPUTE: base_allocs[ResourceType.COMPUTE],
        ResourceType.STORAGE: base_allocs[ResourceType.STORAGE],
        ResourceType.NETWORK: base_allocs[ResourceType.NETWORK],
    }

    # Simulate spending: over-utilize COMPUTE, under-utilize STORAGE
    allocator.allocations[ResourceType.COMPUTE].spent_amount = (
        allocator.allocations[ResourceType.COMPUTE].allocated_budget * 0.95
    )
    allocator.allocations[ResourceType.COMPUTE].remaining_budget = (
        allocator.allocations[ResourceType.COMPUTE].allocated_budget
        - allocator.allocations[ResourceType.COMPUTE].spent_amount
    )

    allocator.allocations[ResourceType.STORAGE].spent_amount = (
        allocator.allocations[ResourceType.STORAGE].allocated_budget * 0.2
    )
    allocator.allocations[ResourceType.STORAGE].remaining_budget = (
        allocator.allocations[ResourceType.STORAGE].allocated_budget
        - allocator.allocations[ResourceType.STORAGE].spent_amount
    )

    # NETWORK near balanced
    allocator.allocations[ResourceType.NETWORK].spent_amount = (
        allocator.allocations[ResourceType.NETWORK].allocated_budget * 0.6
    )
    allocator.allocations[ResourceType.NETWORK].remaining_budget = (
        allocator.allocations[ResourceType.NETWORK].allocated_budget
        - allocator.allocations[ResourceType.NETWORK].spent_amount
    )


def test_rebalance_returns_allocations_and_adjusts_extremes() -> None:
    allocator = BudgetAllocator()
    _seed_allocations(allocator)

    before_compute = allocator.allocations[ResourceType.COMPUTE].allocated_budget
    before_storage = allocator.allocations[ResourceType.STORAGE].allocated_budget

    result = allocator.rebalance_budgets()

    # Should return the allocations mapping
    assert isinstance(result, dict)
    assert ResourceType.COMPUTE in result and ResourceType.STORAGE in result

    compute_alloc = result[ResourceType.COMPUTE]
    storage_alloc = result[ResourceType.STORAGE]

    # After rebalance, allocated budget for compute should increase; storage should decrease
    assert compute_alloc.allocated_budget >= before_compute
    assert storage_alloc.allocated_budget <= before_storage


def test_rebalance_no_change_when_no_over_under_utilized() -> None:
    allocator = BudgetAllocator()
    # Create balanced allocations
    allocations = allocator.allocate_budget(
        total_budget=1000,
        priorities={
            ResourceType.COMPUTE: 0.5,
            ResourceType.STORAGE: 0.3,
            ResourceType.NETWORK: 0.2,
        },
        historical_usage={},
    )

    # Set all utilizations around 0.7 (no extremes)
    for alloc in allocations.values():
        alloc.spent_amount = alloc.allocated_budget * 0.7
        alloc.remaining_budget = alloc.allocated_budget - alloc.spent_amount

    before = {k: v.allocated_budget for k, v in allocations.items()}
    result = allocator.rebalance_budgets()
    after = {k: v.allocated_budget for k, v in result.items()}

    assert before == after


def test_rebalance_handles_zero_budget_allocations() -> None:
    allocator = BudgetAllocator()
    _seed_allocations(allocator)

    # Force one allocation to zero budget/spend
    zero_alloc = allocator.allocations[ResourceType.STORAGE]
    zero_alloc.allocated_budget = 0.0
    zero_alloc.spent_amount = 0.0
    zero_alloc.remaining_budget = 0.0

    # Should not raise ZeroDivisionError
    allocator.rebalance_budgets()
