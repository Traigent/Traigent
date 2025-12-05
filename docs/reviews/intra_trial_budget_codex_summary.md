# Intra-Trial Sample Budget Enforcement Summary

## Overview
This iteration threads a global sample budget through the orchestrator and evaluators so trials stop the moment the shared allowance is exhausted. A lease-based manager mediates consumption across sequential and parallel runs, and trial metadata now records when exhaustion occurs. Local evaluators were refactored to check the lease before each example, ensuring we never overrun `max_total_examples`.

## Key Changes
- Added `SampleBudgetManager`/`SampleBudgetLease` for thread-safe consumption tracking (`traigent/core/sample_budget.py`).
- Orchestrator now acquires per-trial leases, distributes budget fairly across parallel trials, and annotates results with exhaustion metadata (`traigent/core/orchestrator.py`).
- Base/Local evaluators accept an optional `sample_lease`, stop early for both sequential and `max_workers>1` execution paths, and roll back cancelled examples (`traigent/evaluators/base.py`, `traigent/evaluators/local.py`).
- Added dedicated unit tests covering the manager, evaluator budget handling, and parallel ceiling allocation.

## Representative Snippet
```python
from traigent.core.sample_budget import SampleBudgetManager

manager = SampleBudgetManager(total_budget=10)
lease = manager.create_lease("trial-42", ceiling=4)

while lease.try_take(1):
    run_example()

closure = lease.finalize()
if closure.exhausted:
    logger.info("Sample budget exhausted after %s examples", closure.consumed)
```

## Tests Run
- `pytest tests/unit/core/test_sample_budget_manager.py tests/unit/evaluators/test_local_evaluator_budget.py`
- `pytest tests/unit/core/test_orchestrator.py -k "sample_budget"`

## Suggested Next Steps
1. Propagate the lease interface into hybrid/remote adapters and OptiGen DTOs so hosted executions respect the same mid-trial stops.
2. Extend telemetry and CLI output to surface consumed/remaining budget and add orchestrator integration tests (e.g., Optuna pruning on exhaustion).
