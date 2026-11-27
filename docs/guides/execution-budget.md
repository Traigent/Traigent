# Execution Budget Guide (Experimental)

`ExecutionBudget` shares cost, example, and time limits across calls. Pass the
same budget to each phase of a workflow, such as baseline evaluation,
optimization, and holdout evaluation.

## Creating a budget

Set at least one of `max_cost_usd`, `max_examples`, `deadline_seconds`, or
`max_budget_debit_usd`. Nonpositive limits raise `ConfigurationError`.
Unset cost, example, and time limits leave those dimensions unbounded.

## Attaching a budget — explicit keyword

Pass `budget=` to `evaluate()`, `optimize()`, or `optimize_sync()`. The context
manager starts the clock if needed on entry and makes `final_snapshot` available
on exit. Pass the budget explicitly even inside the context manager.

This example shares an example limit across three phases. The optimization
uses a rule-based function and reports zero model cost with `with_usage()`.

```python
import asyncio

import traigent
from traigent.evaluators import Dataset, LocalEvaluator


def baseline_answer(value):
    return value


search_data = Dataset([traigent.EvaluationExample({"value": 2}, 3)])


@traigent.optimize(
    configuration_space={"offset": [0, 1]},
    objectives=["accuracy"],
    evaluation={"eval_dataset": search_data},
    offline=True,
    algorithm="grid",
)
def answer(value):
    offset = traigent.get_config()["offset"]
    return traigent.with_usage(value + offset, total_cost=0)


async def main():
    evaluator = LocalEvaluator(metrics=["accuracy"])
    baseline_data = Dataset([traigent.EvaluationExample({"value": 2}, 3)])
    holdout_data = Dataset([traigent.EvaluationExample({"value": 5}, 6)])

    with traigent.ExecutionBudget(max_examples=4) as budget:
        baseline = await evaluator.evaluate(
            baseline_answer, {}, baseline_data, budget=budget
        )
        search = await answer.optimize(max_trials=2, budget=budget)
        offset = search.best_config["offset"]

        def tuned_answer(value):
            return value + offset

        holdout = await evaluator.evaluate(tuned_answer, {}, holdout_data, budget=budget)

    print(baseline.execution_budget["consumed_examples"])
    print(search.metadata["execution_budget"]["consumed_examples"])
    print(holdout.execution_budget["consumed_examples"])
    print(budget.final_snapshot.as_dict()["remaining_examples"])


asyncio.run(main())
```

## Per-operation limits still apply

At the start of optimization, the SDK can lower the per-run example cap and
timeout to the shared budget's remaining examples and time. The per-run
`cost_limit` still governs cost approval. The SDK checks the shared cost cap
at batch and trial boundaries.

## Explicit allowances for unpriced calls

Use `max_budget_debit_usd` to cap allowances you allocate to calls whose cost
you cannot price. Call `reserve_debit()` before each provider call you want to
cover. The reservation accounts for both settled spend allowances and
outstanding reservations. It releases the lock before the provider call.

```python
import traigent


def example_call():
    return "Example response"


budget = traigent.ExecutionBudget(max_budget_debit_usd=0.21)


def call_with_allowance():
    lease = budget.reserve_debit(0.05)
    try:
        return example_call()
    finally:
        lease.settle_debit()


for _ in range(4):
    print(call_with_allowance())
print(budget.debit_snapshot().remaining_budget_debit_usd)
```

Four settlements of 0.05 from 0.21 leave 0.01. Another 0.05 reservation raises
`BudgetDebitExceeded`. The refusal leaves the balance available for a smaller
reservation. When the call raises or is cancelled, the `finally` block still
settles the allowance. Use `lease.release_debit()` for a request you know was
never placed or was nonbillable. Each lease can be settled or released once.

By default, settlement consumes the amount reserved. Pass a finite,
nonnegative amount to `lease.settle_debit(amount)` to settle explicitly.
A settlement that takes the balance over the cap is recorded in full and
blocks further reservations.

Read allocation amounts in `debit_snapshot()`, including the maximum,
consumed, reserved, remaining, and overrun amounts. The ledger lives in the
current process. Track observed provider charges separately, for example
with `record_external()`. Explicit per-run cost limits still apply to trials
whose actual costs are unmeasured.

## Reading what was consumed

An optimization result exposes the budget snapshot in
`result.metadata["execution_budget"]`. A direct local evaluation exposes it as
`result.execution_budget`. The snapshot includes consumed and remaining cost,
examples, and time, along with trial counts and cost-tracking status. The
serialized snapshot uses `None` for unbounded remaining amounts. The live
properties `remaining_cost`, `remaining_examples`, and `remaining_seconds`
use `float("inf")` for unbounded amounts.

### Recording work the optimizer can't see

Use `record_external()` to count observed costs or examples from production
calls or other work outside a budgeted evaluation or optimization.

```python
import traigent

budget = traigent.ExecutionBudget(max_cost_usd=1.0, max_examples=10)
budget.record_external(cost_usd=0.012, examples=1)
print(budget.snapshot().as_dict())
```

### Agents that make no model calls

For a rule-based or lookup path that incurs no model charges, return
`traigent.with_usage(value, total_cost=0)` during optimization to report
zero cost explicitly. See the shared-budget example above.

## Honesty — what is, and isn't, a hard guarantee

- **Examples.** A direct local evaluation admits examples from the shared
  pool. Optimization clamps its example pool to the remaining allowance at
  run start.
- **Time.** Optimization checks the deadline between trials. A trial already
  in progress can run past it.
- **Cost.** Optimization checks the shared balance before admitting a batch
  and accounts for reported spend as trials finish. The first batch can start
  without a cost estimate that fits. The actual spend of a batch can exceed the
  remaining balance. Concurrent optimization runs can make admission
  decisions against the same remaining balance.
- **Incomplete cost tracking.** A trial with unknown cost can debit its known
  spend while marking cost tracking incomplete. If an optimization finishes
  with `cost_tracking` of `"partial"` or `"untracked"`, its result carries
  `EXECUTION_BUDGET_UNTRACKED_COST` and describes consumed cost as a lower
  bound. Set `enforce_untracked_cost=True` to block further admission when the
  budget records incomplete cost tracking. A trial with unmeasured cost or a
  call to `record_external(examples=1)` can make tracking incomplete. A budget
  denial can also end winner rechecks without changing the run's stop reason.
