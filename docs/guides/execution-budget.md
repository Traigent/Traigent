# Execution Budget Guide (Experimental)

> **Status: experimental (issue #1980).** `ExecutionBudget` is the SDK's first
> cross-call budgeting seam. The API may change in a future minor version — pin
> your version if you depend on it in production.

`ExecutionBudget` puts **one total cap** on cost, examples, and wall-clock time
that is shared across direct `evaluate()` and `optimize()` calls. It is the tool
for a multi-phase workflow — for example a **baseline → search → holdout**
pipeline — where baseline and holdout are evaluations and search is optimization,
but you want the *whole* pipeline to stay under a single budget.

Without it, three phases each with `cost_limit=$1` can spend `$3` total. With one
shared `ExecutionBudget(max_cost_usd=1.0)`, the three phases together stop once
`$1` is spent, wherever in the pipeline that happens.

## Creating a budget

```python
import traigent
from traigent.evaluators import LocalEvaluator

budget = traigent.ExecutionBudget(
    max_cost_usd=5.0,        # total USD across every attached run
    max_examples=2_000,      # total example evaluations across every attached run
    deadline_seconds=1800,   # wall-clock deadline from first attach (30 min)
)
```

At least one of `max_cost_usd`, `max_examples`, or `deadline_seconds` must be set.
Any value `<= 0` raises `ConfigurationError` at construction (a nonpositive cap
would block every trial before it starts). Unset dimensions are unbounded.

`enforce_untracked_cost=False` (default) preserves today's permissive behavior;
set it to `True` to fail closed the moment cost becomes unobservable (see
[Honesty](#honesty-what-is-and-isnt-a-hard-guarantee)).

## Attaching a budget — explicit keyword

The **only** way to attach a budget in this version is the explicit `budget=`
keyword. Pass the **same instance** to every direct evaluator call and
`optimize()` / `optimize_sync()` call in the workflow:

```python
budget = traigent.ExecutionBudget(max_cost_usd=5.0, max_examples=2_000)

evaluator = LocalEvaluator(metrics=["accuracy"])

baseline = await evaluator.evaluate(answer, {}, baseline_dataset, budget=budget)
search = await answer.optimize(budget=budget)  # same instance
holdout = await evaluator.evaluate(answer, {}, holdout_dataset, budget=budget)
```

Each phase spends down the shared remaining. When the shared budget is exhausted,
the current phase stops gracefully with `result.stop_reason == "execution_budget"`,
and any later `evaluate()` or `optimize()` phase stops immediately before doing
work it cannot admit. Direct evaluators expose the same additive
`result.execution_budget` snapshot as their accounting result.

There is **no ambient/implicit attach** (no context-var, no global). This is a
deliberate choice: `optimize_sync()` runs the optimization in a worker thread when
a loop is already running, and thread-local ambient state does not propagate there
— an implicit budget would be silently ignored in exactly that path. An explicit
argument is deterministic and testable.

`ExecutionBudget` **is** a context manager, but only to (1) start the wall-clock
deadline on `__enter__` and (2) freeze a final snapshot on `__exit__`. It never
suppresses exceptions and never ambient-attaches:

```python
with traigent.ExecutionBudget(deadline_seconds=600) as budget:
    await answer.optimize(budget=budget)
    await answer.optimize(budget=budget)
print(budget.final_snapshot.as_dict())
```

## Per-operation limits still apply

Existing per-run limits (`cost_limit`, `max_total_examples`, `timeout`) remain
fully supported and are **combined** with the shared cap: the effective per-run
allowance is `min(per-run limit, shared remaining)`. A per-run limit can only make
a run *tighter*, never let it exceed the shared cumulative cap.

The examples pool and the wall-clock `timeout` are clamped down to the shared
remaining at run start. The **cost** dimension is handled differently: the per-run
`cost_limit` is *not* lowered — it stays your own pre-run approval gate — while the
shared cumulative cost is enforced mid-run, at **batch/trial boundaries**, by the
budget's stop condition and its pre-batch admission gate (which, in parallel mode,
reserves budget for the *whole* next batch — see
[Honesty](#honesty-what-is-and-isnt-a-hard-guarantee)). This is deliberate: it
stops a small shared `max_cost_usd` from manufacturing a spurious pre-run
`CostLimitExceeded` decline against your (larger) `cost_limit`. The run instead
**starts and stops gracefully** with `stop_reason == "execution_budget"` once the
shared cost is spent.

```python
# Per-run cost_limit=2.0, but shared remaining is 0.50 -> this run stops at ~0.50
# with stop_reason == "execution_budget" (never a pre-run CostLimitExceeded).
await answer.optimize(cost_limit=2.0, budget=traigent.ExecutionBudget(max_cost_usd=0.50))
```

## Reading what was consumed

After a run, the budget's state is on the result metadata (the result shape is
unchanged — this is additive):

```python
snap = result.metadata["execution_budget"]
snap["consumed_cost"], snap["remaining_cost"]         # remaining is None when unbounded
snap["consumed_examples"], snap["remaining_examples"]
snap["elapsed_seconds"], snap["remaining_seconds"]
snap["trials"], snap["untracked_trials"], snap["cost_tracking"]
```

You can also read the live object directly: `budget.remaining_cost`,
`budget.remaining_examples`, `budget.remaining_seconds` (each `float("inf")` when
that dimension is unbounded), `budget.consumed_cost`, `budget.snapshot()`.

### Recording work the optimizer can't see

Production calls (`fn.run()` / `fn(...)`) and other raw provider usage are **not**
auto-debited. If you want them to count against the same budget, record them
explicitly — e.g. from `with_usage()` data:

```python
budget.record_external(cost_usd=0.012, examples=1)
```

## Honesty — what is, and isn't, a hard guarantee

- **Examples are a hard limit.** Observable on every execution path, so exhaustion
  is enforced deterministically.
- **The wall-clock deadline is enforced at trial boundaries.** Between trials it is
  a hard stop. A single *hung* trial (a deadlocked sampler, a stuck provider call)
  may overrun the deadline by the orchestrator's watchdog grace — 25% of the
  remaining budget, floored at `1s` and capped at `5min` — before the watchdog
  aborts it. So the deadline is hard at trial granularity, not a mid-trial
  guillotine.
- **The monetary cap is a hard limit only when cost is fully observable.** On raw
  provider paths, self-hosted models, or unpriced models the SDK cannot see the
  true cost of a call, so a trial may debit `$0` it actually cost money for. When
  that happens:
  - `snapshot()["cost_tracking"]` is `"partial"` or `"untracked"` (not
    `"complete"`), and `budget.was_any_cost_untracked` is `True`;
  - the result gains the `EXECUTION_BUDGET_UNTRACKED_COST` warning code and a
    human-readable warning; and
  - **the consumed cost is a lower bound** — the SDK will not claim a hard monetary
    guarantee it cannot honor.

  `enforce_untracked_cost=True` **fails closed mid-run only for trials that report
  no cost** (`cost is None`, e.g. a raw-provider trial with no usage data): the next
  admission after such a trial stops with `stop_reason == "execution_budget"`. A
  model that is merely *unpriced* but reports a concrete `$0` is **not** caught
  mid-run — that debit is indistinguishable from a genuinely free `$0` trial. The
  silent-`$0` path is reconciled at **finalization** (folded into `cost_tracking`
  via the cost enforcer's unknown-cost mode and the `UNPRICED_MODEL_RUNTIME`
  warning) and surfaced as a lower-bound cost, not stopped while running.
- **The cost cap is enforced at batch/trial boundaries, not per token.** The shared
  remaining is checked by the pre-batch admission gate before each batch and by the
  stop condition between trials — never mid-call. In **parallel** mode a batch of up
  to `parallel_trials` trials is admitted together, so the gate only admits a batch
  when the shared remaining can fund the whole batch's *estimated* cost. That makes
  the cap a **tight bound of ± one batch** (± one trial in sequential mode): any
  overshoot is at most one batch's worth of newly-admitted work, and only occurs
  when trials cost more than their pre-batch estimate. It is **not** a per-trial-exact
  hard ceiling. Examples and the wall-clock deadline are unaffected by this (examples
  stay bounded by the sample-pool clamp and per-trial ceilings even under parallelism).

## Out of scope (this version)

- No persistent / cross-process / distributed budget (in-process only).
- No auto-debit of `run()` / `__call__` production calls — use `record_external()`.
- No ambient/implicit attach — the explicit `budget=` keyword is the API.
- No cross-run cost *reservation*: the shared cost is debited from actuals and
  enforced at batch/trial-boundary admission (the per-run `cost_limit` is not
  clamped), so several runs sharing one budget **concurrently** can race on
  admission, and a single run can overshoot the shared cap by about one **batch's**
  actual cost in parallel mode (about one **trial's** in sequential mode). Run
  phases sequentially, with `parallel_trials=1`, if you need the tightest cap.
