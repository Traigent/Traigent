# Agents Under Specification: the missing foundation for production AI agents

Most teams don’t have an “agent problem”.

They have a *software engineering problem*—because agent behavior changes, but their process doesn’t.

We built TraiGent to help teams ship AI agents with the same discipline they apply to production systems: specifications, measurable objectives, constraints, and gates.

> Suggested visual: `docs/demos/output/optimize.svg` (optimization loop) or `docs/demos/output/github-hooks.svg` (gates in practice)

Availability note (so expectations are clear): the open-source SDK runs locally (`edge_analytics`). Managed cloud/hybrid backends are roadmap-compatible but not assumed available.

## TL;DR

- Treat key knobs (model, temperature, retrieval depth, thresholds) as **Tuned Variables**, not “set-and-forget” config.
- Write an **agent spec** that defines objectives, constraints, budgets, and promotion rules.
- Use CI/CD **quality gates** so behavior changes can’t silently regress.

## The uncomfortable truth: “prompt engineering” isn’t a process

In production, the question isn’t “can we make it answer better today?” It’s:

- Can we *prove* the change improved the system on representative inputs?
- Can we *prevent regressions* from merging?
- Can we *control cost and latency* while improving quality?

Most agent stacks are missing a contract that answers these questions.

## Tuned Variables: the config layer that actually matters

Some variables are static and operational:

- API keys
- endpoints
- retry policies

But other variables directly shape behavior, and their best values change over time:

- model selection
- temperature / top_p
- retrieval depth (k) and chunking strategy
- tool thresholds and routing rules
- prompt style / system constraints

These are **Tuned Variables**. They should be optimized against a specification.

## What TraiGent looks like in code (minimal)

At a high level, you define:

- a configuration space (the Tuned Variables you want to explore)
- what you optimize (objectives like accuracy/cost/latency)
- how you evaluate (a small dataset or harness)

Example (simplified pseudo-code; runnable quickstarts are linked below):

```python
import traigent
from traigent.api.decorators import EvaluationOptions, ExecutionOptions

@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.7],
    },
    objectives=["accuracy", "cost", "response_time"],
    evaluation=EvaluationOptions(eval_dataset="examples/datasets/quickstart/qa_samples.jsonl"),
    execution=ExecutionOptions(execution_mode="edge_analytics"),
)
def my_agent(question: str) -> str:
    # Your existing agent code (LLM/RAG/tools) goes here
    ...
```

The point isn’t the decorator. The point is: once the “what matters” is explicit, you can automate the rest.

## Agents under specification: what should be in the “contract”?

A practical agent spec typically includes:

1. **Objectives**: what you optimize (accuracy, cost, latency, safety metrics)
2. **Constraints**: what must never be violated (min accuracy, max cost, max latency)
3. **Budgets**: how much evaluation you’re willing to spend to validate a change
4. **Promotion rules**: how you decide to ship (meaningful deltas + confidence)

This is the mental model behind TVL specs in TraiGent: the spec is the authoritative definition of what matters, and the tooling turns that into a repeatable tuning + gating workflow.

We call this approach **Agents Under Specification**. TraiGent implements it via **Tuned Variable Language (TVL)** specs.

## Where CI/CD fits: quality gates for behavior

Once objectives and constraints are explicit, you can make CI do what it’s good at:

- block regressions on merge
- surface “missed improvements” (so you don’t ship suboptimal configs)
- make cost/latency tradeoffs explicit, not accidental

TraiGent ships an example CI integration that implements two gates:

- **Regression Gate**: fail if you degrade baseline beyond a threshold
- **Improvement Gate**: alert if tuning finds a materially better config than the one you’re shipping

See: `examples/integrations/ci-cd/README.md`

## Privacy + telemetry (what engineers will ask)

- You can run locally in `edge_analytics` mode (no managed backend required).
- You can set `privacy_enabled=True` to minimize logging/telemetry.
- You can opt out completely via `TRAIGENT_DISABLE_TELEMETRY=true`.

See: `docs/api-reference/telemetry.md`

## Practical adoption path (what to do this week)

1. Start with a small evaluation set (10–30 representative examples).
2. Choose 1–2 Tuned Variables (model + temperature is a good start).
3. Add a regression gate in CI (start with accuracy; add cost/latency next).
4. Promote only when the gate passes, then track drift and re-tune.

## Try TraiGent locally (no API spend)

TraiGent includes mock mode so you can test the workflow without provider calls:

- `export TRAIGENT_MOCK_MODE=true`
- `python examples/quickstart/01_simple_qa.py`

Repo: https://github.com/Traigent/Traigent
