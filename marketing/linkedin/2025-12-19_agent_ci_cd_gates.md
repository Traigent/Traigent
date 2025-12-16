# 2025-12-19 — LinkedIn Post

Asset: `docs/demos/output/github-hooks.svg`

Why do we let PRs change agent behavior without a gate?

When code changes, we have tests. When agent behavior changes, many teams still rely on “spot checks”.

Two CI gates we think every agent repo should have:
1. Regression gate: block merges if the current config gets worse than baseline beyond a threshold.
2. Missed-improvement gate: warn if tuning found a materially better config than what you’re about to ship.

This turns “agent quality” into something your pipeline can enforce, not debate.

We put a working CI/CD example in the TraiGent repo (uses SDK evaluation; no custom scoring functions required for the example):
`examples/integrations/ci-cd/README.md`
(link: https://github.com/Traigent/Traigent/blob/main/examples/integrations/ci-cd/README.md)

If you’re already doing something like this, what do you gate on first (and why): accuracy, cost, or latency?

#CICD #AIAgents #MLOps #SoftwareEngineering #LLM
