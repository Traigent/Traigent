# Agent Development Playbook Series (High-Level)

Source material: `docs/agent_development_script.md`

Goal: turn “TraiGent technology” into a **repeatable agent development process** people can adopt (define → evaluate → tune → gate → ship → operate).

## Series format (6 + capstone)

Each episode ships in 3 “language levels”:
- **Exec/Product**: outcomes, risk, governance (no stats terms).
- **Builders (AI/SE)**: implementation patterns + CI gates + code references.
- **ML/Eval**: evaluation design details (metrics, calibration, noise handling).

Each episode publishes across channels:
- **LinkedIn**: 2 posts/week (Exec + Builder). Optional third post = poll/checklist.
- **Medium**: 1 long-form article/week (Builder-first; ML appendix section)
- **GitHub/Docs**: link to one concrete artifact (example, doc, checklist)

## Episode agenda

### Episode 0 — The Agent SDLC (the framing)

- Core: “Agents are software; ship behind specs + quality gates”
- Asset: introduce the Quick Reference table (agent types → metrics)
- CTA: point to `docs/agent_development_script.md` as the playbook entry point
- Cross-cutting spotlight: **Offline evaluation datasets** (test changes before prod)
- Reinforce: Tuned Variables, budgets, CI gates

### Episode 1 — GTM & Acquisition agents (quality before conversion)

- Core: measure message quality + compliance offline; don’t wait weeks for conversion
- What people copy: rubric template + compliance checks checklist
- Cross-cutting spotlight: **Tuned Variables** (templates, cadence, personalization depth)
- Reinforce: budgets, CI gates (compliance as a hard gate)

### Episode 2 — Operations agents (action correctness + routing)

- Core: evaluate action sequences and escalation decisions before automation
- What people copy: “autonomy threshold” spec + escalation labeling guidance
- Cross-cutting spotlight: **Constraints + guardrails** (what must never happen)
- Reinforce: offline evals, budgets, CI gates

### Episode 3 — Knowledge/RAG agents (trustworthiness + abstention)

- Core: separate retrieval quality from answer grounding; optimize abstention behavior
- What people copy: grounded accuracy rubric + “should abstain” dataset recipe
- Cross-cutting spotlight: **Promotion criteria** in plain English (when to ship changes)
- Reinforce: tuned variables (k/thresholds), offline evals, budgets

### Episode 4 — Product/Technical agents (CI-driven evaluation)

- Core: deterministic tests + linters + security scans as first-class evaluation
- What people copy: CI gates template + quality/security objective checklist
- Cross-cutting spotlight: **CI/CD gates** (regressions + missed improvements)
- Reinforce: budgets, rollback, integration surface

### Episode 5 — Customer support agents (resolution + tone + routing)

- Core: resolve correctly, communicate well, escalate appropriately
- What people copy: tone rubric + escalation labeling rubric + policy checks checklist
- Cross-cutting spotlight: **Multi-objective tradeoffs** (quality + cost + latency + policy)
- Reinforce: offline evals, CI gates, budgets

### Episode 6 — Putting it together (AgentsOps at org scale)

- Core: one operating model across teams: Spec → Evaluate → Tune → Gate → Monitor → Re-tune
- What people copy: org adoption playbook (roles, cadences, ownership, escalation paths)
- Cross-cutting spotlight: **Operating cadence** (weekly retros, drift triggers, safe rollouts)
- Reinforce: everything above; plus competitive framing (“why not just build scripts?”)

## Cross-cutting concepts (repeat throughout)

- Tuned Variables (what we tune) vs static config (what we don’t tune)
- Offline evaluation datasets (how we test changes safely)
- CI/CD gates (how we prevent regressions and surface missed improvements)
- Budgets (how we control time/cost and make iteration practical)

## Language rules (so it works for non-technical audiences)

- Replace formal stats terms with intuitive labels:
  - “Pareto front” → “best trade-offs set”
  - “F1” → “balance of false alarms vs misses”
  - “chance constraint” → “confidence threshold”
- Keep one metric per slide/post for exec audiences; move the rest to the appendix.
