# Traigent Launch Content Calendar (6 Weeks)

Start date: **2025-12-15** (adjust freely).

## Launch State (copy/paste into posts as needed)

**Available at launch (OSS):**
- Traigent SDK (current repo) + local execution (`edge_analytics`)
- Mock mode (`TRAIGENT_MOCK_LLM=true`) for zero-cost dry runs
- CI/CD example and docs paths you can point people to (`examples/integrations/ci-cd/README.md`, `docs/api-reference/*`)

**Not promised unless provisioned:**
- Managed cloud/hybrid backend, hosted eval infrastructure, production SLAs

## Cadence (realistic default)

Recommended baseline for a small team:
- **LinkedIn: 2 posts/week** (Mon/Thu)
- **Medium: 1 post/week** (Tue or Wed)

If capacity allows, add a third LinkedIn post as a lightweight poll/checklist.

Audience “variants” should be **adapt-on-publish** (same spine, different language), not pre-written 3x for every piece.

Holiday note: if engagement matters, shift Week 2–3 long-form pieces off **Dec 25 / Jan 1**.

## Ownership (keep it sustainable)

- One owner per week (draft + publish).
- One reviewer (claim hygiene + product truth check).
- Reuse assets: each Medium piece should feed the two LinkedIn posts for that week.

## Audience Lanes (publish in parallel)

- **Exec/Product lane (CTO/CEO/PM)**: LinkedIn frameworks, Medium “why it matters”, 1-pagers, webinars.
- **Builder lane (AI/SE)**: GitHub examples + docs, Medium/Dev.to “how to”, LinkedIn snippets.
- **ML/Evaluation lane**: Medium technical deep dives, docs notes, optional notebooks/talks.

## Anchor Asset to Integrate

Use `docs/agent_development_script.md` as the campaign’s “Agent Development Playbook” and ship it in 3 derivatives:
- Exec brief (no stats language)
- Engineering playbook (CI + eval harness)
- ML appendix (metrics + calibration)

## Weekly retrospective (iteration loop)

Run a 20-minute review every Friday:
- **Track**: impressions, engagement rate, saves, replies, repo stars/forks, click-throughs, DMs.
- **Collect**: top 5 objections/questions (feed into next week’s posts).
- **Decide**:
  - If engagement is low but clicks are high → tighten hooks + visuals.
  - If engagement is high but repo actions are low → strengthen CTAs to one runnable path.
  - If objections repeat → dedicate next week’s Medium post to answering them.

## Week 1 (Paradigm shift + launch framing)

- 2025-12-15 (LI, core): Agents need specs (not vibes). Asset: `docs/demos/output/optimize.svg`
- 2025-12-18 (Medium, core): Agents under specification (approach + how Traigent implements it). Asset: `docs/demos/output/optimize.svg`
- 2025-12-19 (LI, core): Agent CI/CD gates: regression + missed improvement. Asset: `docs/demos/output/github-hooks.svg`
- 2025-12-17 (LI, optional): Tuned Variables: static vs tuned config. Asset: snippet screenshot

Playbook spotlight: **Episode 0 — Agent SDLC framing** (`marketing/AGENT_DEV_PLAYBOOK_SERIES.md`)

## Week 2 (Tuned Variables as the abstraction)

- 2025-12-22 (LI, core): “Not all config is equal” (static vs tuned variables)
- 2025-12-23 or 2025-12-26 (Medium, core): Tuned Variables in practice (adoption path + examples)
- 2025-12-26 (LI, core): Cost as an SLO (stop guessing; measure per config)
- 2025-12-24 (LI, optional): Multi-objective tradeoffs (quality vs cost vs latency) (holiday-safe)

Playbook spotlight: **Episode 1 — GTM & Acquisition agents**

## Week 3 (Agents Under Specification: TVL basics)

- 2025-12-29 (LI, core): What an “agent spec” includes (objectives/constraints/budget/gates)
- 2025-12-30 or 2026-01-02 (Medium, core): TVL 101: executable specs for tuning pipelines
- 2026-01-02 (LI, core): “Specs make teams faster” (less debate, more measurement)
- 2025-12-31 (LI, optional): Typed variables + constraints (holiday-safe)

Playbook spotlight: **Episode 2 — Operations agents**

## Week 4 (Promotion gates + statistical confidence)

- 2026-01-05 (LI, core): “It seems better” isn’t shippable: promotion gates
- 2026-01-08 (Medium, core): Promotion gates for agents: meaningful deltas + confidence (plain-English first)
- 2026-01-09 (LI, core): “Good enough” beats “max”: banded objectives / guardrails
- 2026-01-07 (LI, optional): “Confidence thresholds” (chance constraints, explained simply)

Playbook spotlight: **Episode 3 — Knowledge/RAG agents**

## Week 5 (Agent CI/CD: make it real in pipelines)

- 2026-01-12 (LI, core): PRs change behavior; treat it like code changes
- 2026-01-15 (Medium, core): CI/CD for agents: quality gates that block regressions
- 2026-01-16 (LI, core): “Missed improvement” gate: don’t leave wins on the table
- 2026-01-14 (LI, optional): Minimal CI gates template (what to measure, how to fail)

Playbook spotlight: **Episode 4 — Product/Technical agents**

## Week 6 (AgentsOps: operating model + roadmap)

- 2026-01-19 (LI, core): AgentsOps loop: Spec → Evaluate → Tune → Gate → Monitor → Re-tune
- 2026-01-22 (Medium, core): AgentsOps at org scale (operating model + adoption playbook)
- 2026-01-23 (LI, core): Call for design partners + community + next milestones
- 2026-01-21 (LI, optional): Observability for agents: what signals matter

Playbook spotlight: **Episode 5 — Customer support agents**
Capstone: **Episode 6 — Putting it together (AgentsOps at org scale)**
