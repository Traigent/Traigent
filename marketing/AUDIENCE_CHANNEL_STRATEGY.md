# Audience + Channels Strategy (High-Level)

This plan adds the **agent development workflow** layer (see `docs/agent_development_script.md`) to the launch narrative, and maps messages to distinct audiences and channels.

## Pre-Launch Content Audit (blocking sign-off)

Before Week 1 goes live, do a quick audit pass on any piece that makes a factual/product claim.
Checklist: `marketing/PRE_LAUNCH_CONTENT_AUDIT.md`

**Blockers (must be true or removed/qualified):**
- Budget-dependent claims: “tests thousands”, “hours not weeks”, “zero code changes” → qualify with “within your evaluation budget / depending on your stack”.
- Hard thresholds: any “> 0.7” style numbers → soften or scope to “depending on stakes”.
- Product maturity: anything implying managed backend/cloud/hybrid availability → label as roadmap unless provisioned.
- Integration claims: “works with existing agent code” → scope to supported patterns/frameworks; call out where adapters are needed.

**Sign-off checklist (1 min):**
- Does the post/article clearly separate **what exists today** vs **roadmap**?
- Are code snippets either **runnable** or explicitly labeled as **pseudo-code** with a link to a runnable example?
- Does the CTA link to a real path (`examples/…`, `docs/…`) or one canonical URL?

## Anchor Asset (ties product → real development process)

**Agent Development & Optimization Guide** (current draft): `docs/agent_development_script.md`

Role in the campaign:
- Becomes the “how teams actually build agents” reference.
- Bridges the paradigm shift (“agents are software”) to concrete steps (eval sets, scoring, tuning, gates).
- Provides **agent-type-specific** evaluation patterns so GTM/Ops/Product audiences see themselves.

Planned derivatives (same core ideas, different language level):
- **Exec brief (1 page)**: outcomes, risk, velocity, governance (no stats terms).
- **Engineering playbook**: CI gates + evaluation harness + integration patterns.
- **ML appendix**: deeper evaluation rigor (metrics, calibration, confidence).

## Competitive framing (how to talk about alternatives)

Builders will compare to evaluation/observability tools and to “we can build this ourselves”.

**Position Traigent as:**
- A **spec + tuning + gating** layer for agent behavior (not just logging).
- A way to turn “what matters” into a **declarative contract** (TVL spec) that drives tuning and promotion decisions.

**Comparisons to expect**
- Tracing / observability: LangSmith, Humanloop, “prompt management” platforms
- Evaluation harnesses: Braintrust, bespoke pytest + scripts

**What Traigent does particularly well**
- **TVL spec as contract**: objectives, constraints, budgets, promotion rules (portable + reviewable).
- **Multi-objective tradeoffs**: balance quality/cost/latency rather than optimizing one metric in isolation.
- **Promotion gates**: explicit “ship/no-ship” decisions tied to meaningful improvement thresholds.

**What Traigent does not promise (unless provisioned)**
- Managed eval infrastructure or production SLAs for cloud/hybrid backends (roadmap-compatible in OSS, not “available now”).

**If someone asks “How is this different from X?”**
- Logging/observability tools (e.g., tracing) help you **see** behavior; Traigent helps you **systematically improve** it against a spec and gate changes.
- Custom harnesses can replicate pieces, but Traigent aims to make the workflow **repeatable**, **spec-driven**, and **team-friendly** (shared contract, standard gates).

## Audience Segments (primary messages + channels)

### 1) CTO / CEO / Product (less technical)

**What they care about**
- Predictable shipping, risk management, governance, cost control, trust with customers

**Message framing**
- “Agent SDLC”: acceptance criteria for behavior + quality gates before shipping
- “Offline evaluation”: test changes before they hit production users
- “Cost is an SLO”: control spend with measurable budgets

**Preferred channels**
- LinkedIn (thought leadership + short frameworks)
- Medium (high-level, story + examples; minimal code)
- Webinars / live demos (30–45 min)
- Slides/1-pager PDFs (for internal sharing)

**Language level**
- Replace stats terms with intuitive phrases (“confidence checks”, “false positives”, “guardrails”).

**Anticipated objections → responses**

| Objection | Response angle | Proof/asset |
| --- | --- | --- |
| “This adds process overhead; we need to ship fast.” | Specs + gates reduce rework and regressions; they’re a speed tool past the first week. Start small (10–30 examples) and expand. | `docs/agent_development_script.md` (checklist), `examples/integrations/ci-cd/README.md` |
| “We already measure CSAT/retention; why add evals?” | Outcome metrics lag and confound; offline evals catch regressions before customers do. | Medium “Agent SDLC” episode + demo assets |

### 2) AI / Software Engineers (builders)

**What they care about**
- Adoption friction, reproducible evals, CI integration, observability, rollback, developer ergonomics

**Message framing**
- “AgentsOps”: Spec → Evaluate → Tune → Gate → Deploy → Monitor → Re-tune
- “Zero/low code change”: decorator + injection + framework overrides
- “CI gates”: regression + missed improvement

**Preferred channels**
- GitHub (examples + READMEs)
- Medium / Dev.to (hands-on posts)
- LinkedIn (engineering threads; visuals/snippets)
- Talks / meetups; Hacker News when a deep technical post is ready

**Language level**
- Use engineering terminology; keep stats lightweight unless needed.

**Anticipated objections → responses**

| Objection | Response angle | Proof/asset |
| --- | --- | --- |
| “I can build this with pytest + scripts.” | You can—Traigent packages repeatable patterns: spec-driven knobs, built-in evaluators, and standard gates so teams don’t reinvent. | `examples/integrations/ci-cd/README.md`, `docs/api-reference/decorator-reference.md` |
| “Works only for certain stacks.” | True: low friction for common LangChain/OpenAI SDK patterns; custom stacks may need adapter code. Be explicit early. | (Add matrix) `marketing/INTEGRATION_COMPATIBILITY_MATRIX.md` |

### 3) ML / Evaluation / Applied Research Engineers

**What they care about**
- Evaluation validity, noise, calibration, leakage, tradeoffs, confidence, reproducibility

**Message framing**
- “Measure what matters”: task-specific metrics + failure modes
- “Calibrate judges”: align automated scoring to human intent
- “Promotion gates”: only ship changes that pass meaningful improvement thresholds

**Preferred channels**
- Medium (technical deep dives)
- Docs (reference + examples)
- Notebooks / minimal repros (optional)
- Recorded talks (15–25 min)

**Language level**
- You can use formal terms, but always pair them with a plain-English explanation.

**Anticipated objections → responses**

| Objection | Response angle | Proof/asset |
| --- | --- | --- |
| “Your statistical rigor is too simple for real evaluation.” | Agree for high-stakes domains: start with pragmatic gates, then raise rigor (bigger eval sets, stronger calibration, more conservative thresholds). | TVL promotion docs + “ML appendix” derivative |
| “LLM-as-judge is unreliable.” | Treat it as a measurement device: calibrate rubrics against human labels and use it where it correlates with outcomes. | `docs/agent_development_script.md` (judge calibration guidance) |

## Jargon Ladder (same idea, 3 levels)

Use the same underlying concept with different phrasing:

- **Exec**: “confidence checks”, “guardrails”, “risk of false alarms”
- **Builders**: “quality gates”, “regression thresholds”, “repeatable eval harness”
- **ML**: “statistical testing”, “multiple comparisons”, “calibration”, “tradeoff frontier”

Common substitutions:
- “Pareto front” → “best trade-offs set” / “shortlist of best trade-offs”
- “Chance constraints” → “confidence thresholds” / “probability-of-meeting-the-bar”
- “Cohen’s kappa” → “human–judge agreement score”
- “MRR / NDCG@k” → “how high the first relevant doc appears” / “ranking quality”
