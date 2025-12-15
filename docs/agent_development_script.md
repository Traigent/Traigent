# AI Agent Development Playbook

Systematic evaluation & tuning with TraiGent (for GTM, Ops, Engineering, and product leaders).

## Who this is for

- **Execs / PMs**: use the quick reference + checklist to set acceptance criteria and reduce risk.
- **AI / Software engineers**: use the per-agent-type sections to build eval sets, scoring, and CI gates.
- **ML / evaluation folks**: use this as a starting point; tighten rigor based on stakes and noise.

## How to use this guide

1. Find your agent type in the quick reference table.
2. Copy the “what to tune” + “what to measure” ideas.
3. Build an offline eval set (start small; expand as needed).
4. Map Tuned Variables, objectives, constraints, and budgets into a TVL spec.
5. Tune, compare, and ship behind gates (don’t “just update the prompt”).

## Quick reference: metrics by agent type

| Agent type | Primary metrics (examples) | Key insight |
| --- | --- | --- |
| GTM & acquisition | Message quality score, compliance pass rate | Score message quality directly; don’t wait for conversion |
| Operations | Action sequence accuracy, decision accuracy | Compare proposed actions to ground truth; evaluate routing |
| Knowledge / RAG | Grounded accuracy, abstention balance | Separate retrieval from generation; reward appropriate uncertainty |
| Product / technical | Weighted test pass, quality/security score | Use deterministic CI signals first; judge readability second |
| Customer support | Resolution accuracy, escalation accuracy | Compare to expert responses; evaluate routing directly |

## The TraiGent loop (offline-first)

TraiGent treats agent improvement as an empirical workflow. You define measurable objectives and guardrails, then explore your configuration space **within an evaluation budget**.

1. **Build an evaluation dataset**: representative inputs paired with expected outputs (or expected behaviors).
2. **Define automated scoring**: deterministic checks where possible + calibrated rubrics where needed.
3. **Run configuration search**: test as many variants as your budget allows (could be dozens or thousands depending on cost, dataset size, and runtime).
4. **Review best trade-offs** (often called a “Pareto front”): shortlist configs that improve what you care about without violating guardrails.

Key insight: evaluation happens **offline against your eval set**—you validate changes before you expose them to production users.

## Mapping to TVL (Tuned Variable Language)

Each section maps to TVL concepts:

- **What you tune** → TVL `tvars` (Tuned Variables / configuration space)
- **Objectives & metrics** → TVL objectives (what you optimize)
- **Constraints / guardrails** → TVL constraints (what must not be violated)
- **Budgets** → TVL budgets (how much evaluation you allow)
- **Promotion gate** → TVL promotion rules (ship / don’t ship / manual review)

Note: evaluation wiring can still be code-driven (datasets, evaluators, rubrics). Treat the TVL spec as the contract that drives consistent decisions.

---

## 1) GTM & acquisition agents

### Common failure modes (what your eval should catch)

- Hallucinated company facts (“creepy personalization”)
- Over-claiming outcomes / unsubstantiated testimonials
- Spam triggers / compliance violations (CAN-SPAM/GDPR)
- “Good copy” that’s off-ICP or misaligned tone

### What you tune

- Message templates, subject lines, call scripts
- ICP filters, lead scoring thresholds, firmographic rules
- Channel mix (email vs LinkedIn vs in-app) and follow-up cadence
- Personalization depth (templated → research-intensive)

### Objectives & evaluation

**Objective: Message quality & ICP targeting**
- Primary metric: message quality score (e.g., weighted ICP fit + personalization + value prop clarity)
- Why: conversion data is slow and confounded; score quality directly for fast iteration

**Objective: Compliance & brand safety**
- Primary metric: compliance pass rate (hard gate in many orgs)
- Deterministic checks: policy rules, banned phrases, required disclaimers, spam scoring

### Eval set creation

- Source: historical outbound + CRM exports
- Start: 30–100 leads; expand with more diversity/edge cases as you learn
- Add gold messages for a subset (written by top SDRs / sales leadership)
- Calibration: target **high human–judge agreement**; thresholds depend on stakes (aim for “trustworthy enough to guide iteration”)

---

## 2) Operations agents

### Common failure modes

- Wrong action ordering that breaks workflows
- Silent failures (missing required fields, skipped validations)
- Over-autonomy (should have escalated) vs over-escalation (wastes humans)
- “Correct result” achieved via unsafe/unapproved steps

### What you tune

- Workflow steps, ordering, batching strategies
- Routing rules: assignment logic, escalation triggers, load balancing
- Validation logic: pre-checks, required field enforcement
- Autonomy thresholds: proceed vs request human approval

### Objectives & evaluation

**Objective: Task execution correctness**
- Primary metric: action sequence accuracy (exact match or acceptable alternatives)
- Deterministic validators: schema/business rule checks (e.g., “approval required for > $10K”)

**Objective: Routing / escalation decisions**
- Primary metric: decision accuracy
- Intuition: balance “false alarms” (unnecessary escalations) vs “misses” (missed escalations)

**Objective: Execution efficiency (only after correctness)**
- Primary metric: action economy (minimum required steps / proposed steps) on correct completions

### Eval set creation

- Source: workflow logs + task management systems + escalation/rollback tickets
- Label: correct action sequences (or acceptable alternatives) + correct escalation decisions
- Include boundary cases where experts disagree (helps define your policy)

---

## 3) Knowledge / RAG agents

### Common failure modes

- Confident wrong answers (high trust damage)
- Correct answers without evidence (ungrounded)
- Citations that don’t support the claim
- Over-abstention (“I don’t know” too often) vs under-abstention (hallucinations)

### What you tune

- Retrieval: `top_k`, similarity thresholds, reranker weights, hybrid search balance
- Chunking: size, overlap, semantic boundaries
- Generation: citation requirements, abstention triggers, confidence language

### Objectives & evaluation

**Objective: Answer correctness + grounding**
- Primary metric: grounded accuracy (correct *and* supported by cited passages)

**Objective: Retrieval quality**
- Primary metric: retrieval success rate (at least one relevant doc appears in top-k)
- Optional ranking metrics: “how high does the first relevant doc appear?” (MRR-style), overall ranking quality (NDCG-style)

**Objective: Appropriate abstention**
- Primary metric: abstention balance (catch both “false confidence” and “false humility”)

### Eval set creation

- Source: production query logs + documentation
- Include: paraphrases, multi-hop questions, stale-content questions
- Add “unanswerable” questions (things not in your KB) to validate abstention behavior

---

## 4) Product / technical agents (code-writing agents)

### Common failure modes

- Compiles/passes tests but is semantically wrong (tests incomplete)
- Security vulnerabilities or unsafe patterns
- Over-engineering (excess code/tokens for equivalent quality)
- Style/maintainability regressions that slow teams down

### What you tune

- Coding style conventions and documentation density
- Strategy: test-first vs implementation-first, refactor vs rewrite thresholds
- Tool integration: which tests/linters/scanners run and how strict they are

### Objectives & evaluation

**Objective: Functional correctness**
- Primary metric: weighted test pass rate (weight critical paths higher)
- Optional: mutation testing to detect “weak tests”

**Objective: Quality & security**
- Primary metric: quality/security score from static analysis + scanners (block critical findings)
- Note: run generated code in a sandbox/container in CI to contain risk

**Objective: Efficiency**
- Primary metric: solution economy (quality achieved / resources consumed)

---

## 5) Customer support agents

### Common failure modes

- Policy violations (refund limits, promises, missing disclaimers)
- Tone mismatch for distressed users (technically correct but harmful)
- Wrong escalation decisions (either route too often or miss critical escalations)
- Hallucinated account details / invented policy

### What you tune

- Tone calibration: empathy, formality, persona consistency
- Policy flexibility: when to grant exceptions, refund thresholds, escalation triggers
- Conversation strategy: clarification-first vs solution-first, multi-turn handling

### Objectives & evaluation

**Objective: Resolution quality**
- Primary metric: resolution accuracy vs expert-written gold responses (or labeled outcomes)
- Deterministic checks: policy rules (hard gates where needed)

**Objective: Tone & customer experience**
- Primary metric: tone quality score (empathy + clarity + professionalism)
- Calibration: align judge ratings with human ratings and/or historical CSAT where available (target strong correlation, tuned by stakes)

**Objective: Escalation decision quality**
- Primary metric: escalation accuracy (and track false escalations vs missed escalations)

---

## Implementation checklist (end-to-end)

1. **Choose your agent type** (or mix patterns if your agent spans categories).
2. **Define Tuned Variables**: list the behavior knobs you will explore.
3. **Define objectives + constraints**: pick 1 primary objective + 1–3 secondary objectives; set guardrails.
4. **Build an eval set**: start with 10–30 examples; expand as you increase confidence requirements.
5. **Implement scoring**: deterministic checks first; add calibrated rubrics for subjective qualities.
6. **Tune + gate + ship**: run tuning offline, then add CI gates to block regressions and surface missed improvements.

## Glossary (plain-English first)

| Term | Plain-English meaning |
| --- | --- |
| Evaluation dataset (“eval set”) | A curated set of test cases used to score agent behavior offline |
| LLM-as-judge | Using a model to score outputs against a rubric (treat it like a measuring device; calibrate it) |
| “Best trade-offs set” (Pareto front) | Configs where you can’t improve one goal without hurting another |
| “Balance of false alarms vs misses” (F1-style) | A way to measure decisions like “escalate vs handle” without optimizing only one side |
| Ranking quality (MRR/NDCG@k) | How well retrieval ranks relevant documents near the top |
| TVL (Tuned Variable Language) | TraiGent’s spec format for objectives, constraints, budgets, and Tuned Variables |
| `tvars` | Tuned Variables defined in TVL (the knobs TraiGent explores) |

## Getting started with TraiGent

Once you have an eval set and scoring in place, TraiGent’s SDK helps you explore configs and ship behind gates.

Next steps:
- Review the TVL spec reference for syntax and examples
- Explore runnable examples in the TraiGent repo
