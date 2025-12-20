## Final Glossary Table (Canonical Version 2.0)

---

### Core Primitives

| Term | Short / Symbol | Relations | Definition (Text) | Definition (Formal) |
|------|----------------|-----------|-------------------|---------------------|
| **Tuned Variable** | **TVAR** / tᵢ | Declared in **TSpec**; assigned by **Config** (θ); domain depends on **E_τ** | A single controllable knob influencing behavior (model, prompt template, retrieval settings, tool policy, stop rules). Domain may depend on environment. Informally: "TVARs are the knobs we tune." | Variable tᵢ with typed domain D_tᵢ(E_τ). A configuration assigns θ(tᵢ) ∈ D_tᵢ(E_τ). |
| **Tunable** | **𝒯** | Defined by **TSpec**; governed by **GSpec**; evaluated by **EvalSuite**; promoted/rolled back as a unit | The smallest unit of system behavior that is independently versioned, evaluated, owned, and promoted. May be atomic or a bounded mini-workflow. Has a defined interface. **It is the unit of tuning, evaluation, ownership, and promotion.** | Under E_τ: 𝒯(θ; x, E_τ) ⇒ (y, ξ, **c**), where x ~ I_τ, ξ is optional trace, **c** ∈ ℝᵖ is the resource vector (latency, cost, tokens, tool calls). |
| **Fixed Tunable** | — | Special case of **Tunable** with 0 TVARs | A Tunable with no tunable parameters (singleton search space). Useful as control group, legacy wrapper, or for incremental adoption. | |Θ| = 1 or equivalently all TVAR domains are empty; |𝒳| = 1 (singleton). θ is constant. |
| **Configuration** | **Config** / θ | Assignment to TVARs; element of **ExplorationSpace** | A concrete choice of values for all TVARs in scope. | θ ∈ Θ(E_τ) = ∏ᵢ D_tᵢ(E_τ). |
| **Baseline** | θ₀ | Referenced by **PromotionPolicy**; updated upon successful promotion | The currently promoted configuration against which candidates are evaluated. | π(θ₀, θ₁; evidence) compares candidate θ₁ against baseline θ₀. |

---

### Specification Artifacts

| Term | Short / Symbol | Relations | Definition (Text) | Definition (Formal) |
|------|----------------|-----------|-------------------|---------------------|
| **Tunable Specification** | **TSpec** | Declares TVARs, domains, constraints, objectives, targets; owned by product/AI engineer | Spec artifact describing *what can vary* and *what success means* for the Tunable. The "what." | TSpec := ⟨{tᵢ, D_tᵢ}ᵢ, C^str, C^op, Objectives, Targets, TraceSchema?⟩ |
| **Governance Specification** | **GSpec** (alias: **PromotionPolicy** in CI/UI) | Applies gates + promotion rules; often platform-defined; references **TSpec** | Policy artifact describing *how* promotion decisions are made: gates, thresholds, statistical settings, evidence requirements. The "how we decide." | GSpec := ⟨𝒢, π, α, ε, mult.test, EReq⟩ |
| **Pipeline Specification** | **PSpec** | Composes TSpecs; adds workflow semantics; defines pipeline-level objectives/gates | Spec for composition structure (DAG or bounded cycles), plus pipeline-level evaluation and governance. | PSpec := ⟨{TSpecₗ}ₗ, G, sem(G), Objectives^P, GSpec^P⟩ |
| **Trace Schema** | — | Optional in **TSpec**; defines ξ structure | Declaration of execution trace format for downstream compatibility, credit assignment, and debugging. | Schema for ξ: step IDs, tool calls, intermediate outputs, timing, state snapshots. |

---

### Space & Feasibility

| Term | Short / Symbol | Relations | Definition (Text) | Definition (Formal) |
|------|----------------|-----------|-------------------|---------------------|
| **Domain** | D_tᵢ | For a TVAR; parameterized by **E_τ** | The allowed set of values for a TVAR (typed; may depend on runtime environment). | D_tᵢ(E_τ) ⊆ 𝒯ᵢ where 𝒯ᵢ is the type's value set. |
| **Parameter Space** | Θ | Product of domains; pre-constraints | All possible TVAR assignments before constraints/budgets. | Θ(E_τ) = ∏ᵢ D_tᵢ(E_τ). |
| **Structural Feasible Set** | F_str | Filters Θ via logical constraints from **TSpec** | Configurations satisfying static/logical constraints between TVARs. | F_str(E_τ) = {θ ∈ Θ(E_τ) | C^str_E_τ(θ) = true}. |
| **Operational Feasible Set** | F_op | Filters F_str via **Budget** | Configurations feasible under budgets and operational limits. | F_op(E_τ, B_τ) = {θ ∈ F_str(E_τ) | C^op_E_τ,B_τ(θ) = true}. |
| **Exploration Space** | 𝒳 | What optimizers search; derived from TSpec + Budget | The final feasible configuration set after structural + operational constraints. | 𝒳(E_τ, B_τ) = F_op(E_τ, B_τ). |
| **Structural Constraints** | C^str | Declared in **TSpec**; filter Θ → F_str | Hard logical constraints linking TVARs (e.g., "if model=A then max_tokens ≤ 2k"). | Boolean predicate over θ. |
| **Operational Constraints** | C^op | Derived from **Budget**; filter F_str → F_op | Constraints from budgets/quotas/limits (e.g., max cost/run, tool-call caps, timeouts). | Boolean predicate over θ given B_τ. |
| **Budget** | B_τ | Shapes F_op; limits exploration and runtime | A cap on resources: trials, wallclock, tokens, spend, tool calls, iterations. | B_τ = ⟨max_trials, max_cost, max_time, ...⟩. |

---

### Environment & Evaluation

| Term | Short / Symbol | Relations | Definition (Text) | Definition (Formal) |
|------|----------------|-----------|-------------------|---------------------|
| **Environment Snapshot** | E_τ | Parameterizes domains, costs, tool catalogs, policies | Captures external conditions at time τ: model/tool availability, prices, rate limits, policy versions, index versions. | Conditioning variable: D_tᵢ(E_τ), J(θ; E_τ, I_τ). |
| **Input Distribution** | I_τ | Defines workload; induces **D_τ** | Distribution of real tasks/inputs representing what "good" means in practice. | x ~ I_τ. |
| **Evaluation Dataset** | D_τ | Realized sample; part of **EvalSuite** | Concrete sequence of inputs used to estimate objectives (can repeat). | D_τ = (x₁, ..., xₙ), typically xₖ ~ I_τ. |
| **Evaluation Suite** | **EvalSuite** / S_τ | Contains datasets, rubrics, judges, slicing, seeds; versioned artifact | Versioned bundle defining what gets measured and how; produces evidence for CI gates. | S_τ := ⟨{D}, rubrics, judges, slices, seeds, reporting⟩. |
| **Coverage** | Cov | Referenced by **Gates**; measured during evaluation | Scenario/trajectory coverage metric: fraction of targeted behaviors, call sites, or edge cases observed during evaluation. | Cov(D_τ, 𝒯) ∈ [0, 1]. Required threshold may be specified in GSpec. |

---

### Objectives & Scoring

| Term | Short / Symbol | Relations | Definition (Text) | Definition (Formal) |
|------|----------------|-----------|-------------------|---------------------|
| **Metric** | mⱼ | Per-input measurement; feeds **Objective** | A single measured quantity on one input (accuracy, latency, cost, safety score, etc.). | mⱼ(θ; x, E_τ) ∈ ℝ. |
| **Objective** | — | Declared in **TSpec**; induces **Scoring Functional** | A named intent (quality, cost, latency, safety) defined by metric(s), aggregation, and orientation (min/max). | Objective spec induces Jⱼ(·) via metric mⱼ and aggregator Aⱼ. |
| **Target** | Tgt | Referenced by **Gates**; declared in **TSpec** | A desired performance level for an objective (scalar threshold or band/range). | e.g., J_q(θ) ≥ 0.85 (min quality), J_ℓ(θ) ≤ 2s (max latency), or θ₁ ≤ J(θ) ≤ θ₂ (band). |
| **Orientation** | σ | Per-objective; used in Pareto comparisons | Indicates whether each objective should be maximized (+1) or minimized (−1). | σ ∈ {+1, −1}ᵐ for m objectives. |
| **Scoring Functional** | J | Population-level; used by optimizer + promotion gate | Formal mapping from configuration to score(s) under environment and workload distribution. | J(θ; E_τ, I_τ) = (J₁, ..., Jₘ), e.g., Jⱼ = 𝔼_{x~I_τ}[mⱼ(θ; x, E_τ)]. |
| **Empirical Scoring Estimate** | Ĵ | Computed on **D_τ**; used in gate decisions | Sample-based estimate of the scoring functional from evaluation runs. | Ĵⱼ(θ) = (1/n) Σₖ mⱼ(θ; xₖ, E_τ) for xₖ ∈ D_τ. |

---

### Governance & Promotion

| Term | Short / Symbol | Relations | Definition (Text) | Definition (Formal) |
|------|----------------|-----------|-------------------|---------------------|
| **Gate** | 𝒢 | Declared in **GSpec**; enforced in CI | Pass/fail rules controlling promotion (and often merge). | 𝒢(θ) = Pass ⟺ ⋀ᵣ gᵣ(Ĵ(θ), SLO(θ), Cov). |
| **Chance Constraint (SLO)** | (δ, α) | Special gate type; reliability under uncertainty | A probabilistic constraint (e.g., "p95 latency ≤ 2s with 95% confidence"). | Pr[g(θ; x, E_τ) ≤ 0] ≥ 1 − δ, verified at confidence 1 − α. |
| **Promotion Policy** | π | Part of **GSpec**; yields Promote/Reject/NoDecision | Decision rule for adopting a new config as default (with rollback support). | π(θ₀, θ₁; EB) ∈ {Promote, Reject, NoDecision}. |
| **ε-Pareto Dominance** | ≻_ε | Used for multi-objective promotion | Non-inferiority on all objectives within margins, meaningful improvement on at least one. | θ ≻_ε θ' ⟺ ∀j: σⱼJⱼ(θ) ≥ σⱼJⱼ(θ') − εⱼ ∧ ∃k: σₖJₖ(θ) > σₖJₖ(θ') + εₖ. |
| **Evidence Requirements** | EReq | Part of **GSpec**; schema for what **EB** must contain | Specification of what must be logged for promotion decisions to be reviewable and reproducible. | Required fields: dataset checksum, env snapshot ID, seeds, gate outcomes, coverage. |
| **Evidence Bundle** | EB | Output of evaluation run; validated against **EReq** | The collected artifacts emitted by evaluation: dataset checksum, env snapshot, seeds, metrics, traces, judge configs, gate results. | EB := ⟨D_τ checksum, E_τ ID, seeds, Ĵ, ξ samples, 𝒢 outcomes, Cov⟩. |

---

### Composition

| Term | Short / Symbol | Relations | Definition (Text) | Definition (Formal) |
|------|----------------|-----------|-------------------|---------------------|
| **Pipeline** | P | Composes **Tunables**; governed by **PSpec**; evaluated on trajectories | End-to-end workflow composed of Tunables and control flow (white-box orchestration). | P = compose(𝒯₁, ..., 𝒯ₖ, control) ⇒ (y, ξ, **c**). |
| **Workflow Tunable** | — | A **Pipeline** encapsulated as a **Tunable** | A pipeline that is optimized/promoted as a single unit with unified TSpec + GSpec. Cycles require explicit **execution bounds**. | When a pipeline has a single 𝒳 and is governed as one unit, it is a Workflow Tunable. |
| **Execution Bounds** | — | Required for cyclic workflows to be tunable | Explicit limits that make cyclic workflows comparable: iteration caps, convergence criteria, timeout, or fixed-point rules. | e.g., max_iterations, timeout, convergence_threshold. |
| **Trajectory** | ξ | Produced by **Pipeline** execution; evaluated for credit assignment | The sequence of steps, tool calls, and state updates in a workflow run. Conforms to **TraceSchema** if declared. | ξ = (step₁, ..., stepₙ) with metadata per step. |

---

## Key Principles

> **The Encapsulation Rule:** A Pipeline becomes a Tunable when it is encapsulated with a unified configuration space and governed/promoted as one unit. Pipelines are white-box orchestration; Tunables are black/gray-box optimization targets.

> **The Unit Principle:** A Tunable is the unit of tuning, evaluation, ownership, and promotion. What you tune is what you evaluate is what you own is what you ship.

> **The Semantics Requirement:** You can't tune what you haven't semantically defined. DAGs are tunable by default; cycles require explicit execution bounds.

---

## Notation Summary

| Symbol | Meaning |
|--------|---------|
| θ | Configuration (TVAR assignment) |
| θ₀ | Baseline configuration (currently promoted) |
| x | Input (from workload distribution) |
| 𝒯 | Tunable (formal symbol) |
| P | Pipeline |
| J | Scoring Functional (population) |
| Ĵ | Empirical Scoring Estimate (computed on D_τ) |
| σ | Orientation vector (±1 per objective) |
| E_τ | Environment snapshot at time τ |
| I_τ | Input distribution at time τ |
| D_τ | Evaluation dataset (sampled from I_τ) |
| S_τ | Evaluation Suite |
| Θ | Parameter space (pre-constraints) |
| 𝒳 | Exploration space (post-constraints) |
| 𝒢 | Gate(s) |
| π | Promotion policy |
| ξ | Execution trace / trajectory |
| **c** | Resource vector (latency, cost, tokens, tool calls) |
| EB | Evidence Bundle |
| Cov | Coverage metric |

---

# Front Matter: Notation & Definitions

## 0. How to Use This Section

This section establishes the canonical vocabulary and notation used throughout this book. Every formal definition, code example, and CI/CD reference builds on these primitives.

**Conventions:**
- Terms in **bold** are defined concepts
- Symbols in `monospace` appear in code/specs
- Mathematical notation uses standard LaTeX conventions
- On first use in each chapter, we spell out acronyms (e.g., "Tunable Specification (TSpec)")

If you encounter an unfamiliar term, return here.

---

## 1. The Core Thesis

Classical software engineering assumes **functions are deterministic**: given the same code and inputs, you get the same output. LLM-era systems break this assumption. Once your system includes `llm.invoke(...)` and surrounding configuration (prompts, models, retrieval, tools, policies, budgets), the mapping from inputs to outputs becomes:

- **Stochastic** (same config, different outputs)
- **Environment-dependent** (model availability, prices, and policies shift)
- **Underspecified** (the "real" behavior lives in config files, not code)

This book argues that the correct unit of engineering shifts from **code + tests** to **behavior + evaluation**. We call this paradigm **Evaluation-Driven Development (EDD)**.

The vocabulary below makes this shift precise and actionable.

---

## 2. The Central Abstraction: Tunable

A **Tunable** (symbol: 𝒯) is the smallest unit of system behavior that is independently:

1. **Tuned** — its configuration is optimized over a search space
2. **Evaluated** — it has its own evaluation suite and metrics
3. **Owned** — a team is responsible for its spec and governance
4. **Promoted** — it is shipped/rolled back as a unit

This unification is deliberate. In classical SE, these concerns are often separate (code ownership ≠ test ownership ≠ deployment ownership). In EDD, they converge on the Tunable.

**Formally:**

Under environment snapshot E_τ, a Tunable executes as:

```
𝒯(θ; x, E_τ) ⇒ (y, ξ, c)
```

Where:
- θ is the **configuration** (assignment to all TVARs)
- x is the **input** (sampled from workload distribution I_τ)
- y is the **output**
- ξ is the optional **execution trace**
- **c** ∈ ℝᵖ is the **resource vector** (latency, cost, tokens, tool calls)

A Tunable may be **atomic** (a single LLM call with its surrounding logic) or a **bounded mini-workflow** (multiple steps with control flow, as long as it has defined semantics).

---

## 3. Tuned Variables (TVARs)

A **TVAR** (Tuned Variable) is a single controllable knob that influences behavior. Examples:

| TVAR | Type | Example Domain |
|------|------|----------------|
| `model_id` | enum | {gpt-4o, claude-sonnet-4-20250514, gemini-pro} |
| `prompt_template` | registry | registered prompt IDs |
| `temperature` | float | [0.0, 1.0] |
| `retrieval_k` | int | [1, 20] |
| `tool_allowlist` | set | subsets of {web_search, calculator, code_exec} |
| `max_retries` | int | [0, 5] |

**Key property:** A TVAR's domain may depend on the **environment**. For example, available models change over time; the domain D_model(E_τ) is not static.

**Informal synonym:** "TVARs are the knobs we tune."

**Notation:**
- tᵢ — a single TVAR
- D_tᵢ(E_τ) — its domain under environment E_τ
- θ(tᵢ) — the assigned value in configuration θ

> **Note:** TVAR is unrelated to Haskell's STM TVar (transactional variable). The similarity is coincidental.

---

## 4. Configuration and Search Spaces

A **Configuration** (θ) is a concrete assignment to all TVARs in scope:

```
θ ∈ Θ(E_τ) = ∏ᵢ D_tᵢ(E_τ)
```

The **feasibility funnel** progressively constrains what can be searched:

```
Θ(E_τ)  →  F_str(E_τ)  →  F_op(E_τ, B_τ)  =  𝒳(E_τ, B_τ)
   ↓           ↓              ↓
  all      structural     operational     exploration
 possible  constraints    constraints       space
```

| Space | What filters it | Example |
|-------|-----------------|---------|
| Θ | Nothing (raw product) | All combinations of model × prompt × temperature |
| F_str | Structural constraints (C^str) | "if model=gpt-4o then max_tokens ≤ 8k" |
| F_op | Budget constraints (C^op) | "cost per trial ≤ $0.50" |
| 𝒳 | Final searchable space | What the optimizer actually explores |

The **Exploration Space** (𝒳) is what matters for optimization.

---

## 5. Specification Artifacts

### TSpec (Tunable Specification)

Declares **what** the Tunable is:

```yaml
# triage.tspec.yaml
tunables:
  model_id:
    type: enum
    domain: [gpt-4o, claude-sonnet-4-20250514]
  prompt_template:
    type: registry
    registry: prompts/triage/*
  temperature:
    type: float
    domain: [0.0, 0.7]

constraints:
  - if: model_id == "gpt-4o"
    then: temperature <= 0.5

objectives:
  quality:
    metric: f1_score
    orientation: maximize
  latency:
    metric: p95_latency_ms
    orientation: minimize

targets:
  quality: ">= 0.85"
  latency: "<= 1200"

trace_schema: schemas/triage_trace.json
```

**Owned by:** Product/AI engineers

### GSpec (Governance Specification)

Declares **how** promotion decisions are made:

```yaml
# governance/standard.gspec.yaml
gates:
  - type: target_gate
    objectives: [quality, latency]
  - type: coverage_gate
    min_coverage: 0.90
  - type: slo_gate
    constraint: "p99_latency <= 2000ms"
    confidence: 0.95

promotion:
  policy: epsilon_pareto
  alpha: 0.05
  epsilon:
    quality: 0.02
    latency: 50
  multiplicity_control: benjamini_hochberg

evidence_requirements:
  - dataset_checksum
  - environment_snapshot_id
  - seeds
  - all_traces
```

**Owned by:** Platform/Trust & Safety teams

**Alias in CI/UI:** "PromotionPolicy"

---

## 6. Environment and Evaluation

### Environment Snapshot (E_τ)

Captures external conditions at time τ:
- Model catalog (which models exist, their costs)
- Tool availability
- Rate limits and quotas
- Policy versions
- Retrieval index versions

**Why it matters:** Two evaluations under different E_τ are not directly comparable. The environment must be recorded in every Evidence Bundle.

### Input Distribution (I_τ) and Dataset (D_τ)

- **I_τ** — the distribution of real tasks/inputs ("what users actually send")
- **D_τ** — a concrete sample: D_τ = (x₁, ..., xₙ) where xₖ ~ I_τ

The distinction matters: I_τ is the population; D_τ is what you actually run.

### Evaluation Suite (S_τ)

The versioned artifact containing:
- Datasets (possibly multiple, for different slices)
- Rubrics (scoring criteria)
- Judges (LLM-as-judge configs, human eval protocols)
- Slicing definitions (e.g., by language, difficulty, domain)
- Seeds (for reproducibility)
- Reporting configuration

---

## 7. Objectives and Scoring

### The Hierarchy

```
Metric (mⱼ)          Per-input measurement (e.g., latency of one call)
    ↓
Objective            Named intent + aggregation + orientation
    ↓
Scoring Functional (J)   Mathematical mapping: θ → score vector
    ↓
Empirical Estimate (Ĵ)   Sample-based estimate from D_τ
```

### Orientation (σ)

Each objective has an orientation:
- σⱼ = +1 → maximize (e.g., quality)
- σⱼ = −1 → minimize (e.g., latency, cost)

The orientation vector σ ∈ {+1, −1}ᵐ is used in Pareto comparisons.

### Population vs Estimate

- **J(θ; E_τ, I_τ)** — the "true" scoring functional (expectation over I_τ)
- **Ĵ(θ)** — the empirical estimate computed on D_τ

All gate decisions use Ĵ, with statistical corrections for uncertainty.

---

## 8. Governance and Promotion

### Gates (𝒢)

Pass/fail rules that must be satisfied for promotion:

```
𝒢(θ) = Pass  ⟺  ⋀ᵣ gᵣ(Ĵ(θ), SLO(θ), Cov)
```

Common gate types:
- **Target gates:** Ĵⱼ(θ) meets target threshold
- **SLO gates:** Chance constraint satisfied at confidence level
- **Coverage gates:** Minimum scenario coverage achieved
- **Regression gates:** No degradation vs baseline θ₀

### Promotion Policy (π)

The decision rule:

```
π(θ₀, θ₁; EB) ∈ {Promote, Reject, NoDecision}
```

Where:
- θ₀ is the **baseline** (current production config)
- θ₁ is the **candidate**
- EB is the **Evidence Bundle**

**NoDecision** is a valid outcome — it means uncertainty is too high to decide. This is a feature, not a bug.

### ε-Pareto Dominance

For multi-objective comparison:

```
θ ≻_ε θ'  ⟺
    ∀j: σⱼJⱼ(θ) ≥ σⱼJⱼ(θ') − εⱼ    (non-inferior within margin)
  ∧ ∃k: σₖJₖ(θ) > σₖJₖ(θ') + εₖ    (meaningfully better on at least one)
```

The ε margins prevent "statistically significant but practically meaningless" promotions.

### Evidence Bundle (EB)

The output of an evaluation run:

```
EB := {
  dataset_checksum: "sha256:abc123...",
  environment_snapshot_id: "env-2024-03-15-prod",
  seeds: [42, 1337, 7],
  scores: {quality: 0.87, latency: 1050},
  gate_outcomes: {target: PASS, slo: PASS, coverage: PASS},
  coverage: 0.94,
  trace_samples: [...],
  judge_configs: {...}
}
```

Evidence Requirements (EReq) in GSpec specify what EB must contain.

---

## 9. Composition: Pipelines and Workflows

### Pipeline (P)

A composition of Tunables with control flow:

```
P = compose(𝒯₁, ..., 𝒯ₖ, control) ⇒ (y, ξ, c)
```

Pipelines are **white-box**: you can see and configure the internal structure.

### The Encapsulation Rule

> A Pipeline becomes a Tunable when it is encapsulated with a unified configuration space and governed/promoted as one unit.

This creates a **Workflow Tunable** — a pipeline treated as a black/gray-box for optimization purposes.

### Execution Bounds

Cyclic workflows (agentic loops, retry logic) require explicit bounds to be tunable:

| Bound Type | Example |
|------------|---------|
| Iteration cap | max_iterations = 10 |
| Timeout | timeout = 30s |
| Convergence | stop when Δ < 0.01 |
| Resource limit | max_tool_calls = 5 |

**The Semantics Requirement:** You can't tune what you haven't semantically defined. Without execution bounds, cyclic workflows have undefined comparison semantics.

### Trajectory (ξ)

The execution trace of a pipeline run:

```
ξ = (step₁, step₂, ..., stepₙ)
```

Each step includes: step ID, inputs, outputs, timing, tool calls, state changes.

Trajectories are used for:
- Pipeline-level metrics
- Credit assignment (which step caused a regression?)
- Debugging and replay

---

## 10. CI/CD Mapping

How these concepts appear in Git and CI:

| Concept | Git Artifact | CI Surface |
|---------|--------------|------------|
| TSpec | `*.tspec.yaml` | Validated by pre-commit hooks |
| GSpec | `governance/*.gspec.yaml` | Applied by promotion gate |
| EvalSuite | `eval/` directory + manifest | Run by PR checks |
| Evidence Bundle | CI job output | Stored in artifact registry |
| Promotion Manifest | `promotions/*.json` | Created on successful promotion |
| Baseline | `configs/production.yaml` | Updated on promotion |

### Typical Workflow

```
1. Engineer modifies TSpec (changes prompt template domain)
2. Pre-commit: TSpec validation passes
3. PR opened
4. CI: Evaluation runs on D_τ under pinned E_τ
5. CI: Evidence Bundle generated
6. CI: Gates evaluated against GSpec
7. CI: Promotion policy decides (Promote/Reject/NoDecision)
8. If Promote: baseline updated, manifest stored
9. If Reject: PR blocked with gate failure details
10. If NoDecision: request more evaluation budget or manual review
```

---

## 11. Quick Reference Card

### Symbols

| Symbol | Meaning |
|--------|---------|
| θ | Configuration |
| θ₀ | Baseline |
| x | Input |
| 𝒯 | Tunable |
| P | Pipeline |
| J | Scoring Functional (population) |
| Ĵ | Scoring Functional (empirical) |
| σ | Orientation (±1 per objective) |
| E_τ | Environment snapshot |
| I_τ | Input distribution |
| D_τ | Evaluation dataset |
| S_τ | Evaluation Suite |
| Θ | Parameter space |
| 𝒳 | Exploration space |
| 𝒢 | Gates |
| π | Promotion policy |
| ξ | Trajectory |
| **c** | Resource vector |
| EB | Evidence Bundle |

### Key Equations

**Tunable execution:**
```
𝒯(θ; x, E_τ) ⇒ (y, ξ, c)
```

**Feasibility funnel:**
```
Θ → F_str → F_op = 𝒳
```

**Scoring functional:**
```
J(θ; E_τ, I_τ) = (J₁, ..., Jₘ)
Jⱼ = 𝔼_{x~I_τ}[mⱼ(θ; x, E_τ)]
```

**Empirical estimate:**
```
Ĵⱼ(θ) = (1/n) Σₖ mⱼ(θ; xₖ, E_τ)
```

**Promotion decision:**
```
π(θ₀, θ₁; EB) ∈ {Promote, Reject, NoDecision}
```

**ε-Pareto dominance:**
```
θ ≻_ε θ' ⟺ ∀j: σⱼJⱼ(θ) ≥ σⱼJⱼ(θ') − εⱼ ∧ ∃k: σₖJₖ(θ) > σₖJₖ(θ') + εₖ
```

---

## 12. Glossary Index

For detailed definitions, see the Glossary Table in the Appendix.

**Core:** TVAR, Tunable, Configuration, Baseline

**Specs:** TSpec, GSpec, PSpec, TraceSchema

**Spaces:** Domain, Parameter Space (Θ), Exploration Space (𝒳), Feasibility Funnel

**Environment:** Environment Snapshot (E_τ), Input Distribution (I_τ)

**Evaluation:** Metric, Objective, Target, Scoring Functional (J), Empirical Estimate (Ĵ), Evaluation Suite, Coverage

**Governance:** Gate, SLO, Promotion Policy, ε-Pareto, Evidence Bundle, Evidence Requirements

**Composition:** Pipeline, Workflow Tunable, Execution Bounds, Trajectory

