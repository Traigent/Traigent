# Minimal Agent Spec Template (Copy/Paste)

Use this to define “what good looks like” before you tune or ship changes.

## 1) Agent context

- **Agent name**:
- **User/job-to-be-done**:
- **Where it runs** (product, support, internal ops):
- **Primary risks** (brand, compliance, security, cost, user trust):

## 2) Tuned Variables (what we will tune)

List only variables that change behavior.

- `model`:
- `temperature` / `top_p`:
- retrieval knobs (e.g., `k`, thresholds, reranker):
- tool routing thresholds:
- prompt/policy variants:

## 3) Evaluation set (how we test changes offline)

- **Source** (prod logs, SME-created, synthetic):
- **Size (start)**: 10–30 examples
- **Size (confidence)**: expand as needed based on stakes/cost
- **Edge cases** to include:
- **Ground truth** type (gold answers, labels, acceptable actions):

## 4) Scoring (how we measure)

- **Primary objective** (one): e.g., “resolution accuracy”, “grounded correctness”
- **Secondary objectives**: e.g., cost, latency, tone, policy compliance
- **Deterministic checks** (tests/validators):
- **LLM-as-judge rubric** (if used): what is scored, with anchored levels
- **Calibration plan**: how you’ll decide the judge is “good enough”

## 5) Constraints (what must not happen)

Examples:
- accuracy must not drop below X (or must not regress vs baseline)
- cost per call must stay under Y
- latency must stay under Z
- compliance must pass 100% (hard gate)

## 6) Promotion gate (ship / don’t ship)

Plain-English rule:
- “Ship only if we improve meaningfully on the primary objective without violating constraints.”

Operational details:
- baseline to compare against:
- what counts as “meaningful”:
- what happens if results are mixed (manual review vs block):

## 7) Ops loop (after shipping)

- What signals indicate drift/regression:
- When to re-run tuning:
- Rollback plan:
- Owner (team/person):

