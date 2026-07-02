# Track A — SDK surrogate evaluator (F1)

Part of the evaluator-coevolution mock-results E2E sim (a synthetic
"PolyMath QA agent judged by a primary evaluator plus a cheap surrogate
judge" scenario). This track proves **F1**: the SDK surrogate (pre-screen)
evaluator feature. This demo is self-contained — the small scenario slice it
needs (the 10 dataset examples, the surrogate's name/constant/expected
aggregate score) is inlined in the script itself; nothing is read from disk
at runtime.

## What it proves

`@traigent.optimize(..., evaluation=EvaluationOptions(surrogate_evaluator=..., surrogate_evaluator_name=...))`
attaches a cheap surrogate judge (`judge_surrogate_lenient`, constant score 0.90) alongside
the primary exact-match evaluator, over the SAME captured outputs (no re-execution). The demo
verifies this at **two** layers, both purely offline/in-process:

### 1. Local SDK optimization result (`TrialResult.to_dict()`)

`TrialResult.to_dict()` is the SDK's **in-process optimization result** — what
`optimize_sync` returns to your code. It is **not** the backend POST payload (real
submission rebuilds metadata from scratch; see layer 2). On the local result the demo
asserts, for every trial:

1. `metadata["surrogate_evaluator"]` — the real descriptor built by
   `traigent.core.optimization_pipeline.build_surrogate_descriptor`:
   `evaluator_id == "judge_surrogate_lenient"`, `metric_name == "surrogate_score"`,
   `config.fingerprint_source` matching `^fp1:[0-9a-f]{64}$` (computed by the real
   `compute_surrogate_fingerprint` over the scorer's source).
2. `metrics["surrogate_score"] == 0.90` — the real aggregate computed by
   `traigent.core.trial_lifecycle.apply_surrogate_scoring`.
3. **Per-example** `surrogate_score` present + numeric on **every** scored example, read at
   `metadata["example_results"][i]["metrics"]["surrogate_score"]`. This is the actual
   per-example injection: `apply_surrogate_scoring` writes `surrogate_score` into each
   scored example's metrics (`traigent/core/trial_lifecycle.py` ~187), which
   `trial_result_factory._build_success_trial_metadata` serialises into
   `metadata["example_results"]` (~265-269). Asserting the aggregate alone would not catch a
   broken per-example path; this does.

### 2. Offline wire-carry proof (SDK-built submission dict, no HTTP)

The real backend submission does **not** ship `to_dict()`. It rebuilds the metadata via
`traigent.core.metadata_helpers.build_backend_metadata` and assembles the POSTed dict via
`traigent.cloud.trial_operations.TrialOperations._build_trial_result_data`
(`trial_operations.py` ~557-577, wired at ~836-861). Both are **pure, network-free**
builders — `_build_trial_result_data` reads no instance state, so it is invoked unbound with
`self=None`; no cloud client, session, or socket is created. The demo runs them **offline**
on each trial and asserts the surrogate survives the from-scratch rebuild into the
submission dict:

- `result_data["metadata"]["surrogate_evaluator"]` — descriptor (incl. `fingerprint_source`)
  survives.
- `result_data["metadata"]["surrogate_score"] == 0.90` — aggregate survives.
- `result_data["metadata"]["measures"][i]["metrics"]["surrogate_score"]` — per-example
  surrogate survives on all rebuilt measures.

This runs in **both** full (`privacy=off`) and privacy-redacted (`privacy=ON`) modes: the
surrogate score is a content-free numeric, so it is deliberately privacy-carried
(`metadata_helpers._build_single_measure_privacy` ~678-683).

**Scope of this proof.** This demo shows that the **SDK-built, POST-shaped submission
dict** — the exact dict `TrialOperations` would send — carries the surrogate descriptor and
the per-example/aggregate `surrogate_score`, all the way through the real, network-free
builders. It does **not** make a network call, and it does **not** demonstrate backend
ingest or cell-splitting into a configs x examples x evaluators tensor — that is a separate
concern, demonstrated independently by the backend-side simulation of feature F2 in
`TraigentBackend`. Read this demo as: "the wire payload the SDK would send carries the
surrogate signal," not as "the surrogate reaches the backend evaluator tensor."

## Mock boundary

The "agent" (`polymath_synthetic_agent_impl`) is pure Python — no LLM call, no network
egress. `TRAIGENT_MOCK_LLM`/`TRAIGENT_OFFLINE_MODE` are set as belt-and-suspenders, not
load-bearing. The ONLY synthetic thing is the model's canned output; everything from the
`@traigent.optimize` decorator down through descriptor construction, fingerprinting,
per-example/aggregate surrogate scoring, and the submission-payload builders is the real
shipped SDK code in this repo.

## Run

```bash
cd <this repo root>
TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true ENVIRONMENT=development \
PYTHONPATH=$PWD python3 examples/advanced/surrogate_evaluator_coevolution_demo.py
```

Exit code 0 + `[TRIGGERED] F1 ...` on success; exit code 1 + `[NOT TRIGGERED] F1 ...` if any
trial is missing the descriptor, the aggregate score, a per-example surrogate score, or if
any of those fails to survive the offline submission builders.

## Non-tautology check (empirical)

Ran the same wiring with `surrogate_evaluator` omitted (simulating F1 reverted): confirmed
`metadata["surrogate_evaluator"]`, the aggregate `metrics["surrogate_score"]`, the
per-example `example_results[i].metrics["surrogate_score"]`, AND the rebuilt-wire
`surrogate_evaluator`/`surrogate_score` are all genuinely **absent** — exactly the condition
this demo's assertions catch. The `0.90` and the `fp1:<sha>` value are not hardcoded/echoed
by the demo; they are produced by the SDK's own surrogate-scoring and fingerprinting code
from this demo's own `judge_surrogate_lenient` callable.

## Judge panel construction — diversity needs different models, not prompts

Separate from the surrogate feature above, this is a design implication worth stating
explicitly because it's easy to get backwards. Observed on real Bedrock runs, one task
domain so far — not a universal claim:

- Two instances of the **same** strong judge model (Claude Haiku 4.5), one prompted as a
  normal judge and one prompted as a lax "rubber-stamp" judge, **converged** on the same
  correctness calls. A strong LLM judges the content properly more or less regardless of
  the rubric text it's handed.
- Consequence: a K>=2 judge panel built by varying the **prompt** on a single model has low
  effective diversity — its judges largely agree, their errors are correlated, and the
  panel collapses toward roughly one effective vote instead of K independent ones.
- **Implication**: to build a genuinely diverse panel, vary the **model** (different
  families and/or capability tiers), not just the prompt. Prompt variation alone is
  decorative for diversity purposes.
- This demo's surrogate exists for the **cost** axis, not diversity: a cheap, short-prompt
  judge on the same strong model tracks the authoritative judge's score at a fraction of
  the tokens (see `judge_surrogate_lenient` above). Don't treat a cost-motivated surrogate
  as an independent panelist.
- To check panel diversity empirically rather than assume K judges = K opinions, use the
  shipped ACET evaluator-quality audit's panel "effective-independent-votes" /
  `redundancy_ratio` metric — it will flag a panel with many judges but few effective votes.
