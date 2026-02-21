# Cost Architecture Validation — Branch Objectives

**Branch:** `feat/cost-architecture-validation`
**Created:** 2026-02-20
**Status:** Discovery
**Reviewed by:** Codex (GPT-5.3 xhigh) — 2026-02-20 (2 rounds)

---

## Implemented Hardening (2026-02-21)

The following items from this validation were implemented on
`fix/cost-accounting-bugs-pr1`:

- Canonical runtime strict mode: `TRAIGENT_STRICT_COST_ACCOUNTING` (default `false`).
  - When `true`, runtime post-call paths use strict pricing resolution and fail fast on
    unknown/unpriced models.
  - This surfaces all cost-path failures (not only `UnknownModelError`) in integration
    handlers that intentionally catch generic exceptions in non-strict mode.
- `CostEnforcer` fail-fast behavior now treats either flag as strict:
  - `TRAIGENT_REQUIRE_COST_TRACKING=true` or
  - `TRAIGENT_STRICT_COST_ACCOUNTING=true`.
- Claude alias pricing mappings fixed to dated priced models:
  - `claude-haiku`, `claude-sonnet`, `claude-opus`.
- Hybrid aggregated metrics now emit both `cost` and `total_cost`, and orchestrator
  extraction uses `total_cost` with `cost` fallback.
- CI/pre-commit guardrail added to prevent reintroducing
  `cost_from_tokens(..., strict=False)` in runtime post-call paths
  (allowlisted only for query pricing helper usage).

---

## Problem Statement

The cost calculation subsystem has accumulated organic complexity. A cost query traverses up to **4 fallback tiers**, passes through **11 silent undercount / unknown-cost paths**, and offers **5+ entry points** for the same operation. The result: when cost tracking fails, it fails **invisibly** — the user sees `$0.00` and doesn't know if that's real or broken.

This branch validates the architecture, documents intentional decisions, and eliminates accidental complexity.

---

## Current State (Audit Findings)

### Silent Undercount / Unknown-Cost Paths

Not all of these return literal `0.0` — some preserve a default-zero state, some return `None`, and one returns a non-zero heuristic that undercounts. The common thread: **the caller has no signal that the cost is unreliable.**

| # | Location | Condition | Failure Mode |
|---|----------|-----------|--------------|
| 1 | `cost_calculator.py` `_safe_calculate_prompt_cost` | Any exception | Returns 0.0 silently |
| 2 | `cost_calculator.py` `_safe_calculate_completion_cost` | Any exception | Returns 0.0 silently |
| 3 | `cost_calculator.py` `_calculate_from_tokens` | Any exception | Returns (0.0, 0.0) silently |
| 4 | `cost_calculator.py` `CostCalculator.calculate_cost` | litellm unavailable | Returns empty CostBreakdown (all zeros) |
| 5 | `cost_calculator.py` `_fallback_cost_from_tokens` | Model not in fallback dict | Returns (0.0, 0.0) |
| 6 | `cost_calculator.py:706-710` `calculate_cost` method 2 | **Partial token data** (one of input/output is 0) | Falls through token-based path; total stays 0 |
| 7 | `platforms.py` `_calculate_cost` | Any exception | Returns 0.0 |
| 8 | `metrics_tracker.py` `CostCalculator.calculate_cost` | Any exception (non-debug) | Exception swallowed; default-zero state preserved |
| 9 | `metrics_tracker.py:1015` | Exception swallowed | Same — invisible to grep for `return 0.0` |
| 10 | `handler.py:697` `_fallback_cost_estimate` | All pricing lookups fail | Returns **non-zero heuristic** `(in*1.0 + out*3.0)/1M` — an undercount, not zero |
| 11 | `cost_estimator.py` `extract_trial_cost` | Cost key missing | Returns `None` (triggers unknown-cost mode in CostEnforcer) |

**Consequence:** CostEnforcer tracks 0.0 (or undercount) spend -> budget limits never trigger -> runaway costs.

**Note on #6 (Codex R1 finding):** `cost_calculator.py:706` requires `input_tokens > 0 AND output_tokens > 0` to enter the token-based path. A response with e.g. `input_tokens=500, output_tokens=0` (streaming partial) skips token costing entirely. Reachable from `metrics_tracker.py:1089` via the unified tracking path.

### Token/Cost Estimation Heuristics (ALL locations)

| Location | Heuristic | Type | Call Context |
|----------|-----------|------|--------------|
| `cost_calculator.py:268` | `len(text) // 3` | Char -> tokens | **Post-call** fallback when litellm token_counter fails |
| `metrics_tracker.py:1075` | `length * 0.25` | Char -> tokens | **Post-call** privacy mode token estimation |
| `platforms.py:803` | `len(prompt.split()) * 1.3` | Words -> tokens | **Pre-estimation** when litellm import fails |
| `handler.py:697` | `(input * 1.0 + output * 3.0) / 1_000_000` | Tokens -> cost (hardcoded rates) | **Post-call** last-resort in `on_chat_model_end` callback chain |

Four different heuristics across four files. The first two estimate tokens from text (different ratios). The third estimates tokens from words. The fourth estimates cost from tokens using hardcoded rates. **All four are in scope** — the first three for removal, the fourth for replacement with fail-loud behavior (see R3).

**Correction (Codex R2 finding):** `handler.py:697` is NOT pre-estimation. It's called from `_estimate_cost()` which is invoked by `on_chat_model_end()` at `handler.py:409` — a post-call callback. It must follow post-call fail-loud rules.

### Redundant Entry Points

| Function | Location | Used By |
|----------|----------|---------|
| `calculate_prompt_cost()` | Module-level | metrics_tracker (legacy), platforms.estimate_cost |
| `calculate_completion_cost()` | Module-level | metrics_tracker (legacy), platforms.estimate_cost |
| `CostCalculator.calculate_cost()` | Class method | metrics_tracker (unified path) |
| `calculate_llm_cost()` | Module-level convenience | External callers, handler._estimate_cost |
| `CostCalculator._calculate_from_tokens()` | Private class method | platforms._calculate_cost |
| `get_model_token_pricing()` | Module-level | cost_estimator (pre-estimation) |

### Integration Cost Paths (Who Uses What)

| Integration | Has Token Counts? | Token Source | Cost Path |
|-------------|-------------------|-------------|-----------|
| OpenAI (platforms.py) | Yes (usage object) | `usage.prompt_tokens` / `completion_tokens` | `get_cost_calculator()._calculate_from_tokens()` |
| LangChain handler | Yes (llm_output) | `response.llm_output["token_usage"]` at `handler.py:397` | `_estimate_cost()` -> `calculate_llm_cost()` at `handler.py:668` |
| LangFuse | Yes (response metadata) | Via LangChain callback chain | Same as LangChain |
| **Bedrock streaming** | **No (usage=None)** | `bedrock_client.py:330` | **Falls through all paths -> silent 0** |
| Bedrock non-streaming | Yes (data.get("usage")) | Response body | Standard token path |
| Pre-call estimation | No (prompt only) | N/A | `calculate_prompt_cost` (litellm token_counter) |
| Mock mode | No | N/A | Hardcoded 0.0 + token estimation from lengths |

---

## Design Principles

1. **Fail loud, not silent.** A cost of 0.0 for a real LLM call is almost always a bug. Raise or warn visibly — don't return 0.0 and hope someone notices.

2. **One canonical path.** Token counts -> cost. Text -> tokens is litellm's job (not ours with `len//3`). Pre-estimation is a separate concern from post-call tracking.

3. **Trust the integration, but verify.** Most integrations provide token counts — but not all modes do (Bedrock streaming). Handle missing data explicitly, not silently.

4. **Intentional degradation.** If litellm is missing, that's a clear error — not a "gracefully handle" case. The SDK requires litellm. If a model is unknown, say so — don't guess. Use a `strict` flag to distinguish post-call tracking (must raise) from pre-estimation (may approximate).

5. **Separate concerns.** Pre-call estimation (budget approval) has different accuracy requirements than post-call tracking (cost enforcement). Don't share the same fallback chain.

---

## Requirements

### R1: Eliminate silent undercount in post-call cost tracking

Every code path that handles a real LLM response must either produce a positive cost or raise/warn explicitly. The `_safe_calculate_*` wrapper pattern that swallows exceptions and returns 0.0 must be replaced with visible failure handling. **Includes** the partial-token-data gap at `cost_calculator.py:706` (Codex R1 finding) and the handler.py:697 hardcoded-rate undercount (Codex R2 finding #1).

### R2: Public canonical `cost_from_tokens()` function

Define a **public** `cost_from_tokens(input_tokens, output_tokens, model, *, strict=True) -> (input_cost, output_cost)` as the single entry point for all post-call cost calculation. Route existing callers (`_calculate_from_tokens`, `calculate_llm_cost`, handler._estimate_cost) through it. `strict=True` raises on unknown models; `strict=False` returns (0, 0) with warning (for pre-estimation).

### R3: Remove ALL post-call heuristics

Delete from post-call paths:
- `len(text) // 3` (cost_calculator.py:268) — char-to-token estimation
- `length * 0.25` (metrics_tracker.py:1075) — char-to-token estimation
- `(input * 1.0 + output * 3.0) / 1_000_000` (handler.py:697) — hardcoded cost rates in `on_chat_model_end` callback

The pre-estimation heuristic (`len(prompt.split()) * 1.3` in platforms.py:803) is acceptable as a clearly-labeled approximation for pre-call budget approval only.

### R4: Separate pre-estimation from post-tracking (with migration)

**End state:** Pre-estimation uses `ESTIMATION_MODEL_PRICING` (renamed from current `FALLBACK_MODEL_PRICING` + tier heuristics). Post-call tracking uses litellm only via `cost_from_tokens(strict=True)`.

**Migration path:**
1. **Phase 1 (this branch):** Rename `FALLBACK_MODEL_PRICING` -> `ESTIMATION_MODEL_PRICING`. Post-call paths stop using it (use litellm + strict mode). Pre-estimation continues using it.
2. **Phase 2 (future):** Evaluate whether `ESTIMATION_MODEL_PRICING` can be auto-generated from litellm's bundled data instead of manually maintained.

### R5: Deprecate text-based cost functions

`calculate_prompt_cost(text, model)` and `calculate_completion_cost(text, model)` are tokencost legacy wrappers. Mark deprecated, provide migration path to token-based calculation.

### R6: Make litellm a hard requirement for cost tracking

If litellm is not installed, cost tracking should raise at initialization — not silently return 0.0 for every call.

### R7: Validate integration cost extraction

Each integration (OpenAI, Anthropic, LangChain, **Bedrock non-streaming**) must have a test proving it extracts token counts from real response formats and produces correct costs. **Bedrock streaming** must have a test proving it handles `usage=None` with a visible warning.

### R8: Consolidate metrics_tracker CostCalculator

The nested `CostCalculator` class in `metrics_tracker.py` should be consolidated with `cost_calculator.py`'s `CostCalculator`. Keep a temporary compatibility shim for legacy callers.

---

## Assumptions

### A1: litellm is always available in production

litellm is in the core dependencies. The `LITELLM_AVAILABLE = False` codepath is only for broken installations, not a supported mode.

### A2: Most (not all) supported LLM providers return token usage

OpenAI, Anthropic, Google include `usage` in their API responses. **Exception:** Bedrock streaming returns `usage=None` (`bedrock_client.py:330`). Some Bedrock non-streaming responses may also lack usage depending on the model. Cost tracking must handle missing usage **explicitly** (warn, not silently zero).

### A3: Fallback pricing table becomes estimation-only after migration

**Current state:** `FALLBACK_MODEL_PRICING` is used by both pre-estimation and post-call fallback paths.
**End state (R4):** Renamed to `ESTIMATION_MODEL_PRICING`, used by pre-estimation only. Post-call tracking uses litellm exclusively.

### A4: A cost of 0.0 from a real LLM call is a bug

No production LLM API call is free. If cost calculates to 0.0 with positive token counts, something went wrong.

### A5: Mock mode is not production

In mock mode (`TRAIGENT_MOCK_LLM=true`), cost=0.0 is correct and expected. Silent failure handling for mock mode is fine.

### A6: Two heuristics for the same operation means neither is right

If we need char-to-token estimation, there should be one function, one ratio, clearly labeled as approximate and used only for pre-estimation.

---

## Success Criteria

| # | Criterion | Measurable |
|---|-----------|------------|
| S1 | Zero silent undercount in post-call cost tracking paths | **Structural analysis** (not just grep `return 0.0`): verify every except block in cost paths either re-raises, logs warning, or returns a sentinel that callers check. Includes default-zero-state flows and heuristic undercounts. |
| S2 | Single public `cost_from_tokens()` entry point for all integrations | All integration cost calls trace to one public function |
| S3 | No heuristics in post-call paths | All three post-call heuristics removed: `len // 3`, `* 0.25`, `(in*1.0+out*3.0)/1M`. Pre-estimation heuristic (`split() * 1.3`) documented and labeled. |
| S4 | Pre-estimation and post-tracking use different code paths and pricing dicts | `estimate_cost()` uses `ESTIMATION_MODEL_PRICING`; tracking uses litellm only via `cost_from_tokens(strict=True)` |
| S5 | `calculate_prompt_cost` / `calculate_completion_cost` marked `@deprecated` | Deprecation warnings fire when called |
| S6 | Every integration has a cost extraction test with real response format | Test per integration: OpenAI, Anthropic, LangChain, Bedrock (both modes) |
| S7 | Cost enforcement budget test: unknown model triggers visible warning, not silent 0 | Integration test with unknown model shows warning in logs |
| S8 | `make lint && make format` clean | No regressions |
| S9 | Existing unit tests pass (no behavioral regression for known models) | CI green |

---

## Out of Scope

- Changing the `CostEnforcer` / permit system (that's sound)
- Adding new LLM provider support
- Changing the `CostBreakdown` dataclass shape
- Modifying the SonarCloud pipeline

---

## Resolved Questions (Codex review 2026-02-20)

### Q1: Should `_calculate_from_tokens` raise for unknown models?

**Answer:** Yes for post-call tracking (strict mode). No for pre-estimation. Implement via a `strict` flag on the new public `cost_from_tokens()`. Post-call callers pass `strict=True` (default); pre-estimation passes `strict=False`.

### Q2: Can litellm's `token_counter()` legitimately fail?

**Answer:** Yes — payload shape mismatches or unsupported model names can cause failures. For post-call tracking, the fallback should be an explicit warning/error, NOT char-based heuristics. The caller should see the failure.

### Q3: Should pre-estimation use the same pricing dict as tracking?

**Answer:** No. Rename `FALLBACK_MODEL_PRICING` to `ESTIMATION_MODEL_PRICING` and restrict to pre-estimation. Post-call tracking uses litellm exclusively. See R4 for migration path.

### Q4: Should metrics_tracker's nested CostCalculator be consolidated?

**Answer:** Yes. Consolidate into the canonical `cost_calculator.py` CostCalculator. Keep a temporary compatibility shim for legacy callers that will be removed with the deprecated text-based functions.

### Q5: Should handler.py:697 be removed or retained behind a flag?

**Answer (Codex R2):** Remove from post-call path entirely. It's called from `on_chat_model_end` (handler.py:409) — definitively post-call. When `cost_from_tokens(strict=True)` can't price a model, the handler should log a warning and return `cost=0.0` with a `cost_method="unknown_model"` signal, NOT silently undercount with hardcoded rates.

### Q6: Should the doc define explicit migration phases?

**Answer (Codex R2):** Yes. R4 now defines Phase 1 (this branch: rename + split paths) and Phase 2 (future: auto-generate estimation pricing from litellm data).
