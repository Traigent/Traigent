# Cost Architecture Validation — Implementation Plan

**Branch:** `feat/cost-architecture-validation`
**Depends on:** `docs/architecture/cost-architecture-validation.md` (objectives)
**Estimated scope:** ~15 files modified, ~70 test assertions updated

---

## Implementation Order

Changes are ordered so that **each step is independently shippable** and the test suite passes after every step. No step silently changes behavior — each either adds new code or explicitly migrates callers.

---

### Step 1: Add `cost_from_tokens()` public function (R2)

**What:** Create the new canonical entry point alongside the existing code. Nothing calls it yet.

**File:** `traigent/utils/cost_calculator.py`

**Changes:**
```python
def cost_from_tokens(
    input_tokens: int,
    output_tokens: int,
    model: str,
    *,
    strict: bool = True,
) -> tuple[float, float]:
    """Canonical cost calculation from token counts.

    Args:
        input_tokens: Number of input tokens (0 is valid for output-only)
        output_tokens: Number of output tokens (0 is valid for input-only)
        model: Model identifier (with or without provider prefix)
        strict: If True (default), raise UnknownModelError for unpriced models.
                If False, log warning and return (0.0, 0.0).

    Returns:
        Tuple of (input_cost_usd, output_cost_usd).

    Raises:
        UnknownModelError: When strict=True and model has no pricing.
    """
```

**Implementation:**
1. Try `litellm.cost_per_token(model, input_tokens, output_tokens)` — return if positive
2. Try normalized model name in `litellm.model_cost` dict — return if found
3. If `strict=True`: raise `UnknownModelError`
4. If `strict=False`: log warning, return `(0.0, 0.0)`

No fallback pricing table. No heuristics. No silent zeros in strict mode.

**Tests:** `tests/unit/utils/test_cost_from_tokens.py` (new file)

| Scenario | Input | Expected |
|----------|-------|----------|
| Known model, both tokens positive | `("gpt-4o", 100, 50)` | Positive (input, output) costs |
| Known model, input only | `("gpt-4o", 100, 0)` | `(positive, 0.0)` — NOT skipped |
| Known model, output only | `("gpt-4o", 0, 50)` | `(0.0, positive)` — NOT skipped |
| Known model, zero both | `("gpt-4o", 0, 0)` | `(0.0, 0.0)` — legitimate zero |
| Unknown model, strict=True | `("nonexistent-model", 100, 50)` | Raises `UnknownModelError` |
| Unknown model, strict=False | `("nonexistent-model", 100, 50, strict=False)` | `(0.0, 0.0)` + warning logged |
| Provider-prefixed model | `("openai/gpt-4o", 100, 50)` | Same as unprefixed |
| litellm unavailable, strict=True | Mock `LITELLM_AVAILABLE=False` | Raises `RuntimeError` |
| litellm unavailable, strict=False | Mock `LITELLM_AVAILABLE=False` | `(0.0, 0.0)` + warning |
| Negative tokens | `("gpt-4o", -1, 50)` | Raises `ValueError` |

**Edge cases:**
- Bedrock streaming sends `input_tokens=500, output_tokens=0` — must NOT skip (fixes silent failure #6)
- Model returns `cost_per_token=(0.0, 0.0)` but is known to litellm — return `(0.0, 0.0)` (legitimate free tier)

**Exit criterion:** New function exists with full test coverage. No existing code calls it yet.

---

### Step 2: Rename `FALLBACK_MODEL_PRICING` -> `ESTIMATION_MODEL_PRICING` (R4 Phase 1)

**What:** Rename the dict and update all references. Pure rename — no behavior change.

**Files:**
- `traigent/utils/cost_calculator.py` — rename dict + constant
- `traigent/hooks/validator.py` — update import
- `traigent/analytics/intelligence.py` — update import (if any)
- `tests/unit/utils/test_cost_calculator.py` — update imports
- `tests/unit/utils/test_cost_calculator_pricing.py` — update imports
- `tests/unit/architecture/test_pricing_consistency.py` — update imports
- `tools/lint_pricing_consistency.py` — update AST scanner

**Tests:** All existing tests pass with renamed import. No new tests needed.

**Exit criterion:** `grep -r "FALLBACK_MODEL_PRICING" traigent/` returns zero hits (only `ESTIMATION_MODEL_PRICING`).

---

### Step 3: Wire `CostCalculator._calculate_from_tokens()` through `cost_from_tokens()` (R2)

**What:** The private method becomes a thin wrapper around the public function.

**File:** `traigent/utils/cost_calculator.py`

**Changes:**
```python
def _calculate_from_tokens(self, input_tokens, output_tokens, model):
    try:
        return cost_from_tokens(input_tokens, output_tokens, model, strict=True)
    except UnknownModelError:
        # Fall back to estimation pricing for backward compat (to be removed in Step 6)
        return _estimation_cost_from_tokens(model, input_tokens, output_tokens)
    except Exception:
        if self.logger:
            self.logger.warning("Cost calculation failed for model %r", model)
        return 0.0, 0.0
```

**Also rename:** `_fallback_cost_from_tokens` -> `_estimation_cost_from_tokens` (clarity).

**Tests:** Existing `TestCalculateFromTokens` tests pass. The backward-compat fallback keeps current behavior.

**Exit criterion:** `platforms.py._calculate_cost` (which calls `_calculate_from_tokens`) produces identical results.

---

### Step 4: Wire `handler._estimate_cost()` through `cost_from_tokens()` (R2, R3)

**What:** Replace the handler's custom cost path with the canonical function. Remove `_fallback_cost_estimate` and its hardcoded `(in*1.0 + out*3.0)/1M` rates.

**File:** `traigent/integrations/langchain/handler.py`

**Changes:**
```python
def _estimate_cost(self, model, input_tokens, output_tokens):
    try:
        input_cost, output_cost = cost_from_tokens(
            input_tokens, output_tokens, model, strict=False
        )
        return float(input_cost + output_cost)
    except Exception:
        logger.warning(
            "Cost calculation failed for model %r with %d tokens",
            model, input_tokens + output_tokens,
        )
        return 0.0
```

Delete `_fallback_cost_estimate()` method entirely.

**Tests to update:** `tests/unit/integrations/langchain/test_langchain_handler.py`
- `TestCostEstimation` (10 tests, lines 796-880) — all test the old `_fallback_cost_estimate`
- Tests for known models (gpt-4o, claude) should still return positive costs (litellm handles them)
- `test_fallback_cost_estimate_unknown_model` (line 842) — now returns 0.0 + warning instead of hardcoded `$1/$3` per 1M
- `test_estimate_cost_exception_handling` (line 870) — now returns 0.0 + warning log instead of silent fallback

| Scenario | Before | After |
|----------|--------|-------|
| Known model (gpt-4o) | Positive cost via litellm | Same — positive cost via `cost_from_tokens` |
| Unknown model | `(100*1.0 + 50*3.0)/1M = 0.00025` | `0.0` + warning log |
| Exception | Silent fallback to hardcoded | `0.0` + warning log |

**Exit criterion:** Handler uses `cost_from_tokens`. No hardcoded pricing rates in handler.py.

---

### Step 5: Fix partial-token-data gap in `CostCalculator.calculate_cost()` (R1)

**What:** Remove the `input_tokens > 0 AND output_tokens > 0` gate that silently skips token-based costing.

**File:** `traigent/utils/cost_calculator.py:706-710`

**Before:**
```python
elif input_tokens and output_tokens and input_tokens > 0 and output_tokens > 0:
```

**After:**
```python
elif input_tokens is not None or output_tokens is not None:
    actual_input = input_tokens if input_tokens and input_tokens > 0 else 0
    actual_output = output_tokens if output_tokens and output_tokens > 0 else 0
```

**Tests:**

| Scenario | Before | After |
|----------|--------|-------|
| `input=500, output=0` | Skipped — cost stays 0 | Enters token path — calculates input cost |
| `input=0, output=100` | Skipped — cost stays 0 | Enters token path — calculates output cost |
| `input=None, output=None` | Falls to method 3 | Falls to method 3 (no change) |
| `input=500, output=50` | Works | Works (no change) |

**Exit criterion:** Partial token data produces partial cost (not zero).

---

### Step 6: Remove post-call heuristics (R3)

**What:** Delete the three post-call heuristics. This is the breaking change.

**Files and removals:**
1. `cost_calculator.py:268` — delete `len(text) // 3` block and the `tokens if tokens > 0 else 10` default
2. `metrics_tracker.py:1075` — delete `length * 0.25` token estimation
3. `handler.py:697` — already removed in Step 4

**Behavior change:** When litellm's `token_counter()` fails, `_try_litellm_prompt_cost` propagates the exception instead of falling through to character estimation. The calling `calculate_prompt_cost` logs it and falls to fallback pricing (which uses estimation pricing, not heuristics).

**Tests to update:**
- `test_cost_calculator.py` — tests that relied on `len//3` estimation now see different behavior when litellm token counter is mocked to fail
- `test_metrics_tracker.py` — privacy mode token estimation changes

**Exit criterion:** `grep -rn "len.*// *3\|length.*\* *0.25" traigent/` returns zero hits in cost/metrics code.

---

### Step 7: Make `_calculate_from_tokens` strict (remove estimation fallback) (R1)

**What:** The backward-compat fallback added in Step 3 is removed. Unknown models now raise in post-call tracking.

**File:** `traigent/utils/cost_calculator.py`

**Changes:**
```python
def _calculate_from_tokens(self, input_tokens, output_tokens, model):
    try:
        return cost_from_tokens(input_tokens, output_tokens, model, strict=True)
    except UnknownModelError:
        logger.warning(
            "Unknown model %r in post-call tracking — cost will be 0. "
            "Add model to litellm or report a bug.",
            model,
        )
        return 0.0, 0.0
    except Exception:
        if self.logger:
            self.logger.warning("Cost calculation failed for model %r", model)
        return 0.0, 0.0
```

Note: Still returns (0.0, 0.0) — but now with a **visible warning**. This satisfies R1 (no silent undercount) without breaking the optimization loop.

**Tests to update:**
- `test_platforms.py:test_calculate_cost_zero_cost_warning` — now expects warning log
- `test_cost_calculator.py:858-859` — unknown model fallback now warns

**Exit criterion:** Every except block in `_calculate_from_tokens` has a visible log message.

---

### Step 8: Deprecate text-based cost functions (R5)

**What:** Add `@deprecated` to `calculate_prompt_cost` and `calculate_completion_cost`.

**File:** `traigent/utils/cost_calculator.py`

**Changes:**
```python
import warnings

def calculate_prompt_cost(prompt, model):
    warnings.warn(
        "calculate_prompt_cost() is deprecated. Use cost_from_tokens() with "
        "pre-computed token counts instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # ... existing implementation
```

**Tests:**
```python
def test_calculate_prompt_cost_emits_deprecation():
    with pytest.warns(DeprecationWarning, match="cost_from_tokens"):
        calculate_prompt_cost("hello", "gpt-4o")
```

**Exit criterion:** Calling deprecated functions emits `DeprecationWarning`.

---

### Step 9: Make litellm a hard requirement for cost tracking (R6)

**What:** `cost_from_tokens()` raises `RuntimeError` if litellm is not installed (when `strict=True`). `CostCalculator.__init__` raises if litellm unavailable.

**Tests to update:**
- `test_cost_calculator.py:393` — `litellm unavailable` now raises instead of returning empty CostBreakdown
- `test_cost_calculator.py:104, 209, 385, 599, 644` — tests mocking `LITELLM_AVAILABLE=False` need updating

| Test | Before | After |
|------|--------|-------|
| `test_litellm_unavailable_returns_zero` | `assert total_cost == 0.0` | `pytest.raises(RuntimeError)` |
| `test_cost_calculator_no_litellm` | Returns empty breakdown | Raises at init |

**Exit criterion:** `LITELLM_AVAILABLE=False` + `strict=True` always raises.

---

### Step 10: Consolidate metrics_tracker CostCalculator (R8)

**What:** Replace the nested CostCalculator in `metrics_tracker.py` with imports from `cost_calculator.py`.

**File:** `traigent/evaluators/metrics_tracker.py`

**Changes:**
- Delete nested `CostCalculator` class (~80 lines)
- Import and use `cost_from_tokens` for the unified path
- Keep `_try_backward_compatible_cost_calculation` as a shim that calls the deprecated `calculate_prompt_cost` / `calculate_completion_cost` (these emit deprecation warnings now)

**Tests to update:**
- `test_metrics_tracker.py` — cost-related tests should work identically
- `test_litellm_integration.py` — monkeypatches to `metrics_tracker.calculate_prompt_cost` need path updates

**Exit criterion:** No `class CostCalculator` in metrics_tracker.py. All metrics tests pass.

---

### Step 11: Integration cost extraction tests (R7)

**What:** Add one test per integration verifying real response format -> correct cost.

**File:** `tests/unit/integrations/test_cost_extraction.py` (new)

| Test | Response Format | Expected |
|------|----------------|----------|
| `test_openai_response_cost` | `{"usage": {"prompt_tokens": 100, "completion_tokens": 50}}` | Positive cost matching litellm |
| `test_anthropic_response_cost` | `{"usage": {"input_tokens": 100, "output_tokens": 50}}` | Positive cost |
| `test_langchain_response_cost` | `LLMResult(llm_output={"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}})` | Positive cost |
| `test_bedrock_nonstreaming_cost` | `BedrockChatResponse(usage={"inputTokens": 100, "outputTokens": 50})` | Positive cost |
| `test_bedrock_streaming_no_usage` | `BedrockChatResponse(usage=None)` | Warning logged, cost=0.0 with signal |
| `test_unknown_model_warns` | `{"usage": {"prompt_tokens": 100, ...}, "model": "totally-fake-model"}` | Warning logged |

**Exit criterion:** All 6 integration tests pass.

---

## Test Impact Summary

### Tests That Will Break (by step)

| Step | Files Affected | Tests to Update | Nature of Change |
|------|----------------|-----------------|------------------|
| 2 | 5 test files | ~10 imports | Rename only |
| 4 | test_langchain_handler.py | 10 tests | Unknown model returns 0+warning instead of hardcoded rate |
| 5 | test_cost_calculator.py | 2-3 tests | Partial tokens now produce partial cost |
| 6 | test_cost_calculator.py, test_metrics_tracker.py | 5-8 tests | Heuristic removal |
| 7 | test_platforms.py, test_cost_calculator.py | 3-4 tests | Warning on unknown model |
| 9 | test_cost_calculator.py | 5-6 tests | litellm unavailable raises |
| 10 | test_metrics_tracker.py, test_litellm_integration.py | 5-8 tests | Import path changes |
| **Total** | | **~45 tests** | |

### Tests That Should NOT Break

- `test_pricing_consistency.py` — all 7 tests verify known models; those still work
- `test_cost_tracking.py` — CostTracker/BillingTier is out of scope
- `test_cost_enforcement.py` — CostEnforcer is out of scope
- Any test using mock mode (`TRAIGENT_MOCK_LLM=true`) — A5 exempts mock mode

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking the optimization loop with strict mode | Step 7 still returns (0.0, 0.0) — but with a warning. The loop doesn't crash. |
| Unknown model in production | `strict=False` path exists for pre-estimation. Post-call uses strict but catches `UnknownModelError` and warns. |
| Bedrock streaming usage=None | Step 11 adds explicit test. `cost_from_tokens(0, 0, model)` returns `(0.0, 0.0)` — legitimate zero. |
| 70+ tests assert `cost == 0.0` | Most are in mock mode (A5 exempt) or test initialization (no change). ~45 actually need updating. |
| Regression in cost accuracy for known models | test_pricing_consistency.py validates cross-module consistency — runs after every step. |

---

## Verification After Each Step

After every step, run:
```bash
TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true python -m pytest tests/unit/utils/ tests/unit/architecture/ tests/unit/agents/test_platforms.py tests/unit/integrations/langchain/ tests/unit/evaluators/ -q --timeout=60
```

Full suite before PR:
```bash
TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true python -m pytest tests/unit/ \
  -o "addopts=" \
  --ignore=tests/unit/bridges \
  --ignore=tests/unit/test_bridge_wrapper.py \
  --ignore=tests/unit/evaluators/test_js_evaluator.py \
  --ignore=tests/unit/evaluators/test_js_evaluator_budget.py \
  --ignore=tests/unit/evaluators/test_js_evaluator_stop_conditions.py \
  --ignore=tests/unit/core/test_orchestrator.py \
  --ignore=tests/unit/evaluators/test_litellm_integration.py \
  --cov=traigent --cov-report=xml:coverage.xml -q --timeout=60
```
