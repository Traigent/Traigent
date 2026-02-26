Here's my grounded analysis for Phase 4 automation hardening:

---

## Phase 4 Automation Hardening — Review Findings

### 1. Top Correctness Risks

**CRITICAL — C1: Duplicate `example_outcomes` cardinality breach**
`trial_lifecycle.py:771-777` — The dedup guard on `seen_example_ids` protects `example_ids` but `example_outcomes` appends *before* the dedup check. If an example appears under multiple measures, outcomes list grows unbounded vs the `example_ids` list. Smoke must assert `len(example_ids) == len(example_outcomes)` per span.

**CRITICAL — C2: `failure_classification` enum expansion unvalidated cross-repo**
Phase 3 added `severe_regression`, `regression`, `improved` to `failure_classification_detail`, but the legacy `failure_classification` field still maps only to `unknown | below_trial_mean | stable`. Backend may reject or silently drop unknown values. Smoke must assert all emitted values are in the known allowlist.

**HIGH — H1: Span `trace_linkage` referential integrity**
Each `example_outcome.trace_linkage.trace_id` must equal the parent span's `trace_id`, and `configuration_run_id` must match the span's own `configuration_run_id`. A mismatch silently breaks drill-down in the dashboard. Smoke must validate this linkage.

**HIGH — H2: Timestamp format divergence**
`SpanPayload.__post_init__` normalizes timestamps to ISO 8601 with `+00:00` suffix (no trailing `Z`). Backend may parse `Z` differently. Smoke must assert no timestamp ends with `Z`.

**MEDIUM — M1: `segment` semantics for positive deltas**
`segment` returns `"mild"` for improvements, which is misleading. Not a data corruption risk but creates confusing dashboard UI. Gate should warn but not block.

---

### 2. Deterministic, Low-Flake Implementation Recommendations

**Use fixed-seed mock mode exclusively.** The existing `conftest.py` already forces `TRAIGENT_MOCK_LLM=true` + `TRAIGENT_OFFLINE_MODE=true`. Smoke tests should never hit real APIs.

**Pin scenario inputs for reproducibility:**
```python
# Smoke test fixture — deterministic 3-example dataset
SMOKE_DATASET = [
    {"text": "positive review", "label": "positive"},
    {"text": "negative review", "label": "negative"},
    {"text": "neutral review",  "label": "neutral"},
]
SMOKE_CONFIG_SPACE = {"temperature": [0.0]}  # Single config = deterministic trial count
SMOKE_MAX_TRIALS = 1                          # Exactly 1 trial = deterministic span count
```

**Avoid time-dependent assertions.** Never assert on `duration` or wall-clock values. Assert only on structural invariants (field presence, type, enum membership, referential integrity).

**Use `pytest.mark.smoke` marker.** Register in `pyproject.toml` so CI can run `pytest -m smoke` independently. Keep smoke suite under 10 seconds total.

**Isolate from test_orchestrator.py hang risk.** The known hang issue affects the full orchestrator. Smoke should call `TrialLifecycle.run_trial()` or `_build_example_outcomes()` directly, never the full `optimize()` loop, to stay deterministic and fast.

---

### 3. Fail-Fast Gate Script Requirements

**Three-stage gate, each stage blocks the next:**

| Stage | Scope | Timeout | Fail Action |
|-------|-------|---------|-------------|
| **1. Smoke** | `pytest -m smoke` (payload invariants) | 30s | EXIT 1 immediately |
| **2. Targeted** | `tests/unit/core/test_trial_lifecycle.py tests/unit/integrations/observability/` | 120s | EXIT 1, emit failure report |
| **3. Broad** | `pytest tests/unit/ --ignore=...` (known hangs) | 300s | EXIT 1, emit full report |

**Gate script contract:**
```bash
#!/usr/bin/env bash
set -euo pipefail
# Env guaranteed
export TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true

REPORT_DIR=".release_review/observability_orchestration/verification"
REPORT="$REPORT_DIR/phase-4-automation-hardening_$(date -u +%Y%m%d_%H%M%S).md"

# Stage 1: Smoke (fail-fast, no coverage overhead)
pytest -m smoke --timeout=10 -x --tb=short || { echo "SMOKE FAILED" > "$REPORT"; exit 1; }

# Stage 2: Targeted observability suites
pytest tests/unit/core/test_trial_lifecycle.py \
       tests/unit/integrations/observability/ \
       --timeout=30 -x --tb=short \
       --cov=traigent.core.trial_lifecycle \
       --cov=traigent.integrations.observability \
       --cov-report=term-missing || { emit_report "TARGETED FAILED"; exit 1; }

# Stage 3: Broad unit suite
pytest tests/unit/ --timeout=60 \
       --ignore=tests/unit/core/test_orchestrator.py \
       --ignore=tests/unit/evaluators/test_litellm_integration.py \
       --cov=traigent --cov-report=xml:coverage.xml || { emit_report "BROAD FAILED"; exit 1; }

emit_report "ALL PASSED"  # Write markdown verification report
```

**Markdown report must include:**
- Gate timestamp (UTC)
- Git SHA + branch
- Per-stage pass/fail + duration
- Coverage on new code (from `--cov-report=term-missing`)
- Failing test names if any (from `--tb=short` output)
- List of known-skipped files (for audit trail)

---

### 4. Priority-Tagged Findings Summary

| # | Priority | Finding | Action |
|---|----------|---------|--------|
| C1 | **CRITICAL** | `example_outcomes` duplicates bypass dedup guard | Fix in `_build_example_outcomes` before smoke; smoke asserts `len(ids)==len(outcomes)` |
| C2 | **CRITICAL** | `failure_classification` enum expansion unvalidated | Smoke asserts emitted values ∈ known allowlist; file backend issue for validation |
| H1 | **HIGH** | `trace_linkage` referential integrity untested | Smoke asserts `outcome.trace_linkage.trace_id == span.trace_id` for all outcomes |
| H2 | **HIGH** | ISO 8601 timestamp format (`+00:00` not `Z`) | Smoke asserts no `Z`-terminated timestamps in span payloads |
| H3 | **HIGH** | Gate script must skip known-hang tests | Hardcode `--ignore` list matching `pyproject.toml` addopts |
| H4 | **HIGH** | `MeasuresDict` key validation (Python identifiers, max 50) | Smoke asserts metrics keys pass `^[a-zA-Z_]\w*$` and `len <= 50` |
| M1 | **MEDIUM** | `segment="mild"` for positive deltas is misleading | Warn in report; don't block gate |
| M2 | **MEDIUM** | Span `cost_usd` may be 0.0 in mock mode | Smoke should accept 0.0 but assert field exists and is float |
| M3 | **MEDIUM** | `confidence` semantics differ (delta-based vs sample-based) | Document in report; don't block gate |
| L1 | **LOW** | `_source_priority` internal field leaks to API | Backend-side fix; SDK gate can't catch this |
| L2 | **LOW** | Redaction false-positives on UUID trace IDs | Add regression test with UUID-format IDs |

**Recommended smoke assertion checklist (minimum viable):**

```python
@pytest.mark.smoke
def test_traced_payload_invariants(smoke_trial_span):
    span, outcomes = smoke_trial_span
    # C1: Cardinality
    assert len(span.metadata["lineage"]["example_ids"]) == len(span.metadata["lineage"]["example_outcomes"])
    # C2: Enum allowlist
    ALLOWED = {"unknown", "below_trial_mean", "stable", "severe_regression", "regression", "improved"}
    for o in outcomes:
        assert o["failure_classification_detail"] in ALLOWED
    # H1: Referential integrity
    for o in outcomes:
        assert o["trace_linkage"]["trace_id"] == span.trace_id
        assert o["trace_linkage"]["configuration_run_id"] == span.configuration_run_id
    # H2: Timestamp format
    assert not span.start_time.endswith("Z")
    # H4: Metrics keys
    for key in span.output_data.get("metrics", {}):
        assert re.match(r"^[a-zA-Z_]\w*$", key)
```

This gives you 5 structural invariants that catch the highest-impact bugs with zero flake risk, running in <1 second under mock mode.
