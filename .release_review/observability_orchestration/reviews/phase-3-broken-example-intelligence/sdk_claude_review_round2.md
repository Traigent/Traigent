Permission denied — I'll document the finding instead and continue the review.

---

**Finding 2 — HIGH: `_collect_workflow_span` is ~170 lines of dense inline logic (lines 660–914)**

This method violates separation of concerns. It does: metric extraction, example deduplication, delta computation, classification, confidence scoring, span construction, and collection — all inline. If any of the nested helper closures (`_as_float`, `_example_direction`, `_example_segment`, `_classify_failure`, `_confidence_from_delta`) need reuse or testing, they can't be reached. A bug in classification logic is invisible to unit tests unless the entire span-collection integration test is run.

**Severity:** HIGH (maintainability, testability)
**Remediation:** Extract the five inner functions to module-level (or a small helper class) and add targeted unit tests for the classification logic.

---

**Finding 3 — HIGH: No cap on `example_outcomes` payload size before truncation check (lines 767–836)**

The loop at lines 768–832 builds `example_outcomes` (a list of dicts with ~12 keys each) for *every* measure entry before truncating at line 836 (`example_outcomes = example_outcomes[:50]`). If `measures` contains thousands of entries (large datasets), this allocates a large list and many dicts only to discard them. In a parallel trial scenario this could cause memory pressure.

**Severity:** HIGH (performance/resource)
**Remediation:** Add an early break once 50 unique examples are collected:
```python
if len(seen_example_ids) >= 50:
    break
```

---

### Test File: `tests/unit/core/test_trial_lifecycle.py`

**Finding 4 — HIGH: No test coverage for `_collect_workflow_span` error/pruned paths**

`TestCollectWorkflowSpan` only tests `TrialStatus.COMPLETED`. The method has three status branches (lines 691–696: COMPLETED → COMPLETED, PRUNED → REJECTED, else → FAILED). Neither the PRUNED nor FAILED mapping is tested, meaning a regression in status mapping would go undetected.

**Severity:** HIGH (coverage gap on branch logic)
**Remediation:** Add tests for PRUNED and FAILED trial statuses verifying the span's `status` field.

---

**Finding 5 — HIGH: No test for `APIKeyError` re-raise path (line 425–428)**

`_execute_trial_with_tracing` has a dedicated `except APIKeyError: raise` block to fail-fast on credential errors. There is no test verifying this propagation. If someone accidentally catches it in a broader handler above, trials would silently continue without valid credentials.

**Severity:** HIGH (security-adjacent, fail-fast guarantee)
**Remediation:** Add a test similar to `TestCancelledErrorPropagation` that verifies `APIKeyError` propagates out of `run_trial`.

---

### Summary

| # | Severity | File | Line(s) | Finding |
|---|----------|------|---------|---------|
| 1 | **HIGH** | trial_lifecycle.py | 779 | Variable `metrics` shadowed inside loop in `_collect_workflow_span` |
| 2 | **HIGH** | trial_lifecycle.py | 660–914 | `_collect_workflow_span` is an untestable 170-line monolith with 5 inner closures |
| 3 | **HIGH** | trial_lifecycle.py | 767–836 | Unbounded list allocation before truncation; no early-exit at cap |
| 4 | **HIGH** | test_trial_lifecycle.py | 217–310 | Missing branch coverage for PRUNED/FAILED status in `_collect_workflow_span` |
| 5 | **HIGH** | test_trial_lifecycle.py | — | Missing test for `APIKeyError` fail-fast propagation |

No CRITICAL findings. The variable shadowing (#1) and missing cap (#3) are the most actionable immediate fixes. Want me to apply them?
