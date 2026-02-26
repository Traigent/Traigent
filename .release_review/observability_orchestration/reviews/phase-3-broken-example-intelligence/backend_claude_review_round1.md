Now I have a comprehensive picture. Here's my review:

---

# Phase-3 Broken-Example Intelligence — Code Review

## CRITICAL

### C1. Unbounded full-table scan in `_collect_broken_example_observations` (performance / DoS)

**File:** `observability_service.py:993`

```python
config_runs = ConfigurationRun.query.filter_by(experiment_run_id=run_id).all()
```

This loads **all** configuration runs into memory with no limit. A run with tens of thousands of trials will OOM the worker. The subsequent span query at line 1097 also has no limit — it loads every span row for the run. Combined, these can be weaponized into a denial-of-service by any authenticated user.

**Fix:** Add `.limit(10_000)` (or a configurable cap) to both queries, and return an appropriate warning/error if truncated. Alternatively, paginate the evidence collection itself.

---

### C2. Evidence drill-down re-computes the entire run's observations

**File:** `observability_service.py:1588-1592`

`get_run_broken_example_evidence` calls `_collect_broken_example_observations` which scans **all** examples for the run, then discards everything except the one requested `example_id`. For a run with 5,000 examples × 100 trials, this is O(500K) work per drill-down request.

**Fix:** Add an `example_id` filter parameter to `_collect_broken_example_observations` so the span/measures queries can be narrowed, or at minimum filter `config_run.measures` to only the target example before building observations.

---

## HIGH

### H1. `_matches_metric_name` returns `True` on empty/missing metric names — silent data corruption

**File:** `observability_service.py:926-929`

```python
def _matches_metric_name(cls, metric_name: Any, metric_candidates: list[str]) -> bool:
    if not isinstance(metric_name, str) or not metric_name.strip():
        return True  # <-- includes outcomes with no metric_name
```

If a span outcome has `metric_name: null` or `metric_name: ""`, it's treated as matching **any** requested metric. This means unrelated or corrupt outcomes pollute the broken-example analysis silently.

**Fix:** Return `False` for empty/missing metric names so only explicitly matching outcomes are included. If you want a lenient fallback, at minimum log a warning.

---

### H2. `failure_classification` from untrusted input used without allowlist validation

**File:** `observability_service.py:1077-1085` and `1146-1154`

```python
failure_classification = (
    measure.get("failure_classification")
    if isinstance(measure.get("failure_classification"), str)
    else cls._derive_failure_classification(...)
)
```

Any arbitrary string from the client payload is accepted as `failure_classification` (e.g., `"<script>alert(1)</script>"`). While the API returns JSON, this leaks into `dominant_failure_classification` and `failure_classification_counts` keys in the response, which frontends might render unsafely.

**Fix:** Validate against a known allowlist:
```python
VALID_CLASSIFICATIONS = {"severe_regression", "regression", "below_trial_mean", "improved", "stable", "unknown"}
classification = measure.get("failure_classification", "")
if classification not in VALID_CLASSIFICATIONS:
    classification = cls._derive_failure_classification(...)
```

---

### H3. Sorting is non-deterministic when `average_delta` values are equal

**File:** `observability_service.py:1511-1536`

The sort for `direction == "all"` falls through to the `else` branch (negative sort), which may not be the desired behavior for mixed-direction results. More importantly, the `segment_rank.get(...)` returns `3` for any unrecognized segment, meaning custom/future segments will silently cluster at the end without warning.

**Fix:** Handle `direction == "all"` explicitly with a universal sort (e.g., by `abs(average_delta)` descending). Log unrecognized segments.

---

### H4. `_source_priority` internal field leaks into API response

**File:** `observability_service.py:1038`

The `_source_priority` key is added to observation dicts but never stripped before they flow into the evidence response. While `BrokenExampleEvidencePointDTO.to_dict()` only serializes known fields, the raw `observations` list is used in `_summarize_broken_example` which builds `failure_classification_counts` from the full dict. The `_source_priority` key is benign but represents an information leak.

**Fix:** Strip `_source_priority` after observations are finalized, or use a separate data structure for it.

---

## MEDIUM

### M1. `segment_counts` / `direction_counts` computed from trimmed results, not full results

**File:** `observability_service.py:1538-1543`

```python
trimmed = all_items[:safe_max_items]
segment_counts: dict[str, int] = defaultdict(int)
for item in trimmed:
    segment_counts[...] += 1
```

The counts only reflect the returned page, not the full dataset. A client requesting `max_items=5` from 200 results gets misleading facet counts. This makes the faceted filtering UX unreliable.

**Fix:** Compute `segment_counts` and `direction_counts` from `all_items` before trimming:
```python
for item in all_items:
    segment_counts[...] += 1
    direction_counts[...] += 1
trimmed = all_items[:safe_max_items]
```

---

### M2. DTO `items` field typed as `list[dict[str, Any]]` instead of `list[BrokenExampleInsightDTO]`

**File:** `observability_broken_examples.py:75`

`BrokenExampleListResponseDTO.items` accepts pre-serialized dicts rather than typed DTOs. This defeats the purpose of having frozen dataclasses — the DTO can't validate its own contents. Same issue for `BrokenExampleEvidenceResponseDTO.evidence` at line 109.

**Fix:** Type as `list[BrokenExampleInsightDTO]` and call `to_dict()` inside the response DTO's `to_dict()`.

---

### M3. No test for the `direction="all"` and `segment="all"` filter paths

**File:** `test_observability_correlation_service.py`

The test suite only covers `direction="negative"` and `segment="severe"`. The `"all"` path, `"positive"`, `"neutral"`, and no-segment paths are untested. The sort-order branching at lines 1511-1536 is also uncovered for `positive` and `neutral`.

**Fix:** Add parameterized tests covering each direction/segment combination.

---

### M4. Evidence route uses string heuristic for 404 detection

**File:** `observability_routes.py:297-299`

```python
error_text = str(exc)
status_code = 404 if "not found" in error_text.lower() else 400
```

This is fragile — if the error message wording changes (e.g., "not found" → "does not exist"), the route silently degrades to 400.

**Fix:** Use a custom exception class (e.g., `NotFoundException(ValueError)`) instead of string matching.

---

### M5. Route-level `max_items` cap is 500 but service-level default is 25

**File:** `observability_routes.py:248` vs `observability_service.py:961-967`

The route clamps `max_items` to `max(1, min(max_items, 500))`, then the service's `_normalize_broken_example_filters` also clamps to 500. This is fine but inconsistent with the correlations route which caps at 200. Documenting or unifying the caps would prevent future confusion.

---

## LOW

### L1. `_non_admin_headers` in route tests constructs JWT manually

**File:** `test_observability_v2_routes.py:15-39`

The test helper manually constructs JWTs using `os.getenv("JWT_SECRET_KEY", ...)`. If the secret changes or token structure evolves, every test file with this pattern breaks independently. Consider a shared test fixture.

---

### L2. `SCHEMA_VERSION = "1.0"` is a module-level constant with no versioning strategy

**File:** `observability_broken_examples.py:8`

The schema version is hardcoded but there's no mechanism to negotiate or detect version mismatches between client and server. This is fine for now but should be documented as a contract.

---

### L3. `_coerce_float` called redundantly in evidence serialization

**File:** `observability_service.py:1623-1631`

```python
round(float(row["metric_value"]), 6)
if cls._coerce_float(row.get("metric_value")) is not None
```

`_coerce_float` parses the value, discards it, then `float()` re-parses it. Minor waste — use the coerced result directly.

---

### Summary

| Severity | Count | Key Theme |
|----------|-------|-----------|
| CRITICAL | 2 | Unbounded queries, O(N) drill-down |
| HIGH | 4 | Silent data inclusion, XSS-adjacent field injection, non-deterministic sort, internal field leak |
| MEDIUM | 5 | Wrong facet counts, weak typing, missing test paths, fragile 404 detection |
| LOW | 3 | Test maintainability, schema versioning, redundant parsing |

The most impactful quick wins are **C1** (add query limits), **H1** (fix the metric name match), and **M1** (compute facet counts before trimming). Want me to implement fixes for any of these?
