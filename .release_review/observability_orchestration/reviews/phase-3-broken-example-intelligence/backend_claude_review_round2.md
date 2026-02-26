I've now read all five files completely. Here's my review focused on CRITICAL/HIGH issues only:

---

## Phase-3 Broken-Example Intelligence: Re-Review (CRITICAL/HIGH Only)

### Verdict: **No CRITICAL issues. Two HIGH observations.**

---

### HIGH 1: Unbounded in-memory aggregation in `_collect_broken_example_observations`

**File:** `src/services/observability_service.py:1027-1161`

The method loads **all** `ConfigurationRun` rows and **all** `TraceSpanV2` rows for a run into Python memory before filtering/aggregating. The row caps (`MAX_BROKEN_EXAMPLE_TRIAL_ROWS=10,000` and `MAX_BROKEN_EXAMPLE_SPAN_ROWS=50,000`) prevent unbounded growth, but 50k span rows each containing a `span_metadata` JSON blob could still consume significant memory under load (easily hundreds of MB per request depending on metadata size).

The span query at line 1142 fetches full `span_metadata` for every span in the run, but only a fraction will contain `lineage.example_outcomes`. This pulls far more data from the DB than needed.

**Recommendation:** Add a filter condition to the span query (e.g., `TraceSpanV2.span_metadata.isnot(None)`) or, better, use a JSON path filter if supported by the DB dialect. At minimum, consider whether `span_metadata` content can be narrowed via a column projection rather than fetching the entire blob.

---

### HIGH 2: `_normalize_run_id` performs two DB lookups unconditionally on every broken-example call

**File:** `src/services/observability_service.py:159-174`

`_normalize_run_id` does `_safe_session_get(ExperimentRun, ...)` and if not found, `_safe_session_get(ConfigurationRun, ...)`. Then the callers (`list_run_broken_examples` at line 1512, `get_run_broken_example_evidence` at line 1628) **immediately** call `_collect_broken_example_observations` which itself does `_visible_trial_ids_for_tenant` (another DB query) and then loads all `ConfigurationRun` rows. The original `_normalize_run_id` lookups are not reused.

This is not a correctness bug, but on a hot endpoint it adds 1-2 unnecessary DB round-trips per request. Not critical, but worth noting for performance-sensitive deployments.

---

### Items that look **good** (previously raised concerns now resolved):

1. **Input validation** — `_normalize_broken_example_filters` properly validates `segment` and `direction` with allowlists and raises `ValueError` on invalid input. The route layer at lines 227-302 correctly catches `ValueError` and returns 400/404.

2. **DTO layer** — `BrokenExampleInsightDTO`, `BrokenExampleEvidencePointDTO`, `BrokenExampleListResponseDTO`, `BrokenExampleEvidenceResponseDTO` are all frozen dataclasses with `to_dict()` serialization. Clean contract.

3. **Tenant isolation** — Broken-example queries correctly gate on `_visible_trial_ids_for_tenant`, which filters by `TraceSpanV2.tenant_id`. The route layer calls `_resolve_tenant_id()` with mismatch audit logging.

4. **Idempotent observation merging** — `_upsert_observation` with `source_priority` (measures=1, span metadata=2) correctly deduplicates per `(trial_id, example_id)` pair and prefers span-level evidence.

5. **Test coverage** — Tests cover the happy path for listing, evidence drill-down, direction filtering, sort order, and invalid direction rejection. Route tests verify parameter passthrough and 404 on "not found".

6. **Failure classification** — Derivation logic at `_derive_failure_classification` uses clear, documented thresholds. Normalization validates against `ALLOWED_EXAMPLE_FAILURE_CLASSIFICATIONS`.

7. **Response stability** — Schema version `1.0` included in all response DTOs.

---

**Summary:** The implementation is solid. The two HIGH items are performance concerns, not correctness bugs. No CRITICAL issues found.
