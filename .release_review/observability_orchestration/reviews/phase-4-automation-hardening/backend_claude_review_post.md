I now have a complete picture. Here's the review:

---

## Phase 4 Smoke Test Review — Prioritized Findings

### CRITICAL — None found

The ingest → overview → correlations → broken-examples → evidence pipeline is exercised end-to-end. No data-loss or security bypass issues.

---

### HIGH

**H1. `_enable_model_queries_for_smoke` is a fragile ORM hack** (`test_observability_smoke.py:23-27`)

Manually reassigning `Model.query = db.session.query(Model)` replaces the Flask-SQLAlchemy scoped query property with a non-scoped, instantly-bound query object. This works *only* if no `db.session.remove()` or session rollback occurs between assignment and use. If the test runner or conftest calls `db.session.remove()` (which the `simple_app` fixture does on teardown), any subsequent use of `ConfigurationRun.query` in a different test will get a detached-session error. Today this is benign because there's a single test function using it, but adding a second smoke test that calls `_enable_model_queries_for_smoke()` will likely trigger `DetachedInstanceError`.

*Recommendation:* Replace with explicit `db.session.query(Model)` calls inside the service (or pass a session), or configure the test app to not disable `.query` in the first place.

---

**H2. Smoke test bypasses the HTTP route for 4 of 5 endpoints** (`test_observability_smoke.py:124-187`)

Only the ingest step (`POST /api/observability/traces/ingest`) goes through the Flask test client. The remaining four calls (overview, correlations, broken-examples, evidence) invoke `ObservabilityService.*` class methods directly inside `app_context()`. This means the smoke test does **not** validate:
- Route registration / URL correctness for `GET /runs/{run_id}`, `/correlations`, `/broken-examples`, `/evidence`
- The `_resolve_tenant_id()` middleware on those read endpoints
- HTTP status codes, response envelope structure, or error serialization
- Auth decorator (`@require_resource_read_access`) on read paths

The stated goal is "smoke flow coverage for ingest → overview → correlations → broken examples → evidence," but the coverage is half service-layer, half HTTP-layer.

*Recommendation:* Drive all five steps through `simple_client.get(...)` so the test actually exercises routes, tenant resolution, and response serialization.

---

### MEDIUM

**M1. Tenant isolation test hits `GET /runs/<run_id>` but not the ingest endpoint** (`test_observability_smoke.py:190-211`)

`test_observability_flow_rejects_mismatched_tenant_header` only tests the read path. A mismatched `X-Tenant-Id` on the **write** path (`POST .../ingest`) is equally important but not exercised.

---

**M2. `min_samples=2` masks default-value regressions** (`test_observability_smoke.py:138,152,165`)

The smoke test passes `min_samples=2` everywhere. The service defaults are `5` (correlations) and `3` (broken-examples). If a future change breaks the default clamping logic (e.g., `max(2, ...)` → `max(5, ...)`), the test would still pass because it explicitly overrides the default. Consider adding at least one call that relies on the default `min_samples` so regressions in the default path are caught.

---

**M3. Trial data is minimal — only 4 trials × 2 examples** (`test_observability_smoke.py:50-55`)

Correlations and broken-example detection use statistical thresholds. With only 4 data points the test asserts `total_items > 0` but can't verify confidence levels, sorting order, or segment classification in any meaningful way. A single rounding change in `_summarize_broken_example` could flip the result from 1 item to 0 items without the test catching the cause.

*Recommendation:* Add a targeted assertion on `confidence` or `average_delta` value to pin the expected statistical output.

---

**M4. Shell runner lacks `-m` marker filter and exit-code surfacing** (`run_observability_smoke.sh`)

The script runs the entire test file with `pytest -q`. It doesn't filter by `@pytest.mark.smoke`, so if someone adds a non-smoke test to the same file it will run. It also doesn't pass `--tb=short` or `--junitxml`, limiting CI usefulness.

---

### LOW

**L1. No negative-path test for ingest validation** — e.g., missing `trace_id`, invalid `idempotency_key` format, payload >5MB. These are covered elsewhere in the unit suite but absent from the smoke layer.

**L2. `test-results.xml` is untracked in repo root** — likely a test artifact that should be gitignored.

**L3. Hardcoded datetime (`2026-01-01`)** in `_build_ingest_payload` — trivial, but will look wrong if tests run past that date in a log review. Consider using `datetime.now(timezone.utc)`.

---

## Verdict

**Phase 4 backend scope is acceptable with one caveat.** The ingest endpoint has proper HTTP-level smoke coverage, and the service-layer calls for overview/correlations/broken-examples/evidence prove the data pipeline is wired correctly end-to-end. Tenant isolation is tested on the read path.

However, the fact that 4 of 5 pipeline stages bypass the HTTP routes (**H2**) meaningfully reduces the "smoke" value — a misconfigured route or broken auth decorator would pass this suite silently. Promoting those four calls to `simple_client.get(...)` requests would close the gap and make this a true end-to-end smoke test. Until then, the suite is better characterized as a **service integration test with partial HTTP coverage**.
