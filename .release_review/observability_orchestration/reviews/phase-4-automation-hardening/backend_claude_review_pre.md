Now I have a thorough understanding of both the observability code and test infrastructure. Here's my analysis:

---

## Phase 4 Smoke Suite: Analysis & Recommendations

### 1. Top Risks to Cover

**The end-to-end chain is: ingest → run overview → correlations → broken-examples → evidence**

| Risk | Why it matters |
|------|---------------|
| **Observation data loss between stages** | Broken-examples reads from two sources (`ConfigurationRun.measures[]` AND `TraceSpanV2.span_metadata.lineage`). If ingest silently drops metadata, downstream stages return empty. |
| **Idempotency key format coupling** | The key is `"{trace_id}:{configuration_run_id}:{span_id}"` — computed AND validated. A format mismatch silently rejects spans but returns 201 with `duplicates_skipped`. |
| **Tenant isolation leakage** | Every query filters by `tenant_id`, but `visible_trial_ids` (the pivot set for correlations + broken-examples) is built from a span query. If tenant filtering regresses, one tenant sees another's data. |
| **Empty-result vs 404 semantics** | `broken-examples` returns `[]` for no matches; `evidence` raises 404. A smoke test must verify both paths survive the same dataset. |
| **Attribution trigger timing** | Attribution computes only when `config_run.status` is terminal (`COMPLETED`/`FAILED`). If ingest happens before status update, run-overview returns empty attribution. |

---

### 2. Test Architecture Recommendations

**Create one file:** `tests/integration/test_observability_smoke.py`

Use the **full `app` + `client` + `init_database`** pattern (not `simple_client`) since this is an end-to-end flow test through real DB.

```
Fixture chain:
  app → client → init_database → seed_observability_data
```

**Recommended structure:**

```python
@pytest.mark.integration
class TestObservabilitySmokeE2E:
    """Smoke: ingest → overview → correlations → broken-examples → evidence."""

    @pytest.fixture(autouse=True)
    def seed_observability_data(self, client, init_database, auth_headers):
        """Create experiment → run → 3+ config_runs with measures,
        then ingest spans with lineage metadata."""
        # 1. Create Experiment + ExperimentRun + ConfigurationRuns via DB
        # 2. Set config_runs to COMPLETED status
        # 3. POST /api/observability/traces/ingest with spans
        #    carrying span_metadata.lineage.example_outcomes
        # 4. Store IDs for assertions
        ...

    def test_01_ingest_succeeds(self): ...
    def test_02_run_overview_has_attribution(self): ...
    def test_03_correlations_return_ranked(self): ...
    def test_04_broken_examples_ranked_severe_first(self): ...
    def test_05_evidence_returns_sorted_worst_first(self): ...
    def test_06_evidence_404_for_unknown_example(self): ...
    def test_07_tenant_isolation_blocks_cross_tenant(self): ...
```

**Why numbered tests:** These are ordered smoke checks — if ingest fails, downstream assertions are meaningless. Use `pytest-ordering` or simply prefix names so default alphabetical ordering matches the flow.

**Seed data requirements (minimum viable):**
- 1 experiment, 1 experiment_run, 3+ configuration_runs (COMPLETED)
- Each config_run has `measures` with `example_id` and a score metric
- Each config_run has `experiment_parameters` (for parameter correlations)
- Ingest 2+ spans per config_run, at least one with `span_metadata.lineage.example_outcomes`
- Include 1 "bad" example (`metric_delta <= -0.15`) and 1 "good" example for segment filtering
- Set tenant_id consistently via header and env var

---

### 3. Determinism Pitfalls & Mitigations

| Pitfall | Mitigation |
|---------|------------|
| **UUID collisions** | Use deterministic IDs: `f"smoke_exp_{i}"`, `f"smoke_run_{i}"`, `f"smoke_cfg_{i}"`. Avoid `uuid4()` in smoke fixtures. |
| **Timestamp ordering** | Use `datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)` for span `start_time`/`end_time`. Never use `datetime.now()` in test data — it makes cost-latency bucketing non-deterministic. |
| **Tenant env var coupling** | The route validates `X-Tenant-Id` against `TRAIGENT_TENANT_ID` env var. Set both in fixture: `monkeypatch.setenv("TRAIGENT_TENANT_ID", "smoke_tenant")` and `headers["X-Tenant-Id"] = "smoke_tenant"`. |
| **Idempotency key replay** | If the same test runs twice without DB teardown, spans are silently skipped. Function-scoped `app` fixture with `drop_all()` handles this, but verify `spans_ingested > 0` in assertions. |
| **Correlation `min_samples` threshold** | Default is 5; with only 3 config_runs, correlations return empty. Either seed 5+ trials or pass `?min_samples=2` in the query. |
| **Float comparison** | `metric_delta`, `correlation`, `confidence_score` are floats. Use `pytest.approx()` or assert ranges (`assert -0.35 < item["average_delta"] < -0.25`). |
| **DB session flush timing** | After `db.session.add()` + `db.session.commit()` for config_runs, the ingest endpoint opens its own session query. In SQLite `:memory:` with the same engine this works, but always commit before calling the API. |
| **`visible_trial_ids` gate** | Correlations and broken-examples only see trials that have `TraceSpanV2` rows for the tenant. If you seed config_runs but forget to ingest spans for some, those trials are invisible. Ingest spans for ALL config_runs. |

---

### 4. Priority-Tagged Findings

#### CRITICAL

**C1: No integration test verifies the full ingest→evidence chain.**
Existing tests are either route-level mocks or service-level mocks. No test actually writes spans to DB and then reads broken-examples/evidence from the same DB. A regression in `_collect_broken_example_observations` (the dual-source merge) would be undetected.

**C2: `visible_trial_ids` is the correctness gate for 3 endpoints.**
`correlations`, `broken-examples`, and `evidence` all call `_get_visible_trial_ids()` which queries `TraceSpanV2` by `tenant_id` + `configuration_run_id`. If tenant filtering is wrong here, all three endpoints leak data. Smoke must test cross-tenant query returns 0 items.

#### HIGH

**H1: Attribution compute is only triggered on terminal status.**
`ingest()` checks `config_run.status in ("COMPLETED", "FAILED")` before calling `compute_attributions()`. If the smoke test seeds config_runs as `RUNNING` and then ingests, the run-overview will have empty attribution. Test must verify the ordering: set status → ingest → overview shows attribution.

**H2: Broken-examples `_upsert_observation` priority logic is untested at integration level.**
The dual-source merge (measures priority 1, span metadata priority 2, span metadata wins when richer) is tested with mocks only. A real DB test should verify that span-sourced `failure_classification_detail` overrides measures-sourced classification.

**H3: `min_samples` default of 5 will silently empty results.**
If smoke seeds fewer than 5 trials, correlations return `total_items: 0`. This is correct behavior but misleading in a smoke test. Document the minimum seed count prominently.

#### MEDIUM

**M1: Cost-latency dashboard bucketing differs SQLite vs PostgreSQL.**
The service uses `strftime` for SQLite and `date_trunc` for PostgreSQL. Smoke tests run on SQLite — if the production path (PostgreSQL) regresses, smoke won't catch it. Acceptable for now but flag for future Postgres-backed CI.

**M2: Alert rule creation requires admin role.**
The existing route test verifies 403 for non-admin. Smoke suite should include alert creation as a bonus endpoint check (POST alert rule → GET alert events), but this is secondary to the main flow.

**M3: Ingest 5MB size guard is only tested at route level with mocks.**
A real payload near the limit should be tested at least once, but this is lower priority for smoke.

#### LOW

**L1: `schema_version` field is hardcoded `"1.0"` everywhere.**
All responses include `schema_version: "1.0"`. Assert it in smoke for future-proofing, but it won't regress without an intentional change.

**L2: Cursor pagination in run-overview and dashboard endpoints.**
Smoke should verify `next_cursor` is returned when data exceeds page size, but this is a secondary concern behind correctness of the main flow.

---

### Summary: Implementation Checklist

1. **One file:** `tests/integration/test_observability_smoke.py`
2. **Seed:** 1 experiment, 1 run, 5 config_runs (COMPLETED), 2+ spans each with lineage metadata, 1 "bad" example, 1 "good" example
3. **Deterministic:** Fixed IDs, fixed timestamps, explicit `min_samples=3`, `monkeypatch` tenant env var
4. **7 tests minimum:** ingest OK, overview+attribution, correlations ranked, broken-examples ranked, evidence sorted, evidence 404, tenant isolation
5. **Assert shapes, not just status codes:** Verify `total_items > 0`, `items[0].segment == "severe"`, `evidence` sorted by `metric_delta` ascending
