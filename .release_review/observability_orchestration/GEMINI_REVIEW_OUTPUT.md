# Executive Summary
*   **Phase 2 Implementation is incomplete and cannot exit to Phase 3.**
*   **Critical Backend Blockers:** The new canonical ingest path (`ObservabilityService.ingest`) fails to commit transactions, drops spans, and crashes due to a missing idempotency method.
*   **Critical Frontend Blocker:** The FE still calls the legacy V1 trace endpoint instead of the new canonical `/api/observability/*` namespace.
*   **Attribution Regression:** The new ingest path drops the trigger for attribution computation, breaking downstream analytics.
*   **Scalability Risk:** The correlation endpoint uses an unbounded `IN` clause that will crash the database for experiments with >1000 trials.
*   **Performance Risk:** Span ingestion uses loop-based `db.session.add()` instead of bulk inserts, creating a severe database bottleneck.
*   **UX Risk:** The FE's "large trace mode" naively slices the first 1000 spans, which corrupts the trace tree topology by orphaning child spans.
*   **Security Posture:** Tenant isolation is correctly enforced via headers, but SDK redaction is limited to dictionary keys, risking PII leaks in raw string payloads.
*   **Competitive Edge:** Traigent's built-in correlation to training effectiveness is a strong differentiator against Langfuse/LangSmith.
*   **Next Steps:** 4 Blockers must be patched immediately. 7 tickets are defined for the next sprint to unblock Phase 3.

# Blockers
1. **Severity:** Critical
   **Repo/File:** Backend / `src/services/observability_service.py` (~line 180)
   **Why:** `_expected_idempotency_key` is called but not defined in `ObservabilityService`.
   **Repro:** Any trace ingestion request will crash with `AttributeError: type object 'ObservabilityService' has no attribute '_expected_idempotency_key'`.
   **Patch:** Add `@classmethod def _expected_idempotency_key(cls, trace_id, configuration_run_id, span_id): return f"{trace_id}:{configuration_run_id}:{span_id}"`.

2. **Severity:** Critical
   **Repo/File:** Backend / `src/services/observability_service.py` (~line 130)
   **Why:** `ObservabilityService.ingest` calls `db.session.flush()` but never `db.session.commit()`.
   **Repro:** Send a valid ingest request. It returns 201, but spans are rolled back at the end of the Flask request and never saved to the database.
   **Patch:** Change `db.session.flush()` to `db.session.commit()` in the `try` block of `ingest`.

3. **Severity:** High
   **Repo/File:** Backend / `src/services/observability_service.py` (~line 280)
   **Why:** `_ingest_envelope` drops the attribution computation trigger that existed in the legacy `TraceSpanService`.
   **Repro:** Complete a trial. The spans are ingested, but `TraceAttributionService.compute_attributions` is never called, leaving attribution dashboards empty.
   **Patch:** Add the attribution trigger when `config_run.status` is COMPLETED/FAILED, matching the logic from `TraceSpanService.ingest_spans`.

4. **Severity:** High
   **Repo/File:** Frontend / `src/services/workflowTraceService.ts` (~line 38)
   **Why:** FE still calls the legacy V1 trace endpoint instead of the new canonical namespace.
   **Repro:** Load the Workflow Traces tab. Network tab shows a call to `/api/v1/experiment-runs/runs/${runId}/traces`.
   **Patch:** Update the URL to `/api/observability/runs/${runId}?include_spans=true`.

# High Risks
1. **Severity:** High
   **Repo/File:** Backend / `src/services/observability_service.py` (~line 550)
   **Why:** `get_run_correlations` uses `TraceSpanV2.configuration_run_id.in_(list(trial_quality.keys()))`. For runs with >1000 trials, this exceeds SQLite/Postgres `IN` clause limits and crashes.
   **Patch:** Refactor to use a `JOIN` with `ConfigurationRun` filtered by `experiment_run_id`.

2. **Severity:** High
   **Repo/File:** Backend / `src/services/observability_service.py` (~line 260)
   **Why:** `_ingest_envelope` uses a loop of `db.session.add(span)` instead of `db.session.bulk_save_objects()`. This creates a severe DB bottleneck for 500-span payloads.
   **Patch:** Accumulate spans in a list and use `db.session.bulk_save_objects(prepared_spans)`.

# Medium/Low Risks
1. **Severity:** Medium
   **Repo/File:** Frontend / `src/components/experiment/workflow/WorkflowTracesTab.tsx` (~line 160)
   **Why:** Slicing spans to 1000 (`spans.slice(0, 1000)`) can orphan child spans if their parents are sliced out, breaking the `TraceTree` rendering.
   **Patch:** Implement a tree-aware truncation that preserves parent-child relationships, or slice at the root-span level.

2. **Severity:** Medium
   **Repo/File:** Frontend / `src/services/workflowTraceService.ts` (~line 18)
   **Why:** `WORKFLOW_TRACE_CACHE_TTL_MS = 60_000` means users watching a live trial won't see new spans for a full minute.
   **Patch:** Reduce TTL to 5-10s for active runs, or implement a cache bypass when `status === 'RUNNING'`.

3. **Severity:** Low
   **Repo/File:** SDK / `traigent/integrations/observability/workflow_traces.py` (~line 100)
   **Why:** `_redact_observability_object` only redacts based on dictionary keys. Sensitive data embedded in string values (e.g., raw prompts) will leak.
   **Patch:** Document this limitation or add regex-based value scanning for secrets.

# Contract Consistency Matrix (SDK/BE/FE)
1. **Idempotency Key:** SDK generates `trace_id:config_run_id:span_id`. BE expects this exact format, but the BE method to verify it is missing (Blocker #1).
2. **Identifiers:** SDK uses `trial_id` as `configuration_run_id`. BE correctly maps this to `ConfigurationRun.id`. FE correctly reads `trial_id`. **Consistent.**
3. **Correlation Schema:** BE returns `items` with `feature_key`, `correlation`, `evidence`, etc. FE `CorrelationInsight` interface matches perfectly. **Consistent.**
4. **Error Semantics:** BE returns standard `error_response` (400/403/404/500). FE catches and displays generic error messages. **Consistent.**
5. **Single-version API:** **INCONSISTENT.** FE still calls legacy V1 trace endpoint instead of the new canonical `/api/observability/*` namespace (Blocker #4).

# Performance and Scalability Findings
1. **Ingestion Hotspots:** Loop-based `db.session.add()` instead of bulk inserts will bottleneck the DB under high concurrent trial load.
2. **Query Complexity:** The correlation endpoint loads all `ConfigurationRun` objects into memory and uses an unbounded `IN` clause for spans. This will fail for large experiments.
3. **FE Rendering:** The 1000-span hard limit prevents browser crashes but is implemented naively (array slice), which corrupts the tree topology.
4. **Cache Strategy:** 60s TTL is too long for live observability, leading to perceived system lag.

# Security Findings
1. **Header Spoofing:** `_resolve_tenant_id` correctly validates `X-Tenant-Id` against the configured tenant, preventing cross-tenant spoofing in single-tenant deployments.
2. **Authz Checks:** All new `/api/observability/*` routes have appropriate `@require_resource_read_access` or `write_access` decorators.
3. **Redaction Risk:** SDK redaction is key-based only. High risk of PII/secrets leaking if they are embedded in plain string inputs/outputs.
4. **Admin Endpoints:** `/admin/replay-dlq` is correctly protected with `@require_admin_auth()`.

# Competitive Readiness
1. **Where we are stronger:** Built-in correlation to training effectiveness (quality scores) and automated parameter impact analysis. Langfuse/LangSmith require manual export for this.
2. **Where we remain behind:** Naive tree truncation for large traces (LangSmith handles 100k+ spans via virtualized trees). No value-based PII redaction (Arize has robust PII masking).
3. **Phase 3 Priority:** Implement "Broken Example Intelligence" to automatically flag and cluster spans that caused quality drops, leveraging the correlation engine.

# Phase 2 Exit Criteria (pass/fail checklist)
- [ ] BE: Idempotency key validation executes without errors.
- [ ] BE: Ingested spans are committed to the database.
- [ ] BE: Attribution computation is triggered on trial completion.
- [ ] FE: All trace fetches use the `/api/observability/*` namespace.
- [ ] BE: Correlation endpoint handles >1000 trials without SQL errors.

# Phase 3 Entry Plan (ticketized)
1. **BE-101:** Fix `_expected_idempotency_key` missing method and add `db.session.commit()` to `ObservabilityService.ingest`. (Owner: BE)
2. **BE-102:** Restore attribution computation trigger in `ObservabilityService._ingest_envelope`. (Owner: BE)
3. **BE-103:** Refactor `get_run_correlations` to use `JOIN` instead of `IN` clause for span aggregation. (Owner: BE)
4. **BE-104:** Optimize `_ingest_envelope` to use `db.session.bulk_save_objects`. (Owner: BE)
5. **FE-101:** Update `fetchWorkflowTraces` to use `/api/observability/runs/${runId}`. (Owner: FE)
6. **FE-102:** Implement tree-aware span truncation for large traces. (Owner: FE)
7. **FE-103:** Reduce trace cache TTL to 5s for active runs. (Owner: FE)
