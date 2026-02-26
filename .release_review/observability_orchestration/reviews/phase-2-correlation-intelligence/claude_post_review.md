I've now read enough of all the files to provide a thorough review. Let me compile my findings.

## Phase 2 Observability Review: CRITICAL/HIGH Issues

### 1. **HIGH — Frontend `runId` not sanitized against path traversal in API URLs**

**File:** `workflowTraceService.ts:56-57`, `workflowTraceService.ts:116`, `workflowTraceService.ts:147`

The `runId` is only `.trim()`-ed before being interpolated directly into URL paths:
```ts
const sanitizedRunId = runId.trim();
// ...
const response = await apiClient.get(
  `/api/observability/runs/${sanitizedRunId}?include_spans=true`
);
```

A `runId` containing `../` or encoded path segments could reach unintended endpoints. The `hasWorkflowTraces` and `fetchRunCorrelations` functions have no sanitization at all — `runId` is interpolated raw. Add `encodeURIComponent(runId)` or validate against a strict pattern (e.g., `^[a-zA-Z0-9_-]+$`).

---

### 2. **HIGH — SDK `_collect_workflow_span` silently swallows all exceptions**

**File:** `trial_lifecycle.py:844-845`

```python
except Exception as exc:
    logger.debug(f"Failed to collect workflow span for trial {trial_id}: {exc}")
```

This blanket `except Exception` at `debug` level means **any** bug in the 160-line span construction block (lines 682–843) — including `TypeError`, `KeyError`, serialization errors, or DTO contract violations — is silently discarded. During development/rollout this will mask real bugs. At minimum, log at `warning` level and include the traceback (`logger.warning(..., exc_info=True)`).

---

### 3. **HIGH — Backend correlation endpoint lacks authorization/tenant isolation enforcement**

**File:** `observability_service.py:827-839` (`get_run_correlations`)

The `tenant_id` parameter defaults to `configured_tenant_id()` but is caller-supplied for the node-aggregation query (line 900). The `ConfigurationRun.query.filter_by(experiment_run_id=...)` on line 849 does **not** filter by tenant — it returns all config runs for a run regardless of tenant. This means a caller supplying a different `tenant_id` can still read parameter/example correlation data from any tenant's config runs. The tenant filter only applies to the `TraceSpanV2` node-latency/cost sub-query, not the parameter and example feature extraction.

**Impact:** Cross-tenant data leakage of experiment parameters and per-example metrics in correlation insights.

---

### 4. **HIGH — Backend correlation test uses perfectly linear data, masking edge cases**

**File:** `test_observability_correlation_service.py:48-53`

All 4 test trials use perfectly linearly correlated data (`score=0.2/0.4/0.6/0.8`, `temperature=0.1/0.2/0.3/0.4`). This means `_pearson_correlation` always returns ~1.0, and the direction/confidence filter tests never exercise the `"neutral"` or `"negative"` code paths with realistic data. The `direction="positive"` filter test (line 113-129) passes trivially because **all** features are positive. This is the **IT-CBM** anti-pattern (Condition-Behavior Mismatch) from your test quality guidelines.

---

That's it for CRITICAL/HIGH. Summary:

| # | Severity | Repo | Issue |
|---|----------|------|-------|
| 1 | HIGH | Frontend | `runId` path interpolation without `encodeURIComponent` — potential path traversal |
| 2 | HIGH | SDK | `_collect_workflow_span` blanket `except Exception` at `debug` level silently swallows bugs |
| 3 | HIGH | Backend | `get_run_correlations` leaks cross-tenant parameter/example data (tenant filter only on spans, not config runs) |
| 4 | HIGH | Backend | Correlation test uses perfectly linear data — IT-CBM anti-pattern masks filter logic bugs |
