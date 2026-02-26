Here's the Phase 4 smoke test scope review:

---

## Phase 4 Automation Hardening: Scope Review

### 1. Top UI/Contract Regressions Smoke Tests Must Catch

**CRITICAL — Route-level rendering:**
- WorkflowTracesTab mounts inside `ExperimentResultsTab` (line ~1277) only when `selectedRunId` is truthy. A smoke test must verify the tab renders with a valid `runId` and that all 4 panes appear: GraphView, TraceTree, AttributionTable, Broken Examples panel.
- Empty/not-available state: empty `nodes[]` + empty `trials[]` triggers "Workflow Traces Not Available" — must stay stable.

**CRITICAL — API contract surface (4 endpoints):**
| Endpoint | Key Contract | Regression Risk |
|----------|-------------|-----------------|
| `GET /api/observability/runs/{id}?include_spans=true` | Returns `WorkflowTraceResponse` with `graph`, `trials[]`, `attribution[]`, `recommendations[]` | Shape change breaks entire tab |
| `GET .../correlations` | Returns `CorrelationInsightsResponse` with `items[].evidence[]` | Missing `evidence` array causes `.toFixed()` crash (line 835) |
| `GET .../broken-examples` | Returns `BrokenExamplesResponse` with `items[].failure_classification_counts` | Missing `average_delta` crashes `.toFixed(3)` (line 957) |
| `GET .../broken-examples/{id}/evidence` | Returns `BrokenExampleEvidenceResponse` with `evidence[].metric_delta` | `null` metric_delta handled, but missing `evidence` array isn't |

**HIGH — Broken Examples panel interactions:**
- Filter changes (`metric`, `segment`, `direction`) re-fetch with correct params
- Selecting an example triggers evidence fetch with matching filter context
- URL query params persist and restore (`broken_metric`, `broken_segment`, `broken_direction`, `broken_example_id`)

**HIGH — Correlation panel contract:**
- `point.feature_value.toFixed(3)` and `point.outcome_value.toFixed(3)` (line 835-836) crash if backend returns `null` — no guard exists.

---

### 2. Deterministic Test Design Recommendations

**Eliminate randomness in mock data:**
- Current `workflowTraceService.test.ts` uses `generateMockResponse()` with `Math.random()` for `quality_score`, `total_latency_ms`, etc. These pass today because assertions use `toBeGreaterThan(0)` ranges, but any tighter assertion will flake. **Use deterministic seeded values or static fixtures for smoke tests.**

**Isolate URL state:**
- `WorkflowTracesTab.test.tsx` correctly resets `window.history.replaceState({}, '', '/')` in `beforeEach`. Smoke tests **must** do the same — filter URL params leak between tests and cause phantom assertion failures.

**Mock all 4 service calls atomically:**
- The existing test mocks `fetchWorkflowTraces`, `fetchRunCorrelations`, `fetchRunBrokenExamples`, and `fetchRunBrokenExampleEvidence` in `beforeEach`. Smoke tests must replicate this — missing any one mock causes the component to hit real API or hang in loading state.

**Use `waitFor` with specific text assertions, not timers:**
- All existing tests correctly use `waitFor(() => expect(screen.getByText(...)))`. Continue this pattern. Never use `setTimeout` or `act()` wrappers that depend on timing.

**Avoid ReactFlow measurement dependencies:**
- ReactFlow is correctly mocked as a `div`. Smoke tests should NOT attempt to assert on node positions, layout dimensions, or SVG paths.

---

### 3. Coverage Priorities

**Priority 1 — Service layer contract tests (HIGH value, LOW flakiness):**

| Gap | What to Test | Why |
|-----|-------------|-----|
| Response format normalization | `response.data.data` vs `response.data` unwrapping (lines 71-73, 140-143, 180-183, 229-231) | Backend wraps responses inconsistently; a regression here silently returns `undefined` |
| Validation gate | `validateWorkflowTracePayload()` rejects missing `trials` array | Already tested for `trials`, but **not tested for missing `graph.nodes`** or missing `graph.edges` |
| Cache TTL branching | Active trial status detection (lines 342-355) | Tested, but the status field access pattern `trial.status ?? trial.trial_status` is fragile |
| Input sanitization | `encodeURIComponent` on runId/exampleId | Not tested — special chars in IDs could break URL construction |

**Priority 2 — WorkflowTracesTab interaction tests (HIGH value, MEDIUM flakiness):**

| Gap | What to Test |
|-----|-------------|
| **Broken example selection + evidence fetch** | Click example button -> verify `fetchRunBrokenExampleEvidence` called with correct `(runId, exampleId, options)` |
| **Filter change -> re-fetch** | Change broken segment dropdown -> verify `fetchRunBrokenExamples` re-called with updated `segment` param |
| **Correlation filter interaction** | Change correlation metric -> verify `fetchRunCorrelations` re-called |
| **Evidence display null handling** | Mock evidence with `metric_delta: null` -> verify "n/a" renders (line 1004) |
| **Empty items states** | Mock broken examples with `items: []` -> verify "No broken examples found" text |

**Priority 3 — Edge cases (MEDIUM value):**

| Gap | What to Test |
|-----|-------------|
| Large trace banner | >1000 spans triggers "Large trace mode" banner (already tested, but evidence panel interaction in this state is not) |
| Concurrent runId change | Rapid runId prop changes -> verify no stale state leaks |

---

### 4. Priority-Tagged Findings

**CRITICAL:**
1. **No guard on `correlation.evidence` being undefined.** `selectedCorrelation.evidence` is optional (`evidence?: CorrelationEvidencePoint[]`) in the type, but line 826 does `(selectedCorrelation.evidence || []).map(...)` — this is safe. However, `point.feature_value.toFixed(3)` on line 835 has **no null guard** and will crash if backend sends `null`. **Smoke test needed: mock evidence with `feature_value: null`.**

2. **Response unwrapping is not validated post-unwrap.** Lines 71-73, 140-143, 180-183: after selecting `response.data.data` or `response.data`, there's no check the result isn't `undefined`. If backend returns `{ data: { data: null } }`, the code proceeds with `null` and crashes downstream. **Smoke test needed: mock `{ data: { data: null } }` response.**

**HIGH:**
3. **Broken example filter changes don't cancel in-flight evidence fetch.** If a user changes `brokenSegment` while evidence is loading for a previously selected example, the old evidence response may arrive after the new broken-examples list renders, showing stale evidence. **Smoke test recommendation: verify evidence loading indicator appears on filter change.**

4. **No smoke test for broken example evidence drill-down.** The existing `WorkflowTracesTab.test.tsx` tests that `fetchRunBrokenExampleEvidence` is called (line 578), but never verifies the evidence **renders** (trial_id, metric_delta text). **Add assertion for evidence row content.**

5. **URL param restoration for `broken_example_id` not tested.** Tests verify `broken_metric`, `broken_segment`, `broken_direction` URL restoration, but not `broken_example_id`. If a user shares a URL with a pre-selected example, the evidence panel should auto-load.

**MEDIUM:**
6. **`validateWorkflowTracePayload` doesn't check `graph.loops`.** If backend omits `loops`, code proceeds and `GraphView` tries to render loop badges from undefined data. Not crash-critical (loops are optional in display), but inconsistent with other validation.

7. **Service tests use `Math.random()` in mock factory.** `generateMockResponse()` produces non-deterministic `quality_score` values. Any future assertion tightening will introduce flakiness.

**LOW:**
8. **Retry button on broken-examples error increments `reloadToken`** (line 909), which triggers a full re-fetch of traces, correlations, AND broken examples. This is heavy for a panel-level retry. Not a test concern, but worth noting for UX.

9. **`clearWorkflowTraceCache()` clears both `mockDataCache` and `workflowTraceCache`** — in production `USE_MOCK_DATA=false`, the mock cache clear is a no-op. Harmless but dead code path.

---

### Recommended Smoke Test File Structure

```
src/components/experiment/workflow/__tests__/
  WorkflowTracesTab.smoke.test.tsx    # Route rendering + panel visibility
  WorkflowTracesTab.broken.test.tsx   # Broken example interactions

src/services/__tests__/
  workflowTraceService.contract.test.ts  # Response unwrapping + validation
```

Estimate: ~8-12 test cases total to close the critical and high gaps above.
