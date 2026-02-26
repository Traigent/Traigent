All 5 tests pass. Now I have a complete picture. Here's the review:

---

## Phase 4 Observability Smoke Coverage — Code Review

### All 5 smoke tests pass (2 service + 3 component)

---

### CRITICAL — None found

---

### HIGH

**H1. `quality_score` null-safety crash in trial selector** `WorkflowTracesTab.tsx:684`

```tsx
{(trial.quality_score * 100).toFixed(0)}%
```

`TrialTrace.quality_score` is `number`, but if the backend ever sends `null` or `undefined` (which `WorkflowTraceResponse` doesn't guard against since the validation only checks `graph` and `trials` array shape, not individual trial fields), this will throw `Cannot read properties of null (reading 'toFixed')`. The smoke test uses `quality_score: 0.8` and never exercises a null path.

**H2. `average_metric_value` is `number | null` in the type but not exercised as null in tests** `types.ts` defines `BrokenExampleInsight.average_metric_value: number | null`, but the mock uses `0.45`. No test covers the `null` path. The `formatNumeric` helper does handle it correctly — but this is unverified.

**H3. `BrokenExampleEvidencePoint.metric_delta` is `number | null` but the smoke test regex expects a concrete value** `WorkflowTracesTab.smoke.test.tsx:194`

```tsx
expect(screen.getByText(/delta=-0\.350/)).toBeInTheDocument();
```

If the backend returns `metric_delta: null`, `formatNumeric` returns `'n/a'` and the UI would show `delta=n/a`. No test covers this. This is consistent with the type contract but represents a coverage gap.

---

### MEDIUM

**M1. `unwrapObservabilityPayload` does not distinguish `data: undefined` from absent `data` key** `workflowTraceService.ts:38-48`

If the API returns `{ data: { status: "ok" } }` (no nested `data` key), the function returns the outer object as-is. If it returns `{ data: { data: undefined } }`, it reaches the `payload == null` check and throws. This is probably fine for the current backend, but the `in` check at line 40 treats `{ data: undefined }` as having a `data` key, so `undefined` gets unwrapped and then rejected — good. However, the smoke test for malformed payloads (`workflowTraceService.smoke.test.ts:83-93`) only tests `{ data: { data: null } }`, not `{ data: undefined }` or `{ data: {} }`. Minor gap.

**M2. No smoke test for `hasWorkflowTraces` function** The service exports `hasWorkflowTraces` and the component mock file stubs it, but no smoke test exercises it. This endpoint does a lightweight GET and has its own error-handling logic (404 → false, other errors → false with a `console.warn`). Worth covering.

**M3. Cache TTL dynamic resolution is untested** `resolveWorkflowTraceCacheTtlMs` (line 354) branches on `trial.status` / `trial.trial_status` to select a shorter TTL for active runs. No smoke test covers this — the mock trials have no `status` field, so the function always returns the long TTL. A bug in the active-trial detection would silently cause stale caches during live runs.

**M4. Large trace truncation (`truncateSpansPreservingTopology`) is untested** `WorkflowTracesTab.tsx:212-268` has a non-trivial topology-preserving truncation algorithm with DFS. The smoke test uses `spans: []`, so this path is never exercised. A regression here would break rendering for large traces silently.

**M5. `brokenQueryKey` double-dependency** `WorkflowTracesTab.tsx:497` — the broken examples effect depends on both `[brokenDirection, brokenMetric, brokenSegment]` individually AND on `brokenQueryKey` (which is a memo of those same values). This is harmless but means a filter change fires the effect once (React deduplicates within the same tick), though it could be cleaner.

---

### LOW

**L1. Mock data generator uses `Math.random` without seed** `workflowTraceService.ts:7` has an ESLint disable for `sonarjs/pseudo-random`. The mock generator is behind `USE_MOCK_DATA = false`, so this is dead code in production — but if toggled on for local dev, test output is non-deterministic.

**L2. Metric dropdown options are hardcoded** `WorkflowTracesTab.tsx:738-741, 868-871` — the metric `<Select>` contains only `score`, `accuracy`, `quality_score`. If the backend adds a new metric, the UI won't show it. This is an intentional UX constraint but worth noting.

**L3. `as any` casts in component smoke test** `WorkflowTracesTab.smoke.test.tsx:138-148` — mock return values use `as any`, which could mask type drift between the mock and the actual response shape. Using `satisfies` or explicit typing would catch regressions.

**L4. URL persistence side-effect in tests** `WorkflowTracesTab.smoke.test.tsx:136` resets to `/` in `beforeEach`, which is good. But the component writes URL params via `replaceState` during every test run. If tests ran in parallel with shared `window`, they could interfere. Currently fine since Vitest uses isolated environments per file.

---

### Verdict: Phase 4 is **READY with caveats**

The implementation is solid. The service layer has proper null-guarding (`unwrapObservabilityPayload`), input sanitization, TTL-based caching with eviction, and topology-safe span truncation. The component properly handles loading/error/empty states with cancellation guards on every async effect. The broken-example drill-down flow (URL restoration → list fetch → evidence fetch with query-key gating) is well-sequenced.

**What's strong:**
- Correct `cancelled` flags on all 4 async effects prevent stale-setState
- `resolvedBrokenQueryKey` gating prevents evidence fetches for stale broken-example lists
- Topology-preserving span truncation is a good defensive measure
- `formatNumeric` properly handles `null`/`undefined`/`NaN`

**What to address before merging:**
1. Add a `quality_score` null guard in the trial selector (H1) — one-line fix
2. Add at least one smoke case for `metric_delta: null` evidence and `average_metric_value: null` (H2/H3) — verifies the `formatNumeric` path
3. Consider adding a basic `truncateSpansPreservingTopology` unit test (M4) — the algorithm is non-trivial
