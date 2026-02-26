Here is my CRITICAL/HIGH findings review of the 5 phase-3 files after fixes:

---

## Re-Review: CRITICAL/HIGH Findings Only

### VERDICT: **0 CRITICAL, 1 HIGH** remaining

---

### HIGH-1: Unbounded Module-Level Cache (Memory Leak)

**File:** `src/services/workflowTraceService.ts:283-291`

```typescript
const mockDataCache = new Map<string, WorkflowTraceResponse>();
const workflowTraceCache = new Map<...>();
```

Both `mockDataCache` and `workflowTraceCache` are unbounded `Map` objects at module scope. While the TTL check prevents serving stale data, expired entries are **never evicted** — they accumulate indefinitely. In a long-lived browser session where a user navigates across many runs, this can cause significant memory growth.

**Impact:** Memory leak in long-lived sessions. Each `WorkflowTraceResponse` contains 25 trials with nested spans — these are non-trivial objects.

**Fix:** Add eviction on cache miss (lazy cleanup) or cap the map size. A simple approach:

```typescript
// Before inserting a new entry, evict expired ones
function evictExpiredEntries() {
  const now = Date.now();
  for (const [key, entry] of workflowTraceCache) {
    if (now > entry.expiresAtMs) {
      workflowTraceCache.delete(key);
    }
  }
}
```

Or cap at a max size (e.g., 20 entries) with LRU eviction.

---

### Previously Identified Issues — Now Resolved

| Issue | Status |
|---|---|
| Path traversal via `runId` | **Fixed** — `encodeURIComponent` applied to all URL parameters |
| Preflight `hasWorkflowTraces` double-fetch | **Fixed** — single fetch with `fetchWorkflowTraces` directly |
| Missing cancellation in useEffect | **Fixed** — all 4 data-loading effects have `cancelled` flag + cleanup |
| Stale closure on trial selection after reload | **Fixed** — `useRef` + ref sync pattern for `selectedTrialId`, `selectedCorrelationKey`, `selectedBrokenExampleId` |
| Payload validation | **Fixed** — `validateWorkflowTracePayload` checks graph structure and trials array |
| Large trace rendering perf | **Fixed** — `truncateSpansPreservingTopology` caps at 1000 spans with topology-safe BFS |
| Cache TTL for active runs | **Fixed** — `resolveWorkflowTraceCacheTtlMs` uses 5s TTL for running trials, 60s otherwise |
| Missing `exampleId` encoding | **Fixed** — `encodeURIComponent(sanitizedExampleId)` in `fetchRunBrokenExampleEvidence` |

---

### Items Reviewed & Clean (No Findings)

- **types.ts** — Pure type definitions, no logic, no issues.
- **workflowTraceService.test.ts** — Good coverage including TTL expiry, malformed payload rejection, empty runId, cache clear, concurrent fetches. No test gaps at CRITICAL/HIGH level.
- **WorkflowTracesTab.test.tsx** — Covers loading/error/not-available states, trial selection, broken example filters, URL param restore, large trace banner. Adequate for the component's complexity.

---

**Summary:** The codebase is in good shape after fixes. The only remaining HIGH finding is the unbounded cache growth. Everything else previously flagged has been addressed.
