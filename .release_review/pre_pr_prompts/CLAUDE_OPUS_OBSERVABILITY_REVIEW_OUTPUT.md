# Claude Opus Review Output (Summary Prompt)

Date: 2026-02-26
Model invocation: `claude --model opus --print`
Note: Full extended/high-effort prompt repeatedly timed out in this shell; this output is from a concise summary prompt.

## Raw output

## Change Review: Tracker Cleanup, Idempotency Batch, Bootstrap Gate

### Severity-Tagged Risks

1. **[HIGH] Backend batched prefetch race window** — Between prefetch and insert, a concurrent request can submit a duplicate that the in-request set hasn't seen. The old per-check approach was serialized; batched prefetch introduces a TOCTOU gap under high write concurrency.

2. **[HIGH] SDK final close ordering** — If `final_close()` is called while spool trim is mid-flush, partially-written records could be truncated or lost. Need to confirm trim acquires the same lock as close, or that close waits for in-flight trims.

3. **[HIGH] Frontend double-load bootstrap gate** — Requiring both load calls creates a deadlock risk if either call fails silently (network timeout, 401). If there's no timeout/fallback, the dashboard hangs indefinitely on startup.

4. **[MEDIUM] Spool max trim data loss** — Trimming to a max bound discards records. If the spool contains un-synced events (e.g., during transient cloud outage), trim deletes data the backend never received. Confirm trim only evicts already-acknowledged records.

5. **[MEDIUM] Backend duplicate set memory growth** — In-request duplicate set lives in memory per request. If a single batch submission contains thousands of entries, the set grows unbounded within that request lifecycle. No cap mentioned.

6. **[MEDIUM] Dashboard TTL 60s staleness** — 60s cache means users see up to 1-minute-stale data. If a run completes in <60s, the dashboard may never show the intermediate "running" state, only jumping from nothing to "completed."

7. **[LOW] Frontend 9 tests seems thin** — Bootstrap gate is a critical path change (blocks all rendering). 9 tests may not cover: both-succeed, first-fails, second-fails, both-fail, timeout, retry-then-succeed, auth-expired-mid-load, race between loads, and cache-hit skip.

### Missing Tests

8. **[HIGH] No concurrent duplicate submission test (BE)** — Need a test where two workers submit overlapping event batches simultaneously to verify the prefetch+set approach doesn't allow duplicates through the TOCTOU window.

9. **[HIGH] No SDK close-during-trim test** — Should verify that calling `final_close()` while a spool trim is in progress either waits for trim completion or cleanly aborts without data corruption.

10. **[MEDIUM] No frontend partial-load-failure test** — Should test that if one bootstrap load call succeeds and the other returns 500/timeout, the UI shows an error state rather than hanging.

11. **[MEDIUM] No spool trim with unacked records test (SDK)** — Should verify that trim respects sync acknowledgment boundaries and doesn't discard events the backend hasn't confirmed.

12. **[LOW] No TTL expiry boundary test (FE)** — Should verify dashboard re-fetches at exactly TTL expiry and handles the case where the re-fetch returns different data shape (e.g., new fields from a backend deploy during the 60s window).
