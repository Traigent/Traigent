# Specialized Prompt: Code Review

Append this after `00_META_PROMPT.md`.

```text
Set `review_type` to `code`.

Focus priorities (highest first):
1) Correctness of changed logic against PR intent.
2) Runtime safety (error handling, null/empty/boundary behavior).
3) State/concurrency safety (shared mutable state, async/cancellation, race windows).
4) API compatibility and contract stability (public interfaces, return/error semantics).
5) Reliability/performance cliffs introduced by the diff.

Review method:
- Trace at least one happy path and one failure path per critical change.
- Check whether each external boundary (I/O, DB, network, queue, subprocess) has explicit validation and failure handling.
- Verify cleanup on failure paths (transactions, locks, handles, temp files, retries).
- Verify behavioral compatibility for existing callers where signatures or payloads changed.

Must-fix triggers (usually P0/P1):
- Silent data corruption or data loss risk.
- Crash/panic/uncaught exception in normal production path.
- Incorrect authorization or tenant/data boundary crossing.
- Breaking public API behavior without migration handling.
- Retry loops/timeouts that can cause cascading failure.

Medium-risk examples (usually P2):
- Incomplete validation for edge conditions.
- Error handling that logs but returns inconsistent state.
- Partial backward compatibility gaps with fallback possible.

De-prioritize (usually P3 unless risk-elevated):
- Style-only refactor preferences.
- Naming preferences without behavioral impact.
- Micro-optimizations not tied to realistic load/cost risk.
```
