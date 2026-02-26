No CRITICAL/HIGH findings.

**Readiness verdict:** Both files are clean — the smoke test covers the full ingest→overview→correlations→broken-examples→evidence pipeline with deterministic data, proper teardown via `_scoped_model_query_overrides`, tenant isolation assertion, and a 404 guard; the runner script is minimal and correct. Ready to merge.
