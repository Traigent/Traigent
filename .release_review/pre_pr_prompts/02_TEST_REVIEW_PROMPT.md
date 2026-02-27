# Specialized Prompt: Tests Review

Append this after `00_META_PROMPT.md`.

```text
Set `review_type` to `tests`.

Focus priorities (highest first):
1) Signal quality: tests fail when the behavior is broken.
2) Coverage of changed behavior: changed branches and failure paths are exercised.
3) Assertion strength: assertions validate outcomes, not just presence/truthiness.
4) Determinism and isolation: no flaky, order-dependent, or environment-coupled behavior.
5) Cost and maintainability: fast enough for CI while preserving risk coverage.

Review method:
- Map changed production paths to specific tests and note unmapped paths.
- Identify vacuous tests:
  - tautological assertions,
  - mocks asserting their own return values,
  - tests that cannot fail meaningfully.
- Verify negative/failure-path tests for boundary and error handling.
- Check for flaky patterns:
  - real sleeps/time dependencies,
  - network/filesystem dependence without control,
  - shared mutable global state across tests.
- Validate that fixtures/mocks reflect realistic production behavior.

Must-fix triggers (usually P0/P1):
- No test coverage for critical changed behavior.
- Tests that always pass even when implementation is broken.
- Security/correctness-sensitive code changed with no regression test.
- Non-deterministic tests likely to destabilize CI/release confidence.

Medium-risk examples (usually P2):
- Missing boundary/negative case tests for important paths.
- Weak assertions that check only shape/type.
- Over-mocked tests that bypass real decision logic.

De-prioritize (usually P3 unless risk-elevated):
- Minor test style preferences.
- File naming/layout preferences.
- Non-critical cleanup where reliability is unaffected.
```
