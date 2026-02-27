# Specialized Prompt: PRD / Spec Review

Append this after `00_META_PROMPT.md`.

```text
Set `review_type` to `prd_spec`.

Focus priorities (highest first):
1) Requirement quality: clear, testable, unambiguous, internally consistent.
2) Implementability: enough detail for engineering without hidden assumptions.
3) Acceptance criteria: measurable pass/fail outcomes per key requirement.
4) Risk coverage: failure modes, security/privacy, operational readiness, rollback.
5) Compatibility and migration: impact on existing users/systems is explicit.

Review method:
- Identify ambiguous terms lacking measurable thresholds ("fast", "large", "reliable").
- Check requirement language strength (MUST/SHOULD/MAY style clarity).
- Verify critical scenarios include:
  - happy path,
  - error/failure path,
  - boundary conditions.
- Verify non-functional expectations (latency, scale, reliability, observability) are stated with testable bounds where relevant.
- Check dependency and interface contracts are concrete (inputs, outputs, error behavior, ownership).

Must-fix triggers (usually P0/P1):
- Contradictory requirements likely to produce incompatible implementations.
- Missing acceptance criteria for critical behavior.
- Missing security/privacy constraints for sensitive flows.
- Spec gaps that would force major rework after implementation starts.

Medium-risk examples (usually P2):
- Ambiguous non-critical behavior likely to cause inconsistent implementation.
- Under-specified edge conditions with moderate delivery risk.
- Incomplete rollout/rollback guidance for impactful changes.

De-prioritize (usually P3 unless risk-elevated):
- Preferred formatting/template variations.
- Minor wording improvements with only one reasonable interpretation.
```
