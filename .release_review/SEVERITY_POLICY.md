# Severity and Waiver Policy (v2)

## Severity Classes

- `P0`: Critical release blocker. Must be fixed before release unless emergency waiver is approved.
- `P1`: High-impact blocker. Must be fixed before release unless time-boxed waiver is approved.
- `P2`: Medium risk. Non-blocking by default.
- `P3`: Low risk. Non-blocking.

## SLA and Escalation

- P0: resolve or approved waiver within 24 hours.
- P1: resolve or approved waiver within 72 hours.
- If SLA exceeded, escalate to release owner and security owner.

## Waiver Requirements

Waiver file location:

- `.release_review/runs/<release_id>/waivers/<waiver_id>.json`

Required fields:

- `waiver_id`
- `release_id`
- `finding_id`
- `severity`
- `reason`
- `approved_by` (array, minimum 2 maintainers)
- `expires_at` (UTC ISO-8601)
- `created_at` (UTC ISO-8601)
- `remediation_issue`

## Waiver Guardrails

- Waivers do not bypass unresolved P0 security findings without emergency path.
- Expired waivers are ignored automatically by verdict generation.
- Every waiver requires a remediation issue due within 48h for emergency overrides.
