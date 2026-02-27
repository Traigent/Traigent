# Specialized Prompt: Security Review

Append this after `00_META_PROMPT.md`.

```text
Set `review_type` to `security`.

Focus priorities (highest first):
1) Input-to-sink exploit paths (injection, traversal, unsafe deserialization, command execution).
2) AuthN/AuthZ correctness (access control, IDOR, privilege boundaries, tenant isolation).
3) Secrets and sensitive data handling (storage, transport, logs, errors, telemetry).
4) Cryptography/session/token safety (algorithms, key handling, expiration/revocation).
5) Supply-chain and build integrity (new dependencies, pinning, provenance/signing posture).

Review method:
- Trace attacker-controlled input to sensitive sinks with explicit path evidence.
- Check secure defaults in changed code (deny-by-default, least privilege, fail-safe behavior).
- Validate sensitive values are never exposed in logs/errors.
- Review dependency/config changes for known security risk indicators.
- Check whether security-relevant behavior is covered by tests.

Must-fix triggers (usually P0/P1):
- Unauthenticated unauthorized access to protected data/actions.
- Confirmed secret/token disclosure path.
- Practical injection path (SQL/command/template/deserialization) from user input.
- Disabled or bypassed security controls without equivalent mitigation.
- New dependency with known high/critical advisory and no mitigation plan.

Medium-risk examples (usually P2):
- Missing hardening checks on non-critical boundaries.
- Overly broad permissions with constrained exposure.
- Incomplete audit logging for sensitive actions.

Required caution:
- Do not claim a dependency CVE without advisory ID/source.
- Do not claim exploitability without source-to-sink evidence.
- If exploitability depends on unknown infrastructure assumptions, mark as uncertain.

De-prioritize (usually P3 unless risk-elevated):
- Generic best-practice advice not tied to changed code.
- Theoretical threats without realistic preconditions.
```
