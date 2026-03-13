# Security Monitoring Runbook

This runbook lists the beta-critical security telemetry emitted by Traigent and
the backend services that enforce tenant/project isolation. Operators should wire
these signals into their monitoring stack before onboarding enterprise beta
tenants.

## Configuration Flags
- `TRAIGENT_SECURITY_PROFILE` — `production` (default), `staging`, or `development`.
- `ALLOW_WEAK_CREDS` — permit short credentials (defaults to `false` outside dev).
- `AUTO_DISCOVERY` — allow dynamic framework instantiation when not explicitly set.
- `STRICT_SQL` — enable aggressive SQL quote heuristics (default `true`).
- `TRAIGENT_USE_DECIMAL_RATE_LIMIT` — enable Decimal arithmetic in rate limiting.
- `DISABLE_SECURITY_TELEMETRY` — suppress non-critical telemetry (default `false`).
- `ENABLE_RESOURCE_ACCESS_DUAL_RUN` — emit scoped-vs-legacy discrepancy telemetry
  during Wave 3A/3B rollout validation.

## Beta-Critical Backend Signals

The backend now emits three structured audit event types for scope enforcement:

- `tenant_scope_violation`
- `project_scope_violation`
- `scope_discrepancy`

These are recorded through the central audit logger and surfaced by the audit
API so operators can detect isolation failures or rollout regressions.

### Where these events come from

- `tenant_scope_violation`
  - an actor attempted to read or mutate a resource belonging to another tenant
- `project_scope_violation`
  - an actor attempted to read or mutate an out-of-scope project resource inside
    the same tenant
- `scope_discrepancy`
  - the Wave 3A dual-run path found a difference between the new enforcement
    kernel and the legacy path

### Audit endpoints to monitor

- `GET /api/v1/audit/alerts`
  - active alert summaries and current thresholds
- `GET /api/v1/audit/health`
  - aggregate health plus:
    - `scope_violations_24h`
    - `scope_discrepancies_24h`

Current beta alert IDs:
- `scope_violation_watch`
- `scope_discrepancy_watch`

Escalation guidance:
- any non-zero `tenant_scope_violation` in beta should be treated as a release
  blocker until explained
- any sustained `scope_discrepancy` count means the dual-run rollout is not yet
  stable enough for final cutover

## Credential Store Events
- `short_credential_value` — weak secret stored under relaxed profile.
- `rejected_short_credential_value` — weak secret blocked (increments `short_value_blocked`).
- `weak_credential_detected` — placeholder/known weak phrase rejected.

Monitor `_access_metrics` via `EnhancedCredentialStore.get_security_metrics()` and
alert on `short_value_blocked` > 0 in production.

## Discovery Telemetry
When auto-discovery instantiates a framework because the class had no explicit
opt-in/out flag, a log entry is emitted:

```
Auto discovery instantiating {ClassName} under profile {profile}
```

Aggregate these events to ensure production environments only instantiate classes
that are explicitly opted in.

## Input Validation
- HTML sanitization logs an `INFO` message when content is mutated.
- SQL validation raises `ValidationError` with message `Potentially unsafe quoted SQL input detected` when strict mode catches suspicious quotes.

Integrate these exceptions with application monitoring to trace misuse attempts.

## Rate Limiter Metrics
`SecureRateLimiter.get_metrics()` now includes `rounding_adjustments`, exposing how
often floating point or Decimal corrections were applied. Persistent non-zero values
may indicate precision issues that should be investigated.

## Privacy-Mode Monitoring

Enterprise beta assumes privacy mode is the default on new analytics/export
surfaces. Monitoring should therefore include:

- manifest-only export usage vs. materialized export usage
- policy-denied materialized export attempts
- export artifact cleanup success/failure counts

Materialized/raw-content export is expected to be rare and policy-gated. Any
unexpected increase should be investigated.

## Recommended Alerts
- Weak credential blocked in production.
- Auto-discovery instantiation observed in production.
- SQL quote heuristic triggered in production.
- Rate limiter rounding adjustments exceed a defined SLO.
- Any `scope_violation_watch` alert.
- Any `scope_discrepancy_watch` alert during rollout soak.
- Policy-denied raw-content export attempts above the normal baseline.
- Export artifact cleanup failures or backlog growth.

Keep this runbook alongside your SOC2 evidence to demonstrate proactive monitoring
of client-side and backend isolation controls.
