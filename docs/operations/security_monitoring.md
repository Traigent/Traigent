# Security Monitoring Runbook

This runbook lists key telemetry emitted by the Traigent SDK when security features
trigger defensive behaviors. Operators can wire the log messages or exported
metrics into their observability stack to satisfy SOC2 monitoring controls.

## Configuration Flags
- `TRAIGENT_SECURITY_PROFILE` — `production` (default), `staging`, or `development`.
- `ALLOW_WEAK_CREDS` — permit short credentials (defaults to `false` outside dev).
- `AUTO_DISCOVERY` — allow dynamic framework instantiation when not explicitly set.
- `STRICT_SQL` — enable aggressive SQL quote heuristics (default `true`).
- `TRAIGENT_USE_DECIMAL_RATE_LIMIT` — enable Decimal arithmetic in rate limiting.
- `DISABLE_SECURITY_TELEMETRY` — suppress non-critical telemetry (default `false`).

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

## Recommended Alerts
- Weak credential blocked in production.
- Auto-discovery instantiation observed in production.
- SQL quote heuristic triggered in production.
- Rate limiter rounding adjustments exceed a defined SLO.

Keep this runbook alongside your SOC2 evidence to demonstrate proactive monitoring
of client-side security controls.
