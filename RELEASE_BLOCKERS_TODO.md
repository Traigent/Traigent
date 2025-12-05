# Release Blockers — AI Agent Checklist

Use this list to fix the highest‑risk findings before SDK release. Work from top to bottom. Check off items only when acceptance criteria are satisfied and evidence (diffs/tests) is attached to the PR.

## 🔴 Critical
- [ ] **Rotate and purge exposed API keys**
  - Remove any committed secrets (see reports/SECURITY_ANALYSIS_REPORT.md: “Exposed API Keys in .env”).
  - Add `.env` to `.gitignore` (if not already), delete tracked `.env` files, and purge from git history if present.
  - Confirm rotation of OpenAI, Claude, TraiGent keys and document rotation evidence.
- [ ] **Replace MD5 hashing**
  - `traigent/cloud/resilient_client.py` (around line 217): switch MD5 to SHA-256 (or stronger) and adjust downstream comparisons.
  - `traigent/storage/local_storage.py` (around line 589): same replacement.
  - Acceptance: no MD5 usage remains in codebase; unit tests updated/passing.
- [ ] **Secure temp directory usage**
  - `traigent/utils/batch_processing.py` (around line 363): replace hardcoded `/tmp/traigent_checkpoints` with `tempfile.mkdtemp()` or `TemporaryDirectory`, ensure directory permissions are secure, and clean up after use.

## 🟠 High
- [ ] **Harden HTTP security headers**
  - Add middleware to enforce HSTS, X-Frame-Options (DENY), X-Content-Type-Options (nosniff), Referrer-Policy (strict-origin-when-cross-origin), and a CSP appropriate for the app.
  - Apply wherever HTTP responses are served (API/gateway entrypoint) and add tests for header presence.
- [ ] **Fix linting correctness errors**
  - Run `ruff check` and resolve all 142 errors noted in reports/SECURITY_ANALYSIS_REPORT.md (focus on `raise-without-from`, undefined names, and type comparison issues).
  - Acceptance: `ruff check` passes cleanly; regressions covered by tests where behavior changed.
- [ ] **Strengthen session/token handling**
  - Replace in-memory `_revoked_tokens` with a persistent store (e.g., Redis) and add basic rate limiting on auth endpoints.
  - Document configuration knobs and defaults.

## 🟡 Medium
- [ ] **Input validation sweep**
  - Review SQL usage in `examples/archive/use-cases/by-domain/data/sql/` and any live query paths; add parameterization/escaping and validation.
  - Add tests to prove injection mitigations.
- [ ] **Add security logging guardrails**
  - Ensure sensitive fields are redacted in logs; add a centralized sanitizer and apply in error paths.

## ✅ Final verification
- [ ] Run `python scripts/code_analysis/run_analysis.py` and attach the latest summary under `reports/1_quality/analysis/`.
- [ ] Run full tests (`pytest`) and lint (`ruff check`); attach PASS output.
- [ ] Update `reports/SECURITY_ANALYSIS_REPORT.md` with remediation status or create an addendum noting fixes and residual risks.
