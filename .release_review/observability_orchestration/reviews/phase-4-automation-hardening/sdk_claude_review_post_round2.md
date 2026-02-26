**No CRITICAL/HIGH findings.** All four files are clean:

- **test_observability_phase4_smoke.py** — well-structured smoke test with explicit assertions on dedup, classification enums, timestamp format, metric key regex, and cost invariants. No security issues, no vacuous assertions.
- **run_observability_smoke.sh** — properly uses `set -euo pipefail`, resolves project root correctly, sets mock env vars.
- **run_observability_phase_gate.py** — timeout handling is safe (catches `TimeoutExpired`, decodes bytes properly), report writes to `.release_review/` (untracked), no shell injection (`subprocess.run` with list args), clean early-exit on stage failure.
- **pyproject.toml** — only change is the `smoke` marker addition at line 342; consistent with existing marker conventions.

**Readiness verdict:** Phase 4 automation-hardening changes are ready for PR.
