Now I have a complete picture. Here's the review.

---

## Phase 4 Automation-Hardening Review

### MEDIUM — Smoke test doesn't set `TRAIGENT_MOCK_LLM` / `TRAIGENT_OFFLINE_MODE`

**File:** `scripts/smoke/run_observability_smoke.sh:14`

The shell wrapper runs pytest directly without setting the required mock-mode environment variables. The phase-gate script (`run_observability_phase_gate.py:124-125`) correctly sets them, but anyone running the smoke shell script standalone will hit real backends or fail with missing credentials.

```bash
# Current
"$PYTHON_BIN" -m pytest tests/smoke/test_observability_phase4_smoke.py -q

# Should be
TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true \
  "$PYTHON_BIN" -m pytest tests/smoke/test_observability_phase4_smoke.py -q
```

---

### MEDIUM — Neither runner passes `-m smoke`

**Files:** `run_observability_smoke.sh:14`, `run_observability_phase_gate.py:151`

The smoke test is decorated `@pytest.mark.smoke`, and the marker is properly registered in `pyproject.toml:342`. But neither runner uses `-m smoke` — they rely on targeting the file by path. This works today but is fragile: if non-smoke tests are later added to the same file (or directory), they'll be included. More importantly, running the full suite with `-m smoke` won't pick these up as expected unless the user also targets the file path, since `testpaths` is `["tests"]` and `tests/smoke/` will be auto-discovered anyway. This is a minor consistency issue but worth noting for gate-automation reliability.

---

### MEDIUM — Smoke test couples to a private method (`_collect_workflow_span`)

**File:** `test_observability_phase4_smoke.py:65`

The smoke test calls `lifecycle._collect_workflow_span(...)` directly — a private method. If the method signature or name changes, the smoke test silently breaks with no public-API contract protecting it. For a *gate automation* test that gates releases, this fragility is a concern. Consider adding a thin public entrypoint (e.g., `lifecycle.record_trial_span(...)`) or at minimum documenting the coupling.

---

### LOW — `_SmokeOrchestrator` tracker is a bare `object()`, not `None`

**File:** `test_observability_phase4_smoke.py:36`

```python
self._workflow_traces_tracker = object()
```

In production, `_collect_workflow_span` checks `getattr(orchestrator, "_workflow_traces_tracker", None)` and early-returns if `None`. The smoke mock uses `object()` to bypass this guard. This is correct but means the test doesn't validate the guard path. Fine for a smoke test — just noting for completeness.

---

### LOW — No timeout on phase-gate stages

**File:** `run_observability_phase_gate.py:46-53`

`subprocess.run` is called without a `timeout` parameter. If a test hangs (which is a known issue per MEMORY.md for some test files), the gate script blocks forever with no feedback. Consider adding a per-stage timeout:

```python
completed = subprocess.run(
    stage.command,
    cwd=str(cwd),
    env=env,
    capture_output=True,
    text=True,
    check=False,
    timeout=300,  # 5-minute safety net
)
```

You'd need to catch `subprocess.TimeoutExpired` and produce a TIMEOUT result.

---

### LOW — `_summary_line` could return misleading output

**File:** `run_observability_phase_gate.py:35-41`

The heuristic scans for `passed|failed|error|warnings` in *any* output line. A test docstring or debug print containing those words could be picked as the summary. This is cosmetic-only (doesn't affect pass/fail logic) but could confuse report readers.

---

### LOW — Phase-gate report timestamp racing

**File:** `run_observability_phase_gate.py:133`

The report filename includes a second-precision timestamp. If the gate runs within the same second as a previous run, the file is silently overwritten. Very unlikely in practice but could be addressed with a millisecond suffix or random nonce if auditability matters.

---

### INFO — `pyproject.toml` smoke marker registration looks correct

`markers` list at line 342 properly registers `smoke`. No issues.

---

### INFO — Correctness of invariant assertions

The smoke test validates the right things for a Phase 4 gate:

| Assertion | Validates |
|---|---|
| `len(example_ids) == len(example_outcomes)` | 1:1 lineage cardinality |
| `example_ids == ["ex_1", "ex_2"]` | Deduplication (ex_1 appears twice in measures) |
| `failure_classification in _ALLOWED_*` | Enum closure — no unexpected strings leak through |
| `trace_linkage` fields match span | Cross-reference integrity |
| `not start_time.endswith("Z")` | ISO 8601 format compliance (uses `+00:00` not `Z`) |
| `_METRIC_KEY_PATTERN` on training_outcome keys | Python-identifier-safe metric keys |
| `status == "COMPLETED"`, `cost_usd >= 0.0` | Span correctness |

This is solid coverage for a smoke test.

---

## Verdict

**Phase 4 is ready to merge with minor fixes.** The smoke test correctly validates the critical observability payload invariants (lineage cardinality, deduplication, classification enums, trace linkage, timestamp format, metric key safety). The phase-gate script provides proper fail-fast ordering and markdown reporting.

**Before merging, address:**
1. Add `TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true` to `run_observability_smoke.sh` (MEDIUM — broken for standalone use)
2. Add `timeout=300` to `subprocess.run` in the gate script (LOW — protects against known hanging tests)

The remaining items are optional improvements that can be addressed in a follow-up.
