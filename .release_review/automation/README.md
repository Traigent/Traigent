# Release Review Automation Toolkit (v2)

All commands use `python3`.

## Primary Commands

1. Initialize run + tracking

```bash
python3 .release_review/automation/generate_tracking.py \
  --version v1.0.0 \
  --release-id v1.0.0-20260305_120000 \
  --base-branch main
```

2. Run release gate checks

```bash
python3 .release_review/automation/release_gate_runner.py \
  --release-id v1.0.0-20260305_120000 \
  --strict
```

3. Build canonical verdict

```bash
python3 .release_review/automation/build_release_verdict.py \
  --release-id v1.0.0-20260305_120000
```

## Script Catalog

| Script | Purpose |
|---|---|
| `generate_tracking.py` | Initialize run workspace + slim tracking board |
| `release_gate_runner.py` | Execute gate checks + persist check results |
| `build_release_verdict.py` | Emit canonical `verdict.json` |
| `evidence_validator.py` | Validate evidence JSON against schema and git state |
| `scope_guard.py` | Enforce in-scope file modifications |
| `verify_tests.sh` | Re-run tests and verify claimed results via pytest JSON report |
| `rotation_scheduler.py` | Deterministic model-role scheduling |
| `validate_release_review_consistency.py` | Detect deprecated paths/commands in release-review docs |

## Contracts

- Scope config: `.release_review/scope.yml`
- Evidence schema: `.release_review/automation/schemas/evidence.schema.json`
- Run root: `.release_review/runs/<release_id>/`

## Model Assignment Policy

- Captain/orchestrator: Codex CLI 5.3 xhigh.
- Primary reviewer tasks: Codex CLI 5.3 high.
- Secondary/adversarial tasks: Claude CLI Opus 4.6 extended.
- Optional third reviewer: Copilot CLI Gemini 3.1 Pro.
- If Copilot/Gemini is unavailable, use only Codex 5.3 + Opus 4.6.
