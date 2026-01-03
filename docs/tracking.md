# Documentation Review Tracking

This tracker keeps documentation reviews consistent and prevents drift between
docs and the SDK behavior.

## Review Guidelines

- Accuracy: Verify parameter names, defaults, and behavior against the current
  source of truth (`traigent/api/decorators.py`, `traigent/core/optimized_function.py`,
  `traigent/config/parallel.py`).
- Consistency: Use the same terminology across docs (execution modes, option
  bundles, config access APIs). Avoid mixing legacy and current names.
- Completeness: Include prerequisites, environment variables, and any runtime
  constraints (mock mode, network costs, limits).
- Conciseness: Remove redundant explanations and keep examples focused on one
  concept at a time.
- Clarity: Use short sentences, ordered steps, and explicit parameter names.
  Define acronyms on first use.
- Examples: Ensure code blocks are runnable and include imports. Call out
  `TRAIGENT_MOCK_LLM=true` when examples might incur API costs.
- Links: All local links must resolve; external links should be current.
- Versioning: Align version tags and feature availability with the current
  release. Call out roadmap-only features explicitly.

## Scope

- Track Markdown docs under `docs/` plus reference YAML specs used for
  requirements, feature matrices, and traceability.
- Exclude generated assets (images, video casts, PDFs) unless they are the primary
  deliverable of a tracked doc bundle.
- Use directory rows for large collections; note included subfolders in Notes.

## Review Workflow

1. Identify the doc scope and audience.
2. Compare against the current API and behavior in code.
3. Validate code snippets for imports and API usage.
4. Check local links for drift.
5. Update the tracking table with status, date, and notes.

## Quick Checks

```bash
# Scan docs for deprecated decorator kwargs based on current API validators.
# Source of truth: traigent/api/decorators.py and traigent/api/parameter_validator.py.
rg -n "optimize\\(" docs
```

```bash
# Report missing local .md links
python3 - <<'PY'
import re
from pathlib import Path

root = Path("docs")
link_re = re.compile(r"\\[[^\\]]+\\]\\(([^)]+)\\)")
missing = []

def strip_fenced_blocks(text: str) -> str:
    lines = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return "\\n".join(lines)

for md in root.rglob("*.md"):
    text = strip_fenced_blocks(md.read_text(encoding="utf-8"))
    for match in link_re.findall(text):
        link = match.strip()
        if not link or link.startswith(("#", "http://", "https://", "mailto:")):
            continue
        path = link.split("#", 1)[0]
        if not path.endswith(".md"):
            continue
        target = (md.parent / path).resolve()
        if not target.exists():
            missing.append((md, link))

for md, link in missing:
    print(f"{md}: {link}")
PY
```

```bash
# List docs that are not covered by the tracking tables (directory rows cover their contents).
python3 - <<'PY'
from pathlib import Path

tracking = Path("docs/tracking.md").read_text(encoding="utf-8")
tracked = set()

for line in tracking.splitlines():
    if not line.startswith("|"):
        continue
    parts = [part.strip() for part in line.strip().split("|")[1:-1]]
    if not parts:
        continue
    doc = parts[0]
    if doc.startswith("docs/"):
        tracked.add(doc)

tracked_dirs = {Path(p) for p in tracked if p.endswith("/")}
tracked_files = {Path(p) for p in tracked if not p.endswith("/")}

def covered(path: Path) -> bool:
    if path in tracked_files:
        return True
    return any(parent in tracked_dirs for parent in path.parents)

yaml_allowlist = {Path("docs/requirements.yml"), Path("docs/functionalities.yml")}
yaml_dirs = {Path("docs/feature_matrices"), Path("docs/traceability")}

for path in Path("docs").rglob("*"):
    if path == Path("docs/tracking.md"):
        continue
    if path.suffix == ".md":
        pass
    elif path.suffix in {".yml", ".yaml"}:
        if path not in yaml_allowlist and not any(dir_path in path.parents for dir_path in yaml_dirs):
            continue
    else:
        continue
    if not covered(path):
        print(path.as_posix())
PY
```

## Status Legend

- Pending: Not reviewed in the current cycle
- Reviewed: Checked and aligned with current code
- Needs Update: Known issue or gap exists
- Historical: Planning or archive doc; keep as reference

## Tracking Table (User-Facing Docs)

| Doc | Area | Status | Last Reviewed | Notes |
| --- | --- | --- | --- | --- |
| docs/README.md | Overview | Reviewed | 2026-01-01 | Added guides/API links |
| docs/getting-started/README.md | Getting started index | Reviewed | 2026-01-01 | Added testing link and normalized headings |
| docs/getting-started/GETTING_STARTED.md | Getting started | Reviewed | 2026-01-01 | Links verified; no changes |
| docs/getting-started/installation.md | Getting started | Reviewed | 2026-01-01 | Updated extras list, repo URL, and mock-mode testing |
| docs/getting-started/testing.md | Getting started | Reviewed | 2026-01-01 | Added mock-mode defaults and cleaned coverage guidance |
| docs/api-reference/decorator-reference.md | API reference | Reviewed | 2026-01-01 | Added SE-friendly tuned variables + ConfigSpace/Constraint coverage |
| docs/api-reference/complete-function-specification.md | API reference | Reviewed | 2026-01-01 | Aligned version/runtime overrides; added reps_per_trial fields; updated examples |
| docs/api-reference/interactive_optimizer.md | API reference | Reviewed | 2026-01-01 | New API reference |
| docs/api-reference/thread-pool-examples.md | API reference | Reviewed | 2026-01-01 | Linked parallel configuration guide |
| docs/api-reference/telemetry.md | API reference | Reviewed | 2026-01-01 | Synced telemetry events, privacy notes, and local storage layout |
| docs/guides/evaluation.md | Guide | Reviewed | 2026-01-01 | Added parallel configuration link |
| docs/guides/execution-modes.md | Guide | Reviewed | 2026-01-01 | Clarified analytics controls and privacy alias |
| docs/guides/parallel-configuration.md | Guide | Reviewed | 2026-01-01 | New guide |
| docs/guides/secrets_management.md | Guide | Reviewed | 2026-01-01 | Removed missing scripts; added CLI-based vault examples |
| docs/guides/llm_plugin_migration_guide.md | Guide | Reviewed | 2026-01-01 | Updated plugin list and test commands |
| docs/guides/VSCODE_RESTART_GUIDE.md | Guide | Reviewed | 2026-01-01 | Simplified to generic VS Code reload workflow |
| docs/user-guide/README.md | User guide index | Reviewed | 2026-01-01 | Normalized headings and removed emoji styling |
| docs/user-guide/interactive_optimization.md | User guide | Reviewed | 2026-01-01 | Linked interactive optimizer API |
| docs/user-guide/agent_optimization.md | User guide | Reviewed | 2026-01-01 | Fixed next-steps links |
| docs/user-guide/optuna_integration.md | User guide | Reviewed | 2026-01-01 | Removed broken plan link |
| docs/user-guide/choosing_optimization_model.md | User guide | Reviewed | 2026-01-01 | Added OSS/managed-backend note and clarified remote guidance |
| docs/user-guide/injection_modes.md | User guide | Reviewed | 2026-01-01 | Fixed parameter-mode access patterns and deprecated get_trial_config note |
| docs/user-guide/evaluation_guide.md | User guide | Reviewed | 2026-01-01 | Links verified; no changes |
| docs/features/README.md | Features | Reviewed | 2026-01-01 | Added framework override feature link |
| docs/features/authentication.md | Features | Reviewed | 2026-01-01 | Corrected CLI behavior and storage details; refreshed support guidance |
| docs/features/constraint-dsl.md | Features | Reviewed | 2026-01-01 | Synced builder method names, precedence test, and diagnostics notes |
| docs/features/seamless_injection.md | Features | Reviewed | 2026-01-01 | Rewritten to reflect current AST + runtime shim behavior |
| docs/features/strict_metrics_nulls.md | Features | Reviewed | 2026-01-01 | Verified flag behavior; no doc changes needed |
| docs/features/framework_override_enhanced_features.md | Features | Reviewed | 2026-01-01 | Clarified enhanced mode behavior and validation status |
| docs/operations/azure_openai.md | Operations | Reviewed | 2026-01-01 | Removed unused provider env var; clarified provider flag |
| docs/operations/bedrock.md | Operations | Reviewed | 2026-01-01 | Fixed smoke test formatting and removed unused provider env var |
| docs/operations/google_gemini.md | Operations | Reviewed | 2026-01-01 | Removed unused provider env var; clarified API key names |
| docs/operations/security_monitoring.md | Operations | Reviewed | 2026-01-01 | Verified env flags and telemetry names; no changes needed |

## Tracking Table (Internal and Reference Docs)

| Doc | Area | Status | Last Reviewed | Notes |
| --- | --- | --- | --- | --- |
| docs/agent_development_script.md | Internal tooling | Reviewed | 2026-01-01 | Added next-steps links and mock-mode note |
| docs/case_study_pipeline_overhaul.md | Case study | Reviewed | 2026-01-01 | Marked FEVER-only scope; corrected telemetry schema/imports |
| docs/DEMO_VIDEO_GUIDE.md | Demos | Reviewed | 2026-01-01 | Aligned to docs/demos paths and added GitHub hooks demo |
| docs/demos/README.md | Demos | Reviewed | 2026-01-01 | Aligned paths and referenced TVL demos |
| docs/feature_matrices/README.md | Feature matrices | Reviewed | 2026-01-01 | Clarified stale status and regeneration note |
| docs/feature_matrices/ | Feature matrices | Reviewed | 2026-01-01 | Updated evaluator/optimizer/analytics matrices and timestamps |
| docs/architecture/ARCHITECTURE.md | Architecture | Reviewed | 2026-01-01 | Added streaming invoker and deprecated get_trial_config note |
| docs/architecture/project-structure.md | Architecture | Reviewed | 2026-01-01 | Updated root layout and key directories |
| docs/architecture/execution_mode_follow_up.md | Architecture | Reviewed | 2026-01-01 | Reviewed; no changes |
| docs/architecture/plugin_architecture.md | Architecture | Reviewed | 2026-01-01 | Updated override usage, version management, supported frameworks |
| docs/architecture/stop_conditions.md | Architecture | Reviewed | 2026-01-01 | Reviewed; no changes |
| docs/contributing/README.md | Contributing | Reviewed | 2026-01-01 | Removed emoji headings |
| docs/contributing/CONTRIBUTING.md | Contributing | Reviewed | 2026-01-01 | Updated tooling, test commands, mock mode, and repo details |
| docs/contributing/SECURITY.md | Contributing | Reviewed | 2026-01-01 | Aligned seamless injection, credentials, and optional deps |
| docs/contributing/code_review_instructions.md | Contributing | Reviewed | 2026-01-01 | Added mock-mode note; updated lint tool list |
| docs/contributing/CODE_OF_CONDUCT.md | Contributing | Reviewed | 2026-01-01 | Set enforcement contact |
| docs/contributing/ADDING_NEW_INTEGRATIONS.md | Contributing | Reviewed | 2026-01-01 | Added mock-mode test command |
| docs/testing/RELEASE_READINESS_TESTING.md | Testing | Reviewed | 2026-01-01 | Updated repo URL and expected version |
| docs/testing/CREATIVE_STRESS_TESTING.md | Testing | Reviewed | 2026-01-01 | Swapped get_config usage and updated path |
| docs/features/random_sampler_plan_spec.md | Feature spec | Reviewed | 2026-01-01 | Added sampler factory reference |
| docs/ONBOARDING_HEAD_OF_AI.md | Onboarding | Reviewed | 2026-01-01 | Updated repo, requirements, and subset selection note |
| docs/global_registrations_audit.md | Audits | Reviewed | 2026-01-01 | Noted FEVER-only scope and historical Spider reference |
| docs/grid_search_fix_analysis.md | Analysis | Reviewed | 2026-01-01 | Added historical note for Spider |
| docs/grid_search_model_prioritization.md | Analysis | Reviewed | 2026-01-01 | Historical Spider note and max_trials wording |
| docs/tagging_multiagent_protocol.md | Protocols | Reviewed | 2026-01-01 | Reviewed; no changes |
| docs/requirements.yml | Requirements | Reviewed | 2026-01-01 | Removed OptiGen mention |
| docs/functionalities.yml | Traceability | Reviewed | 2026-01-01 | Reviewed; no changes |
| docs/traceability/schema.md | Traceability | Reviewed | 2026-01-01 | Updated generated report paths |
| docs/traceability/requirements.yml | Traceability | Reviewed | 2026-01-01 | Removed OptiGen mention |
| docs/traceability/ | Traceability | Reviewed | 2026-01-01 | Updated taxonomy cardinality and traceability housekeeping |
| docs/reviews/ | Reviews | Historical | - | Review requests and summaries |
| docs/plans/ | Plans | Historical | - | Planning docs |
| docs/planned_features/ | Plans | Historical | - | Planning docs |
| docs/tvl/ | TVL | Historical | - | TVL spec and site assets |
