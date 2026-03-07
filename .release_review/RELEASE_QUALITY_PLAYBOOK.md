# Release Quality Playbook (Major Revision)

Use this playbook with:

- `.release_review/CAPTAIN_PROTOCOL.md`
- `.release_review/PRE_RELEASE_REVIEW_PLAN.md`
- `.release_review/PRE_RELEASE_REVIEW_TRACKING.md`

## Why This Exists

Major SDK releases need stronger guarantees across:

- Security verification criteria
- Supply-chain integrity and provenance
- Repeatable, auditable evidence
- Reliable AI review prompts/protocols

## External Standards Mapping

| Standard / Guidance | What It Contributes | How It Maps Here |
|---|---|---|
| NIST SSDF (SP 800-218) | Secure software development practices | Maps to release gates, evidence retention, remediation flow |
| SLSA v1.0 | Build integrity and provenance expectations | Guides post-launch attestation roadmap |
| OWASP API Security Top 10 (2023) | API risk priorities | Mandatory lens for API/security component review |
| OWASP ASVS | App security verification requirements | Security checklist baseline and severity language |
| OpenSSF Scorecard | CI/CD and repo hygiene signals | Advisory hardening gate after v2 launch |

## Mandatory Gate Stack (v2 Blocking)

| Gate | Command Pattern | Output Artifact | Stop-Ship Rule |
|---|---|---|---|
| Lint/type | `python3 -m ruff ... && python3 -m mypy ...` | gate logs + summary | Any non-zero exit |
| Unit tests | `python3 -m pytest tests/unit -q` | gate logs + summary | Any non-zero exit |
| Integration tests | `python3 -m pytest tests/integration -q` | gate logs + summary | Any non-zero exit |
| Security baseline | `make security-check` | gate logs + summary | Unresolved blocking findings |
| Dependency review | `pip-audit --progress-spinner off --desc` | gate logs + summary | Known vuln without accepted waiver |
| CodeQL | CI-native (`release-gate/codeql`) | code scanning results | Failed required check |
| Protocol consistency | `python3 .release_review/automation/validate_release_review_consistency.py` | check log | Any deprecated protocol usage |
| Peer-review completeness | `python3 .release_review/automation/build_release_verdict.py --release-id <RELEASE_ID>` | `gate_results/verdict.json` (`failed_required_reviews`) | Missing primary/secondary/tertiary/reconciliation evidence, non-approved peer decisions, same-family P0/P1 reviewers, missing per-file 4-angle coverage, missing per-file artifact matrix (`file + role + agent`), or low-substance artifacts (missing strengths/checks/review summary/notes) |

Run all local gates with:

```bash
python3 .release_review/automation/release_gate_runner.py --release-id <RELEASE_ID> --strict
```

## Evidence Protocol (Non-Negotiable)

For each approved component and gate, keep:

- Command line used (exact)
- Exit code
- Timestamp (UTC ISO-8601)
- Commit SHA under review
- Artifact paths (`summary.md`, JSON reports, logs)
- Explicit disposition:
  - `Resolved`
  - `Accepted risk` (with owner + ticket)
  - `Blocked release`
- Per-file review proof:
  - each in-scope changed file appears in `files_reviewed[]` for primary, secondary, tertiary, and reconciliation evidence.
  - each in-scope changed file has artifacts under `.release_review/runs/<release_id>/file_reviews/` for required lanes:
    - `primary/codex_cli`
    - `secondary/claude_cli`
    - `tertiary/codex_cli` (or `tertiary/copilot_cli`)
    - `reconciliation/codex_cli`
  - each component/file artifact includes positive findings (`strengths[]`) and explicit checks (`checks_performed[]`) to support meta-analysis of both strengths and defects.

## AI Prompt Pack (Required for READY)

Use prompt templates under `.release_review/prompts/`.

- `codex_cli__captain.md`
- `codex_cli__primary.md`
- `claude_cli__secondary.md`
- `codex_cli__tertiary.md`
- `copilot_cli__tertiary.md`
- `codex_cli__reconciliation.md`

`READY` states require completed evidence from this peer-review flow for every component.

## Cost-Controlled Source Waves

Use source-wave mode when the per-file review matrix is too slow or expensive to execute as one
monolithic batch.

- Canonical source inventory: `.release_review/inventories/source_files.txt`
- Canonical wave definitions: `.release_review/components.yml`
- Ordered execution plan: `.release_review/PRIORITY_REVIEW_WAVES.md`
- Coverage verifier: `python3 .release_review/automation/verify_source_wave_coverage.py`

Rules:

- The wave plan is a batching strategy, not a weaker readiness bar.
- Every file in the source inventory must appear in exactly one wave inventory.
- Every wave still requires the same primary, secondary, tertiary, and reconciliation coverage.
- Completing only the first few waves is useful for risk triage, but it is never enough for
  `READY`.

## Suggested Rollout

1. Enforce `release-review.yml` required checks for tag releases.
2. Run two dry-run releases (`workflow_dispatch`) and stabilize verdict artifacts.
3. Track false positives for first three release cycles.
4. Add attestations/Scorecard in post-launch hardening milestone.

## Primary Sources

- NIST SSDF: https://csrc.nist.gov/Projects/ssdf
- NIST SP 800-218 PDF: https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218.pdf
- SLSA levels: https://slsa.dev/spec/v1.0/levels
- OWASP API Security Top 10: https://owasp.org/API-Security/editions/2023/en/0x11-t10/
- OWASP ASVS project: https://github.com/OWASP/ASVS/releases
- GitHub rulesets: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets
- GitHub dependency review action: https://github.com/actions/dependency-review-action
- GitHub CodeQL: https://docs.github.com/en/code-security/concepts/code-scanning/about-code-scanning
- GitHub artifact attestations: https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations
