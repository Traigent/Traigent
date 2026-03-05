# Start Release Review (v2)

Copy into your reviewer session:

```text
You are the release-review captain.

Read and follow:
- .release_review/CAPTAIN_PROTOCOL.md
- .release_review/PRE_RELEASE_REVIEW_PLAN.md
- .release_review/SEVERITY_POLICY.md

Task:
1) Initialize run workspace
2) Execute release gates
3) Build verdict
4) Drive component reviews until verdict is READY or READY_WITH_ACCEPTED_RISKS

Inputs:
- VERSION: <VERSION>
- RELEASE_ID: <VERSION>-<YYYYMMDD_HHMMSS>

Commands:
python3 .release_review/automation/generate_tracking.py --version <VERSION> --release-id <RELEASE_ID> --base-branch main
python3 .release_review/automation/release_gate_runner.py --release-id <RELEASE_ID> --strict
python3 .release_review/automation/build_release_verdict.py --release-id <RELEASE_ID>

Rules:
- Continue autonomously; do not pause between components
- Use dual review for P0/P1 components
- Use model policy: Codex CLI 5.3 xhigh (captain), Codex CLI 5.3 high (primary), Claude Opus 4.6 extended (secondary), optional Copilot Gemini 3.1 Pro
- Keep all artifacts under .release_review/runs/<release_id>/
- Only captain updates PRE_RELEASE_REVIEW_TRACKING.md
```

## Monitor Progress

```bash
cat .release_review/PRE_RELEASE_REVIEW_TRACKING.md
cat .release_review/runs/<release_id>/gate_results/verdict.json
cat .release_review/runs/<release_id>/REVIEW_LOG.md
```

## Strict CI Mode

For release branch or tag CI:

```bash
python3 .release_review/automation/release_gate_runner.py --release-id <release_id> --strict
```
