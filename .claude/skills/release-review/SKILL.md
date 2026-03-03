---
name: release-review
description: Execute a full pre-release review of the Traigent SDK. Coordinates multi-agent review with rotation scheduling, tracking, and evidence validation. Involves Codex for cross-model review.
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep, Skill(consult-codex), Skill(pr-local-checks)
---

# Release Review Skill

Execute a complete pre-release review of the Traigent SDK using the Captain Protocol. This skill replaces the manual `START_REVIEW.md` copy-paste workflow.

## Usage

- `/release-review v0.11.0` - Start a new release review for v0.11.0
- `/release-review v0.11.0 --round 5` - Specify the rotation round explicitly
- `/release-review --resume` - Resume an interrupted review session

## Prerequisites

Before starting a release review, ensure:

1. **Clean git state** - No uncommitted changes (`git status` is clean)
2. **On latest main** - `git pull origin main` is up to date
3. **Python venv activated** - `.venv/bin/python` works
4. **Codex CLI installed** - `codex --version` works (for cross-model review)
5. **SonarQube available** - Local or cloud instance accessible

## Workflow Overview

```
Step 1: Initialize   → Create branch, tag baseline
Step 2: Gen Tracking  → Run generate_tracking.py, create version workspace
Step 3: Rotation      → Run rotation_scheduler.py, determine round
Step 4: Read Protocol → Load CAPTAIN_PROTOCOL.md + PRE_RELEASE_REVIEW_PLAN.md
Step 5: Review Loop   → Priority-ordered dispatch with parallel agents
Step 6: Validation    → Run /pr-local-checks, full test suite, SonarQube
Step 7: Finalize      → Audit trail, tracking summary, human sign-off
```

---

## Step 1: Initialize Release Review Branch

Parse version from the argument or detect from `pyproject.toml`:

```bash
# Detect current version if not provided
VERSION=$(grep '^version = ' pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
echo "Current version: $VERSION"

# Create release review branch
git checkout main && git pull origin main
git checkout -b release-review/$VERSION

# Tag the baseline
git tag -a ${VERSION}-rc1 -m "Release candidate 1 for $VERSION"
echo "Tagged ${VERSION}-rc1 at $(git rev-parse --short HEAD)"
```

---

## Step 2: Generate Fresh Tracking File

**MANDATORY**: Always generate a fresh tracking file before every review. Never reuse old tracking.

```bash
# Create version workspace
mkdir -p .release_review/$VERSION/artifacts

# Generate fresh tracking with file hashes and component matrix
.venv/bin/python .release_review/automation/generate_tracking.py \
  --version $VERSION \
  --output .release_review/$VERSION/PRE_RELEASE_REVIEW_TRACKING.md

# Create symlink to canonical location
ln -sf $VERSION/PRE_RELEASE_REVIEW_TRACKING.md .release_review/PRE_RELEASE_REVIEW_TRACKING.md

# Create trace log
cat > .release_review/$VERSION/TRACE_LOG.md << 'TRACE'
# Release Review Trace Log

- **Version**: $VERSION
- **Captain**: Claude Code (Opus 4.6)
- **Started**: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- **Status**: IN PROGRESS

## Agent Spawns
| Timestamp | Agent | Model | Component | Status |
|-----------|-------|-------|-----------|--------|
TRACE

# Create user questions file
cat > .release_review/$VERSION/USER_QUESTIONS.md << 'UQ'
# User Questions — Release Review $VERSION

No questions yet. Captain will write questions here if user input is needed.
UQ

echo "Tracking generated. Review workspace at .release_review/$VERSION/"
```

---

## Step 3: Generate Rotation Schedule

Determine the round number from rotation history and generate a fresh schedule:

```bash
# Check how many previous rounds exist
ROUND=$(.venv/bin/python -c "
import json
from pathlib import Path
history = Path('.release_review/rotation_history.json')
if history.exists():
    data = json.loads(history.read_text())
    print(len(data.get('rounds', [])) + 1)
else:
    print(1)
")
echo "This is round $ROUND"

# Generate rotation schedule
.venv/bin/python .release_review/automation/rotation_scheduler.py generate $ROUND $VERSION

# Save rotation history for this version
.venv/bin/python .release_review/automation/rotation_scheduler.py generate $ROUND $VERSION \
  > .release_review/$VERSION/ROTATION_HISTORY.md
```

---

## Step 4: Read Protocol and Plan

Read the full protocol and review plan to understand the process:

```
Read these files in order:
1. .release_review/CAPTAIN_PROTOCOL.md    (orchestration rules, evidence format, conflict tiers)
2. .release_review/PRE_RELEASE_REVIEW_PLAN.md  (component checklists, release gates, scope)
3. .release_review/$VERSION/PRE_RELEASE_REVIEW_TRACKING.md  (generated tracking with components)
4. .release_review/$VERSION/ROTATION_HISTORY.md  (which models review which categories)
```

Key rules from the protocol:
- You (Claude Code) are the **captain** — you orchestrate, agents review
- Max **3 concurrent agents** at a time
- **Priority order**: P0 (90-100) first, then P1 (75-89), P2 (60-74), P3 (<60)
- Evidence must be **machine-validated JSON** in the tracking table
- **Cross-model review**: P0/P1 secondary reviewer MUST be a different model than lead

---

## Step 5: Execute Review Loop (Captain Tick)

This is the core loop. Repeat until all components are Approved:

### 5a. Select Work

Sort components by priority score (descending). Select the top 3 unreviewed components that can be assigned to different agents without scope overlap.

### 5b. Dispatch Agents

**For Claude Code worker agents** (Task tool):

```
Use the Task tool to spawn a review agent with this prompt:

"You are reviewing the <COMPONENT> component for Traigent SDK $VERSION release.

Scope (ONLY touch these files): <FILE_PATHS>

Checklist:
<PASTE_COMPONENT_CHECKLIST_FROM_PLAN>

Instructions:
1. Read all files in scope
2. Run relevant tests: pytest <TEST_PATHS> -q
3. Check for: security issues, API correctness, thread safety, error handling
4. Record findings in this format:
   - Issue title, severity, file:line, whether you fixed it
5. Return your findings and test results"
```

**For Codex reviews** (via `/consult-codex`):

```bash
# Use the consult-codex skill for Codex agent dispatching
# Prepare the review request:

codex exec -m GPT-5.3-Codex -c 'model_reasoning_effort="xhigh"' "
# Release Review: <COMPONENT>
## Version: $VERSION
## Scope
<FILE_PATHS>

## Checklist
<PASTE_COMPONENT_CHECKLIST>

## Instructions
1. Review all files in scope for: security, correctness, thread safety, error handling
2. Run tests: pytest <TEST_PATHS> -q
3. Report findings with severity, file:line, and whether you can fix them
4. Provide evidence JSON with test results

## Evidence Format
{
  \"format\": \"standard\",
  \"commits\": [],
  \"tests\": {\"command\": \"...\", \"status\": \"PASS/FAIL\", \"passed\": N, \"total\": N},
  \"models\": \"GPT-5.3\",
  \"reviewer\": \"codex\",
  \"timestamp\": \"ISO-8601\",
  \"followups\": \"None\",
  \"accepted_risks\": \"None\"
}
" 2>&1 | tee .release_review/$VERSION/artifacts/<component>/codex/$(date +%Y%m%d)_findings.md
```

**For GitHub Copilot / Gemini 3 Pro** (manual, optional):

Copilot reviews require a separate terminal. Write the prompt to a file and instruct the user:

```bash
# Write prompt for Copilot review
cat > /tmp/copilot_review_prompt.md << 'EOF'
Review <COMPONENT> for release readiness...
EOF
echo "Run in another terminal: gh copilot explain < /tmp/copilot_review_prompt.md"
```

### 5c. Integrate Results

After agents return:

1. **Validate scope** — Run scope guard to ensure agents only touched allowed files:
   ```bash
   .venv/bin/python .release_review/automation/scope_guard.py \
     --allowed "<FILE_PATTERNS>" --diff HEAD~1
   ```

2. **Validate evidence** — Parse and validate the evidence JSON:
   ```bash
   .venv/bin/python .release_review/automation/evidence_validator.py \
     --evidence '<JSON>'
   ```

3. **Spot-check tests** — Re-run tests on 20% of reviewed components:
   ```bash
   bash .release_review/automation/verify_tests.sh <TEST_COMMAND>
   ```

4. **Update tracking** — Record status, evidence, and notes in the tracking file

5. **Commit progress** — One commit per component:
   ```bash
   git add -A && git commit -m "release-review($VERSION): approve <component>"
   ```

### 5d. Repeat

Continue the loop until:
- All P0 and P1 components are Approved
- Then process P2 and P3
- Stop when all components are Approved or a Tier 3 conflict requires human escalation

---

## Step 6: Run Final Validation

After all components are approved, run the full validation suite:

```bash
# Use /pr-local-checks skill for comprehensive validation
# This covers: formatting, linting, security, tests, coverage, SonarQube

# Or run manually:
make format && make lint

# Full test suite
TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true pytest tests/ \
  --ignore=tests/unit/core/test_orchestrator.py \
  --ignore=tests/unit/evaluators/test_litellm_integration.py \
  -q

# Version consistency check
python -c "
import traigent
import tomllib
with open('pyproject.toml', 'rb') as f:
    v = tomllib.load(f)['project']['version']
print(f'pyproject.toml: {v}')
print(f'traigent.__version__: {traigent.__version__}')
assert v == traigent.__version__, 'VERSION MISMATCH!'
print('Version check: PASS')
"

# Packaging smoke test
pip install -e ".[dev,all]" && traigent --version
```

### Release Gates Checklist

- [ ] All P0/P1 components Approved in tracking file
- [ ] `pytest` passes (unit + integration + e2e)
- [ ] `ruff check` and `mypy` pass
- [ ] Mock mode works fully offline
- [ ] Version consistency across pyproject.toml and `traigent.__version__`
- [ ] Packaging smoke test passes (build + install + CLI)
- [ ] SonarQube Quality Gate passes

---

## Step 7: Finalize and Audit Trail

```bash
# Generate audit trail
cat > .release_review/$VERSION/AUDIT_TRAIL.md << EOF
# Release Review Audit Trail — $VERSION

## Summary
- **Captain**: Claude Code (Opus 4.6)
- **Round**: $ROUND
- **Duration**: [start] to [end]
- **Components Reviewed**: [count]
- **Issues Found**: [count]
- **Issues Fixed**: [count]
- **Accepted Risks**: [count or None]

## Release Gates
- pytest: PASS ([N] passed, [N] skipped)
- ruff check: PASS
- mypy: PASS
- Mock mode: PASS
- Version consistency: PASS
- Packaging smoke: PASS
- SonarQube: PASS

## Sign-Off
- Release Captain: Claude Code (Opus 4.6) — [timestamp]
- Human Release Owner: __________________ — ____/____/____
EOF

# Commit final state
git add -A && git commit -m "release-review($VERSION): finalize audit trail"

echo "Release review complete. Awaiting human sign-off."
echo "Please review: .release_review/$VERSION/AUDIT_TRAIL.md"
```

---

## Resume Protocol

To resume an interrupted review session:

```bash
# Pull the review branch
git checkout release-review/$VERSION
git pull origin release-review/$VERSION

# Read current state
# 1. Check tracking file for component statuses
# 2. Read the Review Notes Log for last activity
# 3. Check USER_QUESTIONS.md for pending questions

# Continue from where you left off — the tracking file is the source of truth
```

---

## Async Question Protocol

When you need user input but can continue working:

1. Write the question to `.release_review/$VERSION/USER_QUESTIONS.md`
2. Continue reviewing components that don't depend on the answer
3. Check the file every 2-3 batches for answers
4. If no answer after the deadline, use best judgment and document the decision

---

## Model Policy (Quick Reference)

| Tool | Model | CLI Flag | Tier |
|------|-------|----------|------|
| Claude Code (Captain) | Claude Opus 4.6 | Default | Tier 1 |
| Codex CLI | GPT-5.3 | `codex exec -m GPT-5.3-Codex -c 'model_reasoning_effort="xhigh"'` | Tier 1 |
| GitHub Copilot CLI | Gemini 3 Pro | `gh copilot` (manual) | Tier 2 |

**Assignment guidelines:**
- **GPT-5.3**: Most thorough. Use for complex/large components (P0 items)
- **Claude Opus 4.6**: Excellent depth + speed. Security analysis, orchestration
- **Gemini 3 Pro**: Fast. Good for packaging, docs, P2/P3 items

**Cross-model rule**: P0/P1 secondary reviewer MUST be a different model than the lead.

---

## Conflict Resolution

| Tier | Type | Resolution | Time Limit |
|------|------|------------|------------|
| 1 | Minor (style, non-critical) | Captain decides | < 10 min |
| 2 | Moderate (severity disagreement) | Captain verifies, documents | < 30 min |
| 3 | BLOCKING (security, data risk) | Escalate to human | Release blocked |

---

## Troubleshooting

### Codex CLI times out
Use shorter prompts or split the component into smaller sub-scopes. Reduce `model_reasoning_effort` to `"high"` for faster results.

### Flaky tests
Known flaky tests to always skip:
- `tests/unit/core/test_orchestrator.py` (hangs at ~33%)
- `tests/unit/evaluators/test_litellm_integration.py` (pydantic KeyError)
Add `--ignore=` flags for these.

### Scope guard false positive
If `scope_guard.py` flags legitimate changes (e.g., shared __init__.py), override with `--allow-extra <path>` and document the reason.

### SonarQube not available
At minimum, run coverage locally:
```bash
pytest tests/unit/ --cov=traigent --cov-report=term-missing --cov-fail-under=80
```

---

## Related Skills

- `/pr-local-checks` — Run comprehensive local CI checks
- `/consult-codex` — Dispatch work to Codex (GPT-5.3)
- `/code-review` — Request structured code reviews from Codex
- `/write-tests` — Write high-quality tests with anti-pattern detection
- `/sonarcloud-issues` — Browse and analyze SonarCloud issues
