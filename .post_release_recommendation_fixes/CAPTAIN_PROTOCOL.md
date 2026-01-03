# Captain Protocol: Multi-Agent Post-Release Recommendation Fixes

This protocol is written for **Claude Code CLI** acting as the fix orchestration captain ("captain"). The captain coordinates up to **3 concurrent fix threads** using:
- Claude Code (captain + optional worker instance)
- Codex CLI (OpenAI)
- GitHub Copilot CLI (via `gh copilot` or your local Copilot CLI)

The goal is to systematically implement post-release recommendations identified during release reviews, prioritized by impact and effort.

## Model Policy (Fixed Tool -> Model Mapping)

Configure the tools to use these models/settings for consistency:
- **Claude Code (captain + any Claude workers)**: **Claude Opus 4.5**
- **Codex CLI**: **ChatGPT 5.2 (GPT-5.2)** with **reasoning/effort = `xhigh`**
- **GitHub Copilot CLI**: **Gemini 3.0**

If a CLI cannot explicitly select the requested model, the captain must note the actual model used in the task's **Evidence** field.

### Model Capability Tiers (for task assignment)

| Tier | Models | Strengths | Best For |
|------|--------|-----------|----------|
| **Tier 1** | Claude Opus 4.5, ChatGPT 5.2 (xhigh) | Deep reasoning, complex analysis, thoroughness | Security fixes, complex refactoring, architecture changes |
| **Tier 2** | Gemini 3.0 | Fast, good at targeted edits, solid comprehension | Config updates, docs, simple fixes, test additions |

**Assignment Guidelines**:
- **ChatGPT 5.2**: Most thorough but slowest. Use for complex fixes where depth matters.
- **Claude Opus 4.5**: Excellent balance of depth and speed. Good for security fixes, refactoring.
- **Gemini 3.0**: Faster, slightly less thorough. Good for documentation, test additions, simple changes.

## Workflow vs Release Review Differences

| Aspect | Release Review | Post-Release Fixes |
|--------|---------------|-------------------|
| **Goal** | Find and fix issues before release | Implement known recommendations |
| **Input** | Codebase scan | `.release_review/<version>/POST_RELEASE_TODO.md` |
| **Scope** | Full codebase | Specific files per TODO item |
| **Blocking** | Can block release | Never blocks release |
| **Timeline** | Time-boxed (hours) | Ongoing (days/weeks) |
| **Branch** | release-review/<version> | fix/<issue-id>/<description> (version tracked in metadata) |

## Canonical Inputs / Artifacts

- **Source of TODOs**: `.release_review/<version>/POST_RELEASE_TODO.md`
- **Tracking board**: `.post_release_recommendation_fixes/<version>/TRACKING.md`
- **Session progress**: `.post_release_recommendation_fixes/<version>/sessions/<date>/PROGRESS.md`

## Versioning Rules (Required)

- Set `RR_VERSION=<version>` before running automation.
- Use only `.post_release_recommendation_fixes/<version>/...` paths for tracking and sessions.
- The version must match `.release_review/<version>/POST_RELEASE_TODO.md`.
- Legacy unversioned sessions under `.post_release_recommendation_fixes/sessions/` should be migrated manually.

## Workspace Structure (Versioned)

```
.post_release_recommendation_fixes/
├── CAPTAIN_PROTOCOL.md          # This file
├── START_WORKFLOW.md            # How to trigger workflow
├── USER_QUESTIONS.md            # Template for async questions
├── templates/                   # Shared templates
│   ├── PROGRESS.md              # Session progress template
│   └── artifacts/README.md      # Artifact template
├── automation/                  # Automation scripts
│   ├── todo_importer.py         # Import from POST_RELEASE_TODO.md
│   ├── progress_tracker.py      # Track completion status
│   └── effort_estimator.py      # Estimate fix effort
│
└── <version>/                   # Per-release workspace (required)
    ├── TRACKING.md              # Master tracking for this release
    ├── USER_QUESTIONS.md        # Async questions for this release
    └── sessions/                # Per-session workspace
        ├── TEMPLATE/            # Copied template (optional)
        └── <YYYYMMDD>/          # Session date
            ├── PROGRESS.md      # Session progress
            ├── artifacts/       # Agent outputs (git-ignored)
            │   └── <issue-id>/
            │       └── <model>/
            │           ├── analysis.md
            │           ├── implementation.md
            │           └── test_results.md
            └── USER_QUESTIONS.md  # Session-specific questions
```

## Preconditions (Before Starting)

Captain must verify:
- Source TODO file exists (e.g., `.release_review/v0.8.0/POST_RELEASE_TODO.md`)
- `RR_VERSION` is set and matches `.post_release_recommendation_fixes/<version>/TRACKING.md`
- `.release_review/<version>/PRE_RELEASE_REVIEW_TRACKING.md` exists and evidence is JSON-validated
- POST_RELEASE_TODO.md items align with release review followups/accepted risks
- Repo is clean: no uncommitted changes; on the correct base branch
- Test infrastructure is working (`make test` or `pytest` runs)
- Tool model settings match **Model Policy**

## Priority Framework

TODOs from POST_RELEASE_TODO.md already have priorities. Map them to fix order:

| Original Priority | Fix Priority | Time Box | Agent Tier |
|-------------------|--------------|----------|------------|
| High | P0 | 2-4 hours | Tier 1 |
| Medium | P1 | 1-2 hours | Tier 1 or 2 |
| Low | P2 | 30-60 min | Tier 2 |

## Orchestration Constraints (Hard Rules)

1. **Max 3 concurrent threads**. No more.
2. Each thread works on a **non-overlapping fix scope** (avoid merge conflicts).
3. **Only the captain updates** `.post_release_recommendation_fixes/<version>/TRACKING.md`.
4. Every completed fix is committed with:
   - A fix-scoped branch
   - Appropriate tests added/verified
   - Evidence recorded in tracking
5. The captain does **not** assume first-pass success: iterate, test, verify.
6. Evidence entries must be **machine-validated JSON** before marking Complete.

---

## ⚠️ CRITICAL: Agent Usage Requirement

**THE CAPTAIN MUST USE AGENTS TO IMPLEMENT FIXES. DO NOT IMPLEMENT FIXES DIRECTLY.**

This is the most important rule in the protocol. The captain's role is to:
- **Orchestrate** agents (dispatch, monitor, verify)
- **Track** progress in TRACKING.md
- **Verify** that fixes are correct
- **NOT** write code directly (except for minor tracking updates)

### Exception (Last Resort Only)

If all agents are blocked after 3 iterations (or the work is truly trivial), the captain may implement directly, but MUST:
- Document the exception and rationale in the session `PROGRESS.md`
- Record the exception in the fix Evidence (include reason + timestamp)
- Require a **secondary review** by a different model before marking Complete

### Why Agents?

1. **Parallelism**: Multiple fixes can run simultaneously
2. **Specialization**: Different agents have different strengths
3. **Auditability**: Agent outputs are tracked per-fix
4. **Efficiency**: Captain can focus on coordination, not implementation

### How to Spawn Agents

Use the **Task tool** with appropriate agent types:

```
# For implementing a fix
Task tool → subagent_type: "general-purpose"
Prompt: "Implement fix for issue 001: <description>. Branch: fix/001/desc. Files: <paths>"

# For exploring codebase
Task tool → subagent_type: "Explore"
Prompt: "Find all usages of <function> to understand impact of fix 001"

# For planning complex fixes
Task tool → subagent_type: "Plan"
Prompt: "Design implementation approach for fix 001: <description>"
```

### Agent Dispatch Checklist

Before dispatching an agent for a fix:
- [ ] Create the fix branch first (`git checkout -b fix/<id>/<desc>`)
- [ ] Identify exact file paths for the fix
- [ ] Write clear, complete prompt with all context
- [ ] Specify expected deliverables
- [ ] Tell agent NOT to update TRACKING.md

### Example Agent Prompt

```
Implement fix for issue 001: Add atomic file writes to storage module.

**Branch**: fix/001/atomic-writes (already created)
**Files to modify**: traigent/storage/file_ops.py, tests/unit/test_storage.py

**Issue**: File writes can be corrupted if interrupted mid-write.
**Recommendation**: Use temp file + atomic rename pattern.

**Requirements**:
1. Modify write_file() to use tempfile + os.replace()
2. Add test for interrupted write scenario
3. Run tests: TRAIGENT_MOCK_LLM=true .venv/bin/python -m pytest tests/unit/test_storage.py
4. Report test results

**Constraints**:
- Do NOT edit TRACKING.md
- Keep changes minimal
- Follow existing code patterns

**Deliverables**:
- Summary of changes made
- Test command and results
- Any issues encountered
```

## Standard Loop ("Captain Tick")

Repeat until all targeted fixes are complete.

### Step 1: Import TODOs (First Session Only)

```python
# Import from release review
from todo_importer import TodoImporter
version = "v0.8.0"
importer = TodoImporter(
    source=f".release_review/{version}/POST_RELEASE_TODO.md",
    version=version,
)
importer.import_to_tracking(f".post_release_recommendation_fixes/{version}/TRACKING.md")
```

Or manually copy items from POST_RELEASE_TODO.md to TRACKING.md.

### Step 2: Select Work (3 threads max)

From TRACKING.md:
- Pick the highest Priority items with Status = Pending
- Prefer fixes that affect the same subsystem (batch for efficiency)
- Ensure chosen fixes are disjoint (different files)

### Step 3: Claim + Branch

For each selected fix:
- Create a branch: `fix/<issue-id>/<short-description>`
  - Example: `fix/001/atomic-writes-storage`
  - Base branch: `RR_BASE_BRANCH` (defaults to `main`)
- In TRACKING.md, set:
  - Owner = assigned agent
  - Status = In Progress
  - Start Time = ISO-8601 timestamp

### Step 4: Dispatch Agents

Dispatch up to 3 agents (parallel). For each agent, provide:
- **Issue ID and description** from TRACKING.md
- **Exact file paths** to modify
- **Expected deliverable** (code changes + tests + evidence)
- **No tracking-file edits** (captain owns tracking)

### Step 5: Verify + Merge (Captain)

For each agent's output:
- Run tests to verify fix works
- Check that fix doesn't introduce regressions
- Review code quality
- If acceptable:
  - Merge fix branch to main (or target branch)
  - Update TRACKING.md:
    - Status = Complete
    - Commit SHA
    - Tests run
    - Evidence (machine-validated JSON)
  - Validate evidence: `.release_review/automation/evidence_validator.py --file .post_release_recommendation_fixes/<version>/TRACKING.md`
- If not acceptable:
  - Send back to agent for iteration OR
  - Reassign to different agent

### Step 6: Update Source TODO

After fix is merged:
- Check the item as complete in `.release_review/<version>/POST_RELEASE_TODO.md`
- Update the tracking counts table

### Step 7: Repeat

Continue until all targeted fixes are done or session ends.

## Continuous Execution Protocol

**Core principle**: The captain runs start-to-finish without waiting for user approval between fixes. User interaction is asynchronous and non-blocking.

### Autonomous Decision Making

The captain SHOULD make judgment calls autonomously for:
- Fix prioritization within the same tier
- Agent assignment (following capability tiers)
- Minor scope adjustments needed for the fix
- Test strategy decisions

The captain MUST escalate to user for:
- Fixes that would change public API
- Fixes that require new dependencies
- Fixes that exceed estimated effort by 3x
- Architectural decisions not covered by original recommendation

### Non-Blocking User Questions

When the captain needs user input but can continue working:

1. **Write question** to `.post_release_recommendation_fixes/<version>/USER_QUESTIONS.md`:
   ```markdown
   ## Question <N> (<ISO-8601 timestamp>)
   **Status**: PENDING
   **Blocking**: Yes/No
   **Issue ID**: <affected fix>
   **Question**: <clear, specific question>
   **Options**:
   - Option A: <description>
   - Option B: <description>
   **Captain's Recommendation**: <what captain would do if no answer>
   **Deadline**: <when captain will proceed with recommendation>

   ### User Answer
   (User fills this in)
   ```

2. **Continue working** on fixes not affected by the question

3. **Check periodically** (every 2-3 fixes)

4. **Document decision** in TRACKING.md

### Stop Conditions

Captain continues until ONE of:
- All targeted fixes are Complete
- User explicitly says "stop" or "pause"
- A fix requires human architectural decision (document and wait)
- Session time limit reached (user-defined)

## Agent Prompt Templates

### Template A: Implement Fix

**Task**: Implement fix for issue: `<ISSUE_ID>` - `<DESCRIPTION>`

**Source**: From `.release_review/<version>/POST_RELEASE_TODO.md`

**Scope**: `<PATHS>` (do not edit outside scope unless strictly required)

**Goals**:
1. Implement the recommended fix
2. Add/update tests to cover the change
3. Run tests and verify no regressions
4. Document what was changed

**Constraints**:
- Do NOT edit TRACKING.md (captain will update)
- Keep changes minimal and focused
- Follow existing code patterns
- Add tests for the fix

**Deliverable**:
- Git-ready patch on branch `fix/<ISSUE_ID>/<description>`
- Test evidence (commands run, results)
- Brief summary of changes

### Template B: Review Fix (Secondary)

**Task**: Review the fix on branch `<BRANCH_NAME>` for issue `<ISSUE_ID>`.

**Focus**:
- Does fix actually address the recommendation?
- Any regressions introduced?
- Test coverage adequate?
- Code quality acceptable?

**Deliverable**:
- "Approve" or "Request changes"
- If changes needed: specific actionable feedback

## Evidence Policy

Evidence MUST be machine-validated JSON (same schema as release review).
- Use `format=standard` for new fixes (required)
- `format=legacy` is allowed only for historical backfills

Required fields:
- `format`, `commits`, `tests`, `models`, `reviewer`, `timestamp`, `followups`, `accepted_risks`
Optional fields:
- `legacy_summary` (legacy format only, for human-readable backfill notes)

Example (standard):
```json
{"format":"standard","commits":["def5678"],"tests":{"command":"TRAIGENT_MOCK_LLM=true .venv/bin/python -m pytest tests/unit/test_storage.py -q","status":"PASS","passed":12,"total":12},"models":"Claude Opus 4.5","reviewer":"captain","timestamp":"2025-12-14T10:30:00Z","followups":"None","accepted_risks":"None"}
```
Record before/after details in fix artifacts or Notes (not inside evidence JSON).

### Evidence Backfill (Legacy)

If legacy entries lack commit/test details, backfill before closing the release:
- Use `git log -- <paths>` or `git blame` to identify commit SHAs
- Update legacy evidence to `format=standard` with real commits/tests/reviewer/timestamp
- Re-validate with `.release_review/automation/evidence_validator.py --file .post_release_recommendation_fixes/<version>/TRACKING.md`

## Session Handoff Protocol

Before ending a session, the captain MUST:

1. **Commit all work**: Ensure all fix branches are pushed
2. **Update tracking**: Set correct Status for all in-flight fixes
3. **Log progress**: Update session PROGRESS.md with:
   - Fixes completed
   - Fixes in progress
   - Next priority items
4. **Note blockers**: Document any issues blocking progress

When resuming:
1. Read TRACKING.md to understand current state
2. Read last session's PROGRESS.md
3. Continue from highest-priority pending fix

## Iteration Limits

- Max **3 iterations** per fix per agent
- If agent fails 3 times: switch agents or escalate
- Don't spend >1 hour on a single fix before reassessing

## Verification Checklist

Before marking any fix as Complete:

- [ ] Fix addresses the original recommendation
- [ ] Tests pass (new and existing)
- [ ] No regressions in related functionality
- [ ] Code follows project conventions
- [ ] Changes are documented (if public API)
- [ ] Original POST_RELEASE_TODO.md updated

## Metrics Tracking

Track effectiveness across sessions:

| Metric | Description |
|--------|-------------|
| Fixes completed | Total fixes done |
| Time per fix | Average time to complete |
| Agent success rate | First-attempt success by agent |
| Iteration rate | Fixes needing multiple attempts |
| Regression rate | Fixes that caused new issues |

## Integration with Release Review

This workflow feeds back into the release review process:

1. **Version lock**: Post-release fixes must target the same `<version>` as the source release review
2. **Evidence schema**: Use the same JSON evidence schema and validator as release review
3. **Fixes inform future reviews**: Patterns in fixes guide what to check
4. **Regression tracking**: If a fix causes issues, note for next review
5. **Effort estimation**: Actual vs estimated effort improves planning
6. **Agent performance**: Track which agents are best for which fix types

---

## Rollback Protocol

When a merged fix causes issues:

### Step 1: Identify the Problem
- Confirm regression is caused by specific fix
- Note the commit SHA to revert

### Step 2: Revert the Fix
```bash
.venv/bin/python .post_release_recommendation_fixes/automation/git_helper.py revert <commit>
```

### Step 3: Update Tracking
In TRACKING.md, change fix status:
```
Status: Reverted
Evidence: Reverted in <revert-sha>. Reason: <description>
```

### Step 4: Document in History
Add to item details:
```markdown
#### History
- <timestamp> - Completed by <agent> (commit <sha>)
- <timestamp> - Reverted: <reason>
```

### Step 5: Re-queue if Needed
If fix is still needed:
1. Create new TODO with "Fix #N (retry)"
2. Assign to different agent
3. Reference original failure in issue

### Revert Status Flow
```
Pending → In Progress → Complete → Reverted → Re-queued
```

---

## Escalation Protocol

When agents fail repeatedly:

### Escalation Ladder
1. **Retry same agent** (different prompt, more context)
2. **Switch to different agent** (prefer higher tier)
3. **Mark as Blocked** with detailed blockers
4. **Escalate to user** via `.post_release_recommendation_fixes/<version>/USER_QUESTIONS.md`

### Blocked Status Template
```markdown
Status: Blocked
Blocker: <description>
Attempted: <agent1> (3x), <agent2> (2x)
Needs: <what would unblock this>
```

### User Escalation Template
```markdown
## Question N (timestamp)
**Status**: PENDING
**Blocking**: Yes
**Issue ID**: <fix-id>
**Question**: Fix blocked after multiple attempts
**Context**:
- Agent 1 failed because: <reason>
- Agent 2 failed because: <reason>
**Options**:
- Option A: Defer to next release
- Option B: Accept partial fix
- Option C: Manual implementation needed
**Captain's Recommendation**: <recommendation>
**Deadline**: <timestamp>
```

---

## Automation Toolkit

Location: `.post_release_recommendation_fixes/automation/`

**IMPORTANT**: Always use `.venv/bin/python` (not bare `python`) to ensure correct environment.
All scripts read `RR_VERSION` to resolve versioned paths (override with `--version` where supported).

| Script | Purpose | Usage |
|--------|---------|-------|
| `todo_importer.py` | Import from POST_RELEASE_TODO.md | `.venv/bin/python todo_importer.py .release_review/<version>/POST_RELEASE_TODO.md --yes --version <version>` |
| `progress_tracker.py` | Track completion status | `.venv/bin/python progress_tracker.py --version <version> stats` |
| `effort_estimator.py` | Estimate fix effort | `.venv/bin/python effort_estimator.py --version <version> report` |
| `session_init.py` | Initialize session | `.venv/bin/python session_init.py --version <version> init high` |
| `preflight_check.py` | Verify prerequisites | `.venv/bin/python preflight_check.py --version <version>` |
| `conflict_detector.py` | Detect parallel conflicts | `.venv/bin/python conflict_detector.py batch` |
| `git_helper.py` | Branch/merge/revert utilities | `.venv/bin/python git_helper.py branch 001 desc` |

### Script Options (Automation-Friendly)

Supported flags (automation-friendly):
- `--yes` / `-y`: Skip confirmation prompts
- `--quiet` / `-q`: Suppress verbose output
- `--dry-run`: Preview changes without executing

Notes:
- `todo_importer.py` supports `--yes`, `--quiet`, `--dry-run`.
- `preflight_check.py` supports `--quiet`.
- `git_helper.py test` uses `RR_TEST_COMMAND` if set.

Example (fully automated import):
```bash
export RR_VERSION=v0.8.0
.venv/bin/python .post_release_recommendation_fixes/automation/todo_importer.py \
  .release_review/$RR_VERSION/POST_RELEASE_TODO.md --yes --quiet --version "$RR_VERSION"
```

### Pre-Session Checklist
```bash
# Version lock
export RR_VERSION=v0.8.0
export RR_BASE_BRANCH=main

# 1. Pre-flight check
.venv/bin/python .post_release_recommendation_fixes/automation/preflight_check.py --version "$RR_VERSION"

# 2. Initialize session (auto-confirm)
.venv/bin/python .post_release_recommendation_fixes/automation/session_init.py --version "$RR_VERSION" init high

# 3. Check for conflicts
.venv/bin/python .post_release_recommendation_fixes/automation/conflict_detector.py batch
```

### During Session
```bash
# Create fix branch
.venv/bin/python .post_release_recommendation_fixes/automation/git_helper.py branch 001 "atomic-writes"

# Run tests (use mock mode!)
export RR_TEST_COMMAND="TRAIGENT_MOCK_LLM=true .venv/bin/python -m pytest tests/ -q"
TRAIGENT_MOCK_LLM=true .venv/bin/python -m pytest tests/ -v

# Format and lint
make format && make lint

# Check progress
.venv/bin/python .post_release_recommendation_fixes/automation/progress_tracker.py --version "$RR_VERSION" stats
```

### Post-Fix
```bash
# Merge fix
.venv/bin/python .post_release_recommendation_fixes/automation/git_helper.py merge fix/001/atomic-writes

# Validate evidence JSON
.venv/bin/python .release_review/automation/evidence_validator.py \
  --file .post_release_recommendation_fixes/$RR_VERSION/TRACKING.md

# Clean up branch
.venv/bin/python .post_release_recommendation_fixes/automation/git_helper.py delete fix/001/atomic-writes
```

---

## Quick Reference

### Create New Session (Automated)
```bash
export RR_VERSION=v0.8.0
.venv/bin/python .post_release_recommendation_fixes/automation/session_init.py --version "$RR_VERSION" init high
```

### Create New Session (Manual)
```bash
mkdir -p .post_release_recommendation_fixes/$RR_VERSION/sessions/$(date +%Y%m%d)/artifacts
```

### Start Fix Branch (Automated)
```bash
.venv/bin/python .post_release_recommendation_fixes/automation/git_helper.py branch 001 "description"
```

### Start Fix Branch (Manual)
```bash
git checkout -b fix/<issue-id>/<description> "$RR_BASE_BRANCH"
```

### Run Tests
```bash
TRAIGENT_MOCK_LLM=true .venv/bin/python -m pytest tests/ -v
```

### Format and Lint
```bash
make format && make lint
```

### Mark Fix Complete
Update TRACKING.md with Status = Complete, add JSON evidence.

### Update Original TODO
Check the box in `.release_review/<version>/POST_RELEASE_TODO.md`.

---

## Lessons Learned (Common Pitfalls)

This section captures issues encountered during actual workflow execution.

### 1. Environment Issues

**Problem**: `python: command not found` or wrong Python used.

**Solution**: Always use `.venv/bin/python` explicitly:
```bash
# WRONG
python script.py

# CORRECT
.venv/bin/python script.py
```

### 2. Temp Directory Issues (Pytest)

**Problem**: `pytest` fails with `FileNotFoundError: No usable temporary directory found ...`

**Solution**: Use a repo-local temp dir:
```bash
mkdir -p .post_release_recommendation_fixes/$RR_VERSION/tmp
export TMPDIR=$PWD/.post_release_recommendation_fixes/$RR_VERSION/tmp
```

Then re-run tests (still in mock mode):
```bash
TRAIGENT_MOCK_LLM=true .venv/bin/python -m pytest tests/ -v
```

### 3. Interactive Prompts in Automation

**Problem**: Scripts requiring confirmation fail with `EOFError` when run non-interactively.

**Solution**: Use `--yes` flag for automated execution:
```bash
# WRONG (will fail in non-interactive context)
.venv/bin/python todo_importer.py source.md

# CORRECT
.venv/bin/python todo_importer.py source.md --yes
```

### 4. Captain Implementing Directly

**Problem**: Captain writes code directly instead of dispatching agents, losing parallelism and auditability.

**Solution**: ALWAYS use agents. See "⚠️ CRITICAL: Agent Usage Requirement" section above.

**Self-check question**: "Am I about to write code? If yes, spawn an agent instead."

### 5. Forgetting Mock Mode for Tests

**Problem**: Tests fail or make real API calls without mock mode.

**Solution**: Always prefix test commands with `TRAIGENT_MOCK_LLM=true`:
```bash
TRAIGENT_MOCK_LLM=true .venv/bin/python -m pytest tests/ -v
```

### 6. Not Running Pre-flight Checks

**Problem**: Session starts with missing prerequisites, causing failures mid-fix.

**Solution**: Always run pre-flight check first:
```bash
.venv/bin/python .post_release_recommendation_fixes/automation/preflight_check.py --version "$RR_VERSION"
```

### 7. Tracking File Conflicts

**Problem**: Multiple agents update TRACKING.md simultaneously, causing conflicts.

**Solution**: ONLY the captain updates TRACKING.md. Tell agents explicitly:
> "Do NOT edit TRACKING.md"

### 8. Branches Not Cleaned Up

**Problem**: Fix branches accumulate, cluttering the repo.

**Solution**: Delete branches after successful merge:
```bash
.venv/bin/python .post_release_recommendation_fixes/automation/git_helper.py delete fix/001/desc
```

### 9. Using `.venv/bin/pytest` Instead of `python -m pytest`

**Problem**: `.venv/bin/pytest` has a shebang pointing to a non-existent Python path (happens when venv is copied/moved).

**Solution**: ALWAYS use `python -m pytest`:
```bash
# WRONG (shebang may be broken)
.venv/bin/pytest tests/
TRAIGENT_MOCK_LLM=true .venv/bin/pytest tests/

# CORRECT
.venv/bin/python -m pytest tests/
TRAIGENT_MOCK_LLM=true .venv/bin/python -m pytest tests/
```

### 10. Skipping Conflict Check Before Parallel Dispatch

**Problem**: Dispatching agents in parallel without checking for file conflicts can cause merge conflicts.

**Solution**: ALWAYS run conflict check before dispatching parallel agents:
```bash
.venv/bin/python .post_release_recommendation_fixes/automation/conflict_detector.py batch
```

### 11. Not Saving Agent Artifacts

**Problem**: Agent outputs not saved, losing audit trail and making debugging harder.

**Solution**: Save agent outputs to session artifacts:
```bash
mkdir -p .post_release_recommendation_fixes/$RR_VERSION/sessions/<date>/artifacts/<fix-id>/
# Save agent output to implementation.md in that folder
```

### 12. Non-JSON Evidence

**Problem**: Evidence in TRACKING.md is plain text, causing validation failures.

**Solution**: Always record JSON evidence and validate:
```bash
.venv/bin/python .release_review/automation/evidence_validator.py \
  --file .post_release_recommendation_fixes/$RR_VERSION/TRACKING.md
```

### 13. File Lock Portability

**Problem**: `progress_tracker.py` uses `fcntl` locks, which are Unix-only.

**Solution**: Run workflow on Unix-like systems, or replace locks with a cross-platform mechanism.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-13 | Initial protocol |
| 1.1 | 2025-12-13 | Added agent usage requirement, lessons learned, venv instructions |
| 1.2 | 2025-12-13 | Added pitfalls 9-11: pytest shebang, conflict check, artifacts |
| 1.3 | 2025-12-13 | Versioned workspace, JSON evidence, release review coupling |
| 1.4 | 2025-12-13 | Template location clarified, evidence backfill guidance, portability note |
