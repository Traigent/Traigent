# Captain Protocol: Multi-Agent Pre-Release Review Orchestration

This protocol is written for **Claude Code CLI** acting as the release-review captain (“captain”). The captain coordinates up to **3 concurrent review threads** using:
- Claude Code (captain + optional worker instance)
- Codex CLI (OpenAI)
- GitHub Copilot CLI (via `gh copilot` or your local Copilot CLI)

The goal is to drive the repo to release readiness by reviewing/fixing components, updating tracking, and committing progress incrementally.

## Model Policy (Fixed Tool → Model Mapping)

Configure the tools to use these models/settings for consistency:
- **Claude Code (captain + any Claude workers)**: **Claude Opus 4.5**
- **Codex CLI**: **ChatGPT 5.2 (GPT-5.2)** with **reasoning/effort = `xhigh`**
- **GitHub Copilot CLI**: **Gemini 3.0**

If a CLI cannot explicitly select the requested model, the captain must note the actual model used in the component's **Evidence** field in `.release_review/PRE_RELEASE_REVIEW_TRACKING.md`.

### Model Capability Tiers (for task assignment)

| Tier | Models | Strengths | Best For |
|------|--------|-----------|----------|
| **Tier 1** | Claude Opus 4.5, ChatGPT 5.2 (xhigh) | Deep reasoning, complex analysis, thoroughness | Security-critical code, complex orchestration, cross-cutting concerns, P0 components |
| **Tier 2** | Gemini 3.0 | Fast, good at targeted edits, solid comprehension | Config files, docs, simpler components, P2/P3 items |

**Assignment Guidelines**:
- **ChatGPT 5.2**: Most thorough but slowest. Use for complex/large components where depth matters more than speed.
- **Claude Opus 4.5**: Excellent balance of depth and speed. Good for security analysis, orchestration review, captain duties.
- **Gemini 3.0**: Faster, slightly less thorough. Good for packaging, docs, smaller scopes.

### Cross-Model Review Policy

For P0/P1 components, the **secondary reviewer MUST be a different model** than the lead:

| Lead Model | Secondary Reviewer |
|------------|-------------------|
| ChatGPT 5.2 (Codex) | Claude Opus 4.5 |
| Claude Opus 4.5 | ChatGPT 5.2 (Codex) |
| Gemini 3.0 (Copilot) | Claude Opus 4.5 or ChatGPT 5.2 |

This ensures diverse perspectives catch issues a single model might miss.

### Role Rotation Protocol (Multi-Round Reviews)

**Purpose**: Eliminate model bias by rotating which model reviews which component type across releases or review rounds.

#### When to Use Role Rotation

| Scenario | Rotation Required |
|----------|-------------------|
| First release review | No (establish baseline) |
| Subsequent releases | Yes (rotate from previous) |
| Re-review after major changes | Yes (fresh perspective) |
| Audit/compliance review | Yes (mandatory rotation) |
| Post-incident review | Yes (different model than original) |

#### Rotation Matrix

Define component categories and rotate model assignments:

```
Round 1 (e.g., v0.8.0):
┌─────────────────────┬─────────────┬─────────────┬─────────────┐
│ Category            │ Primary     │ Secondary   │ Spot-Check  │
├─────────────────────┼─────────────┼─────────────┼─────────────┤
│ Security/Core       │ Claude      │ GPT-5.2     │ Gemini      │
│ Integrations        │ GPT-5.2     │ Claude      │ Gemini      │
│ Packaging/CI        │ Gemini      │ Claude      │ GPT-5.2     │
│ Docs/Examples       │ Gemini      │ GPT-5.2     │ Claude      │
└─────────────────────┴─────────────┴─────────────┴─────────────┘

Round 2 (e.g., v0.9.0 or re-review):
┌─────────────────────┬─────────────┬─────────────┬─────────────┐
│ Category            │ Primary     │ Secondary   │ Spot-Check  │
├─────────────────────┼─────────────┼─────────────┼─────────────┤
│ Security/Core       │ GPT-5.2     │ Gemini      │ Claude      │
│ Integrations        │ Claude      │ Gemini      │ GPT-5.2     │
│ Packaging/CI        │ Claude      │ GPT-5.2     │ Gemini      │
│ Docs/Examples       │ GPT-5.2     │ Claude      │ Gemini      │
└─────────────────────┴─────────────┴─────────────┴─────────────┘

Round 3 (rotate again):
┌─────────────────────┬─────────────┬─────────────┬─────────────┐
│ Category            │ Primary     │ Secondary   │ Spot-Check  │
├─────────────────────┼─────────────┼─────────────┼─────────────┤
│ Security/Core       │ Gemini      │ Claude      │ GPT-5.2     │
│ Integrations        │ Gemini      │ GPT-5.2     │ Claude      │
│ Packaging/CI        │ GPT-5.2     │ Gemini      │ Claude      │
│ Docs/Examples       │ Claude      │ Gemini      │ GPT-5.2     │
└─────────────────────┴─────────────┴─────────────┴─────────────┘
```

#### Generating a Rotation Schedule

Use the automation script to generate rotations:

```python
from rotation_scheduler import RotationScheduler

scheduler = RotationScheduler(
    models=["Claude Opus 4.5", "GPT-5.2", "Gemini 3.0"],
    categories=["Security/Core", "Integrations", "Packaging/CI", "Docs/Examples"]
)

# Get schedule for round N
schedule = scheduler.get_schedule(round_number=2)
print(schedule.to_markdown())

# Or auto-rotate from previous release
schedule = scheduler.rotate_from("v0.8.0")
```

#### Tracking Rotation History

Maintain `.release_review/<version>/ROTATION_HISTORY.md`. The rotation scheduler writes an auto-generated block between markers; keep any manual notes outside the block:

```markdown
# Rotation History

## v0.8.0 (Round 1)
| Category | Primary | Secondary | Issues Found |
|----------|---------|-----------|--------------|
| Security/Core | Claude | GPT-5.2 | 2 |
| Integrations | GPT-5.2 | Claude | 0 |
| Packaging/CI | Gemini | Claude | 4 |
| Docs/Examples | Gemini | GPT-5.2 | 0 |

## v0.9.0 (Round 2) - Rotated
| Category | Primary | Secondary | Issues Found |
|----------|---------|-----------|--------------|
| Security/Core | GPT-5.2 | Gemini | TBD |
| Integrations | Claude | Gemini | TBD |
| ... | ... | ... | ... |
```

#### Benefits of Rotation

1. **Catches blind spots**: Each model has different strengths/weaknesses
2. **Prevents stale reviews**: Fresh eyes find new issues
3. **Model performance tracking**: Compare issue detection rates across assignments
4. **Audit compliance**: Demonstrates diverse review coverage
5. **Reduces bias**: No model "owns" a component permanently

#### Constraints

- **Tier 1 models** (Claude Opus, GPT-5.2) should always review P0 components
- **Tier 2 models** (Gemini) can be primary for P2/P3 but need Tier 1 secondary for P0/P1
- Captain can override rotation if a model has known weakness for specific component type
- Document any rotation overrides in tracking file

## Canonical Inputs / Artifacts

- Review plan: `.release_review/PRE_RELEASE_REVIEW_PLAN.md`
- Tracking board (source of truth): `.release_review/PRE_RELEASE_REVIEW_TRACKING.md`
- Release blockers: `RELEASE_BLOCKERS_TODO.md`

## Release Review Workspace (Versioned + Ignored Artifacts)

This repo uses `.release_review/` as a dedicated home for the review system:
- **Versioned (commit these)**: plan, tracking, captain protocol, rotation schedule/history (share status with the team).
- **Ignored (do not commit)**: `.release_review/<version>/artifacts/` (agent notes, scratch outputs).

Store any agent-generated files under:
- `.release_review/<version>/artifacts/<component>/<agent>/<YYYYMMDD>/...`

Examples:
- `.release_review/v0.9.0/artifacts/traigent-integrations/codex/20251213/notes.md`
- `.release_review/v0.9.0/artifacts/release-blockers/claude/20251213/fix-plan.md`

**Rule**: `.release_review/PRE_RELEASE_REVIEW_TRACKING.md` is the canonical state across sessions. If chat history is lost, restart from that file.

## Preconditions (Before Starting)

Captain must verify:
- Tooling installed and reachable: `git`, `pytest`, `ruff`, `mypy` (as configured), `codex` CLI, Claude Code CLI, Copilot CLI.
- Repo is clean: no uncommitted changes; on the correct branch.
- You have permission to run tests/builds and (if needed) access network/API keys.
- You know which branch to merge into (e.g., `main` / `release`).
- Tool model settings match **Model Policy** (or are documented as exceptions).

Suggested setup:
- Create a dedicated branch: `release-review/v0.8.0` (or `release-review/<version>`), based on the target base branch.
- Configure commit identity: `git config user.name`, `git config user.email`.

## Orchestration Constraints (Hard Rules)

1. **Max 3 concurrent threads**. No more.
2. Each thread works on a **non-overlapping component scope** (avoid merge conflicts).
3. **Only the captain updates** `.release_review/PRE_RELEASE_REVIEW_TRACKING.md`.
4. Every completed unit of **code work** is committed with:
   - A component-scoped branch
   - A component-scoped commit(s)
   - Evidence recorded in tracking (commit SHA, tests run, notes)
5. The captain does **not** assume first-pass success: iterate, re-run agents, or assign a second reviewer.

## Parallel Execution Model (Single Branch, Multiple Terminals)

**Preferred approach**: All agents work on the same `release-review/<version>` branch in the same repo folder, using separate terminal sessions.

### Why single-branch parallel?
- No need to clone the repo multiple times
- No complex branch merging
- Simpler conflict avoidance via strict scope discipline
- Easier to coordinate and track progress

### Setup for parallel execution:

```
Terminal 1 (Captain - Claude Code):     Works on Component A
Terminal 2 (Codex CLI):                  Works on Component B
Terminal 3 (Copilot CLI):                Works on Component C
```

All terminals point to the same directory: `/path/to/repo`
All terminals are on the same branch: `release-review/v0.8.0`

### Scope discipline (CRITICAL):
- Assign **strictly non-overlapping file scopes** to each agent
- If Component A touches `traigent/security/`, no other agent touches those files
- Captain coordinates commits to avoid conflicts

### Example parallel assignment:

| Agent | Scope | Files (exclusive) |
|-------|-------|-------------------|
| Claude (Captain) | Release blockers | `traigent/cloud/resilient_client.py`, `traigent/storage/local_storage.py`, `traigent/utils/batch_processing.py` |
| Codex CLI | Integrations | `traigent/integrations/**` |
| Copilot CLI | Packaging | `pyproject.toml`, `requirements/`, `MANIFEST.in` |

## Merge Strategy (Recommended)

- Use a dedicated `release-review/<version>` branch as the integration branch.
- Prefer **squash merges** of component branches into the integration branch to keep history readable.
- Require the squash commit message to include **evidence footers** (see “Commit Protocol”).

## Work Unit Definition (Per Component)

For each component row in `.release_review/PRE_RELEASE_REVIEW_TRACKING.md`, “Done” means:
- Code review performed against the checklist in `.release_review/PRE_RELEASE_REVIEW_PLAN.md`.
- Issues found are either fixed (preferred) or logged + explicitly accepted with rationale.
- Appropriate tests executed (or a justified note why not possible).
- Captain records evidence and marks component **Approved**.

## Standard Loop (“Captain Tick”)

Repeat until all P0/P1 items are Approved (then proceed down the list).

### Step 1: Select Work (3 threads max)

From `.release_review/PRE_RELEASE_REVIEW_TRACKING.md`:
- Pick the highest Priority components with Status ≠ Approved.
- Prefer “user-hit-first” and cross-cutting components (P0/P1).
- Ensure chosen components are disjoint (e.g., don’t run `traigent/core/` and `traigent/config/` in parallel if you expect shared changes).

### Step 2: Claim + Branch

For each selected component:
- Create a branch: `review/<component-slug>/<agent>/<YYYYMMDD>`
  - Example: `review/integrations/codex/20251213`
- In `.release_review/PRE_RELEASE_REVIEW_TRACKING.md`, set:
  - Owner = assigned agent
  - Approver = captain
  - Status = In progress
  - (Leave checkboxes until completion)

### Step 3: Dispatch Agents

Dispatch up to 3 agents (parallel). For each agent, provide:
- **Component scope** (paths + constraints)
- **Checklist** reference (`.release_review/PRE_RELEASE_REVIEW_PLAN.md`)
- **Expected deliverable** (PR-ready patch + summary + tests)
- **No tracking-file edits** (captain owns tracking)

### Step 4: Integrate (Captain)

For each agent’s output:
- Review diffs for correctness + scope discipline.
- Run the suggested tests (or the nearest equivalent).
- If results are acceptable:
  - Squash merge to the `release-review/<version>` branch.
  - Update `.release_review/PRE_RELEASE_REVIEW_TRACKING.md`:
    - Mark Review/Tests/Docs checkboxes as appropriate
    - Set Status = Approved
    - Add Evidence (commit SHA(s), test commands, notes)
  - Commit the updated tracking file to the `release-review/<version>` branch so the team sees progress.
- If results are not acceptable:
  - Send the failing output back to the same agent to iterate OR
  - Spin a second agent to review/repair the patch.

### Step 5: Repeat

Select next highest priority items until complete.

## Conflict Avoidance Policy

- Do not assign multiple threads to modify the same subsystem files.
- Prefer “vertical slices” that don’t share code paths.
- If conflicts are unavoidable:
  - Sequence those components (run them in separate ticks).

## Cross-Component Change Protocol

When a fix requires changes outside the assigned component scope:

1. **Minor spillover** (≤3 lines in adjacent module): agent may implement, but must explain why and list touched files in the deliverable summary.
2. **Significant spillover** (>3 lines or different subsystem): agent proposes the change but does **not** implement it. Captain either sequences the dependent component next or creates a dedicated cross-cutting branch.
3. **Shared files** (e.g., `pyproject.toml`, `traigent/__init__.py`): agents propose diffs; **captain** applies and commits to avoid conflicts.

## Evidence Policy (What Goes in the Tracking File)

For each component, Evidence MUST be **machine-validated JSON** with these fields:
- `format`: `standard` or `legacy`
- `commits`: list of commit SHAs
- `tests`: `{command,status,passed,total}`
- `models`: model string used for review
- `reviewer`: `<agent> + <captain>`
- `timestamp`: ISO-8601 date/time of approval
- `followups`: links/IDs or `None`
- `accepted_risks`: explicit statement or `None`

Rules:
- Use `format=standard` for all new reviews.
- `format=legacy` is only for backfilled historical entries; it may use `UNKNOWN` values.

Example (single line is required in the Evidence column):
`{"format":"standard","commits":["abc1234"],"tests":{"command":"TRAIGENT_MOCK_LLM=true pytest tests/unit/integrations/ -q","status":"PASS","passed":42,"total":42},"models":"Codex/GPT-5.2/xhigh","reviewer":"codex + @captain","timestamp":"2025-12-13T14:30:00Z","followups":"#1234","accepted_risks":"None"}`

## Iteration Limits and Escalation

- Max **3 iterations** per component per agent.
- If an agent fails 3 times: switch agents or escalate to a human reviewer; mark Status = **Blocked** with blocker notes.
- Don't spend >30 minutes debugging a single agent failure before escalating.

## Continuous Execution Protocol

**Core principle**: The captain runs start-to-finish without waiting for user approval between steps. User interaction is asynchronous and non-blocking.

### Autonomous Decision Making

The captain SHOULD make judgment calls autonomously for:
- Component prioritization within the same tier
- Agent assignment (following rotation schedule)
- Minor fixes that don't change public API
- Accepting risks that match previous release patterns
- Batch sizing and parallelization strategy

The captain MUST escalate to user for:
- Tier 3 conflicts (security disagreements)
- Public API changes
- New dependencies or breaking changes
- Decisions that contradict explicit user instructions

### Non-Blocking User Questions

When the captain needs user input but can continue working:

1. **Write question** to `.release_review/<version>/USER_QUESTIONS.md`:
   ```markdown
   ## Question <N> (<ISO-8601 timestamp>)
   **Status**: PENDING
   **Blocking**: Yes/No (does this block the review?)
   **Component**: <affected component>
   **Question**: <clear, specific question>
   **Context**: <relevant background>
   **Options**:
   - Option A: <description>
   - Option B: <description>
   **Captain's Recommendation**: <what captain would do if no answer>
   **Deadline**: <when captain will proceed with recommendation if no answer>

   ### User Answer
   (User fills this in)
   ```

2. **Continue working** on components not affected by the question

3. **Check periodically** (every 2-3 component batches):
   - Read `USER_QUESTIONS.md`
   - If answer provided: incorporate and update Status to RESOLVED
   - If deadline passed and no answer: proceed with recommendation, document decision

4. **Document decision** in tracking file:
   ```
   Accepted risks: Captain decided <X> per USER_QUESTIONS.md Q3 (no user response by deadline)
   ```

### Stop Conditions

Captain continues until ONE of:
- ✅ ALL components are Approved → Generate audit trail, announce completion
- ⏸️ Tier 3 conflict requires human escalation → Document conflict, wait for resolution
- 🛑 User explicitly says "stop" or "pause"
- 💥 Unrecoverable error (git corruption, test infrastructure down)

Captain does NOT stop for:
- Individual component failures (retry or escalate, continue with others)
- Agent timeouts (switch agents, continue)
- Non-blocking questions (ask async, continue)
- Minor ambiguities (use best judgment, document)

### Progress Checkpoints

Every 30 minutes OR after each batch completion:
1. Update tracking file with current status
2. Append entry to Review Notes Log
3. Check `USER_QUESTIONS.md` for answers
4. Commit progress to branch

### How to Trigger Review

See `.release_review/START_REVIEW.md` for copy-paste prompts to start a review.

Quick start:
```
You are the release-review captain for Traigent SDK version <VERSION>.
Read and follow: .release_review/CAPTAIN_PROTOCOL.md
Execute complete review start-to-finish. Do NOT stop until complete.
Start now.
```

## Session Handoff Protocol

Before ending a session, the captain MUST:

1. **Commit all work**: Ensure all code changes are committed on their component branches (or merged to integration branch).
2. **Update tracking**: Set correct Status for all in-flight components in `.release_review/PRE_RELEASE_REVIEW_TRACKING.md`.
3. **Log progress**: Append a timestamped entry to the "Review Notes Log" section summarizing:
   - Components worked on
   - Current blockers or pending items
   - Next priority items to pick up
4. **Push branch**: Push the `release-review/<version>` branch so other reviewers can sync.
5. **Verify state**: Run `git status` to confirm no uncommitted changes remain.

When resuming a session:
1. Pull latest `release-review/<version>` branch.
2. Read `.release_review/PRE_RELEASE_REVIEW_TRACKING.md` to understand current state.
3. Check the "Review Notes Log" for the last session's summary.
4. Continue from the highest-priority non-Approved component.

## Commit Protocol

- Branch per component per agent.
- Squash merge into `release-review/<version>` with a final commit that includes evidence.
- Commit messages (recommended):
  - `fix(<component>): <summary>`
  - `refactor(<component>): <summary>` (avoid unless necessary)
  - `docs(<component>): <summary>`
- Keep commits small; prefer multiple commits over one giant commit.

### Required Commit Evidence Footers (on squash merge commits)

Include these lines at the bottom of the squash-merge commit message:

```
RR-Component: <component>
RR-Tests: <command(s)> → <PASS/FAIL summary>
RR-Models: <Claude/Codex/Copilot + actual models>
RR-Reviewer: <agent> + <captain>
RR-Timestamp: <ISO-8601>
RR-Followups: <links/IDs or None>
RR-Accepted-Risks: <None or description>
```

## Re-Review Protocol (Second Pass)

For high-risk components (P0/P1), require at least one of:
- A second agent reviews the final diff (no code changes unless needed), OR
- Captain performs a dedicated “adversarial review” (edge cases, security, failure modes).

## Agent Prompt Templates

Use these templates when invoking sub-agents.

### Template A: Component Review + Fix (Primary)

**Task**: Review and fix component: `<COMPONENT_NAME>`

**Scope**: `<PATHS>` (do not edit outside scope unless strictly required; if you must, explain why)

**Goals**:
1. Identify likely issues (correctness, security, reliability, DX).
2. Fix the top issues with minimal, surgical changes.
3. Run relevant tests and report exact commands/results.

**Constraints**:
- Do NOT edit `.release_review/PRE_RELEASE_REVIEW_TRACKING.md` (captain will update).
- If you create any files (notes, reports, scratch outputs), put them under `.release_review/<version>/artifacts/<component>/<agent>/<YYYYMMDD>/...`.
- Keep changes minimal; no broad refactors unless required to fix a real issue.
- If you find a large issue, propose a ticket + smallest safe mitigation for release.

**Deliverable**:
- A git-ready patch (diff) on branch `<BRANCH_NAME>`
- Short summary of findings + fixes
- Tests run + results

### Template B: Patch Review (Secondary Reviewer)

**Task**: Review the changes on branch `<BRANCH_NAME>` for component `<COMPONENT_NAME>`.

**Focus**:
- Hidden regressions, missed edge cases
- Security/privacy/logging leaks
- API/behavior compatibility
- Whether tests sufficiently cover the change

**Deliverable**:
- “Approve” or “Request changes”
- If changes needed: specific actionable diffs or instructions

## Invocation Examples (Adjust to Your Local CLIs)

Because CLI flags differ across installs, treat these as patterns:

- **Claude Code (Opus 4.5)**: captain and any Claude worker sessions use Opus 4.5; use for orchestration + deep patch review.
- **Codex CLI (GPT-5.2, xhigh)**: use for heavy code comprehension and surgical refactors within a component scope.
- **Copilot CLI (Gemini 3.0)**: use for quick suggestions/explanations and small targeted edits; avoid repo-wide changes.

## Message to Send to Claude Code CLI (Copy/Paste)

You are the release-review captain for this repo.

Model policy for all work:
- Claude Code (you): **Opus 4.5**
- Codex CLI: **ChatGPT 5.2 (GPT-5.2)** with **xhigh** effort
- GitHub Copilot CLI: **Gemini 3.0**

1) Read these files and treat them as canonical:
- `.release_review/PRE_RELEASE_REVIEW_PLAN.md`
- `.release_review/PRE_RELEASE_REVIEW_TRACKING.md`
- `RELEASE_BLOCKERS_TODO.md`

2) Create/checkout the integration branch: `release-review/v0.8.0` (or `release-review/<version>`). This branch is the shared source of truth for tracking progress.

3) Any agent-generated notes/reports must go under `.release_review/<version>/artifacts/<component>/<agent>/<YYYYMMDD>/...` (git-ignored).

4) Use `.release_review/PRE_RELEASE_REVIEW_TRACKING.md` priority order and run up to **3 concurrent threads** at a time. Each thread must be a distinct component scope to avoid conflicts.

5) For each selected component:
- Create a branch `review/<component>/<agent>/<YYYYMMDD>`
- Assign the component to one of: (a) you (Claude), (b) Codex CLI, (c) GitHub Copilot CLI
- Send the assigned agent “Template A: Component Review + Fix” with the exact scope and checklist reference.

6) Integration rules:
- Only you update `.release_review/PRE_RELEASE_REVIEW_TRACKING.md`.
- Every code change is committed per component branch.
- Squash merge approved component branches into `release-review/<version>` and include commit evidence footers.
- After merging, update tracking with Status=Approved + Evidence, commit the tracking update, and push the branch.

7) Expect iteration:
- If an agent’s output fails tests or seems risky, send it back for a fix or spin a different agent to review/repair it.
- For P0/P1 items, require a second review pass (Template B) before approval.

8) Continue until all P0/P1 items are Approved, then proceed down the list. Stop only when we (captain + user) agree release gates are satisfied.

---

## LESSONS LEARNED (v0.8.0 Review)

The following additions were identified during the v0.8.0 release review and incorporated based on feedback from multiple models (Claude Opus 4.5, GPT-5.2).

---

## Artifact Persistence Requirements

**Problem solved**: Agent reports were ephemeral during v0.8.0 review. Findings weren't automatically saved.

### Canonical Directory Structure (Per-Version)

**IMPORTANT**: All release review artifacts live under `.release_review/<version>/`:

```
.release_review/
├── CAPTAIN_PROTOCOL.md          # This file (versioned, shared)
├── PRE_RELEASE_REVIEW_TRACKING.md  # Tracking board (versioned)
├── PRE_RELEASE_REVIEW_PLAN.md   # Review plan (versioned)
├── automation/                  # Automation scripts (versioned)
│
└── <version>/                   # Per-release workspace (e.g., v0.8.0/)
    ├── TRACE_LOG.md             # Agent spawn/completion log
    ├── AUDIT_TRAIL.md           # Post-release audit
    ├── DASHBOARD.md             # Real-time status (optional)
    ├── POST_RELEASE_TODO.md     # Backlog from recommendations
    └── artifacts/               # Agent outputs
        └── <component>/         # e.g., core_orchestrator/
            └── <model>/         # e.g., claude/
                ├── YYYYMMDD_findings.md
                ├── YYYYMMDD_fixes.md
                └── YYYYMMDD_recommendations.md
```

### Mandatory Artifact Path

Every agent review MUST write to:
```
.release_review/<version>/artifacts/<component>/<model>/YYYYMMDD_<type>.md
```

**Example**: `.release_review/v0.8.0/artifacts/core_orchestrator/claude/20251213_findings.md`

**Types**:
- `findings` - Main review output
- `fixes` - Code changes proposed/applied
- `recommendations` - Post-release suggestions

### Standardized Report Template

Agents must use this structure for `findings.md`:

```markdown
# Review Findings: <Component>

**Agent**: <Model Name + Version>
**Agent ID**: <Unique session ID>
**Timestamp**: <ISO-8601>
**Scope**: <File paths reviewed>

## Tests Executed

| Command | Exit Code | Passed | Failed | Duration |
|---------|-----------|--------|--------|----------|
| `pytest tests/...` | 0 | 47 | 0 | 12.3s |

## Issues Found

### Issue 1: <Title>
- **Severity**: Critical/High/Medium/Low
- **File**: `path/to/file.py:123`
- **Description**: ...
- **Fix Applied**: Yes/No (if yes, include commit SHA)

## Recommendations (Post-Release)

- [ ] Item 1...
- [ ] Item 2...

## Evidence

- Terminal output hash: sha256:...
- Git state at review: <commit SHA>
- Environment: TRAIGENT_MOCK_LLM=true, Python 3.11.x
```

### Trace Log

Captain maintains `.release_review/<version>/TRACE_LOG.md`:

```markdown
# Trace Log: v0.8.0

| Timestamp | Agent | Component | Action | Status | Artifact Link |
|-----------|-------|-----------|--------|--------|---------------|
| 2025-12-13T10:00:00Z | Claude Opus 4.5 | core/orchestrator | spawn | started | - |
| 2025-12-13T10:15:00Z | Claude Opus 4.5 | core/orchestrator | complete | passed | [findings](artifacts/core/claude/20251213_findings.md) |
```

---

## Captain Role Boundaries

**Problem solved**: Captain mixed orchestration with direct review work.

### Captain DOES (Orchestration)

- Spawn/dispatch agents via Task tool or CLI
- Monitor agent progress, unblock stuck agents
- Merge component branches, resolve git conflicts
- Final approval based on agent evidence
- Update tracking file and release notes
- Perform spot-checks (verify 20% of agent claims)
- Make final call on conflict resolution

### Captain DOES NOT (Direct Review)

- Perform detailed code review (delegate to agents)
- Write fixes directly (agents propose, captain approves)
- Make judgment calls on technical debt (agents classify, captain prioritizes)

### Exception Handling

If captain MUST review (e.g., all agents blocked):
1. Log exception in Review Notes Log with justification
2. Assign a secondary reviewer for the captain's work
3. Record "Captain reviewed directly" in Evidence field

---

## Conflict Resolution Tiers

**Problem solved**: "Captain resolves" was too vague for disagreements.

### Tier 1: Minor Disagreements
**Trigger**: Style suggestions, non-critical issues, different phrasings

**Resolution**:
- Captain documents both perspectives in artifact
- Captain makes final call based on project standards
- Resolution time: <10 minutes

**Example**: Agent A suggests refactoring, Agent B says leave as-is → Captain decides based on scope.

### Tier 2: Moderate Disagreements
**Trigger**: Different interpretations of bug severity, test coverage adequacy

**Resolution**:
- Captain runs independent verification
- Captain documents reasoning for chosen resolution
- Losing reviewer's concerns noted in `ACCEPTED_RISKS.md`
- Resolution time: <30 minutes

**Example**: Agent A says "Medium priority", Agent B says "High priority" → Captain verifies impact.

### Tier 3: Major Disagreements (BLOCKING)
**Trigger**: Security vulnerability detected vs. not detected, data corruption risk

**Resolution**:
- **Automatic escalation to human reviewer**
- Both agents must provide detailed evidence
- Captain documents conflict but does NOT resolve
- **Release blocked** until human decision
- Create issue/ticket for tracking

**Template for escalation**:
```markdown
## BLOCKING CONFLICT: <Component>

- **Primary Agent**: <Model> - <Finding>
- **Secondary Agent**: <Model> - <Finding>
- **Evidence**: [links to both reports]
- **Status**: ESCALATED TO HUMAN
- **Assigned to**: @<human-reviewer>
- **Blocking release**: YES
```

---

## Peer Review Independence Protocol

**Problem solved**: Secondary reviewers could be biased by primary findings.

### Two-Phase Independent Review (P0/P1 Only)

**Phase 1: Primary Review**
- Agent A reviews component
- Writes findings to `artifacts/<component>/<model>/YYYYMMDD_findings.md`
- Captain does NOT share findings with secondary reviewer yet

**Phase 2: Independent Secondary Review**
- Agent B (different model family) reviews SAME component
- Works in separate session (cannot see Agent A's conversation)
- Writes findings independently
- Focuses on: edge cases, security, missed issues

**Phase 3: Reconciliation**
- Captain compares both reports
- Identifies agreements and disagreements
- Applies Conflict Resolution Tiers as needed
- Final approval requires both reviewers' sign-off (or documented override)

### Tiered Dual Review (Efficiency Optimization)

Full dual review for all components is expensive. Use risk-based approach:

| Priority | Dual Review Requirement |
|----------|------------------------|
| P0 (90-100) | Full dual review required |
| P1 (75-89) | Dual review for security/core components; sampling (30%) for others |
| P2 (60-74) | Captain spot-check only |
| P3 (<60) | Single review sufficient |

---

## Test Verification Mechanisms

**Problem solved**: No way to verify agents actually ran tests vs. claiming they did.

### Required Test Evidence

Agents must provide:

```markdown
## Test Evidence (REQUIRED)

- **Command**: `TRAIGENT_MOCK_LLM=true pytest tests/unit/test_X.py -v`
- **Exit code**: 0
- **Summary**: 47 passed, 0 failed, 2 skipped
- **Duration**: 12.3s
- **Git state**: abc123 (HEAD at time of test)
- **Environment**: Python 3.11.5, TRAIGENT_MOCK_LLM=true
```

### Captain Spot-Checks

Captain randomly selects **20% of component reviews** and:
1. Re-runs the exact test command
2. Compares results with agent's claimed results
3. If mismatch: escalate to human review + document agent failure

### Flaky Test Policy

**Problem**: How to handle tests that pass sometimes but fail intermittently.

**Definition**: A test is "flaky" if it fails on first run but passes on retry without code changes.

**Protocol**:

| Scenario | Action | Status |
|----------|--------|--------|
| Test fails once, passes on retry | Document as flaky, continue review | **Approved** (with note) |
| Test fails 2+ consecutive times | Investigate root cause | **In Progress** |
| Test fails consistently (3+ times) | Mark component blocked | **Blocked** |
| Known flaky test (documented) | Skip with justification | **Approved** (with note) |

**Required Documentation for Flaky Tests**:
```markdown
## Flaky Test Note

- **Test**: `tests/unit/test_example.py::test_name`
- **Behavior**: Fails ~10% of runs due to race condition
- **Root cause**: Test isolation issue (not production bug)
- **Action**: Retry passed, documented for post-release fix
- **Tracking**: Added to POST_RELEASE_TODO.md
```

**Rules**:
1. Never mark a component Approved if tests fail consistently
2. Flaky tests due to **test isolation** (not production bugs) can be accepted with documentation
3. Flaky tests that indicate **real race conditions** in production code = **Blocked** until fixed
4. Captain must verify flaky classification with independent run

### Pytest JSON Reports (Recommended)

For tamper-evident results:
```bash
pytest --json-report --json-report-file=.release_review/<version>/artifacts/<component>/test_results.json
```

Captain validates JSON structure matches agent claims.

---

## Scope Enforcement

**Problem solved**: Agents doing more than assigned (scope creep).

### Pre-flight Checklist

Agents must acknowledge before starting:

```markdown
## Agent Pre-flight Checklist

- [ ] I have read my assigned component list
- [ ] I will NOT modify files outside my scope
- [ ] I will NOT merge branches (Captain-only)
- [ ] I will flag cross-component issues to Captain instead of fixing directly
- [ ] My assigned scope: `<paths>`
```

### Scope Validation (Captain)

Before approving any agent work:

```bash
# Run scope guard
git diff --name-only main...<agent-branch> | while read file; do
  if [[ ! "$file" =~ ^<allowed-pattern> ]]; then
    echo "SCOPE VIOLATION: $file"
  fi
done
```

### Time Boxing

| Priority | Expected Time | Max Time (before intervention) |
|----------|---------------|-------------------------------|
| P0 | 30-45 min | 90 min |
| P1 | 20-30 min | 60 min |
| P2 | 10-20 min | 40 min |
| P3 | 5-15 min | 30 min |

If agent exceeds 2x expected time, captain intervenes to check for scope creep or blockers.

---

## Failure Recovery

**Problem solved**: No mechanism for agent crashes or interrupted sessions.

### Incremental Checkpoints

Agents should save progress every 10-15 minutes:

```
.release_review/<version>/artifacts/<component>/<model>/.checkpoint_<timestamp>.json
```

Checkpoint format:
```json
{
  "timestamp": "2025-12-13T10:15:00Z",
  "status": "in_progress",
  "files_reviewed": ["file1.py", "file2.py"],
  "issues_found": 2,
  "tests_run": ["test_a.py"],
  "resumable": true
}
```

### Resume Protocol

If agent crashes:
1. Check for checkpoint file
2. If <1 hour old and `resumable: true`: continue from checkpoint
3. Otherwise: restart review from beginning

---

## Automation Toolkit

Location: `.release_review/automation/`

| Script | Purpose |
|--------|---------|
| `artifact_manager.py` | Auto-generate artifact paths, validate report structure |
| `scope_guard.py` | Validate agent changes are within assigned scope |
| `evidence_validator.py` | Parse and validate evidence format |
| `verify_tests.sh` | Captain re-runs tests to verify agent claims |
| `checkpoint.py` | Manage incremental progress saving |
| `metrics.py` | Track agent effectiveness across releases |

See `.release_review/automation/README.md` for usage.

---

## Post-Release Audit Trail

After release approval, captain generates:

```markdown
# .release_review/<version>/AUDIT_TRAIL.md

## Release Approval
- **Version**: v0.8.0
- **Approved by**: Captain (Claude Opus 4.5)
- **Approval Time**: 2025-12-13T21:00:00Z
- **Final Commit**: abc123def456

## Agent Contributions
| Agent | Components | Issues Found | Fixes Applied | False Positives |
|-------|------------|--------------|---------------|-----------------|
| Claude Opus 4.5 | 10 | 3 | 3 | 0 |
| GPT-5.2 | 8 | 1 | 1 | 0 |

## Spot-Check Results
- **Components verified**: 6/30 (20%)
- **Mismatches found**: 0

## Conflicts & Resolutions
(List any Tier 2/3 conflicts and how they were resolved)

## Accepted Risks
(List any risks accepted for this release)
```

---

## Release Dashboard (Optional)

For real-time visibility, captain can maintain:

```markdown
# .release_review/<version>/DASHBOARD.md

**Status**: IN_PROGRESS / COMPLETE
**Progress**: 18/30 components (60%)

## Quick Stats
- Critical issues: 0
- Blocking conflicts: 0
- Tests passing: 12,000+

## Component Status
| Component | Agent | Status | Blockers |
|-----------|-------|--------|----------|
| core/orchestrator | Claude | ✅ Done | - |
| integrations | GPT-5.2 | 🔄 In Progress | - |
| security | Claude | ⏸️ Blocked | Conflict with GPT-5.2 |
```
