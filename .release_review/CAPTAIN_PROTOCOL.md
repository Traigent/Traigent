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

## Canonical Inputs / Artifacts

- Review plan: `.release_review/PRE_RELEASE_REVIEW_PLAN.md`
- Tracking board (source of truth): `.release_review/PRE_RELEASE_REVIEW_TRACKING.md`
- Release blockers: `RELEASE_BLOCKERS_TODO.md`

## Release Review Workspace (Versioned + Ignored Artifacts)

This repo uses `.release_review/` as a dedicated home for the review system:
- **Versioned (commit these)**: plan, tracking, captain protocol (share status with the team).
- **Ignored (do not commit)**: `.release_review/artifacts/` (agent notes, scratch outputs).

Store any agent-generated files under:
- `.release_review/artifacts/<component>/<agent>/<YYYYMMDD>/...`

Examples:
- `.release_review/artifacts/traigent-integrations/codex/20251213/notes.md`
- `.release_review/artifacts/release-blockers/claude/20251213/fix-plan.md`

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

For each component, Evidence MUST include:
- **Commit(s)**: SHA(s) and branch name
- **Tests**: exact command(s) run + PASS/FAIL summary
- **Models**: confirm model policy used (or record actual model)
- **Reviewer**: assigned agent + approving captain
- **Timestamp**: ISO-8601 date/time of approval
- **Follow-ups**: links/IDs to any tickets created (or “None”)
- **Accepted risks**: explicit statement (or “None”)

Example (single line is fine):
`Commits: abc1234 on review/integrations/codex/20251213 | Tests: TRAIGENT_MOCK_MODE=true pytest tests/unit/integrations/ -q → PASS (42 passed) | Models: Codex/GPT-5.2/xhigh | Reviewer: codex + @captain | Timestamp: 2025-12-13T14:30:00Z | Follow-ups: #1234 | Accepted risks: None`

## Iteration Limits and Escalation

- Max **3 iterations** per component per agent.
- If an agent fails 3 times: switch agents or escalate to a human reviewer; mark Status = **Blocked** with blocker notes.
- Don't spend >30 minutes debugging a single agent failure before escalating.

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
- If you create any files (notes, reports, scratch outputs), put them under `.release_review/artifacts/<component>/<agent>/<YYYYMMDD>/...`.
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

3) Any agent-generated notes/reports must go under `.release_review/artifacts/<component>/<agent>/<YYYYMMDD>/...` (git-ignored).

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
