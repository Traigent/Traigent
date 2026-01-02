# Start Post-Release Fixes Workflow

## Quick Start (Copy This)

```
You are the fix orchestration captain for Traigent SDK.

Read and follow: .post_release_recommendation_fixes/CAPTAIN_PROTOCOL.md

export RR_VERSION=v0.8.0
export RR_BASE_BRANCH=main

MANDATORY STEPS (do not skip any):
1. Pre-flight check: .venv/bin/python .post_release_recommendation_fixes/automation/preflight_check.py --version "$RR_VERSION"
2. Initialize session: .venv/bin/python .post_release_recommendation_fixes/automation/session_init.py --version "$RR_VERSION" init high
3. Conflict check: .venv/bin/python .post_release_recommendation_fixes/automation/conflict_detector.py batch
4. Implement fixes using AGENTS (spawn Task tools, do not implement directly)
5. Save agent outputs to session artifacts folder

CRITICAL REMINDERS:
- ALWAYS use `.venv/bin/python -m pytest` (NEVER bare `pytest` or `.venv/bin/pytest`)
- ALWAYS run conflict check BEFORE dispatching parallel agents
- ALWAYS save agent outputs to: .post_release_recommendation_fixes/$RR_VERSION/sessions/<date>/artifacts/<fix-id>/
- ALWAYS validate evidence JSON: .venv/bin/python .release_review/automation/evidence_validator.py --file .post_release_recommendation_fixes/$RR_VERSION/TRACKING.md
- Shared templates live at: .post_release_recommendation_fixes/templates/

Source: .release_review/$RR_VERSION/POST_RELEASE_TODO.md
Continue start-to-finish. Use async questions if blocked.

Start now.
```

---

## First-Time Setup

If TRACKING.md is empty, first import TODOs:

```bash
# Import from release review
export RR_VERSION=v0.8.0
.venv/bin/python .post_release_recommendation_fixes/automation/todo_importer.py \
    .release_review/$RR_VERSION/POST_RELEASE_TODO.md --yes --version "$RR_VERSION"
```

---

## Resume Previous Session

```
You are the fix orchestration captain for Traigent SDK.

Read: .post_release_recommendation_fixes/CAPTAIN_PROTOCOL.md

Resume from previous session:
export RR_VERSION=v0.8.0
.venv/bin/python .post_release_recommendation_fixes/automation/session_init.py --version "$RR_VERSION" resume

Continue with pending fixes. Do not re-implement completed ones.

Start now.
```

---

## Target Specific Scope

### High Priority Only (4 fixes)
```
.venv/bin/python .post_release_recommendation_fixes/automation/session_init.py --version "$RR_VERSION" init high
```

### Medium Priority Only
```
.venv/bin/python .post_release_recommendation_fixes/automation/session_init.py --version "$RR_VERSION" init medium
```

### All Fixes
```
.venv/bin/python .post_release_recommendation_fixes/automation/session_init.py --version "$RR_VERSION" init all
```

---

## Monitor Progress

```bash
# Pre-flight check
.venv/bin/python .post_release_recommendation_fixes/automation/preflight_check.py --version "$RR_VERSION"

# Progress stats
.venv/bin/python .post_release_recommendation_fixes/automation/progress_tracker.py --version "$RR_VERSION" stats

# Full report
.venv/bin/python .post_release_recommendation_fixes/automation/progress_tracker.py --version "$RR_VERSION" report

# Conflict check for parallel work
.venv/bin/python .post_release_recommendation_fixes/automation/conflict_detector.py batch

# Latest session
.venv/bin/python .post_release_recommendation_fixes/automation/session_init.py --version "$RR_VERSION" latest
```

---

## Answering Captain Questions

If the captain writes to `USER_QUESTIONS.md`:
1. Open `.post_release_recommendation_fixes/$RR_VERSION/USER_QUESTIONS.md`
2. Find questions with `Status: PENDING`
3. Add your answer under `### User Answer`
4. Save the file

---

## Workflow at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│                     FIX WORKFLOW                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. preflight_check.py  →  Verify prerequisites             │
│                             ↓                               │
│  2. todo_importer.py    →  Import TODOs (first time only)   │
│                             ↓                               │
│  3. session_init.py     →  Create session, select fixes     │
│                             ↓                               │
│  4. conflict_detector   →  Check BEFORE parallel dispatch   │  ← MANDATORY
│                             ↓                               │
│  5. Captain dispatches  →  Spawn agents (Task tool)         │  ← USE AGENTS
│                             ↓                               │
│  6. Save artifacts      →  Store agent outputs per fix      │
│                             ↓                               │
│  7. Verify + merge      →  Run tests, merge to branch       │
│                             ↓                               │
│  8. progress_tracker    →  Update status, generate report   │
│                                                             │
│  ALWAYS use: .venv/bin/python -m pytest (not bare pytest)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
