# Start Post-Release Fixes Workflow

## Quick Start (Copy This)

```
You are the fix orchestration captain for TraiGent SDK.

Read and follow: .post_release_recommendation_fixes/CAPTAIN_PROTOCOL.md

Steps:
1. Run pre-flight check: .venv/bin/python .post_release_recommendation_fixes/automation/preflight_check.py
2. Initialize session: .venv/bin/python .post_release_recommendation_fixes/automation/session_init.py init high
3. Implement fixes from TRACKING.md in priority order
4. Run conflict check: .venv/bin/python .post_release_recommendation_fixes/automation/conflict_detector.py batch

Source: .release_review/v0.8.0/POST_RELEASE_TODO.md
Continue start-to-finish. Use async questions if blocked.

Start now.
```

---

## First-Time Setup

If TRACKING.md is empty, first import TODOs:

```bash
# Import from release review
.venv/bin/python .post_release_recommendation_fixes/automation/todo_importer.py \
    .release_review/v0.8.0/POST_RELEASE_TODO.md --yes
```

---

## Resume Previous Session

```
You are the fix orchestration captain for TraiGent SDK.

Read: .post_release_recommendation_fixes/CAPTAIN_PROTOCOL.md

Resume from previous session:
.venv/bin/python .post_release_recommendation_fixes/automation/session_init.py resume

Continue with pending fixes. Do not re-implement completed ones.

Start now.
```

---

## Target Specific Scope

### High Priority Only (4 fixes)
```
.venv/bin/python .post_release_recommendation_fixes/automation/session_init.py init high
```

### Medium Priority Only
```
.venv/bin/python .post_release_recommendation_fixes/automation/session_init.py init medium
```

### All Fixes
```
.venv/bin/python .post_release_recommendation_fixes/automation/session_init.py init all
```

---

## Monitor Progress

```bash
# Pre-flight check
.venv/bin/python .post_release_recommendation_fixes/automation/preflight_check.py

# Progress stats
.venv/bin/python .post_release_recommendation_fixes/automation/progress_tracker.py stats

# Full report
.venv/bin/python .post_release_recommendation_fixes/automation/progress_tracker.py report

# Conflict check for parallel work
.venv/bin/python .post_release_recommendation_fixes/automation/conflict_detector.py batch

# Latest session
.venv/bin/python .post_release_recommendation_fixes/automation/session_init.py latest
```

---

## Answering Captain Questions

If the captain writes to `USER_QUESTIONS.md`:
1. Open `.post_release_recommendation_fixes/USER_QUESTIONS.md`
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
│  4. conflict_detector   →  Check for parallel conflicts     │
│                             ↓                               │
│  5. Captain implements  →  Spawn agents, verify, merge      │
│                             ↓                               │
│  6. progress_tracker    →  Update status, generate report   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
