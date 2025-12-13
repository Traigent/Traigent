# Post-Release Recommendation Fixes Tracking

**Source**: (Import from `.release_review/<version>/POST_RELEASE_TODO.md`)
**Created**: (Run `todo_importer.py` to populate)
**Total Items**: 0

## Summary

| Priority | Total | Pending | In Progress | Complete |
|----------|-------|---------|-------------|----------|
| High (P0) | 0 | 0 | 0 | 0 |
| Medium (P1) | 0 | 0 | 0 | 0 |
| Low (P2) | 0 | 0 | 0 | 0 |
| **Total** | **0** | **0** | **0** | **0** |

---

## How to Populate This File

Run the TODO importer to populate from a release review:

```bash
python .post_release_recommendation_fixes/automation/todo_importer.py \
    .release_review/v0.8.0/POST_RELEASE_TODO.md
```

Or manually add items below following this format:

---

## High Priority (P0)

| ID | Title | Component | Status | Owner | Evidence |
|----|-------|-----------|--------|-------|----------|
| (none) | - | - | - | - | - |

---

## Medium Priority (P1)

| ID | Title | Component | Status | Owner | Evidence |
|----|-------|-----------|--------|-------|----------|
| (none) | - | - | - | - | - |

---

## Low Priority (P2)

| ID | Title | Component | Status | Owner | Evidence |
|----|-------|-----------|--------|-------|----------|
| (none) | - | - | - | - | - |

---

## Item Details

(Items will be populated here with full details)

---

## Status Legend

| Status | Meaning |
|--------|---------|
| Pending | Not started |
| In Progress | Agent working on fix |
| Complete | Merged and verified |
| Blocked | Cannot proceed (see notes) |

## Evidence Format

```
Commit: <sha7> | Tests: <command> -> <result> | Time: <timestamp>
```

Example:
```
Commit: abc1234 | Tests: pytest tests/unit/test_storage.py -> PASS (12) | Time: 2025-12-14T10:30:00Z
```
