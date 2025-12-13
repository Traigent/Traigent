# User Questions: Post-Release Fixes

This file is used for asynchronous communication between the captain and the user during fix sessions.

**How it works**:
1. Captain writes questions here when user input is needed
2. Captain continues working on non-dependent fixes
3. User answers by editing this file
4. Captain checks periodically and incorporates answers

---

## Template for New Questions

Copy this template when adding new questions:

```markdown
## Question <N> (<ISO-8601 timestamp>)

**Status**: PENDING
**Blocking**: Yes/No
**Issue ID**: <affected fix ID>
**Question**: <clear, specific question>
**Context**: <relevant background>
**Options**:
- Option A: <description>
- Option B: <description>

**Captain's Recommendation**: <what captain will do if no answer by deadline>
**Deadline**: <ISO-8601 timestamp, typically 30-60 min from question time>

### User Answer
(User fills this in)
```

---

## Monitoring Instructions

To watch for new questions:
```bash
watch -n 30 'cat .post_release_recommendation_fixes/USER_QUESTIONS.md | grep -A 20 "Status: PENDING"'
```

To answer: Edit this file directly and save.
