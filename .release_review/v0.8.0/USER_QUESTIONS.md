# User Questions: v0.8.0 Release Review

This file is used for asynchronous communication between the captain and the user during the release review.

**How it works**:
1. Captain writes questions here when user input is needed
2. Captain continues working on non-dependent components
3. User answers by editing this file
4. Captain checks periodically and incorporates answers

---

## Question 1 (2025-12-13T19:30:00Z) - RESOLVED

**Status**: RESOLVED
**Blocking**: No
**Component**: Security & privacy
**Question**: Should we accept the in-memory token revocation store risk for v0.8.0?
**Context**: The JWT validator uses an in-memory set for revoked tokens. This means revoked tokens are lost on restart and not shared across instances. Redis integration is possible but not implemented.
**Options**:
- Option A: Accept risk for SDK use case (single instance, not multi-server)
- Option B: Block release until Redis integration added
- Option C: Add file-based persistence as stopgap

**Captain's Recommendation**: Option A - SDK is typically single-instance, this matches other SDK patterns
**Deadline**: 2025-12-13T20:00:00Z

### User Answer
Accepted. Option A is fine for SDK use case. Document as accepted risk. Plan Redis for v1.0.0 if we add server mode.

---

## Template for New Questions

Copy this template when adding new questions:

```markdown
## Question <N> (<ISO-8601 timestamp>)

**Status**: PENDING
**Blocking**: Yes/No
**Component**: <component name>
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
watch -n 30 'cat .release_review/v0.8.0/USER_QUESTIONS.md | grep -A 20 "Status: PENDING"'
```

To answer: Edit this file directly and save.
