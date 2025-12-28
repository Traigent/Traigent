# Start Release Review

Copy and paste this prompt to your Claude Code CLI to start a release review.

---

## Quick Start (Copy This)

```
You are the release-review captain for Traigent SDK.

Read and follow the protocol in `.release_review/CAPTAIN_PROTOCOL.md`.

Your task:
1. Create branch `release-review/<version>` from latest main
2. Tag baseline as `<version>-rc1`
3. Execute the review loop until ALL components are Approved
4. DO NOT STOP until complete or I explicitly tell you to stop

Key rules:
- You orchestrate, agents review (use Task tool to spawn agents)
- Continue start-to-finish without waiting for me
- If you need my input, write to `.release_review/USER_QUESTIONS.md` and check periodically
- Make judgment calls when reasonable; document decisions in tracking file

Start with version: v<VERSION>
```

---

## Full Initialization Prompt

For more control, use this expanded prompt:

```
You are the release-review captain for Traigent SDK version <VERSION>.

## Protocol
Read and strictly follow: `.release_review/CAPTAIN_PROTOCOL.md`

## Your Mission
Execute a complete pre-release review from start to finish:
1. Branch from latest main → `release-review/<VERSION>`
2. Tag baseline → `<VERSION>-rc1`
3. Review ALL components (30 total) using priority order
4. Apply fixes as needed
5. Generate audit trail when complete

## Execution Model
- **Continuous**: Do NOT pause for my approval between batches
- **Autonomous**: Make judgment calls; document in tracking file
- **Parallel**: Use Task tool to spawn review agents (max 3 concurrent)
- **Non-blocking**: If you need input, use the async question protocol (see below)

## Async Question Protocol
When you need user input but can continue working:
1. Write question to `.release_review/<VERSION>/USER_QUESTIONS.md`
2. Continue with other components that don't depend on the answer
3. Check the file periodically (every 2-3 batches)
4. If answer provided, incorporate it; if not, use best judgment

## Rotation
This is round <N>. Use rotation schedule from:
`.release_review/automation/rotation_scheduler.py generate <N> <VERSION>`

## Stop Conditions
Only stop when:
- ALL components are Approved, OR
- A Tier 3 conflict requires human escalation (document and wait), OR
- I explicitly say "stop" or "pause"

## Start Now
Begin immediately. First action: create branch and read tracking file.
```

---

## Example Usage

### Starting v0.9.0 (Round 2)

```
You are the release-review captain for Traigent SDK version v0.9.0.

Read and follow: `.release_review/CAPTAIN_PROTOCOL.md`

This is round 2. Generate rotation schedule:
.venv/bin/python .release_review/automation/rotation_scheduler.py generate 2 v0.9.0

Execute complete review start-to-finish. Use async question protocol if needed.
Do NOT stop until all components are Approved or you hit a Tier 3 conflict.

Start now.
```

### Re-review After Major Changes

```
You are the release-review captain for Traigent SDK.

Task: Re-review components affected by recent security changes.

Scope: Only these components:
- Security & privacy (traigent/security/)
- Core orchestration (traigent/core/)
- Cloud client (traigent/cloud/)

Use DIFFERENT models than v0.8.0 review (rotation required).
Reference: `.release_review/v0.8.0/ROTATION_HISTORY.md`

Start now. Continue until complete.
```

---

## Monitoring Progress

While the captain works, you can monitor:

```bash
# Watch tracking file
watch -n 30 'tail -50 .release_review/PRE_RELEASE_REVIEW_TRACKING.md'

# Check for questions
cat .release_review/<VERSION>/USER_QUESTIONS.md

# View trace log
cat .release_review/<VERSION>/TRACE_LOG.md
```

---

## Answering Questions

If the captain writes questions to `USER_QUESTIONS.md`:

1. Open the file
2. Add your answer under each question
3. Save the file
4. Captain will pick up answers on next check

Example:
```markdown
## Question 1 (2025-12-13T10:30:00Z)
**Status**: PENDING
**Question**: Should we accept the in-memory token revocation risk for v0.9.0?
**Context**: Same as v0.8.0, Redis integration not ready.

### User Answer
Yes, accept the risk. Document in ACCEPTED_RISKS.md. Plan Redis for v1.0.0.
```
