---
name: consult-codex
description: Consult with Codex (GPT-5.3 xhigh reasoning) for second opinions on technical decisions, documentation, architecture, and cross-team communications. Use when you need expert review before finalizing important deliverables.
allowed-tools: Bash(codex:*), Bash(mkdir:*), Bash(cat:*), Bash(date:*), Bash(tee:*), Read, Write, Glob
---

# Consult Codex Skill

Get expert second opinions from Codex (GPT-5.3) on technical decisions, documentation, and cross-team communications.

## Usage

- `/consult-codex` - Interactive consultation
- `/consult-codex "review these FE instructions"` - With context
- "ask codex about this architecture"
- "get codex's opinion on this document"

## When to Use

- Before sending instructions to other teams (FE, BE, QA)
- Reviewing API contracts or schema definitions
- Architecture decisions with trade-offs
- Documentation that will be shared externally
- Any deliverable where a second opinion adds value

## Consultation Protocol

### 1. Prepare Context

Gather all relevant information:
- The document/decision to review
- Background context (what problem it solves)
- Specific questions or concerns
- Constraints or requirements

### 2. Structure the Request

```text
# Consultation Request

## Context
[Brief background - 2-3 sentences max]

## Document/Decision Under Review
[The actual content to review - paste or summarize]

## Specific Questions
1. [Question about correctness/completeness]
2. [Question about clarity for target audience]
3. [Question about potential issues]

## Constraints
- [Target audience: e.g., "Frontend team with TypeScript expertise"]
- [Timeline: e.g., "Needs to be actionable immediately"]
- [Other relevant constraints]
```

### 3. Execute Consultation

```bash
mkdir -p tmp/codex-consult
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
CONSULT_FILE="tmp/codex-consult/${TIMESTAMP}_consultation.md"
```

**For document review:**
```bash
codex exec -m GPT-5.3-Codex -c 'model_reasoning_effort="xhigh"' "
[CONSULTATION REQUEST HERE]

Please review and provide:
1. Factual errors or inconsistencies
2. Missing information the audience needs
3. Clarity improvements
4. Potential confusion points
5. Overall assessment (ready to send / needs revision)
" 2>&1 | tee "$CONSULT_FILE"
```

**For decision review:**
```bash
codex exec -m GPT-5.3-Codex -c 'model_reasoning_effort="xhigh"' "
[DECISION CONTEXT HERE]

Please evaluate:
1. Technical soundness
2. Risks or edge cases missed
3. Alternative approaches worth considering
4. Recommendation (proceed / reconsider / needs more info)
" 2>&1 | tee "$CONSULT_FILE"
```

### 4. Summarize Findings

After receiving feedback, present to user:

```text
## Codex Consultation Summary

**Model:** Codex GPT-5.3 (o3, xhigh reasoning)
**Topic:** [what was reviewed]
**Verdict:** [Ready / Needs Revision / Major Issues]

### Key Feedback
1. [Most important point]
2. [Second point]
3. [Third point]

### Recommended Changes
- [ ] [Change 1]
- [ ] [Change 2]

### My Assessment
- **Agree with:** [items and why]
- **Disagree with:** [items and why]
- **Will incorporate:** [specific changes]

### Next Steps
[What to do now]
```

## Model Selection

Default: `codex exec -m GPT-5.3-Codex -c 'model_reasoning_effort="xhigh"'`

Options:
- `GPT-5.3-Codex` - Best model for complex reasoning (default)
- `-c 'model_reasoning_effort="xhigh"'` - Maximum reasoning effort
- `-c 'model_reasoning_effort="high"'` - Faster, still thorough

Override with args: `/consult-codex -c 'model_reasoning_effort="high"' "quick check on this"`

## Important Rules

1. **No secrets** - Never include API keys, tokens, or credentials in requests
2. **Concise context** - Codex works better with focused requests
3. **Specific questions** - Vague requests get vague answers
4. **Apply judgment** - Codex suggestions are advisory, not mandatory
5. **Document rationale** - Explain why you accept/reject suggestions
6. **Cross-team comms** - Always consult before sending instructions to other teams

## Example: Reviewing FE Instructions

```bash
codex exec -m GPT-5.3-Codex -c 'model_reasoning_effort="xhigh"' "
# Consultation: Frontend Instructions Review

## Context
We're sending Phase 3 implementation instructions to the Frontend team for
multi-agent workflow metadata support. Schema and backend are complete.

## Document
[paste the FE instructions markdown]

## Questions
1. Are the TypeScript interfaces correct and complete?
2. Is anything missing that FE would need to implement this?
3. Are there any breaking changes not clearly called out?
4. Is the priority ordering logical for FE workflow?

## Constraints
- FE team has strong TypeScript skills
- They haven't seen the schema changes yet
- Backend API is already deployed
"
```

## Cleanup

Consultation files are kept for reference. To clean up:
```bash
rm -rf tmp/codex-consult/
```
