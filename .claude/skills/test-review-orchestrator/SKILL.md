---
name: test-review-orchestrator
description: Orchestrate test reviews using multiple AI CLI tools (claude, codex). Coordinates parallel review workflows, tracks progress in test_tracking.json, and synthesizes findings. Use for reviewing optimizer validation tests, validating test coverage, or running multi-agent test analysis.
allowed-tools: Read, Write, Edit, Grep, Glob, Bash(python:*), Bash(.venv/bin/python:*), Bash(claude:*), Bash(codex:*), Bash(cat:*), Bash(ls:*), Bash(mkdir:*), Bash(git:*)
---

# Test Review Orchestrator

Orchestrate comprehensive test reviews using multiple AI CLI tools working in parallel.

## Available Commands

```bash
# Show current progress (run this first)
.venv/bin/python tests/optimizer_validation/tools/orchestrator.py status

# Live dashboard with auto-refresh
.venv/bin/python tests/optimizer_validation/tools/live_display.py --watch

# Run a review batch with Claude
.venv/bin/python tests/optimizer_validation/tools/orchestrator.py review --batch-size 10 --tool claude

# Run parallel reviews with both Claude and Codex
.venv/bin/python tests/optimizer_validation/tools/orchestrator.py review --parallel --batch-size 10

# Validate completed reviews
.venv/bin/python tests/optimizer_validation/tools/orchestrator.py validate

# Generate summary report
.venv/bin/python tests/optimizer_validation/tools/orchestrator.py report
```

## Overview

This skill coordinates multiple AI agents to review the Optimizer Validation Test suite:

| Agent            | CLI Tool       | Role                              | Model           |
| ---------------- | -------------- | --------------------------------- | --------------- |
| **Reviewer 1**   | `claude`       | Primary test reviewer             | Claude Sonnet 4 |
| **Reviewer 2**   | `codex`        | Secondary reviewer with code focus| Codex o4-mini   |
| **Validator**    | `claude`       | Cross-validates reviewer findings | Claude Sonnet 4 |
| **Synthesizer**  | (this agent)   | Orchestrates and synthesizes      | Claude Opus 4.6 |

## Quick Start

To start orchestrating test reviews:

```text
Orchestrate test reviews for the optimizer validation suite
```

Or with specific options:

```text
Orchestrate test reviews with 3 parallel workers, reviewing dimensions tests first
```

## Workflow

### Phase 1: Initialize

1. Load `test_tracking.json` to get current progress
2. Identify tests needing review (`review.status == "not_started"`)
3. Display current progress summary

### Phase 2: Dispatch Reviews

1. Claim batch of 10-20 tests
2. Dispatch to reviewers in parallel:
   - Claude CLI for primary review
   - Codex CLI for secondary validation
3. Wait for completions
4. Parse and merge results

### Phase 3: Validate

1. Send completed reviews to validator agent
2. Cross-reference findings against actual code
3. Resolve disagreements
4. Update final verdicts

### Phase 4: Report

1. Update `test_tracking.json` with results
2. Display progress summary
3. Generate synthesis report

## Orchestrator Commands

Run these Python scripts from the skill:

```bash
# Check status
.venv/bin/python tests/optimizer_validation/tools/orchestrator.py status

# Start review batch
.venv/bin/python tests/optimizer_validation/tools/orchestrator.py review --batch-size 10

# Run validation pass
.venv/bin/python tests/optimizer_validation/tools/orchestrator.py validate

# Generate report
.venv/bin/python tests/optimizer_validation/tools/orchestrator.py report
```

## Progress Display

The orchestrator shows live progress:

```text
=== Test Review Orchestrator ===
Progress: [████████░░░░░░░░░░░░] 42% (375/892)

Review Status:
  ✓ Completed: 320
  → In Progress: 55  (claude: 30, codex: 25)
  ○ Not Started: 517

Validation Status:
  ✓ Confirmed: 280
  ✗ Overridden: 15
  ? Pending: 25

Current Batch: dimensions-batch-012
  Reviewer: claude-sonnet-4
  Tests: 10
  ETA: ~3 minutes
```

## File Locations

| File                                               | Purpose               |
| -------------------------------------------------- | --------------------- |
| `tests/optimizer_validation/test_tracking.json`    | Master tracking file  |
| `tests/optimizer_validation/REVIEW_PROTOCOL.md`    | Reviewer instructions |
| `tests/optimizer_validation/VALIDATION_PROTOCOL.md`| Validator instructions|
| `tests/optimizer_validation/tools/orchestrator.py` | Orchestration engine  |

## CLI Tool Usage

### Claude CLI (Primary Reviewer)

```bash
claude -p --output-format json --system-prompt "$(cat REVIEW_PROTOCOL.md)" \
  "Review these tests: [test_ids]" \
  --allowed-tools "Read,Grep"
```

### Codex CLI (Secondary Reviewer)

```bash
codex exec --json \
  "Review these tests following the protocol: [test_ids]"
```

## Error Handling

- **Tool timeout**: Retry up to 3 times with exponential backoff
- **Parse failure**: Log raw output, mark batch as failed
- **Disagreement**: Escalate to human review if unresolvable

## Integration with Existing Protocols

This skill uses the existing protocols:

- `REVIEW_PROTOCOL.md` - Defines the four-question review checklist
- `VALIDATION_PROTOCOL.md` - Defines cross-reference methodology
- `test_tracking.json` - Tracks all test reviews and validations

The orchestrator simply coordinates multiple agents following these protocols.
