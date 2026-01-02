# Session Artifacts

This directory contains agent outputs for this fix session.

## Structure

```
artifacts/
└── <issue-id>/
    └── <model>/
        ├── analysis.md      # Initial analysis of the fix
        ├── implementation.md # Implementation details
        └── test_results.md   # Test evidence
```

## Example

```
artifacts/
└── 001/
    └── claude/
        ├── analysis.md
        ├── implementation.md
        └── test_results.md
```

## Creating Artifact Directories

```bash
mkdir -p artifacts/<issue-id>/<model>
```

## Required Sections for implementation.md

```markdown
# Implementation: <Issue ID> - <Title>

**Agent**: <Model Name>
**Branch**: fix/<issue-id>/<description>
**Timestamp**: <ISO-8601>
**Release Version**: <version>

## Changes Made

| File | Change Type | Lines Changed |
|------|-------------|---------------|
| path/to/file.py | Modified | +10, -3 |

## Implementation Details

(Describe what was changed and why)

## Test Evidence

- **Command**: `pytest tests/...`
- **Exit code**: 0
- **Summary**: X passed, 0 failed

## Evidence (JSON)

- Copy the JSON evidence recorded in TRACKING.md for this fix.

## Verification Checklist

- [ ] Fix addresses the original recommendation
- [ ] Tests pass (new and existing)
- [ ] No regressions
- [ ] Code follows project conventions
```
