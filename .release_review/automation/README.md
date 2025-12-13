# Release Review Automation Toolkit

This directory contains automation scripts to support the multi-agent release review protocol.

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `artifact_manager.py` | Auto-generate artifact paths, validate report structure | Captain uses to assign paths to agents |
| `scope_guard.py` | Validate agent changes are within assigned scope | Captain runs before approving any agent work |
| `evidence_validator.py` | Parse and validate evidence format | Captain validates all evidence before approval |
| `verify_tests.sh` | Re-run tests to verify agent claims | Captain spot-checks 20% of reviews |
| `checkpoint.py` | Manage incremental progress saving | Agents use for failure recovery |
| `metrics.py` | Track agent effectiveness across releases | Post-release analysis |
| `rotation_scheduler.py` | Generate model rotation schedules | Captain uses for multi-round reviews |

## Quick Start

### For Captain

```python
from artifact_manager import ArtifactManager
from scope_guard import ScopeGuard
from evidence_validator import EvidenceValidator

# Initialize for release
manager = ArtifactManager("v0.8.0")
guard = ScopeGuard()
validator = EvidenceValidator()

# Get artifact path for agent assignment
path = manager.get_component_path("core/orchestrator", "claude")
# Returns: .release_review/v0.8.0/artifacts/core_orchestrator/claude/20251213_findings.md

# Validate agent's changes are in scope
result = guard.validate_changes(
    branch="review/core/claude/20251213",
    allowed_paths=["traigent/core/"]
)
if not result["valid"]:
    print(f"SCOPE VIOLATION: {result['violations']}")

# Validate evidence format
evidence = "Tests: 47/47 | Commits: abc123 | Model: Claude/Opus4.5 | Time: 2025-12-13T10:00:00Z"
parsed = validator.validate(evidence)
if not parsed["valid"]:
    print(f"Invalid evidence: {parsed['error']}")
```

### For Agents

```python
from checkpoint import CheckpointManager

# Initialize checkpoint for your review
checkpoint = CheckpointManager("core/orchestrator", "claude_001")

# Resume or start fresh
state = checkpoint.resume_or_start()

# Save progress periodically
state["files_reviewed"].append("orchestrator.py")
state["issues_found"] = 2
checkpoint.save(state)

# Final save when complete
state["status"] = "complete"
checkpoint.save(state)
```

### Rotation Scheduling

```python
from rotation_scheduler import RotationScheduler

scheduler = RotationScheduler()

# Generate schedule for round 2 (rotated from round 1)
schedule = scheduler.get_schedule(round_number=2, version="v0.9.0")
print(schedule.to_markdown())

# Or auto-rotate from previous release
schedule = scheduler.rotate_from("v0.8.0")

# Compare two rounds
round1 = scheduler.get_schedule(1, "v0.8.0")
round2 = scheduler.get_schedule(2, "v0.9.0")
print(scheduler.generate_comparison(round1, round2))

# View model statistics across all rounds
stats = scheduler.get_model_stats()
```

**CLI Usage**:
```bash
# Generate schedule for round 2
python rotation_scheduler.py generate 2 v0.9.0

# Rotate from previous version
python rotation_scheduler.py rotate v0.8.0

# Compare two rounds
python rotation_scheduler.py compare 1 2

# View assignment statistics
python rotation_scheduler.py stats

# View history
python rotation_scheduler.py history
```

### Verify Test Claims

```bash
# Captain verifies agent's test claims
./verify_tests.sh "traigent/core" 20 20
# Output: VERIFIED or MISMATCH
```

## Integration with Protocol

1. **Before dispatching agents**: Use `artifact_manager.py` to generate artifact paths
2. **During review**: Agents use `checkpoint.py` to save progress
3. **Before approval**: Captain runs `scope_guard.py` and `evidence_validator.py`
4. **Spot-checks**: Captain uses `verify_tests.sh` on 20% of reviews
5. **Post-release**: Use `metrics.py` to analyze agent effectiveness
