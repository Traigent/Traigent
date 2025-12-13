# Post-Release TODO: v0.8.0

**Generated from**: Release review findings
**Date**: 2025-12-13
**Owner**: TBD (assign before next sprint)

This file tracks recommendations identified during the v0.8.0 release review that were not blocking but should be addressed in future releases.

---

## High Priority (Address in v0.8.1 or v0.9.0)

### 1. Atomic Writes for Storage
- **Component**: Storage & Persistence
- **Location**: [traigent/utils/persistence.py](../../traigent/utils/persistence.py), [traigent/storage/local_storage.py](../../traigent/storage/local_storage.py)
- **Issue**: Direct file writes without temp+rename pattern
- **Risk**: Data corruption if process crashes during save
- **Recommendation**:
  ```python
  temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp.{os.getpid()}")
  try:
      with open(temp_path, "w") as f:
          json.dump(data, f, indent=2)
      temp_path.replace(file_path)  # Atomic rename
  finally:
      if temp_path.exists():
          temp_path.unlink()
  ```
- **Effort**: Small (1-2 hours)
- **Assignee**: TBD
- [ ] Implemented
- [ ] Tested
- [ ] Merged

### 2. CI Python Matrix Expansion
- **Component**: CI Workflows
- **Location**: [.github/workflows/tests.yml](../../.github/workflows/tests.yml)
- **Issue**: Only tests Python 3.11, missing 3.8-3.10, 3.12
- **Risk**: Compatibility issues on other Python versions
- **Recommendation**: Expand matrix to cover all supported versions per pyproject.toml
- **Effort**: Small (30 min)
- **Assignee**: TBD
- [ ] Implemented
- [ ] Tested
- [ ] Merged

### 3. RAGAS Config Thread-Safety
- **Component**: Metrics
- **Location**: [traigent/metrics/ragas_metrics.py](../../traigent/metrics/ragas_metrics.py)
- **Issue**: `configure_ragas_defaults()` modifies global `_GLOBAL_RAGAS_CONFIG` without lock
- **Risk**: Race condition in multi-threaded usage
- **Recommendation**:
  ```python
  _RAGAS_CONFIG_LOCK = threading.Lock()

  def configure_ragas_defaults(...):
      global _GLOBAL_RAGAS_CONFIG
      with _RAGAS_CONFIG_LOCK:
          _GLOBAL_RAGAS_CONFIG = RagasConfig(...)
  ```
- **Effort**: Small (30 min)
- **Assignee**: TBD
- [ ] Implemented
- [ ] Tested
- [ ] Merged

### 4. Telemetry Opt-Out Flag
- **Component**: Telemetry
- **Location**: [traigent/telemetry/optuna_metrics.py](../../traigent/telemetry/optuna_metrics.py)
- **Issue**: No environment variable to disable Optuna telemetry
- **Risk**: Users cannot opt out of telemetry
- **Recommendation**: Add `TRAIGENT_DISABLE_TELEMETRY` or `DISABLE_OPTUNA_TELEMETRY` env var
- **Effort**: Small (30 min)
- **Assignee**: TBD
- [ ] Implemented
- [ ] Tested
- [ ] Merged

---

## Medium Priority (Address in v0.9.0+)

### 5. Unused Statement in Error Handler
- **Component**: Utilities
- **Location**: [traigent/utils/error_handler.py:155](../../traigent/utils/error_handler.py)
- **Issue**: `cls.IMPORT_FIXES[module_name]` retrieved but not used
- **Impact**: Installation command never displayed to users on import errors
- **Effort**: Small (15 min)
- **Assignee**: TBD
- [ ] Implemented
- [ ] Tested
- [ ] Merged

### 6. ExecutionMode Enum Consistency
- **Component**: Adapters
- **Location**: [traigent/adapters/execution_adapter.py](../../traigent/adapters/execution_adapter.py)
- **Issue**: RemoteExecutionAdapter and HybridPlatformAdapter return plain strings instead of ExecutionMode enum
- **Risk**: Type inconsistency
- **Recommendation**: Harmonize to use enum values
- **Effort**: Medium (1-2 hours)
- **Assignee**: TBD
- [ ] Implemented
- [ ] Tested
- [ ] Merged

### 7. Agent Resource Cleanup
- **Component**: Agents
- **Location**: [traigent/agents/executor.py](../../traigent/agents/executor.py)
- **Issue**: No explicit cleanup/shutdown method for executors
- **Risk**: Resource leaks if executors acquire persistent resources
- **Recommendation**: Add `async def cleanup()` or `__aexit__`
- **Effort**: Medium (2-3 hours)
- **Assignee**: TBD
- [ ] Implemented
- [ ] Tested
- [ ] Merged

---

## Low Priority (Future Enhancement)

### 8. Documentation Improvements
- Add public API reference guide for decorator signatures
- Document telemetry data collection behavior and retention
- Add more examples of thread pool usage with context propagation
- **Effort**: Medium-Large
- [ ] Started
- [ ] Completed

### 9. Testing Improvements
- Add RemoteExecutionAdapter unit tests
- Add property-based tests (hypothesis) for config validation
- Expand integration tests for additional evaluation types
- Fix 3 flaky tests (test isolation issue, not production bug)
- **Effort**: Large
- [ ] Started
- [ ] Completed

---

## Accepted Risks (Documented)

### In-Memory Token Revocation Store
- **Component**: Security
- **Location**: `traigent/security/jwt_validator.py`
- **Risk**: Token revocation stored in memory, lost on restart
- **Mitigation**: Acceptable for SDK use case (not multi-instance server)
- **Future**: Redis integration is optional/future enhancement
- **Decision Date**: 2025-12-13
- **Approved By**: Release review captain

---

## Tracking

| Priority | Total | Done | Remaining |
|----------|-------|------|-----------|
| High | 4 | 0 | 4 |
| Medium | 3 | 0 | 3 |
| Low | 2 | 0 | 2 |
| **Total** | **9** | **0** | **9** |

**Last Updated**: 2025-12-13
