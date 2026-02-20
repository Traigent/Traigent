# Refactoring Plan: Separate "Mock LLM" from "Offline Backend" Terminology

## Current State: All TRAIGENT Environment Variables

### Mode/Behavior Flags (FOCUS OF THIS REFACTOR)

| Variable | Purpose | Current Behavior |
|----------|---------|------------------|
| `TRAIGENT_MOCK_LLM` | Confusing: Mocks LLM + implies offline backend | Check `is_mock_mode()` |
| `TRAIGENT_MOCK_LLM` | Already exists but unused | Not implemented |
| `TRAIGENT_OFFLINE_MODE` | Already exists | Explicit offline backend mode |
| `TRAIGENT_ALLOW_HTTP_IN_MOCK` | Bridge: allow backend in mock mode | Used in trial_operations |
| `TRAIGENT_REAL_MODE` | Debug script only | For reproducibility tracking |

### Other flags (NOT changing)

- API/Backend: `TRAIGENT_API_KEY`, `TRAIGENT_API_URL`, `TRAIGENT_BACKEND_URL`, JWT vars
- Cost: `TRAIGENT_COST_APPROVED`, `TRAIGENT_RUN_COST_LIMIT`, etc.
- Logging/Debug: `TRAIGENT_LOG_LEVEL`, `TRAIGENT_DEBUG`, etc.

---

## Problem Statement

The codebase uses `TRAIGENT_MOCK_LLM` for two distinct purposes:

1. **Mock LLM** - Mocking LLM provider calls with simulated responses
2. **Offline Backend** - Skipping Traigent backend communication

This is confusing. The term "mock" should ONLY refer to LLM mocking.

## Goals

1. Rename `TRAIGENT_MOCK_LLM` → `TRAIGENT_MOCK_LLM` (LLM mocking only)
2. Use `TRAIGENT_OFFLINE_MODE` for backend offline behavior (already exists)
3. Remove `TRAIGENT_ALLOW_HTTP_IN_MOCK` (no longer needed with separate flags)
4. **NO backward compatibility** - clean break with explicit error if old var is set
5. **Raise error** if `TRAIGENT_MOCK_LLM` is still set to make breaking change explicit

## New Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `TRAIGENT_MOCK_LLM=true` | Mock LLM API calls with simulated responses | `false` |
| `TRAIGENT_OFFLINE_MODE=true` | Skip Traigent backend communication | `false` |

**Removed:**

- `TRAIGENT_MOCK_LLM` - replaced by `TRAIGENT_MOCK_LLM`
- `TRAIGENT_ALLOW_HTTP_IN_MOCK` - no longer needed (use `OFFLINE_MODE=false`)

## Behavior Matrix

```text
MOCK_LLM=T, OFFLINE=T  → Mock LLM calls, Skip backend (local testing)
MOCK_LLM=T, OFFLINE=F  → Mock LLM calls, Send to backend (dev with tracking)
MOCK_LLM=F, OFFLINE=T  → Real LLM calls, Skip backend (air-gapped env)
MOCK_LLM=F, OFFLINE=F  → Real LLM calls, Send to backend (production)
```

## Semantic Decisions

### Cost Enforcement → Tied to `MOCK_LLM` (not OFFLINE)

Cost enforcement bypasses tracking when LLMs are mocked because there are no real costs:

- `MOCK_LLM=true` → Skip cost tracking (no real LLM costs)
- `OFFLINE=true` → Still track costs (real LLM calls may happen)

### JWT Secret Generation → Tied to `MOCK_LLM` + development mode

Development/mock environments can use generated ephemeral secrets. Keep tied to `is_mock_llm() or is_development()`.

### .env Warning Suppression → Tied to `MOCK_LLM`

**Decision:** Suppress when `MOCK_LLM=true`

**Justification:** When mocking LLMs, you don't need real API keys, so the warning is noise. In offline mode with real LLMs, you DO need API keys, so the warning should still appear.

### Backend Warning Logging → Tied to `OFFLINE` (use `is_backend_offline()`)

Backend connection warnings should be suppressed in offline mode (you expect no backend), NOT in mock LLM mode. This includes:

- `api_operations.py` line 389: Use `is_backend_offline()` not `is_mock_llm()`
- `session_operations.py` line 276: Use `is_backend_offline()`
- `backend_session_manager.py` line 82: Use `is_backend_offline()`
- API key warnings: Suppress in offline mode since you don't need backend auth

### Session ID strings → Keep `mock_session_*` naming

The backend API expects these field names. Changing them would be a cross-repo contract change.

---

## Files to Modify

### Phase 1: Core Configuration

**File: `traigent/utils/env_config.py`**

1. **At module import time**: Check if `TRAIGENT_MOCK_LLM` is set and raise error with migration message
2. Rename `is_mock_mode()` → `is_mock_llm()`
3. Update to check `TRAIGENT_MOCK_LLM` instead of `TRAIGENT_MOCK_LLM`
4. Simplify `is_backend_offline()` to only check `TRAIGENT_OFFLINE_MODE`
5. Remove `TRAIGENT_ALLOW_HTTP_IN_MOCK` handling
6. Update line 32 (suppress .env warning): Change from direct `os.getenv("TRAIGENT_MOCK_LLM")` to call `is_mock_llm()` (AFTER the deprecation check)
7. Update line 181 (`get_jwt_secret`) to keep `is_mock_llm() or is_development()` (both conditions!)

**Error message for migration:**

```text
TRAIGENT_MOCK_LLM is deprecated and no longer supported.
Please migrate to:
  - TRAIGENT_MOCK_LLM=true (to mock LLM API calls)
  - TRAIGENT_OFFLINE_MODE=true (to skip backend communication)
For local testing, set both: TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true
```

### Phase 2: Cost Enforcement

**File: `traigent/core/cost_enforcement.py`**

1. Line 11: Update docstring `TRAIGENT_MOCK_LLM` → `TRAIGENT_MOCK_LLM`
2. Line 196 (`_check_mock_mode`): Check `TRAIGENT_MOCK_LLM`
3. Property name stays `is_mock_mode` (internal, refers to LLM mock semantics)

### Phase 3: Backend Communication Files

**File: `traigent/cloud/api_operations.py`**

- Remove import of `is_mock_mode`
- Line ~389: Use `is_backend_offline()` for warning suppression (NOT `is_mock_llm()`)
- Lines 217-219: Already uses `is_backend_offline()` ✓
- Update comment at line 216 to remove `TRAIGENT_ALLOW_HTTP_IN_MOCK` reference

**File: `traigent/cloud/trial_operations.py`**

- Remove `_allow_http_in_mock_mode()` helper (lines 38-50)
- Update lines 250, 557, 698 to use `is_backend_offline()` directly
- Remove import of `is_mock_mode`, add `is_backend_offline`

**File: `traigent/cloud/session_operations.py`**

- Line 276: Use `is_backend_offline()` for backend warning suppression

**File: `traigent/core/backend_session_manager.py`**

- Line 82: Use `is_backend_offline()` for backend warning suppression

**File: `traigent/core/orchestrator.py`**

- Update any `is_mock_mode()` → appropriate function

### Phase 4: LLM Mocking Files

**File: `traigent/core/optimized_function.py`**

- Lines 107, 1369: Update `TRAIGENT_MOCK_LLM` → `TRAIGENT_MOCK_LLM`

**File: `traigent/integrations/utils/mock_adapter.py`**

- Line 52, 98: Update `TRAIGENT_MOCK_LLM` → `TRAIGENT_MOCK_LLM`

**File: `traigent/evaluators/local.py`**

- Update mock mode config references to use `TRAIGENT_MOCK_LLM`

**File: `traigent/evaluators/base.py`**

- `_apply_mock_delay_if_enabled()` → use `is_mock_llm()`

### Phase 5: Other Core Files (CORRECTED file paths)

**File: `traigent/cloud/api_key_manager.py`** (NOT security/)

- Line 161: Use `is_backend_offline()` for API key warning suppression (backend auth not needed in offline mode)

**File: `traigent/cli/optimization_validator.py`** (NOT utils/)

- Line 253: Update direct `os.environ.get("TRAIGENT_MOCK_LLM")` → use `is_mock_llm()` from env_config

**File: `traigent/evaluators/metrics_tracker.py`**

- Line 966: Update direct `os.environ.get("TRAIGENT_MOCK_LLM")` → use `is_mock_llm()` from env_config

**File: `traigent/config/backend_config.py`** (NEW - not in original plan)

- Line 185: Use `is_backend_offline()` for backend warning suppression (matches semantics)

**File: `traigent/utils/diagnostics.py`**

- Line 129: Update ENVIRONMENT_VARIABLES list to include both `TRAIGENT_MOCK_LLM` and `TRAIGENT_OFFLINE_MODE`
- Line 254: Update logic that checks `TRAIGENT_MOCK_LLM` → use `is_mock_llm()`
- Line 315: Update logic that checks `TRAIGENT_MOCK_LLM` → use `is_mock_llm()`

**File: `traigent/utils/reproducibility.py`**

- Lines 72-73: Update safe_env_vars to track both `TRAIGENT_MOCK_LLM` and `TRAIGENT_OFFLINE_MODE`
- Remove `TRAIGENT_MOCK_LLM` from tracked vars
- Keep `TRAIGENT_REAL_MODE` for now (used by debug_evaluation.py script)

**File: `traigent/utils/error_handler.py`**

- Line 191: Update error message to reference new env vars

**File: `traigent/api/decorators.py`**

- Line 1109: Update any mock mode messages

**File: `traigent/core/optimized_function.py`** (COMPLETE list)

- Line 107: Update direct `os.getenv("TRAIGENT_MOCK_LLM")` → use `is_mock_llm()`
- Line 129: Update user-facing string mentioning `TRAIGENT_MOCK_LLM`
- Line 140: Update user-facing string mentioning `TRAIGENT_MOCK_LLM`
- Line 1369: Update direct `os.environ.get("TRAIGENT_MOCK_LLM")` → use `is_mock_llm()`
- Line 1929: Update `is_mock_mode()` → `is_mock_llm()` for CI approval bypass

**File: `traigent/evaluators/local.py`**

- Line 153: Update user-facing string mentioning `TRAIGENT_MOCK_LLM`
- Line 1100: Update direct `os.environ.get("TRAIGENT_MOCK_LLM")` → use `is_mock_llm()`

**File: `traigent/evaluators/base.py`**

- Line 779: Update direct `os.environ.get("TRAIGENT_MOCK_LLM")` → use `is_mock_llm()`
- Line 1097: Update user-facing string mentioning `TRAIGENT_MOCK_LLM`

**File: `traigent/integrations/utils/mock_adapter.py`**

- Line 98: Update direct `os.getenv("TRAIGENT_MOCK_LLM")` → use `is_mock_llm()`

### Phase 6: Documentation and Configuration

**File: `CLAUDE.md`**

- Line 56: `TRAIGENT_MOCK_LLM=true` → `TRAIGENT_MOCK_LLM=true`
- Add note about `TRAIGENT_OFFLINE_MODE` for backend offline

**File: `Makefile`**

- Lines 44, 47, 50, 53: `TRAIGENT_MOCK_LLM=true` → `TRAIGENT_MOCK_LLM=true`

**File: `README.md`**

- Update any mock mode references

**Files: `docs/*`**

- Search and update documentation

**Files: `.env.example`, `configs/env-templates/*`**

- Update env var examples

### Phase 7: Test Files

**IMPORTANT: Behavior change for tests**

Previously, tests running under pytest with `TRAIGENT_MOCK_LLM=true` automatically got HTTP backend calls allowed via `_allow_http_in_mock_mode()` which checked `PYTEST_CURRENT_TEST`. After removing this helper:

- Tests that need backend code paths must explicitly set `TRAIGENT_OFFLINE_MODE=false`
- Tests that want to skip backend must set `TRAIGENT_OFFLINE_MODE=true`

**File: `tests/conftest.py`**

- Update monkeypatch to use `TRAIGENT_MOCK_LLM` and `TRAIGENT_OFFLINE_MODE`
- For most tests: Set both `TRAIGENT_MOCK_LLM=true` and `TRAIGENT_OFFLINE_MODE=true`

**All test files using mock mode:**

Global search and replace:

- `TRAIGENT_MOCK_LLM` → `TRAIGENT_MOCK_LLM`
- `is_mock_mode` → `is_mock_llm` (where referring to function)

Key test files:

- `tests/unit/utils/test_env_config.py` - Update tests for renamed function
- `tests/unit/test_safeguards.py` - Lines 516, 577, 659
- `tests/unit/cli/test_validation_*.py` - Multiple locations
- `tests/unit/integrations/test_mock_adapter.py`
- `tests/integration/test_mock_mode_metrics.py`

### Phase 8: Scripts, CI, and Examples

**File: `.github/workflows/traigent-ci-gates.yml`**

- Line 12: `TRAIGENT_MOCK_LLM` → `TRAIGENT_MOCK_LLM`

**Files: `scripts/*`**

- `scripts/validation/verify_installation.py`
- `scripts/setup/quickstart.py`
- `scripts/hooks/performance_check.py`
- `scripts/examples/run_examples.py`

**Files: `examples/*` and `use-cases/*`**

- All files checking `TRAIGENT_MOCK_LLM`

---

## Implementation Order

1. **Phase 1:** Update `env_config.py` (core configuration)
2. **Phase 2:** Update `cost_enforcement.py` (uses mock for cost bypass)
3. **Phase 3:** Update backend communication files (use offline mode)
4. **Phase 4:** Update LLM mocking files (use mock_llm)
5. **Phase 5:** Update other core files
6. **Phase 6:** Update documentation
7. **Phase 7:** Update tests
8. **Phase 8:** Update scripts, CI, and examples

---

## Critical Files Summary

| File | Key Changes |
|------|-------------|
| `traigent/utils/env_config.py` | Import-time error for old env var, rename `is_mock_mode` → `is_mock_llm`, simplify `is_backend_offline` |
| `traigent/core/cost_enforcement.py` | Update env var name to `TRAIGENT_MOCK_LLM` |
| `traigent/cloud/trial_operations.py` | Remove `_allow_http_in_mock_mode()`, use `is_backend_offline()` |
| `traigent/cloud/api_operations.py` | Use `is_backend_offline()` for backend warnings |
| `traigent/cloud/session_operations.py` | Use `is_backend_offline()` for backend warnings |
| `traigent/cloud/api_key_manager.py` | Use `is_backend_offline()` for API key warnings |
| `traigent/config/backend_config.py` | Use `is_backend_offline()` for backend warnings |
| `traigent/core/backend_session_manager.py` | Use `is_backend_offline()` for backend warnings |
| `traigent/core/optimized_function.py` | Update all direct env var reads + user-facing strings + CI approval |
| `traigent/cli/optimization_validator.py` | Update direct env var read → `is_mock_llm()` |
| `traigent/evaluators/metrics_tracker.py` | Update direct env var read → `is_mock_llm()` |
| `traigent/evaluators/local.py` | Update direct env var read + user-facing strings |
| `traigent/evaluators/base.py` | Update direct env var read + user-facing strings |
| `traigent/integrations/utils/mock_adapter.py` | Update direct env var read → `is_mock_llm()` |
| `traigent/utils/diagnostics.py` | Track both new flags + update logic at lines 254, 315 |
| `traigent/utils/reproducibility.py` | Track both new env vars (keep `TRAIGENT_REAL_MODE`) |
| `Makefile` | Update test commands to use both new env vars |
| `CLAUDE.md` | Update documentation |
| `tests/conftest.py` | Update fixtures to set both `TRAIGENT_MOCK_LLM` and `TRAIGENT_OFFLINE_MODE` |

---

## Search Patterns for Comprehensive Update

```bash
# Find all TRAIGENT_MOCK_LLM references
grep -r "TRAIGENT_MOCK_LLM" --include="*.py" --include="*.md" --include="*.yml" --include="*.yaml" --include="Makefile" --include="*.sh"

# Find all is_mock_mode function calls
grep -r "is_mock_mode" --include="*.py"

# Find TRAIGENT_ALLOW_HTTP_IN_MOCK references
grep -r "TRAIGENT_ALLOW_HTTP_IN_MOCK" --include="*.py"
```

---

## Testing Strategy

After refactoring:

1. Run `TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true pytest tests/` - should work as before
2. Verify `TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=false` mocks LLMs but allows backend
3. Verify `TRAIGENT_OFFLINE_MODE=true` skips backend regardless of LLM mocking
4. Ensure cost enforcement only bypasses when `TRAIGENT_MOCK_LLM=true`
5. Run `make lint` and `make format` to ensure code quality
