# Changelog

All notable changes to Traigent SDK are documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.11.1] - 2026-04-01

### Fixed
- Quickstart fails when run from any directory other than project root - set `TRAIGENT_DATASET_ROOT` to package directory (#636)
- PyPI publish verification broken pipe - save curl response to file instead of piping (#635)
- Consistent `resolve()` for dataset path in quickstart

## [0.11.0] - 2026-03-30

### Added
- **DeepEval metric integration** - `DeepEvalScorer` bridges DeepEval metrics into Traigent's `metric_functions` system
- **Rich results table** - Formatted ASCII table with box-drawing characters for optimization output
- **Interactive vendor error handling** - Pause/resume prompts on rate limits and budget exhaustion
- **Evaluation kwargs** - Configurable per-evaluation parameters for hybrid wrapper service
- **Run labeling** - Optional `run_label` on `OptimizationResult` for experiment tagging
- **Session creation circuit breaker** - Graceful fallback when backend is unreachable
- **RAG walkthrough demo** - New `walkthrough/demo/optimize_rag.py` example

### Changed
- Default backend URL changed from `localhost:5000` to `https://portal.traigent.ai` - SDK devs should set `TRAIGENT_BACKEND_URL=http://localhost:5000` explicitly
- `ExperimentRunDTO.status` default changed from `"pending"` to `"not_started"`
- `HybridEvaluateRequest.config` type narrowed from `dict[str, Any]` to `dict[str, ScalarValue]` (`config` is now a deprecated alias for `kwargs`)
- Keyring credential storage removed in favor of file-only storage (`~/.traigent/credentials.json` with `0600` permissions) to avoid macOS Keychain prompts
- Schema validator package renamed from `optigen_schemas` to `traigent-schema`

### Removed
- **`benchmark_id` from `HybridExecuteRequest` and `HybridEvaluateRequest`** - Breaking change for external hybrid service consumers
- **`benchmarks_revision` from hybrid request DTOs**
- `hello.py` replaced by `hello_world.py`
- `traigent/experimental/` package (simple_cloud platforms)
- `traigent/conftest.py` (moved to `tests/conftest.py`)

### Fixed
- Credential file write TOCTOU race fixed with atomic `os.open()` + mode `0o600`
- Credential file read now verifies and tightens permissions if overly permissive
- Auth rejection log spam suppressed with instance-scoped deduplication at DEBUG level
- `pydantic` added to core dependencies
- `traigent-schema` made optional for public installs
- Missing `aiohttp.ClientTimeout` added to `DatasetConverter` and OAuth2 sessions
- Vendor error classification anchored to contextual patterns (`"status 429"` instead of bare `"429"`)
- `_safe_copy` in DeepEval integration now raises `TypeError` instead of silently returning shared mutable instance
- Silent `except Exception: pass` in `_finalize_optimization` replaced with debug logging
- Dead methods removed from orchestrator (`_check_batch_vendor_failures`, `_maybe_pause_on_cost_limit`, `_handle_vendor_pause_in_loop`)

### Breaking Changes
- `create_session()` return type changed from `str` to `SessionCreationResult` dataclass.
  `SessionCreationResult.__str__` returns the session ID so most existing code continues
  to work without changes, but explicit `isinstance(result, str)` checks will need updating.

  **Before (v0.10.x):**
  ```python
  session_id: str = create_session()
  ```
  **After (v0.11.0+):**
  ```python
  result = create_session()
  session_id: str = result.session_id   # explicit
  # or: str(result)                      # via __str__
  ```
- `HybridExecuteRequest` and `HybridEvaluateRequest` no longer accept `benchmark_id` parameter
- Default backend URL is now cloud instead of localhost

## [0.10.0] - 2026-02-07

### Added
- **LangChain/LangGraph native callback handler** - Native instrumentation for LangChain and LangGraph workflows via `TraigentCallbackHandler`
- **Langfuse observability bridge** - Integration with Langfuse for unified observability across platforms
- **Namespace parsing utilities** - Agent namespace extraction from span names for multi-agent optimization
- **Agent-specific metrics utilities** - Automatic per-agent metric aggregation for multi-agent workflows
- **Workflow traces visualization** - Graph-based visualization of multi-agent workflow execution
- Multi-agent parameter and measure mapping system
- `AgentConfiguration` types for explicit agent groupings
- `agent` parameter on `Range`, `Choices`, `IntRange` classes
- `agent_prefixes` for prefix-based agent inference
- TVL support for `agent` field on tvars
- Content scoring and data integrity improvements
- Pre-rendered architecture diagram (replaces inline Mermaid)
- Click-to-play demo thumbnails with animated SVG playback

### Changed
- **Root directory reorganization** - Consolidated 40+ root items to ~14
- Moved `baselines/` to `configs/baselines/`, `runtime/` to `configs/runtime/`
- Consolidated `mypy.ini` and `pytest.ini` into `pyproject.toml`
- Backend session metadata now includes `agent_configuration`
- Improved type safety across API types
- Improved node context restoration on chain end to prevent metric misattribution
- README trimmed ~30% with all examples fixed to include required params

### Fixed
- Agent ID validation to fix edge cases in multi-agent workflows
- Mock mode metrics simulation for cost and accuracy in demos

## [0.9.0] - 2025-01-09

### Added
- Core optimization decorator (`@traigent.optimize`)
- Grid, Random, and Bayesian optimization strategies
- TVL (Traigent Variable Language) specification support
- LangChain and DSPy integration adapters
- Edge analytics execution mode
- Cloud and hybrid execution modes
- Privacy-preserving optimization mode
- Parallel trial execution support
- Stop conditions (plateau detection, max trials)
- Plugin architecture for extensibility

### Changed
- Refactored plugin architecture for modularity

### Security
- JWT-based authentication for cloud operations
- Input validation and sanitization
