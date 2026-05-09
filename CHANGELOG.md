# Changelog

All notable changes to Traigent SDK are documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Removed
- **Breaking in 0.12.0:** removed Python-orchestrated JavaScript optimization through the temporary JS bridge. `ExecutionOptions.runtime`, all `ExecutionOptions.js_*` fields, `traigent.bridges.*`, and `traigent.evaluators.JSEvaluator` are no longer available. JavaScript/TypeScript users should migrate to native `@traigent/sdk` optimization with `optimize(spec)(agentFn)` and `await wrapped.optimize(...)`; see https://github.com/Traigent/traigent-js/blob/main/docs/getting-started/minimal-integration.md and https://github.com/Traigent/traigent-js/blob/main/docs/MIGRATION_FROM_PYTHON.md.

### Security
- Bump `litellm` floor from `>=1.83.0` to `>=1.83.7` to fix [GHSA-xqmj-j6mv-4862](https://github.com/BerriAI/litellm/security/advisories/GHSA-xqmj-j6mv-4862) — prompt-template SSTI in `POST /prompts/test` that could run arbitrary code in the LiteLLM Proxy process. 1.83.7 renders templates in a sandboxed Jinja environment.
- Bump `langchain-core` floor `>=1.2.22` → `>=1.2.28` (CVE-2026-40087 — improper element neutralization in templates).
- Bump `langchain-openai` floor `>=0.3.30` → `>=1.1.14` (GHSA-r7w7-9xr2-qq2r — SSRF). Major-version-line bump; lock currently resolves to 1.2.x.
- Lock-only bumps for transitive deps to clear Aikido CVEs: `fonttools` 4.60.1 → 4.62.1 (CVE-2025-66034), `langgraph-checkpoint` 3.0.0 → 4.0.2 (CVE-2026-27794, unsafe deserialization → RCE), `filelock` 3.20.0 → 3.29.0 (CVE-2025-68146, CVE-2026-22701), `sqlparse` 0.5.3 → 0.5.5 (GHSA-27jp-wm6q-gp25), `pygments` 2.19.2 → 2.20.0 (CVE-2026-4539). Two further uv.lock advisories — `ragas` (CVE-2026-6587, SSRF) and `diskcache` (CVE-2025-69872, code injection) — have no upstream patch yet and remain open; tracked separately.

### Migration notes
- **`langgraph-checkpoint` 3.x → 4.x:** the 4.0 release drops default deserialization of payloads serialized with the legacy `"json"` serde mode. Users running `langgraph` with a persistent checkpoint saver (SQLite, Postgres, Redis, etc.) whose stored state contains JSON-format blobs will see deserialization failures unless they configure `serde` with an explicit `allowed_json_modules` list. Traigent does not bind `langgraph-checkpoint` directly — it arrives transitively through the `langgraph` dependency — so this only affects projects that also use `langgraph` checkpointers in their own code. See [CVE-2026-27794](https://github.com/langchain-ai/langgraph/security/advisories) for the underlying RCE that motivated removing the default.

### Changed
- Cost enforcement invariant checks now treat the cost-limit bound as an admission-time permit rule. `assert_invariants()` still detects stranded permits and reservation drift, but it no longer flags valid post-trial actual-cost overruns while other admitted permits remain in flight.
- `metric_limit` in parallel mode is evaluated after each batch and may overshoot by up to `parallel_trials - 1` trials; use `cost_limit` when a hard spend bound is required.
- Deprecated `budget_limit` stop-condition aliases now report `OptimizationResult.stop_reason="metric_limit"` instead of `"cost_limit"`. Use `cost_limit` for hard USD spend control and `metric_limit` with `metric_name` for soft cumulative metric stopping.
- Built-in hypervolume convergence now reports `OptimizationResult.stop_reason="convergence"`; arbitrary custom stop conditions still report the generic `"condition"` reason unless explicitly mapped.

## [0.11.2] - 2026-04-01

### Fixed
- Quickstart accuracy always 0% with real LLMs - add system prompt for concise answers and use contains-match instead of exact-match (#642)
- CI publish verification retry improvements (#640)

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
