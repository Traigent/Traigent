# Reliability Sweep Post-Sync Status

Date: 2026-02-21
Branch: `feat/reliability-sweep`
Synced to: `origin/develop` @ `0c7fcd3`

## Summary

- Total findings reviewed: 35
- Open: 0
- Partially fixed: 2
- Fixed: 31
- Stale/superseded (report drift): 2

## Status Matrix

| Finding | Status | Notes |
|---|---|---|
| F-001 | FIXED | `_calculate_cost()` now re-raises in strict mode via `is_strict_cost_accounting()` (`traigent/agents/platforms.py:734`). |
| F-002 | FIXED | `estimate_cost()` no longer returns magic `0.001`; it logs and re-raises (`traigent/agents/platforms.py:763`). |
| F-003 | FIXED | `_safe_calculate_*` now propagate; no broad swallow in these helpers (`traigent/utils/cost_calculator.py:768`). |
| F-004 | FIXED | Cost estimator now uses model-aware pricing via `get_model_token_pricing()` with conservative fallback (`traigent/core/cost_estimator.py:48`). |
| F-005 | FIXED | Resilient wrapper now propagates critical exceptions (`CostLimitExceeded`, `OptimizationAborted`, `CostTrackingRequiredError`) (`traigent/integrations/wrappers.py:24`). |
| F-006 | FIXED | `pre_trial_validate_config()` now raises wrapped `TVLConstraintError` on constraint exceptions (`traigent/core/orchestrator_helpers.py:331`). |
| F-007 | PARTIAL | Inconsistency reduced after cost refactor, but alias sources are still duplicated (`traigent/utils/cost_calculator.py:498`, `traigent/hooks/validator.py:22`). |
| F-008 | FIXED | Retry defaults now use shared `MAX_RETRIES` constant in all targeted auth/retry paths (`traigent/cli/auth_commands.py:200`, `traigent/cloud/auth.py:1430`, `traigent/cloud/password_auth_handler.py:228`, `traigent/cloud/token_manager.py:326`). |
| F-009 | FIXED | Heuristic tier fallback removed from primary pricing path; `get_model_token_pricing()` is now fail-fast (`traigent/utils/cost_calculator.py:590`). |
| F-010 | FIXED | Divergence threshold is now cached at init and invalid env values fail fast (`traigent/core/cost_enforcement.py:231`, `traigent/core/cost_enforcement.py:263`). |
| F-011 | FIXED | Default model fallback now imports `DEFAULT_MODEL` for the previously flagged paths (`traigent/agents/platforms.py:125`, `traigent/agents/platforms.py:511`, `traigent/utils/constraints.py:498`). |
| F-012 | STALE | Cited paths no longer implicitly return `None`; they return fallback IDs / booleans with logging (`traigent/cloud/session_operations.py:385`, `traigent/cloud/trial_operations.py:352`, `traigent/cloud/trial_operations.py:926`). |
| F-013 | FIXED | Tier-pricing duplication removed from current implementation (no `_TIER_PRICING` path remains). |
| F-014 | FIXED | Default run budget now centralized as `DEFAULT_COST_LIMIT_USD` and reused by enforcer + orchestrator (`traigent/core/cost_enforcement.py:45`, `traigent/core/cost_enforcement.py:112`, `traigent/core/orchestrator.py:389`). |
| F-015 | FIXED | Removed sampler abstraction layers (`BaseSampler`/`SamplerFactory`) and switched to direct random sampler construction via `create_sampler()` + `RandomSampler` (`traigent/core/samplers/__init__.py:38`, `traigent/core/samplers/random_sampler.py:81`). |
| F-016 | FIXED | `custom_params` extraction is now centralized via `_extract_custom_params()` with typed fallback + warning (no broad swallow) and adopted by all targeted plugins (`traigent/integrations/llms/base_llm_plugin.py:169`, `traigent/integrations/llms/openai_plugin.py:177`, `traigent/integrations/llms/langchain_plugin.py:337`). |
| F-017 | FIXED | Consolidated lifecycle naming/structure: one public lifecycle manager (`SessionLifecycleManager`) backed by private registry (`_SessionStateRegistry`) with compatibility alias retained (`traigent/cloud/sessions.py:378`, `traigent/cloud/sessions.py:944`, `traigent/cloud/sessions.py:1242`). |
| F-018 | FIXED | `SATConstraintValidator` is now a concrete compatibility adapter delegating validation/satisfiability checks to `PythonConstraintValidator` (no `NotImplementedError`) (`traigent/api/validation_protocol.py:431`, `traigent/api/validation_protocol.py:443`, `traigent/api/validation_protocol.py:452`). |
| F-019 | PARTIAL | Logger facade logs exceptions, but still degrades to no-op logger state on init failure (`traigent/core/logger_facade.py:34`). |
| F-020 | STALE | Cited `TrialResult` coercion properties no longer exist in current structure; finding needs re-scoping. |
| F-021 | FIXED | Complex fuzzy matcher removed from current cost calculator implementation. |
| F-022 | FIXED | Removed deprecated pricing alias `FALLBACK_MODEL_PRICING`; canonical map remains `ESTIMATION_MODEL_PRICING` (`traigent/utils/cost_calculator.py:50`). |
| F-023 | FIXED | Removed deprecated alias `_fallback_cost_from_tokens`; callers/tests now use `_estimation_cost_from_tokens` (`traigent/utils/cost_calculator.py:538`). |
| F-024 | FIXED | Ragas optional import guard now catches `ImportError` specifically (no broad module-level `except Exception`) (`traigent/evaluators/base.py:88`). |
| F-025 | FIXED | Extracted shared permit/cost core logic into private helpers used by both sync/async wrappers (`_acquire_permit_locked`, `_release_permit_locked`, `_track_cost_locked`) (`traigent/core/cost_enforcement.py:884`, `traigent/core/cost_enforcement.py:919`, `traigent/core/cost_enforcement.py:964`). |
| F-026 | FIXED | Session comparison now respects inferred objective direction for minimize metrics via shared helper (`traigent/utils/objectives.py:17`, `traigent/cloud/sessions.py:341`, `traigent/cloud/sessions.py:918`). |
| F-027 | FIXED | Legacy normalization no longer clips to `[0,1]`; out-of-range signal preserved (`traigent/api/types.py:727`). |
| F-028 | FIXED | Orchestrator `best_result` now uses min/max based on objective direction (`traigent/core/orchestrator.py:530`). |
| F-029 | FIXED | `_calculate_cost()` now uses canonical token-based path (`traigent/agents/platforms.py:734`). |
| F-030 | FIXED | Case/alias handling now normalized with canonical candidate resolution (`traigent/utils/cost_calculator.py:590`, `traigent/utils/cost_calculator.py:188`). |
| F-031 | FIXED | Added one-time cold-start warning when estimates remain at seed value after repeated unknown-cost trials with zero samples (`traigent/core/cost_enforcement.py:46`, `traigent/core/cost_enforcement.py:1128`, `traigent/core/cost_enforcement.py:1132`). |
| F-032 | FIXED | `extract_trial_cost()` now reads metadata parity keys `total_cost` and `cost` in addition to `total_example_cost` (`traigent/core/cost_estimator.py:180`). |
| F-033 | FIXED | Override path now emits a DEBUG log when replacing caller-provided kwargs with config values (`traigent/integrations/wrappers.py:138`). |
| F-034 | FIXED | Removed validation theater (`errors.append` + `errors.clear`) for unknown-model checks; plugins now warn without fabricating/clearing errors (`traigent/integrations/llms/anthropic_plugin.py:104`, `traigent/integrations/llms/gemini_plugin.py:113`, `traigent/integrations/llms/mistral_plugin.py:122`, `traigent/integrations/llms/openai_plugin.py:109`). Regression warning coverage is in `tests/unit/integrations/test_model_validation_warnings.py:27`. |
| F-035 | FIXED | Encryption/decryption now use mutable key buffers with best-effort zeroization in `finally` (`_to_mutable_key_buffer`, `_zeroize_key_buffer`) (`traigent/security/encryption.py:145`, `traigent/security/encryption.py:187`, `traigent/security/encryption.py:281`). |

## Priority Now

Highest-priority still-open correctness/budget items:
- None
