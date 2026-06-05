# Traigent SDK Credential and Data Trust Model

This page documents what the Python SDK sends out of the local process today. It is intentionally narrow: it describes the SDK boundary and the backend payloads the SDK builds, not the full Traigent backend retention policy.

## Summary

- Provider credentials such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and AWS provider credentials are client-side provider credentials. The SDK does not use them as Traigent backend credentials.
- The Traigent API key, including the `sk_` key returned by device login, authenticates Traigent backend and portal operations only.
- `edge_analytics` means trials execute locally. It does not mean "no network" by itself: user code can still call model providers, and backend tracking or analytics can run if configured. Set `TRAIGENT_OFFLINE_MODE=true` for the SDK's no-Traigent-backend-egress mode.
- `hybrid` means local trials plus backend session, trial, metrics, and portal tracking. Provider credentials are not part of the backend auth path, but trial configs and metadata are submitted to the backend, so do not put secrets in configs or metadata.
- Per-example local content logging is opt-out today. Set `TRAIGENT_LOG_EXAMPLE_CONTENT=false` or pass `log_example_content=False` to omit query, response, and expected content from optimization log files.

## Credential Boundary

### Provider credentials

Provider keys are used by provider SDKs or user code. The SDK's provider key helper reads provider-specific environment variables such as `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`, or keys explicitly set for a provider, and returns them to local provider integrations (`traigent/config/api_keys.py:23-76`). Provider validation also reads provider env vars and constructs provider clients locally, for example `OpenAI(api_key=key)` and `Anthropic(api_key=key)` (`traigent/providers/validation.py:439-461`, `traigent/providers/validation.py:512-538`). Bedrock support follows the AWS SDK boundary: `BedrockChatClient` creates a local `boto3` or `aioboto3` session, optionally with `profile_name` and `region_name`, then invokes `bedrock-runtime` directly (`traigent/integrations/bedrock_client.py:101-113`, `traigent/integrations/bedrock_client.py:152-157`, `traigent/integrations/bedrock_client.py:351-364`).

The LiteLLM and LangChain interceptors wrap the provider library call and capture metadata after the call returns. They do not route the model call through Traigent: `litellm.completion` / `acompletion` are called first, then response metadata is captured (`traigent/utils/litellm_interceptor.py:48-81`, `traigent/utils/litellm_interceptor.py:95-129`); `ChatAnthropic.invoke` and `ChatOpenAI.invoke` are likewise called first, then metadata is captured (`traigent/utils/langchain_interceptor.py:193-241`, `traigent/utils/langchain_interceptor.py:281-328`).

### Traigent backend credentials

Traigent backend auth resolves Traigent credentials, not provider credentials. The backend credential resolver loads `TRAIGENT_API_KEY`, stored CLI credentials, `TRAIGENT_JWT_TOKEN`, OAuth client credentials, service credentials, or explicit Traigent API key parameters (`traigent/cloud/credential_resolver.py:185-241`). `AuthManager` also seeds its API-key manager from an explicit API key or `TRAIGENT_API_KEY` (`traigent/cloud/auth.py:391-401`).

Backend HTTP headers are generated from the Traigent auth state. API-key requests emit `X-API-Key`; JWT requests emit `Authorization: Bearer <jwt>`; common SDK/backend headers are added afterward (`traigent/cloud/auth.py:812-930`). Backend callers merge those auth headers and fail closed rather than issuing unauthenticated requests when auth header generation fails (`traigent/cloud/backend_components.py:138-171`).

Device login also returns a Traigent backend key. The CLI validates that the device-token response contains an `api_key` with an `sk_` prefix plus tenant/project metadata (`traigent/cli/auth_commands.py:568-615`), writes it as `TRAIGENT_API_KEY` when requested (`traigent/cli/auth_commands.py:405-424`), and validates API keys against the backend `/keys/validate` endpoint with `X-API-Key` (`traigent/cli/auth_commands.py:462-487`). The SDK API-key auth path validates keys against the backend and fails closed on validation or transport failure (`traigent/cloud/auth.py:1074-1155`, `traigent/cloud/auth.py:1386-1420`).

## What Data Leaves the Machine

### `edge_analytics`

Trials and provider calls execute in the local process. The SDK does not make provider calls on Traigent's behalf; provider calls happen only because user code, provider integrations, or validation code call provider SDKs with user provider credentials.

`edge_analytics` is not a blanket no-egress switch. The backend session manager skips backend-client initialization only when `TRAIGENT_OFFLINE_MODE=true` (`traigent/core/backend_session_manager.py:173-175`). Without offline mode, a backend client can be initialized and, when a Traigent API key is configured, session tracking can send function metadata, search space, dataset size, and related session metadata (`traigent/core/backend_session_manager.py:206-237`, `traigent/core/backend_session_manager.py:360-399`; session payload shape in `traigent/cloud/session_operations.py:305-318` and `traigent/cloud/api_operations.py:270-291`). After a backend session is created, the SDK may upload deterministic `simhash_v1` example features for backend-side analytics, not raw example text (`traigent/core/backend_session_manager.py:277-306`, `traigent/metrics/content_features.py:29-63`, `traigent/cloud/backend_client.py:979-1035`).

Local usage analytics can submit aggregated usage stats when `enable_usage_analytics=True`, `execution_mode="edge_analytics"`, a Traigent API key exists, and offline mode is not set (`traigent/utils/local_analytics.py:60-74`, `traigent/utils/local_analytics.py:221-310`; orchestrator gate in `traigent/core/orchestrator.py:1822-1836`).

To make Traigent backend communication skip deliberately, set:

```bash
export TRAIGENT_OFFLINE_MODE=true
```

That flag is evaluated per call and is documented in code as skipping Traigent backend communication (`traigent/utils/env_config.py:450-485`). Session creation and trial/result submission paths check it and return local/offline behavior instead of posting (`traigent/cloud/api_operations.py:216-231`, `traigent/cloud/trial_operations.py:222-230`, `traigent/cloud/trial_operations.py:536-547`). Standalone backend clients also fail closed before transport in offline mode (`tests/unit/test_offline_mode_clients.py:1-7`, `tests/unit/test_offline_mode_clients.py:100-117`).

If you also need no provider calls or provider spend, do not call provider-backed user code, or use explicit mock/test paths such as `TRAIGENT_MOCK_LLM=true` where appropriate. Offline mode is a Traigent backend-egress switch, not a provider-egress switch (`traigent/utils/env_config.py:459-462`).

### `hybrid`

Hybrid is local trial execution plus backend/portal tracking. The backend-tracked privacy session path says the client executes trials locally and uses backend session APIs for tracking; it does not create a remote cloud execution session (`traigent/cloud/privacy_operations.py:43-57`). Session creation sends:

- function/problem statement
- configuration search space
- optimization settings, objectives, and max trials
- dataset metadata, with `dataset.examples` explicitly set to an empty list in the API session payload
- caller-provided session metadata

The session payload is built at `traigent/cloud/api_operations.py:262-291`; the higher-level session operation builds dataset metadata with size and `privacy_mode=True` at `traigent/cloud/session_operations.py:305-318`.

Trial result submission sends the tested config, validated metrics, backend status, mode metadata, optional error string after sanitization, and optional top-level `measures` and `summary_stats` (`traigent/cloud/trial_operations.py:385-405`, `traigent/cloud/trial_operations.py:509-631`). It validates the submission before posting (`traigent/cloud/trial_operations.py:590-596`).

Per-example measures contain stable example IDs and numeric metrics. In non-privacy hybrid mode, all numeric example metrics can be included (`traigent/core/metadata_helpers.py:183-199`, `traigent/core/metadata_helpers.py:430-490`). With `privacy_enabled=True` or legacy `execution_mode="privacy"`, measures are limited to score, timing, token, and cost metrics (`traigent/core/metadata_helpers.py:200-205`, `traigent/core/metadata_helpers.py:510-571`). Summary stats are aggregate statistics, not raw examples (`traigent/core/metadata_helpers.py:145-162`, `traigent/evaluators/metrics_tracker.py:349-439`).

### `hybrid_api`

`hybrid_api` is an external-service execution mode. It requires either a user-configured `hybrid_api_endpoint` or a preconfigured transport (`traigent/core/optimization_pipeline.py:392-423`). The evaluator sends trial config and dataset examples to that external transport in `HybridExecuteRequest` (`traigent/evaluators/hybrid_api.py:620-731`, `traigent/evaluators/hybrid_api.py:733-778`). Treat that endpoint as part of your own trusted runtime boundary.

The Traigent backend credential boundary remains separate: a `hybrid_api_auth_header` is for the external service transport, while `TRAIGENT_API_KEY` / `sk_` is for Traigent backend auth.

### `cloud`

`execution_mode="cloud"` is reserved and not currently supported. The config validator raises and tells users to use hybrid for portal-tracked optimization (`traigent/config/types.py:66-101`). There is no supported SDK path today where Traigent Cloud runs the user's model calls.

## Content Logging Defaults

Local optimization logs persist per-example query, response, and expected content by default. `_should_log_example_content()` returns `True` when `TRAIGENT_LOG_EXAMPLE_CONTENT` is unset, and false-like values (`0`, `false`, `no`, `off`, or an empty string) disable content fields (`traigent/utils/optimization_logger.py:188-205`).

When content logging is disabled, the serializer writes `None` for query, response, and expected fields while retaining IDs and metrics (`traigent/utils/optimization_logger.py:208-257`). The `OptimizationLogger` constructor reads the env switch unless `log_example_content` is passed explicitly (`traigent/utils/optimization_logger.py:260-288`), and trial flushing uses that setting when serializing `example_results` (`traigent/utils/optimization_logger.py:524-557`). Regression tests confirm the default-on behavior, env opt-out values, explicit override, and absence of prompt/expected strings on disk when disabled (`tests/unit/utils/test_optimization_logger_content_optout.py:64-78`, `tests/unit/utils/test_optimization_logger_content_optout.py:84-99`, `tests/unit/utils/test_optimization_logger_content_optout.py:125-141`).

Disable local per-example content logging with either:

```bash
export TRAIGENT_LOG_EXAMPLE_CONTENT=false
```

or an explicit logger setting:

```python
OptimizationLogger(..., log_example_content=False)
```

This flag controls local optimization log files. It does not automatically sanitize arbitrary user-provided backend metadata or payloads sent to a user-configured `hybrid_api` endpoint.

## What the Backend Stores From the SDK

From the SDK side, the backend may receive and store:

- session metadata: function/problem name, search space, optimization settings, max trials, dataset metadata, and caller-provided metadata
- trial lifecycle and results: trial IDs, tested configs, metrics, status, timing, mode metadata, sanitized errors, optional measures, and optional summary stats
- content-analysis features: allowed feature kinds currently include `simhash_v1`, uploaded as stable example IDs plus 64-bit feature strings
- local usage analytics: aggregated counts and usage metrics when enabled and authenticated

The SDK does not send provider credentials as part of the Traigent backend auth path. However, it does send `config` and `metadata` dictionaries supplied by user code. The code redacts sensitive fields for debug logging (`traigent/cloud/trial_operations.py:147-201`) but posts the JSON-sanitized result payload itself (`traigent/cloud/trial_operations.py:385-405`, `traigent/cloud/trial_operations.py:628-631`). Do not place provider keys, secrets, prompts, or sensitive raw content in configuration values, custom metrics, or metadata fields unless you intend that data to be transmitted.

## Guarantees and Limits

Verified guarantees:

- Traigent backend auth uses Traigent credentials (`TRAIGENT_API_KEY`, device-login `sk_` key, JWT/OAuth/service credentials), not provider credentials (`traigent/cloud/credential_resolver.py:185-241`, `traigent/cloud/auth.py:812-930`).
- Provider validation and provider key lookup happen locally against provider-specific env vars and provider clients, including Bedrock's local AWS SDK session/client path (`traigent/config/api_keys.py:51-76`, `traigent/providers/validation.py:439-461`, `traigent/providers/validation.py:512-538`, `traigent/integrations/bedrock_client.py:101-113`).
- Hybrid mode does not send dataset examples in the session-creation payload; it sends empty `dataset.examples` plus dataset metadata (`traigent/cloud/api_operations.py:270-291`).
- Offline mode skips Traigent backend communication at the documented backend request boundaries (`traigent/utils/env_config.py:450-485`, `traigent/cloud/api_operations.py:216-231`, `traigent/cloud/trial_operations.py:222-230`, `traigent/cloud/trial_operations.py:536-547`).
- Local per-example content logging is currently opt-out, not opt-in (`traigent/utils/optimization_logger.py:188-205`).

Softened or not guaranteed:

- `edge_analytics` does not guarantee no provider calls. Provider calls still occur if user code or provider validation invokes provider SDKs.
- `edge_analytics` does not guarantee no Traigent backend communication unless `TRAIGENT_OFFLINE_MODE=true` is set or backend credentials/configuration are absent.
- The SDK cannot prove that arbitrary user-supplied secrets hidden in `config`, custom metrics, or metadata under benign field names will be removed before backend submission.
- This page does not verify backend retention, deletion, access-control, or database storage semantics in `TraigentBackend`; it documents what the SDK sends.
