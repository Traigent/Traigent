# Bank install profile: no vendor-cloud fallback

Goal: an install that never contacts Traigent's cloud unless you named the endpoint yourself.

## Settings
```
export TRAIGENT_REQUIRE_EXPLICIT_ENDPOINT=1      # strict endpoint mode (or TRAIGENT_DEPLOYMENT_PROFILE=self_hosted)
export TRAIGENT_BACKEND_URL=https://tg.your-bank.internal   # explicit endpoint (TRAIGENT_API_URL also accepted)
export TRAIGENT_WEB_URL=https://portal.your-bank.internal   # only if a portal link is needed
```
Optional egress-shaping modes (opt-in, off by default): `TRAIGENT_OPTIMIZATION_ONLY=1` and `TRAIGENT_CONTENT_LESS=1`.

## What strict mode does
- Code: `traigent/config/backend_config.py` `is_strict_endpoint_mode()`, `get_default_cloud_url_or_raise()`, `EndpointNotConfiguredError` (a `ConfigurationError`).
- With no endpoint set, every URL resolver (`get_backend_url`, `get_cloud_backend_url`, `get_*_api_url`, `get_cloud_web_url`) raises instead of returning `https://portal.traigent.ai`. Signup hints, the usage-analytics endpoint, the dataset converter and the CSP origin also drop the vendor default.
- It does not change default (non-strict) behaviour.

## Proof of no fallback
`tests/unit/config/test_strict_endpoint_mode.py::TestBankInstallNoVendorFallback::test_bank_install_missing_endpoint_refuses_before_transport`:
strict mode, no endpoint, a real `TraigentCloudClient` construction plus `create_session`, with the HTTP request method and DNS resolution replaced by tripwires. It asserts the configuration error and zero HTTP calls and zero DNS lookups. A control run in default mode reaches DNS for `portal.traigent.ai`, so the tripwire does detect the fallback. Run:
`TRAIGENT_MOCK_LLM=true python -m pytest -q -n0 tests/unit/config/test_strict_endpoint_mode.py`

## Optional egress-shaping modules: what they do and do not guarantee
- `traigent/utils/optimization_only_egress.py`: applies the backend's closed metadata contract locally, just before a payload reaches a transport, and refuses operations the server forbids (trace, dataset, hosted-LLM) for an `optimization_only` workspace. It is a metadata projector: it does not encrypt, and it does not cover what an ordinary session sends outside that contract.
- `traigent/cloud/content_less.py`: session create carries opaque references and closed enums instead of agent key, dataset id, names and evaluator ids; a backend without the `content_less_v1` capability is refused (no plaintext fallback). It covers session-create identity only (phase 3, group 1), not every request.

## Limits you must keep in mind
- Ordinary connected sessions (neither mode on) send configuration-space text and names to the endpoint you configured.
- Local evaluators run as your own code in your own process; the SDK does not sandbox them.
- Strict mode controls where traffic can go, not what the SDK sends to the endpoint you chose.
- Observability and telemetry are not a privacy guarantee. Disable what you do not want sent (for example `TRAIGENT_DISABLE_TELEMETRY`) and review the endpoint it targets.
