# Safety Gates

Traigent includes several fail-closed or explicit-opt-in gates around real
optimization, CI runs, dataset paths, backend validation, and persisted example
content.

## Cost Coverage Preflight

Real runs check model cost coverage before trials start. Unknown or unpriced
models are detected from the configuration space or current/default config.

With `TRAIGENT_STRICT_COST_ACCOUNTING=true`, unpriced models always raise
`UnknownModelError` before any trial and no prompt is shown.

When the variable is unset, runs whose objectives include `cost` are strict at
runtime only: this preflight still follows the approval rules below, because it runs
before any response exists and cannot see provider-reported costs such as
OpenRouter's. An approved run then stops at the first call that has neither a table
price nor a reported cost, instead of recording `$0` for it. Set
`TRAIGENT_STRICT_COST_ACCOUNTING=false` to keep recording `$0` with a warning.

To proceed after acknowledging that affected results will report `$0` cost while
your provider may still bill you, use one explicit approval:

- pass `cost_approved=True` as a real Python boolean; strings such as `"true"` are ignored,
- set `TRAIGENT_COST_APPROVED=true`; `1`, `yes`, and `on` do not approve this gate,
- or create the XDG cost approval token (`approved` or `approved:<limit>`).

Approved runs still emit the unpriced-model warning. Mock LLM quickstart mode
does not skip this model-pricing preflight (raw provider calls can still be real
under the mock flag); pass `cost_approved=True` or set `TRAIGENT_COST_APPROVED=true`
for a keyless demo that uses unpriced model names.

To fail before any trial starts, enable strict cost accounting:

```bash
export TRAIGENT_STRICT_COST_ACCOUNTING=true
```

Provide explicit pricing for private or unsupported models with:

```bash
export TRAIGENT_CUSTOM_MODEL_PRICING_JSON='{"my-model":{"input_cost_per_token":0.000001,"output_cost_per_token":0.000002}}'
export TRAIGENT_CUSTOM_MODEL_PRICING_FILE=.traigent/model-pricing.json
```

## CI Run Approval

Optimization in CI requires explicit approval. The recommended
environment approval is:

```bash
export TRAIGENT_RUN_APPROVED=1
export TRAIGENT_APPROVED_BY="your_name"
```

`TRAIGENT_MOCK_LLM` does not bypass this CI approval gate.

## Mock Mode and the Cost Preflight

Mock mode alone does not skip the pre-run cost approval or the model-price
coverage preflight, so an agent that mixes real and mock calls is still checked.
Keyless walkthroughs and tests that are genuinely fully mocked opt out with
`TRAIGENT_COST_PREFLIGHT_MOCK_EXEMPT=true`. The exemption applies only when mock
LLM mode is on and the environment (`ENVIRONMENT`, `TRAIGENT_ENV` or
`TRAIGENT_ENVIRONMENT`) is explicitly non-production, e.g. `ENVIRONMENT=development`.
An unset or unknown environment counts as production and never gets the
exemption. It is test/walkthrough-only and must not be set for real-LLM runs. Runtime cost accounting stays active.

## Dataset Path Containment

Evaluation datasets must resolve under the trusted dataset root. Set the root
explicitly when a process should only read datasets from one directory:

```bash
export TRAIGENT_DATASET_ROOT="$PWD/evals"
```

When `TRAIGENT_DATASET_ROOT` is not set, the current working directory is the
trusted root.

## Backend Validation Loopback Gate

Loopback backend validation is default-closed. For local backend development,
use an explicit non-production environment:

```bash
export ENVIRONMENT=development
```

Do not use this override for production.

### Self-hosted private backend (`TRAIGENT_ALLOW_PRIVATE_BACKEND`)

A self-hosted Traigent backend on a private network (RFC 1918, ULA or loopback
address) is refused by default, because the SDK sends your API key to the
configured backend. To allow one, set both variables:

```bash
export TRAIGENT_BACKEND_URL=https://traigent.internal.example:8443
export TRAIGENT_ALLOW_PRIVATE_BACKEND=1
```

Security notes:

- The opt-in is exact: `1` or `true` (case-insensitive). Any other value is off.
- It is pinned to the scheme, host and port of `TRAIGENT_BACKEND_URL`. Any other
  URL, including a different port on the same host, stays refused. With
  `TRAIGENT_BACKEND_URL` unset it does nothing. It never reads `TRAIGENT_API_URL`.
- HTTPS is still required outside a development environment. Plain HTTP is not
  relaxed.
- Cloud metadata and link-local addresses (169.254.0.0/16, fe80::/10 and the
  known metadata hosts) stay blocked in every environment.
- Redirects are refused for the private backend. Any 3xx response from the
  pinned origin is an error, and the redirect target is never requested. This
  covers the SDK's project-management client and the SDK's HTTP clients that
  carry the API key to that origin.
- DNS failures are refused for the pinned host. The opt-in does not accept a
  name it cannot resolve, and it does not skip the lookup for `localhost`
  names. Every resolved answer must pass the always-blocked check.
- A port of `0`, or any port outside 1-65535, is invalid in the configured URL
  and in the candidate URL.
- DNS rebinding is not closed by this check. The SDK validates the resolved
  address but cannot pin the connection to it, so an answer that changes between
  the validation lookup and the connection is not caught.

## Content Logging Opt-Out

Per-example prompt, response, and expected-output content is persisted by
default for local developer experience. To keep IDs and metrics while omitting
content-bearing fields from disk:

```bash
export TRAIGENT_LOG_EXAMPLE_CONTENT=false
```

## Copy-Paste Example

```bash
export TRAIGENT_STRICT_COST_ACCOUNTING=true
export TRAIGENT_DATASET_ROOT="$PWD/evals"
export TRAIGENT_LOG_EXAMPLE_CONTENT=false
```

Honesty note: Traigent cost limits and estimates are local guardrails, not
provider billing caps. Set provider-side billing limits in your LLM or cloud
account for hard spend protection.
