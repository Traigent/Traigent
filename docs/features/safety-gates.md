# Safety Gates

Traigent 0.12.0 adds several fail-closed or explicit-opt-in gates around real
optimization, CI runs, dataset paths, backend validation, and persisted example
content.

## Cost Coverage Preflight

Real runs check model cost coverage before trials start. Unknown or unpriced
models are detected from the configuration space or current/default config.

With `TRAIGENT_STRICT_COST_ACCOUNTING=true`, unpriced models always raise
`UnknownModelError` before any trial and no prompt is shown.

To proceed after acknowledging that affected results will report `$0` cost while
your provider may still bill you, use one explicit approval:

- pass `cost_approved=True` as a real Python boolean; strings such as `"true"` are ignored,
- set `TRAIGENT_COST_APPROVED=true`; `1`, `yes`, and `on` do not approve this gate,
- or create the XDG cost approval token (`approved` or `approved:<limit>`).

Approved runs still emit the unpriced-model warning. Mock LLM quickstart mode
skips this model-pricing preflight, but it is a test/demo mode, not a production
billing bypass.

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

Edge analytics optimization in CI requires explicit approval. The recommended
environment approval is:

```bash
export TRAIGENT_RUN_APPROVED=1
export TRAIGENT_APPROVED_BY="your_name"
```

`TRAIGENT_MOCK_LLM` does not bypass this CI approval gate.

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
use an explicit non-production environment and opt in:

```bash
export ENVIRONMENT=development
export TRAIGENT_ALLOW_INSECURE_BACKEND=true
```

Do not use this override for production.

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
