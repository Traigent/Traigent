# Minimum Integration (API or SDK)

This guide defines the minimum implementation needed for a new user to run one successful optimization flow.

## Path A: API Integration (Hybrid Service)

Implement these endpoints first:

1. `GET /traigent/v1/capabilities`
2. `GET /traigent/v1/config-space`
3. `POST /traigent/v1/execute`

Optional (recommended for two-phase quality evaluation):

1. `POST /traigent/v1/evaluate`

Minimum behavior contract:

1. `capabilities` advertises whether `/evaluate` is supported.
2. `config-space` returns at least one tunable parameter with a valid domain.
3. `execute` accepts `tunable_id`, `config`, `examples`, and returns:
   - `status`
   - `outputs` (with `example_id` and either `output` or `output_id`)
   - `operational_metrics` (cost/latency fields)
4. If combined mode is used, return non-empty `quality_metrics`.
5. If two-phase mode is used, return `quality_metrics: null` (or omit it) and implement `/evaluate`.

## Path B: SDK Integration (Function Optimization)

Minimum components:

1. One callable function to optimize.
2. One evaluation dataset.
3. One objective (`accuracy`, `cost`, or `latency`).
4. One configuration space with at least one tunable.

Minimal checklist:

1. Function can run with deterministic inputs.
2. Dataset includes representative samples.
3. Objective can be computed from returned outputs.
4. At least 3 trial configurations can be sampled.

## Environment Minimum

1. `TRAIGENT_API_KEY` configured.
2. `TRAIGENT_API_URL` configured (if not using default backend URL).
3. For hybrid mode scripts using `requests`, set `User-Agent: Traigent-SDK/1.0`.

## 15-Minute Acceptance Test

A new integration is considered ready when all pass:

1. Service health/capabilities endpoints respond successfully.
2. Config space is discovered without schema errors.
3. One trial executes on a small dataset (3-5 examples).
4. Trial returns non-empty metrics and appears in backend monitoring.
5. Re-running with a different config changes at least one quality or cost metric.

## Troubleshooting Priority

1. Authentication failures (`401/403`): validate key and backend URL.
2. Endpoint mismatch (`404`): verify `/traigent/v1/*` paths.
3. Timeouts (`408` or client timeout): reduce batch size or increase timeout.
4. Rate limits (`429`): retry with backoff.
