# Cross-SDK Validation

Native JS correctness and performance are validated against Python-owned fixtures and benchmark specs in the sibling `Traigent` repo.

## Correctness Oracles

Python exports deterministic oracle payloads from:

- `../Traigent/tests/cross_sdk_oracles/generate_native_js_oracles.py`

The JS parity tests consume that payload in:

- `tests/cross-sdk/oracle.test.ts`

Covered oracle cases:

- `grid_3x3`: exact config order, best config, best metrics, normalized stop reason
- `random_seed_42`: exact seeded config sequence for deterministic discrete sampling
- `conditional_grid`: exact config order, default fallback application, best config, best metrics
- `conditional_random_seed_7`: exact seeded config sequence for conditional native sampling
- `budget_cutoff`: same budget stop point, total cost, best config, normalized stop reason
- `bayesian_branin`: optional outcome-envelope check when the Python Bayesian environment is available

If the sibling Python repo is absent, the cross-SDK oracle test skips locally. Internal release validation must run it.

## Async Benchmark Methodology

Cross-language async comparison uses a synthetic workload, not real provider calls.

Shared benchmark inputs come from:

- `../Traigent/tests/cross_sdk_oracles/generate_native_js_oracles.py`
- `../Traigent/tests/cross_sdk_oracles/run_async_scheduler_benchmark.py`
- `scripts/run-cross-sdk-benchmark.mjs`

Method:

1. Use the same seeded random config space and the same sleep schedule in both runtimes.
2. Run warmups first, then measured runs at concurrency `1`, `2`, `4`, and `8`.
3. Compare normalized scheduler behavior rather than absolute language speed.

Reported metrics:

- wall-clock time
- throughput
- theoretical minimum makespan
- normalized overhead ratio
- duplicate-config rate
- context-leak count
- RSS delta
- best config
- best latency
- stop reason
- trial count

## Commands

```bash
npm run test:cross-sdk
npm run benchmark:cross-sdk
```

The benchmark command builds the SDK, runs the JS benchmark harness, executes the Python benchmark script, and prints a combined JSON report.

## Hybrid Session Contract Checks

Hybrid JS optimization adds mocked contract validation around the Python backend session API:

- session creation payload shape
- next-trial suggestion decoding
- result submission payload shape
- stop-reason normalization

These checks live in:

- `tests/unit/optimization/hybrid.test.ts`
