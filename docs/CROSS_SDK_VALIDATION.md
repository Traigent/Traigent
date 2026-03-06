# Cross-SDK Validation

Phase-1 native JS correctness is validated against Python-owned fixtures and benchmark specs in the sibling `Traigent` repo.

## Correctness Oracles

Python exports deterministic oracle payloads from:

- `../Traigent/tests/cross_sdk_oracles/generate_native_js_oracles.py`

The JS parity tests consume that payload in:

- `tests/cross-sdk/oracle.test.ts`

Covered oracle cases:

- `grid_3x3`: exact config order, best config, best metrics, normalized stop reason
- `random_seed_42`: exact seeded config sequence for deterministic discrete sampling
- `budget_cutoff`: same budget stop point, total cost, best config, normalized stop reason
- `bayesian_branin`: optional outcome-envelope check when the Python Bayesian environment is available

If the sibling Python repo is absent, the cross-SDK oracle test skips locally. Internal release validation must run it.

## Async Benchmark Methodology

Phase 1 does not expose public native trial concurrency yet, so the JS benchmark is a standalone async scheduler harness rather than an `.optimize()` benchmark.

Shared benchmark inputs come from:

- `../Traigent/tests/cross_sdk_oracles/generate_native_js_oracles.py`
- `../Traigent/tests/cross_sdk_oracles/run_async_scheduler_benchmark.py`
- `scripts/run-cross-sdk-benchmark.mjs`

Method:

1. Use a shared fixed config list and deterministic sleep schedule in both runtimes.
2. Run warmups first, then measured runs at concurrency `1`, `2`, `4`, and `8`.
3. Compare normalized scheduler behavior rather than absolute language speed.
4. Treat the output as release reporting in phase 1, not release gating.

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

The benchmark command builds the SDK, runs the JS scheduler harness, executes the Python benchmark script, and prints a combined JSON report.
