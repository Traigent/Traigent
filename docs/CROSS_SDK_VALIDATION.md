# Cross-SDK Validation

Native JS correctness and performance are validated against Python-owned fixtures and benchmark specs in the sibling `Traigent` repo.

## LangChain E2E Parity

Agent-level parity now has a separate report-first benchmark lane built around a shared LangChain simple chain and a neutral benchmark asset package:

- neutral assets: `../traigent-cross-sdk-benchmarks/benchmarks/langchain/simple_chain/v1`
- JS runner: `../traigent-cross-sdk-benchmarks/runners/js/run_langchain_report.mjs`
- Python runner: `../traigent-cross-sdk-benchmarks/runners/python/run_langchain_report.py`
- neutral comparator: `../traigent-cross-sdk-benchmarks/scripts/compare_reports.py`
- orchestration entrypoint: `../traigent-cross-sdk-benchmarks/scripts/run-cross-sdk-langchain-parity.mjs`
- live JS runner: `../traigent-cross-sdk-benchmarks/runners/js/run_langchain_online_report.mjs`
- live Python runner: `../traigent-cross-sdk-benchmarks/runners/python/run_langchain_online_report.py`
- live orchestration entrypoint: `../traigent-cross-sdk-benchmarks/scripts/run-cross-sdk-langchain-online.mjs`

The benchmark uses the same fixture-backed LangChain chain in both runtimes:

- `ChatPromptTemplate -> fixture chat model -> string output parser -> evaluator`
- one shared dataset manifest
- one shared response fixture table
- one shared report schema

Current run matrix:

- `native-grid`: exact parity required
- `native-random`: exact seeded parity required
- `hybrid-optuna-random`: runs only when `TRAIGENT_API_KEY` and `TRAIGENT_BACKEND_URL` or `TRAIGENT_API_URL` are set
- `hybrid-optuna-tpe`: runs only when `TRAIGENT_API_KEY` and `TRAIGENT_BACKEND_URL` or `TRAIGENT_API_URL` are set

The comparator checks:

- trial count
- ordered config sequence
- normalized stop reason
- best config
- best metrics
- per-trial aggregated metrics
- per-trial neutral trace hash
- TrialContext semantics inside evaluation

Performance reporting includes:

- wall-clock time
- throughput
- mean/p50/p95 trial duration
- orchestration overhead
- hybrid backend round-trip latency when hybrid lanes run

Outputs land under `../traigent-cross-sdk-benchmarks/tmp/langchain-e2e/`:

- `js-report.json`
- `python-report.json`
- `comparison.json`
- `comparison.md`

## Live Provider Lane

There is now a separate live-provider lane for observing real-provider behavior with `TRAIGENT_OFFLINE_MODE=false` in both native and hybrid modes.

- default env file: `../Traigent/walkthrough/examples/real/.env`
- preferred provider order: `OPENROUTER_API_KEY`, then `OPENAI_API_KEY`
- current checked-in walkthrough env provides `OPENAI_API_KEY` but not `OPENROUTER_API_KEY`, so OpenAI is the practical default unless OpenRouter is exported in the shell
- native mode creates a typed tracking session and mirrors each completed local trial to `/api/v1/sessions/{id}/results`
- hybrid mode uses the typed interactive session API directly

Outputs land under `../traigent-cross-sdk-benchmarks/tmp/langchain-e2e-online/`:

- `js-report.json`
- `python-report.json`
- `comparison.json`
- `comparison.md`

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
npm run report:langchain-e2e
npm run report:langchain-e2e:online -- --env-file ../Traigent/walkthrough/examples/real/.env --max-trials 2 --dataset-size 2
npm run benchmark:cross-sdk
```

`report:langchain-e2e` builds the SDK, then delegates to the neutral cross-SDK harness in `../traigent-cross-sdk-benchmarks` to run the JS report, the Python report, and the comparison step.

`report:langchain-e2e:online` does the same against real providers and a live backend. It is report-only and may surface mismatches because provider outputs can drift across separate client calls even when the prompts match.

`benchmark:cross-sdk` remains the lower-level synthetic scheduler benchmark.

## Hybrid Session Contract Checks

Hybrid JS optimization adds mocked contract validation around the Python backend session API:

- session creation payload shape
- next-trial suggestion decoding
- result submission payload shape
- stop-reason normalization

These checks live in:

- `tests/unit/optimization/hybrid.test.ts`
