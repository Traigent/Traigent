# Priority Review Waves (Source-First)

This is the staged review order for the shipped Python source in this repository.

Important repo-specific note:

- There is no repo-root `src/` here.
- The canonical source roots are `traigent/**/*.py` and `traigent_validation/**/*.py`.
- The exact inventory lives in `.release_review/inventories/source_files.txt`.

## Coverage Contract

- Source inventory size: 362 files as of March 6, 2026
- Wave inventories live under `.release_review/inventories/priority_wave*.txt`
- Every source file appears in exactly one wave inventory
- Use `python3 .release_review/automation/verify_source_wave_coverage.py` before relying on the
  plan after source-tree changes

## Selection Principles

The ordering is optimized for expensive CLI review runs:

1. Review the external contract and security boundaries first.
2. Review shared foundations before high-fanout product surfaces.
3. Keep tightly coupled packages in the same review graph where practical.
4. Split oversized packages into multiple inventories only where that materially reduces rerun cost.
5. Make every wave resumable so a stopped review can continue without re-partitioning the tree.

## Wave Summary

| Wave | Priority | Graph | Inventory | Files | Approx lines |
|---|---:|---|---|---:|---:|
| 01 | P0 | Public API contract | `.release_review/inventories/priority_wave01_public_api_contract.txt` | 20 | 11,782 |
| 02 | P0 | Security/auth boundary | `.release_review/inventories/priority_wave02_security_auth_boundary.txt` | 22 | 9,554 |
| 03 | P0 | Config/runtime foundation | `.release_review/inventories/priority_wave03_config_runtime_foundation.txt` | 17 | 5,813 |
| 04 | P0 | Utils foundation | `.release_review/inventories/priority_wave04_utils_foundation.txt` | 31 | 13,924 |
| 05 | P0 | Core orchestration/runtime | `.release_review/inventories/priority_wave05_core_orchestration_runtime.txt` | 23 | 13,410 |
| 06 | P1 | Core metrics/selection | `.release_review/inventories/priority_wave06_core_metrics_selection.txt` | 21 | 6,839 |
| 07 | P1 | Cloud auth/client surface | `.release_review/inventories/priority_wave07_cloud_auth_client_surface.txt` | 16 | 11,054 |
| 08 | P1 | Cloud operations/data graph | `.release_review/inventories/priority_wave08_cloud_operations_data_graph.txt` | 18 | 10,419 |
| 09 | P1 | Integrations provider graph | `.release_review/inventories/priority_wave09_integrations_provider_graph.txt` | 44 | 12,276 |
| 10 | P1 | Execution/transport graph | `.release_review/inventories/priority_wave10_execution_transport_graph.txt` | 45 | 14,320 |
| 11 | P1 | Config generation/tuning | `.release_review/inventories/priority_wave11_config_generation_tuning.txt` | 34 | 11,015 |
| 12 | P1 | Optimizers | `.release_review/inventories/priority_wave12_optimizers.txt` | 20 | 8,901 |
| 13 | P1 | Evaluators + metrics | `.release_review/inventories/priority_wave13_evaluators_metrics.txt` | 13 | 8,744 |
| 14 | P2 | CLI/agents/experimental | `.release_review/inventories/priority_wave14_cli_agents_experimental.txt` | 26 | 11,114 |
| 15 | P2 | Analytics/observability | `.release_review/inventories/priority_wave15_analytics_observability.txt` | 12 | 8,506 |

## Wave Detail

### Wave 01: Public API Contract

Inventory: `.release_review/inventories/priority_wave01_public_api_contract.txt`

Why first:

- public decorators, parameter contracts, and validation protocol
- exported package entry points and client-facing SDK surface
- highest backward-compatibility blast radius

### Wave 02: Security/Auth Boundary

Inventory: `.release_review/inventories/priority_wave02_security_auth_boundary.txt`

Why next:

- auth, credentials, crypto, rate limiting, session handling
- explicit privilege and tenancy boundaries
- highest security risk concentration in shipped code

### Wave 03: Config/Runtime Foundation

Inventory: `.release_review/inventories/priority_wave03_config_runtime_foundation.txt`

Why next:

- runtime configuration ingress
- provider validation and local storage behavior
- small, shared foundation that influences many later graphs

### Wave 04: Utils Foundation

Inventory: `.release_review/inventories/priority_wave04_utils_foundation.txt`

Why next:

- very high fan-out helper layer used across core, cloud, integrations, and optimizers
- common source of subtle determinism and error-handling regressions

### Wave 05: Core Orchestration/Runtime

Inventory: `.release_review/inventories/priority_wave05_core_orchestration_runtime.txt`

Why next:

- optimization pipeline, orchestration, session lifecycle, stop conditions
- release-critical runtime behavior and budget guardrails

### Wave 06: Core Metrics/Selection

Inventory: `.release_review/inventories/priority_wave06_core_metrics_selection.txt`

Why next:

- result selection, statistics, tracing, meta-types, and metric aggregation
- lower external exposure than Wave 05, but still tightly coupled to release correctness

### Wave 07: Cloud Auth/Client Surface

Inventory: `.release_review/inventories/priority_wave07_cloud_auth_client_surface.txt`

Why next:

- remote auth, credentials, sessions, token flow, resilient client behavior
- highest-risk cloud-facing SDK boundary

### Wave 08: Cloud Operations/Data Graph

Inventory: `.release_review/inventories/priority_wave08_cloud_operations_data_graph.txt`

Why next:

- DTOs, synchronization, trial operations, billing, privacy operations
- cloud-side contract integrity and state-transition correctness

### Wave 09: Integrations Provider Graph

Inventory: `.release_review/inventories/priority_wave09_integrations_provider_graph.txt`

Why next:

- model-provider plugins and model discovery logic
- deterministic parameter normalization and optional-dependency behavior

### Wave 10: Execution/Transport Graph

Inventory: `.release_review/inventories/priority_wave10_execution_transport_graph.txt`

Why next:

- hybrid transport, wrappers, bridges, invokers, vector stores, observability adapters
- async, transport, and resource-safety risk concentration

### Wave 11: Config Generation/Tuning

Inventory: `.release_review/inventories/priority_wave11_config_generation_tuning.txt`

Why next:

- generated config presets, tuned-variable discovery, and TVL gating
- correctness matters, but external blast radius is lower than Waves 01-10

### Wave 12: Optimizers

Inventory: `.release_review/inventories/priority_wave12_optimizers.txt`

Why next:

- optimization engines and remote optimizer flows
- performance and state-transition risk without being the first external ingress path

### Wave 13: Evaluators + Metrics

Inventory: `.release_review/inventories/priority_wave13_evaluators_metrics.txt`

Why next:

- evaluation correctness, dataset registry behavior, and metric plumbing
- directly impacts outcome quality and regression confidence

### Wave 14: CLI/Agents/Experimental

Inventory: `.release_review/inventories/priority_wave14_cli_agents_experimental.txt`

Why next:

- important product surface, but less foundational than the runtime and transport graphs
- experimental code is separated so it does not inflate earlier blocking waves

### Wave 15: Analytics/Observability

Inventory: `.release_review/inventories/priority_wave15_analytics_observability.txt`

Why last:

- lower release-blocking risk than contract, runtime, security, or transport layers
- still must be reviewed before claiming full source coverage

## Immediate Start Point

Start with `Wave 01`, then continue in numeric order. If a run stops, resume from the first
incomplete wave rather than creating a new split.
