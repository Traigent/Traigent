# Traceability Taxonomy Reference

**Source**: "Established Taxonomies & Standards for Traceability Tagging" (2024-11-22)

This document captures the industry taxonomies and standards we actively reference when tagging Traigent code artifacts. Use it as the canonical lookup when selecting `CONC-*` identifiers or when justifying new tags inside RFCs, trace reports, or CodeSync metadata. All vocabularies below are **closed/controlled**; updates require an RFC plus CodeSync validator changes.

## 0. Tag dimensions & constraints

- **Layer (required, exactly 1):** `CONC-Layer-*` — choose from: API, Core, Integration, Infra, Data, Tooling
- **Quality (optional, 0–3):** `CONC-Quality-*` ordered by priority. Files with 4+ quality tags should be split or reduced.
- **Optional, closed vocabularies:** `CONC-View-*`, `CONC-Lifecycle-*` (if used), `CONC-Domain-*`, `CONC-Compliance-*`, `CONC-ML-*`
- **Deprecated layers:** CrossCutting, Experimental, Util, Test — see `docs/traceability/taxonomy.yaml` for migration guidance.
- **Deprecated quality tag:** `CONC-Quality-Compliance` — use a normal quality plus a `CONC-Compliance-*` tag when needed.
- **If a file spans layers,** move to symbol-level entries in `trace_links.json` instead of stacking layer tags.

## 1. Quick Comparison Table

| Taxonomy / Standard | Scope & Key Terms | How it influences tags | Adoption notes |
| --- | --- | --- | --- |
| **Layered Architecture + C4 Containers** (Bass/Clements/Kazman, Simon Brown) | Structural breakdown: API, Core/Service, Integration/Adapter, Infrastructure, Data/Schema, Tooling, CrossCutting; C4 adds Context → Container → Component → Code levels. | Drives the `CONC-Layer-*` namespace: every file must resolve to one structural layer/component. | Mirrors how LangChain, Ray Serve, SageMaker, Vertex AI surface APIs vs runtime vs connectors; keeps docs aligned with C4 diagrams. |
| **ISO/IEC 25010** (SQuaRE quality model) | Eight characteristics + sub-attributes: Functional Suitability, Performance Efficiency, Compatibility, Usability, Reliability, Security, Maintainability, Portability. | Drives the `CONC-Quality-*` namespace (0–3 per file). Sub-attributes can be captured in trace metadata when specificity is required. | Referenced by ML frameworks when labeling latency, reliability, portability, etc.; provides common language with QA teams. |
| **ISO/IEC/IEEE 42010** (Architecture Description) | Core concepts: stakeholder, concern, viewpoint, view, correspondence, decision rationale. | Supplies concern vocabulary when explaining why a file carries a given layer/quality tag; tie CodeSync nodes back to stakeholder concerns. | Azure/GCP architecture centers and internal Traigent RFCs use 42010-style viewpoint tables. |
| **ISO/IEC/IEEE 24765** (Vocabulary) | Canonical glossary of SE terms: requirement, verification, traceability, etc. | Ensures descriptive labels in tags/metadata use standard terminology (e.g., "verification", "architecture design"). | Prevents ad-hoc wording across repos; aligns with other ISO docs we reference. |
| **SEI Quality Attribute Taxonomy / ATAM** | Ten core attributes split between runtime (performance, availability, security, usability, etc.) and design/operation (modifiability, testability, deployability, maintainability). | Use scenario phrasing (stimulus, response, response measure) when documenting why a `CONC-Quality-*` tag applies. | ATAM is common in reliability reviews for orchestrators, message buses, and ML serving stacks. |
| **IEEE 1471 / Viewpoint Libraries (4+1, arc42)** | Multi-view architecture descriptions: Logical, Development, Process, Physical, Scenarios, etc. | Provide optional `CONC-View-*` metadata when a file is called out in multiple views (e.g., Security view). | Keeps links between code and architecture diagrams consistent. |
| **INCOSE MBSE / ISO 15288 process taxonomy** | Lifecycle artifacts: Stakeholder Requirements, System Requirements, Architecture, Implementation, Integration, Verification, Validation. Also SoS classifications. | When mapping requirements/functionalities, reuse these lifecycle terms for `REQ-*`, `FUNC-*`, or optional `CONC-Lifecycle-*` tags. | Used heavily in regulated industries (aerospace, automotive, medtech) and in Traigent partner audits. |
| **Enterprise Architecture Frameworks (Zachman, TOGAF)** | Domains (Business, Data, Application, Technology) and interrogatives (What, How, Where, Who, When, Why) across roles. | Optional cross-cutting tags when code must be traceable to EA domains or compliance controls. | Helpful when sharing trace data with enterprise architecture or compliance teams. |
| **ML/MLOps Artifact Ontologies (TFX MLMD, Kubeflow, Vertex AI)** | Artifact types: Dataset, Model, Metrics, Hyperparameters; pipeline steps: Ingestion, Training, Evaluation, Deployment. | Provide secondary tags (`CONC-ML-*`) when a module implements a pipeline phase or produces a first-class ML artifact; integrate with CodeSync graph for lineage. | Built into TensorFlow Extended, Kubeflow Pipelines, Vertex AI – ensures our trace graph can interoperate with ML metadata stores. |
| **NIST AI RMF / SOC 2 / ISO 27001 controls** | Risk governance functions (Govern, Map, Measure, Manage) and control families (access control, audit logging, retention). | Introduces compliance-focused concept tags (e.g., `CONC-Compliance-RiskMgmt`) paired with relevant quality tags (e.g., Security, Observability). | Matches how cloud providers label compliance-critical services; supports customer audits. |

## 2. Structural Layers (Layered Architecture + C4)

Use **exactly one** `CONC-Layer-*` tag per file.

| Tag | Definition | Example files | Mapping rationale |
| --- | --- | --- | --- |
| `CONC-Layer-API` | Public entry points: decorators, CLI, SDK surface. | `traigent/api/decorators.py`, `traigent/cli/main.py` | C4 "Container" boundary exposed to users. |
| `CONC-Layer-Core` | Optimization orchestration, objectives, lifecycle controllers. | `traigent/core/orchestrator.py`, `traigent/core/optimized_function.py` | C4 Component responsible for domain logic. |
| `CONC-Layer-Integration` | External adapters, plugins, telemetry bridges. | `traigent/integrations/*` | C4 Components that connect to other systems. |
| `CONC-Layer-Infra` | Cloud/hybrid plumbing, storage backends, network clients. | `traigent/cloud/backend_client.py`, `traigent/storage/local_storage.py` | Maps to C4 deployment/process views. |
| `CONC-Layer-Data` | Schemas, TVL specs, dataset loaders, DTOs. | `traigent/tvl/spec_loader.py`, `optigen_schema/*` | Aligns with TOGAF Data domain. |
| `CONC-Layer-Tooling` | Scripts, trace analyzers, dev automation. | `docs/traceability/scripts/*`, `tools/traceability/*` | Tracks supporting software not shipped to end users. |
| ~~`CONC-Layer-CrossCutting`~~ | **DEPRECATED** — Migrate to Infra (plumbing), Core (domain logic), or Data (metadata). See `taxonomy.yaml`. | — | Use decision tree; avoid catch-all buckets. |
| ~~`CONC-Layer-Test`~~ | **DEPRECATED** — Tests remain untagged by convention. | `tests/*` | Only tag if explicitly approved. |
| ~~`CONC-Layer-Experimental`~~ | **DEPRECATED** — Migrate to Integration (adapters) or Core (domain logic). See `taxonomy.yaml`. | — | Sandbox code should mature or be removed. |

**Layer decision tree (apply in order):**
1) Exposed to end users (decorator/CLI/SDK)? → API  
2) Talks directly to external services/platforms? → Integration  
3) Manages storage/network/runtime plumbing? → Infra  
4) Encodes domain/lifecycle logic? → Core  
5) Primarily schemas/specs/DTOs? → Data  
6) Developer-only tooling/scripts? → Tooling  
7) Cross-layer concerns (logging/retry/config)? → Infra (if plumbing) or Core (if domain-specific)  
8) Tests? → Remain untagged by convention  
If none apply, escalate for review—do not default to catch-all buckets.

When a file truly spans multiple layers, use symbol-level entries in `trace_links.json` rather than stacking layer tags.

## 3. Quality Attributes (ISO/IEC 25010 + SEI scenarios)

Select **0–3** quality tags per file, listed in priority order. If a file seems to need 4+, consider whether it should be split or if some concerns are secondary.

| Tag | Description | Example concerns | Typical modules |
| --- | --- | --- | --- |
| `CONC-Quality-Performance` | Time behavior, throughput, resource usage. | Latency SLAs, budget schedulers. | Orchestrator loops, optimizer kernels. |
| `CONC-Quality-Reliability` | Availability, fault tolerance, recovery. | Retry logic, circuit breakers, stop conditions. | `utils/retry_consolidated.py`, lifecycle controllers. |
| `CONC-Quality-Security` | Confidentiality, integrity, authn/z. | JWT validation, secret management. | `security/jwt_validator.py`, cloud invokers. |
| `CONC-Quality-Compatibility` | Interoperability, co-existence, plugin contracts. | Multi-provider adapters, SDK shims. | `integrations/*`, plugin registries. |
| `CONC-Quality-Observability` | Logging, tracing, metrics, telemetry (runtime insights). | Experiment analytics, log sinks. | `analytics/intelligence.py`, telemetry emitters. |
| `CONC-Quality-Maintainability` | Modularity, configurability, testability (ease of change). | Registries, config builders, dependency injection. | `core/registry.py`, config helpers. |
| `CONC-Quality-Usability` | Learnability, operability, API ergonomics. | Decorators, CLI UX, documentation helpers. | `api/decorators.py`, CLI utilities. |
| `CONC-Quality-Portability` | Adaptability, installability, hybrid mobility. | Execution mode resolvers, path abstraction. | `config/resolve_execution_mode.py`. |

Guidance for tricky boundaries:
- Logging/metrics/exporters → Observability, **not** Maintainability.
- Plugin registries/config injection to ease change → Maintainability, **not** Observability.

When documenting why a tag applies, reference the relevant SEI quality attribute scenario (stimulus, environment, response, response measure) in RFCs or review notes.

## 4. Architecture Views & Lifecycle Links (42010, IEEE 1471, INCOSE)

- **Views (closed set):** `CONC-View-Logical`, `CONC-View-Development`, `CONC-View-Process`, `CONC-View-Physical`, `CONC-View-Security`, `CONC-View-Scenarios`. Use only if the module is explicitly called out in that view; store in `trace_links.json`.
- **Lifecycle (optional, closed set if used):** `CONC-Lifecycle-Requirements`, `CONC-Lifecycle-Architecture`, `CONC-Lifecycle-Implementation`, `CONC-Lifecycle-Integration`, `CONC-Lifecycle-Verification`, `CONC-Lifecycle-Validation`. If unused, omit rather than inventing values.
- **Decisions & rationale:** Keep references to ADRs/RFCs per ISO 42010. The taxonomy reference ensures the `Concern` column uses standard terms (performance, compliance, usability, etc.).

## 5. Enterprise & Compliance Domains (Zachman, TOGAF, NIST)

Use domain tags when code must be traceable into enterprise governance artifacts:

- `CONC-Domain-Business`, `CONC-Domain-Data`, `CONC-Domain-Application`, `CONC-Domain-Technology`, `CONC-Domain-Motivation`, `CONC-Domain-Process`.
- `CONC-Compliance-*` for NIST AI RMF functions (Govern, Map, Measure, Manage) or specific control families (e.g., `CONC-Compliance-GDPR-Retention`, `CONC-Compliance-SOC2-Audit`).

Compliance is **not** a quality tag: pair the relevant quality (Security, Observability, etc.) with a `CONC-Compliance-*` tag when the code exists to satisfy a control.

## 6. ML/MLOps Phase Tags

For repositories that include ML optimization or evaluation flows, align modules with the standard MLOps pipeline phases:

| Tag | Meaning | Example |
| --- | --- | --- |
| `CONC-ML-Ingestion` | Data acquisition, validation, feature prep. | Dataset loaders, feature builders. |
| `CONC-ML-Training` | Hyperparameter tuning, training loops. | Optimizers interfacing with model trainers. |
| `CONC-ML-Evaluation` | Metric computation, comparison, bias tests. | Benchmark suites, evaluator plugins. |
| `CONC-ML-Deployment` | Serving, routing, canarying. | Model invocation bridges, deployment scripts. |
| `CONC-ML-Lineage` | Metadata, provenance, artifact tracking. | Trace export, metadata sync. |

These tags are **orthogonal** to layers: e.g., a Core file can also be `CONC-ML-Training`. Map them to Kubeflow/TFX artifact types where applicable so CodeSync can align with external ML metadata stores.

## 7. Governance & Enforcement

- **Who tags:** Authors add/update tags; reviewers enforce. New vocab requires an RFC and updates to this file and CodeSync validators.
- **CI/lint:** Fail if layer missing or >1, >3 qualities, unknown/deprecated tags, or tags outside closed vocab. Provide remediation hints pointing to this doc.
- **Conflicts:** If Core vs Infra vs Integration is disputed, follow the decision tree; escalate before merging rather than defaulting to CrossCutting.
- **Tests/experimental:** Tag only if explicitly in scope; otherwise leave untagged and document the policy.

## 8. Usage Checklist

1. **Select structural layer** (exactly 1) and add the tag to the file header or `trace_links.json`.
2. **Select quality tags** (0–3), referencing SEI scenarios for rationale.
3. **Optional:** Add view, lifecycle (if used), domain, compliance, or ML phase tags from the closed vocabularies.
4. **Document** in RFCs/PR notes when introducing or modifying tags; cite this reference.
5. **Verify** via CodeSync reports (scan) that new tags pass validation and produce expected clusters; keep zero-orphan status.

Keep this file synchronized across all Traigent-owned repositories to ensure consistent traceability semantics.
