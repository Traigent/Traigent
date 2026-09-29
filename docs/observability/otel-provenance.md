# Provenance: OpenTelemetry observability layer

Independent implementation from public specifications and package metadata.
No competitor or instrumentation-library source code was read, copied or
translated; no third-party documentation text, fixtures or assets were
reproduced. Verified on 2026-09-30.

| Source | Version | Licence | URL | Purpose | Covered by |
|--------|---------|---------|-----|---------|-----------|
| OTLP specification | 1.x | spec (CC-BY 4.0 site) | https://opentelemetry.io/docs/specs/otlp/ | wire format, retry status codes, partial success, 413/Retry-After | `test_otel_exporter_retry.py`, `test_otel_api_init.py` |
| OTel SDK trace specification | 1.x | spec | https://opentelemetry.io/docs/specs/otel/trace/sdk/ | sampler decisions, RECORD_ONLY meaning, parent-based sampling | `test_otel_sampling.py`, `test_otel_processor.py` |
| OpenTelemetry GenAI semantic conventions | Development status | spec | https://github.com/open-telemetry/semantic-conventions-genai | attribute NAMES only (interoperability) | `test_otel_content_policy.py`, `test_otel_contract_conformance.py` |
| OpenInference semantic conventions | main, 2026-09-30 | spec | https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md | attribute and span-kind NAMES only | `test_otel_contract_conformance.py` |
| Traigent receiver contract snapshot | TraigentBackend feat/otlp-ingest @ f780d4ef2 | Traigent | (internal) | shared attribute contract, hash-locked | `test_otel_contract_conformance.py` |

Packages (versions and licences from PyPI metadata; "used" = imported by the
SDK code and tests, "optional" = named in an extra only, not exercised here):

| Package | Version | Licence | Role |
|---------|---------|---------|------|
| opentelemetry-api / -sdk / -proto / -exporter-otlp-proto-common | 1.44.0 (tested); 1.45.0 is current on PyPI | Apache-2.0 | used |
| opentelemetry-semantic-conventions | 0.65b0 (transitive) | Apache-2.0 | used (transitive) |
| protobuf | 5.29.6 (tested) | BSD-3-Clause | used (transitive) |
| openinference-instrumentation-openai | 0.1.61 | Apache-2.0 | optional |
| openinference-instrumentation-anthropic | 2.1.7 | Apache-2.0 | optional |
| openinference-instrumentation-langchain | 0.1.76 | Apache-2.0 | optional |
| openinference-instrumentation-bedrock | 0.1.54 | Apache-2.0 | optional |
| openinference-instrumentation | 0.1.66 | Apache-2.0 | optional (transitive) |
| openinference-semantic-conventions | 0.1.39 | Apache-2.0 | optional (transitive) |
| opentelemetry-instrumentation | 0.65b0 in uv.lock (0.66b0 on PyPI) | Apache-2.0 | optional (transitive) |
| wrapt | 2.5.0 (PyPI) | BSD-2-Clause | optional (transitive) |
| dacite | 1.9.2 | MIT | optional (bedrock transitive) |
| typing-extensions | 4.16.0 | PSF-2.0 | optional (transitive) |

Open items: the optional OpenInference `TraceConfig` masking call is written
from the package's documented public API and is exercised in tests only
against a stand-in (the packages are not installed in this repo's test
environment). It must be verified against the real packages before release.
