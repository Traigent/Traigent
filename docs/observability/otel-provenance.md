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
SDK code and tests, "optional" = named in an extra only, not exercised by default; resolution verified by optional tests):

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

Real-package verification (2026-09-30, throwaway venv, Python 3.13, exact pins
above plus opentelemetry-api/-sdk 1.44.0, semantic-conventions 0.65b0,
instrumentation 0.65b0; licences read from each installed package's metadata,
including transitive deps): every module path and class name in `INSTRUMENTORS`
(`OpenAIInstrumentor`, `AnthropicInstrumentor`, `LangChainInstrumentor`,
`BedrockInstrumentor`) imports, constructs, and `instrument(tracer_provider=...,
config=TraceConfig(...))` succeeds; `openinference.instrumentation.TraceConfig`
accepts all six `hide_*` flags used by `_masking_config`. The bedrock
instrumentor imports only when `botocore` is present, and each other one only
instruments when its host library (openai / anthropic / langchain-core) is
installed; those host libraries are the caller's, not Traigent extras. Covered
by `test_registered_instrumentor_names_resolve_in_real_package` and
`test_real_openinference_trace_config_accepts_every_masking_flag`, which skip
cleanly when the extras are absent.
