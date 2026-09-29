# OpenTelemetry observability (`traigent.observability.otel`)

Install: `pip install "traigent[observability]"` (add `observability-openai`,
`-anthropic`, `-langchain` or `-bedrock` for third-party instrumentors).

```python
import traigent.observability.otel as otel

otel.init(api_key="tg_...", service_name="support-bot")   # metadata-only by default

@otel.observe("answer", as_type="chain")
def answer(question): ...

with otel.attributes(session_id="s-42", user_id="u-7"):
    answer("hi")

otel.flush(5)      # returns FlushOutcome; also runs automatically at exit
otel.stats()       # queued / exported / dropped_* / rejected_by_server counters
```

The legacy client API in `traigent.observability` is unchanged; this layer is
additive.

## What leaves the process

Traigent's exporter rebuilds every span before serialisation from a typed,
bounded, exact-key allowlist (`traigent/observability/otel/contract.py`, no
wildcards). Span names, event names, link attributes, resource and scope
fields are covered as well. In the default `metadata` mode no prompt,
completion, tool argument, exception message, tag or user metadata leaves.

Modes (most restrictive of argument / environment wins; `observe(content_mode=)`
can only tighten):

| mode | span attributes that carry content |
|------|-----------------------------------|
| `metadata` (default) | dropped; only allowlisted structural fields sent |
| `redacted` | content keys kept with the value `[REDACTED]` |
| `record` | content sent after secret scrubbing, 64 KiB per attribute cap |

The SDK declares its mode as the resource attribute `traigent.content_mode.v1`.
The declaration never grants permission. **The current receiver drops content
attributes in every mode**, so `record` sends content that is then discarded
server-side; storing content will require a future authenticated project
policy.

### Boundaries you must know

* The content policy protects only spans that leave through Traigent's
  exporter. When you attach Traigent to your own `TracerProvider`, any other
  exporter on it still receives whatever your instrumentors capture.
* Therefore `instrument()` refuses to install instrumentors onto a provider
  with exporters Traigent cannot verify (or cannot inspect), unless you pass
  `allow_unverified_exporters=True`, which is your consent to that exposure.
* Instrumentors are also asked to hide content at the source outside `record`
  mode (best effort; depends on the instrumentor accepting a config).
* Names and scope names are heuristically identifier-checked; a single-token
  string placed in a span name cannot be told apart from an identifier.

`OTEL_EXPORTER_OTLP_*` variables are ignored on purpose: an environment
variable must not be able to redirect the Traigent API key.

## Delivery

Best-effort, no durable outbox. Bounded queue (drop-new, counted), one export
in flight, retries only on HTTP 429/502/503/504 and network errors with
full-jitter backoff (`Retry-After` honoured, capped at 60 s, at most 5
attempts, batches older than 120 s dropped). 413 splits the batch once.
Partial success is final. `flush(timeout)` and the exit hook have a hard
deadline that does not depend on the retry budget. After `fork` the child
starts with an empty queue and its own worker. Losses are visible in `stats()`
and through `on_drop`, which are process-local (a crash loses them too).

## Sampling

`sample_rate` (or `TRAIGENT_OBSERVABILITY_SAMPLE_RATE`, default 1.0) applies to
providers Traigent creates: parent-based, deciding on the low 64 bits of the
trace id (`low64 < floor(round(rate*1e6) * 2^64 / 1e6)`), identical in every
Traigent SDK (shared vectors). A provider you pass in keeps its own sampler.
There is no error-rescue in this release: unsampled traces are dropped.

## Lineage

Every span started while an optimizer trial is active - including spans made by
third-party instrumentors - is stamped with `traigent.trial_id` and
`traigent.optimization_session_id`. Only identifiers are stamped; optimizer
configs and metrics never go on spans. Context flows through asyncio tasks
automatically; a bare thread needs `contextvars.copy_context().run(...)`.
