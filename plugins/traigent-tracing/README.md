# Traigent Tracing Plugin

OpenTelemetry distributed tracing support for Traigent optimization runs.

## Installation

```bash
pip install traigent-tracing
```

For Jaeger support:
```bash
pip install traigent-tracing[jaeger]
```

## Features

- Automatic span creation for optimization sessions
- Trial-level tracing with configuration and metrics
- Integration with popular backends (Jaeger, Zipkin, OTLP)
- Baggage propagation for distributed optimization

## Usage

Tracing is opt-in. Configure it via environment variables:

```bash
# Enable tracing
export TRAIGENT_TRACE_ENABLED=true

# Configure OTLP endpoint (default: localhost:4317)
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Set service name
export OTEL_SERVICE_NAME=my-optimization-service
```

`TRAIGENT_TRACE_ENABLED` is the canonical tracing flag and defaults to
`false`. The older plural spelling `TRAIGENT_TRACES_ENABLED` is **no longer
supported** as of 0.13.0 and is silently ignored. Migrate any use of
`TRAIGENT_TRACES_ENABLED` to `TRAIGENT_TRACE_ENABLED`.

Or configure programmatically:

```python
from traigent_tracing import configure_tracing

configure_tracing(
    enabled=True,
    endpoint="http://localhost:4317",
    service_name="my-optimization-service",
)
```

## Spans Created

- `traigent.optimization_session` - Root span for entire optimization run
- `traigent.trial` - Per-trial spans with config and metrics
- `traigent.evaluation` - Evaluation function execution

## Attributes

Each span includes relevant attributes:

- `traigent.optimization_id` - Unique optimization ID
- `traigent.trial_number` - Trial index
- `traigent.config.*` - Configuration parameters
- `traigent.metrics.*` - Evaluation metrics
- `traigent.algorithm` - Optimization algorithm used
