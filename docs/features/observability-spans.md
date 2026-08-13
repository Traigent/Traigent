# Agent Observability Spans

Traigent exposes `add_agent_span(...)` as a public API for recording
agent workflow spans inside active optimization trials.

## Public API

```python
from traigent.observability import add_agent_span

add_agent_span(
    "retriever",
    span_type="agent",
    input_tokens=120,
    output_tokens=35,
    cost_usd=0.00042,
    latency_ms=180.5,
    model="anthropic.claude-3-5-sonnet",
    metadata={"documents": 4},
)
```

Signature:

```python
add_agent_span(
    node_id: str,
    *,
    span_type: str = "agent",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
    latency_ms: float | None = None,
    model: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SpanResult
```

The helper is safe to call from user code: it never raises. It always returns a
`SpanResult` receipt describing what happened to the span, so a caller can detect
lost telemetry instead of inferring it from an empty dashboard.

```python
@dataclass(frozen=True)
class SpanResult:
    accepted: bool
    reason: str | None = None                  # set when accepted is False
    dropped_metadata_keys: tuple[str, ...] = ()
```

`SpanResult` is falsy when the span was not accepted, so a check reads naturally:

```python
result = add_agent_span("retriever", input_tokens=1200)
if not result:
    log.warning("span dropped: %s", result.reason)
```

`reason` is one of `no_trace_context` (no active optimization trial),
`no_trace_manager`, `tracing_disabled`, `invalid_node_id`, `no_trace_linkage`, or
`collection_failed`.

`dropped_metadata_keys` names any `metadata` entries that were removed. Metadata is
numeric-only, and credential- or content-shaped keys are always dropped — so
`{"prompt": "...", "auth_token_count": 5, "score": 0.9}` keeps only `score` and
reports the other two. This is a security boundary, not a bug: pass token and cost
figures through the dedicated parameters rather than through `metadata`.

## Sanitization

`add_agent_span` keeps numeric metadata only. Sensitive content-like metadata
keys are dropped. Unsafe model identifiers, negative numbers, and non-finite
numbers are ignored.

## Bedrock Capture

Traigent captures usage and latency from:

- `langchain_aws.ChatBedrock`
- `langchain_aws.ChatBedrockConverse`
- the SDK `BedrockChatClient` wrapper for `bedrock-runtime`

Captured responses are normalized into token and cost tracking paths when usage
metadata is present. Bedrock mock mode uses the same response-capture path, so
tests can exercise token and span behavior without live AWS calls.

## Mock Interception

For local development, mock mode can be enabled in code:

```python
from traigent.testing import enable_mock_mode_for_quickstart

enable_mock_mode_for_quickstart()
```

## Copy-Paste Example

```python
from traigent.observability import add_agent_span

def run_node(query: str) -> str:
    output = "mock answer"
    add_agent_span(
        "answer_node",
        input_tokens=20,
        output_tokens=8,
        cost_usd=0.0,
        latency_ms=12.0,
        metadata={"candidate_count": 2},
    )
    return output
```

Honesty note: spans are collected only when an active optimization trial has
workflow trace collection enabled. Calling `add_agent_span` outside that context
is intentionally a no-op.
