# Agent Observability Spans

Traigent 0.12.0 exposes `add_agent_span(...)` as a public API for recording
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
) -> None
```

The helper is safe to call from user code. If no optimization trial or workflow
trace transport is active, it logs at debug level and returns.

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
