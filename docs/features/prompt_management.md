# Prompt Management Surface

The SDK prompt-management surface now includes the critical parity endpoints
needed for prompt registry workflows:

- prompt listing and filtering
- prompt detail lookup
- prompt version resolution
- prompt analytics
- prompt creation and versioning
- prompt label updates
- prompt playground execution and preview

## Playground API

Use `PromptManagementClient.run_playground(...)` to preview or execute a stored
prompt version.

```python
from traigent.prompts import PromptManagementClient

client = PromptManagementClient()

result = client.run_playground(
    "support/welcome",
    version=2,
    variables={"customer_name": "Ada"},
    model="gpt-4.1-mini",
    provider="openai",
    dry_run=False,
)

print(result.executed)
print(result.trace_id)
print(result.output)
```

## Returned DTOs

The SDK exposes typed DTOs for the playground flow:

- `PromptPlaygroundConfig`
- `PromptPlaygroundTokenUsage`
- `PromptPlaygroundResult`

These are exported from both `traigent.prompts` and top-level `traigent`.

## Cross-Surface Demo

For a compact SDK demo, chain the three parity surfaces together:

1. Use `PromptManagementClient.run_playground(...)` to execute a stored prompt version.
2. Read the returned `trace_id` and fetch trace detail from `ObservabilityClient`.
3. Fetch the project observability summary dashboard from `CoreMetricsClient`.

This shows that prompt execution, observability trace retrieval, and project-scoped analytics all line up across the SDK surface.

```python
from traigent import CoreMetricsClient, ObservabilityClient, PromptManagementClient

prompts = PromptManagementClient()
observability = ObservabilityClient()
metrics = CoreMetricsClient()

result = prompts.run_playground(
    "support/welcome",
    version=2,
    variables={"customer_name": "Ada"},
    provider="openai",
    model="gpt-4.1-mini",
    dry_run=False,
)

trace = observability.get_trace(result.trace_id)
dashboard = metrics.get_observability_summary_dashboard(days=7, limit=3)

print(result.output)
print(trace.name)
print(dashboard.summary_cards.traces_in_range)
```

## Verification

Validated by:

- `tests/unit/prompts/test_prompt_management_client.py`

Targeted coverage after this implementation:

- `traigent.prompts.client`: 85% line coverage
- `traigent.prompts.dtos`: 98% line coverage
