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

## Projects, pagination and errors

Prompts belong to a project, and the project id is part of every request path
(`/api/v1/projects/{project_id}/prompts`). Pass it explicitly or set
`TRAIGENT_PROJECT_ID`; there is no default project.

```python
from traigent.prompts import PromptManagementClient, PromptManagementConfig

client = PromptManagementClient(PromptManagementConfig(project_id="my-project"))

page = client.list_prompts(limit=50, label="production")
print(page.pagination.has_more, page.pagination.next_cursor)

# Walk every page; the cursor is followed for you and filters are kept.
for prompt in client.iter_prompts(label="production"):
    print(prompt.name)
```

Lists are newest first and have no total count. A cursor only works with the
project and filters it was issued for.

Errors from the prompts service raise `PromptApiError` subclasses with `code`,
`status_code` and `problem_id` attributes:

| Exception | Codes |
|---|---|
| `PromptNotFoundError` | `not_found` (missing, or not visible to you; the two look the same) |
| `PromptInvalidRequestError` | `invalid_page_size`, `invalid_cursor`, `validation_failed` |
| `PromptConflictError` | `prompt_already_exists`, `prompt_version_conflict` |
| `PromptNotPermittedError` | `feature_not_entitled`, `provider_not_allowed` |
| `PromptPaymentRequiredError` | `insufficient_balance`, `spend_blocked` |
| `PromptUpstreamError` | `upstream_failed` |

Authentication failures raise `AuthenticationError`, rate limits raise
`RateLimitError` and temporary unavailability raises `ServiceUnavailableError`.

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
