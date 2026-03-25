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

## Verification

Validated by:

- `tests/unit/prompts/test_prompt_management_client.py`

Targeted coverage after this implementation:

- `traigent.prompts.client`: 85% line coverage
- `traigent.prompts.dtos`: 98% line coverage
