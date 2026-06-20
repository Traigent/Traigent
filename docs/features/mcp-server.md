# Local MCP Server

Traigent 0.12.0 includes a local stdio MCP server for coding agents.

```bash
python3 -m pip install "traigent[mcp]"
traigent mcp serve
```

The server runs locally over stdio and is created with the name `traigent`. Its
instructions describe a local-first SDK server where tools run against local
code and local auth storage.

## Tools

The current server source registers these tools:

- `auth_status`
- `list_recommendation_agent_types`
- `recommend_configuration_space`
- `detect_tvars`
- `scaffold_eval`
- `validate_dataset`
- `estimate_cost`
- `run_optimization`
- `get_results`
- `export_evidence`

## Security Boundary

`auth_status` returns local auth state and masked key metadata only. The agent
never receives the raw API key. A live backend validation call is made only when
the tool is called with `check=true`.

Path-taking tools constrain paths to the MCP server's current working directory,
or to `TRAIGENT_DATASET_ROOT` for dataset validation and cost estimation.

## Dry-Run-First Optimization

`run_optimization` defaults to mock mode:

```text
mode = "mock"
```

Mock mode sets local safe defaults for the run:

```bash
TRAIGENT_MOCK_LLM=true
TRAIGENT_OFFLINE=1
```

Real mode is refused unless `confirm=true` and `cost_limit` is set. Real mode
also refuses local draft evaluation manifests until a human reviews or replaces
the dataset and sets the manifest approval status to `user_approved` or
`approved`.

## Copy-Paste Example

```bash
python3 -m pip install "traigent[mcp]"
traigent mcp serve
```

Then configure your coding agent to launch that command as a stdio MCP server.

Honesty note: the server is a local single-client v1 surface. A running
optimization blocks the MCP stdio loop until it returns.
