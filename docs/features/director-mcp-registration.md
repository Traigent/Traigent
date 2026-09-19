# Registering the Director v0 tools (analytics MCP)

The Director v0 tools (`director_start`, `director_turn`, `director_state`) ship
in the same `traigent-analytics-mcp` server as the existing analytics/observability
tools (`analytics_get_run_report`, `analytics_get_run_decision_brief`, etc.) — see
[`traigent/analytics_mcp/server.py`](../../traigent/analytics_mcp/server.py). This is
a **separate server** from the local `traigent mcp serve` server documented in
[mcp-server.md](mcp-server.md); that one only sees `.traigent/` results on disk,
this one is an authenticated cloud-read/write client for the backend.

There is currently no `.mcp.json` shipped with the SDK (tracked as a follow-up,
backlog item B9). Until then, register the server with your coding agent
manually, the same way you would register any other stdio MCP server.

## 1. Install

```bash
python3 -m pip install "traigent[mcp,hybrid]"
```

`mcp` provides the server framework; `hybrid` provides `httpx`, the transport
`BackendAnalyticsClient` uses to call the backend.

## 2. Provide credentials

The server resolves credentials the same way the rest of the SDK does — no
separate configuration step:

1. `TRAIGENT_API_KEY` environment variable, or
2. stored CLI credentials (`traigent auth login`), or
3. `TRAIGENT_JWT_TOKEN` (bearer auth), if you use JWTs instead of an API key.

Optionally set `TRAIGENT_BACKEND_URL` to point at a non-default backend
(defaults to the SDK's normal resolved backend URL).

The MCP tools never return credential material, and no tool accepts a
`tenant_id` argument — tenancy is derived from the credential.

## 3. Register the stdio server

The installed console entry point is `traigent-analytics-mcp`
(`traigent.analytics_mcp.server:main`). Point your coding agent's MCP client
config at that command. For an agent that reads a JSON config of
`{"mcpServers": {...}}` shape (Claude Code, Claude Desktop, and most others
follow this convention):

```json
{
  "mcpServers": {
    "traigent-analytics": {
      "command": "traigent-analytics-mcp",
      "env": {
        "TRAIGENT_API_KEY": "<YOUR_API_KEY>"
      }
    }
  }
}
```

Omit the `env` block if you rely on stored CLI credentials instead of an
environment variable.

## 4. Director tools available after registration

All three are **advisory only** (v0): a Director response never authorizes
spend, promotion, a lease, a budget reservation, or any change. Every
response carries `advisory_only: true`.

| tool | purpose |
|---|---|
| `director_start` | Open an advisory Director session for a project. Optionally attaches one run at create time. |
| `director_turn` | Advance a session by one turn. Takes `session_id`, `session_revision` (from the last state/turn read), the closed `intent` enum (`ask_next_step` \| `report_progress`), and an optional typed `report` block (`{instruction_id, status, run_id?}`) that reports the outcome of a previously issued instruction. No free-text field is accepted anywhere on this call. |
| `director_state` | Read a session's current state (pure read; never advances `revision`, never calls the model). Use this to recover from a `409 stale_revision` result and re-read the current revision before retrying `director_turn`. |

See the frozen contract at `~/.claude/plans/director-v0-contract/` for the
full request/response schemas and the state-machine rules these tools are a
thin pass-through to.
