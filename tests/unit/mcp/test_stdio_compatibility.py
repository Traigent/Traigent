"""Exercise the real MCP stdio channel across supported SDK majors."""

import os
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    ("module", "tool", "arguments"),
    [
        ("traigent.mcp.server", "detect_tvars", {"file_path": "agent.py"}),
        ("traigent.analytics_mcp.server", "health_check", {}),
    ],
)
async def test_real_stdio_server_lists_and_calls_local_tool(
    tmp_path, module, tool, arguments
):
    (tmp_path / "agent.py").write_text(
        "def answer(query: str) -> str:\n    temperature = 0.1\n    return query\n"
    )
    # Preserve this test interpreter's dependency order, including isolated MCP
    # installations, rather than accidentally starting the shared environment.
    bootstrap = (
        f"import sys; sys.path.extend({sys.path!r}); "
        f"from {module} import run_stdio_server; run_stdio_server()"
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "PYTHONPATH", "SYSTEMROOT", "TEMP", "TMPDIR"}
    }
    environment.update(
        TRAIGENT_MOCK_LLM="true",
        TRAIGENT_OFFLINE_MODE="true",
        TRAIGENT_SKIP_DOTENV="1",
    )
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-c", bootstrap],
        env=environment,
        cwd=str(tmp_path),
    )
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listing = await session.list_tools()
            assert tool in {registered.name for registered in listing.tools}
            result = await session.call_tool(tool, arguments)
            structured = (
                result.structured_content
                if hasattr(result, "structured_content")
                else result.structuredContent
            )
            assert structured is not None
            assert structured["ok"] is True
