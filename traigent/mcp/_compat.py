"""Lazy compatibility with the supported MCP 1.x and 2.x server APIs."""

from typing import Any


def mcp_server_class() -> Any:
    """Load the official server class without importing the optional extra early."""
    try:
        from mcp.server.mcpserver import MCPServer
    except ModuleNotFoundError as exc:
        if exc.name != "mcp.server.mcpserver":
            raise
        from mcp.server.fastmcp import FastMCP

        return FastMCP
    return MCPServer
