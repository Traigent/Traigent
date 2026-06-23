"""Agent-facing Traigent analytics MCP (cloud-read optimization analytics).

A separate MCP server from :mod:`traigent.mcp` (the local-results server): this
one is an authenticated thin client over the backend analytics endpoints plus a
local chart-render helper. Console entry point: ``traigent-analytics-mcp``.
"""

from __future__ import annotations

from traigent.analytics_mcp.tools import ANALYTICS_TOOL_NAMES

__all__ = ["ANALYTICS_TOOL_NAMES"]
