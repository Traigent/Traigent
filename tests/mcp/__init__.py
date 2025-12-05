"""MCP (Model Context Protocol) testing framework for TraiGent SDK.

This module provides comprehensive testing for MCP integration, including:
- Unit tests for individual MCP endpoints
- Integration tests for end-to-end scenarios
- LLM task interpretation tests
- Proper cleanup and resource management
"""

from .conftest import cleanup_mcp_resources, mock_mcp_service

__all__ = ["mock_mcp_service", "cleanup_mcp_resources"]
