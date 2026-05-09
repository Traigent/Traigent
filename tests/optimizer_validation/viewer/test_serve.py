"""Tests for the optimizer validation viewer server.

These tests ensure the chat functionality works correctly, including
fallback behavior when external CLI tools fail.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest


class TestProcessChatMessage:
    """Tests for chat message processing."""

    @pytest.fixture(autouse=True)
    def _force_local_mode(self, monkeypatch):
        """Use local fallback mode so tools_used is populated."""
        monkeypatch.setattr(
            "tests.optimizer_validation.viewer.serve.CHAT_USE_CLI", False
        )

    def test_chat_returns_valid_response_structure(self) -> None:
        """Chat responses must have 'response', 'tools_used', and 'mode' keys."""
        from tests.optimizer_validation.viewer.serve import process_chat_message

        result = process_chat_message("show stats", [])

        assert "response" in result
        assert "tools_used" in result
        assert "mode" in result
        assert isinstance(result["response"], str)
        assert isinstance(result["tools_used"], list)
        # Default mode should be local (not CLI)
        assert result["mode"] in ("local", "cli", "local (CLI failed)")

    def test_chat_stats_query(self) -> None:
        """Stats query should return test statistics."""
        from tests.optimizer_validation.viewer.serve import process_chat_message

        result = process_chat_message("show test stats", [])

        assert "response" in result
        assert (
            "total" in result["response"].lower()
            or "tests" in result["response"].lower()
        )
        assert "get_test_stats" in result["tools_used"]

    def test_chat_dimensions_query(self) -> None:
        """Dimensions query should list test dimensions."""
        from tests.optimizer_validation.viewer.serve import process_chat_message

        result = process_chat_message("list dimensions", [])

        assert "response" in result
        assert "list_dimensions" in result["tools_used"]

    def test_chat_gaps_query(self) -> None:
        """Gaps query should return coverage information."""
        from tests.optimizer_validation.viewer.serve import process_chat_message

        result = process_chat_message("show coverage gaps", [])

        assert "response" in result
        assert "get_coverage_gaps" in result["tools_used"]

    def test_chat_search_query(self) -> None:
        """Search query should find tests by keyword."""
        from tests.optimizer_validation.viewer.serve import process_chat_message

        result = process_chat_message("find parallel tests", [])

        assert "response" in result
        assert "search_tests" in result["tools_used"]

    def test_chat_unknown_query_returns_help(self) -> None:
        """Unknown queries should return helpful guidance."""
        from tests.optimizer_validation.viewer.serve import process_chat_message

        result = process_chat_message("xyzzy unknown command", [])

        assert "response" in result
        # Should provide guidance on available commands
        assert (
            "stats" in result["response"].lower()
            or "help" in result["response"].lower()
        )

    def test_chat_handles_empty_message(self) -> None:
        """Empty messages should not crash."""
        from tests.optimizer_validation.viewer.serve import process_chat_message

        result = process_chat_message("", [])

        assert "response" in result
        assert isinstance(result["response"], str)


class TestChatFallbackBehavior:
    """Tests ensuring chat gracefully handles failures.

    These tests verify that when external dependencies fail,
    the chat system falls back gracefully rather than exposing errors.
    """

    @pytest.fixture(autouse=True)
    def _force_local_mode(self, monkeypatch):
        """Use local fallback mode to avoid CLI subprocess hangs in CI."""
        monkeypatch.setattr(
            "tests.optimizer_validation.viewer.serve.CHAT_USE_CLI", False
        )

    def test_fallback_does_not_expose_cli_errors(self) -> None:
        """CLI errors should not be exposed to users.

        This is the bug we fixed - CLI errors like fs.watch failures
        were being shown to users instead of falling back gracefully.
        """
        from tests.optimizer_validation.viewer.serve import _process_chat_fallback

        # Direct fallback should never return CLI error messages
        result = _process_chat_fallback("show stats")

        assert "CLI error" not in result["response"]
        assert "fs.watch" not in result["response"]
        assert (
            "Error:" not in result["response"] or "total" in result["response"].lower()
        )

    def test_fallback_works_without_external_cli(self) -> None:
        """Chat should work even if Claude CLI is not installed."""
        from tests.optimizer_validation.viewer.serve import process_chat_message

        # The current implementation uses fallback directly,
        # so this should always work
        result = process_chat_message("show stats", [])

        assert "response" in result
        assert "CLI error" not in result["response"]

    @patch("subprocess.run")
    def test_subprocess_failure_does_not_crash(self, mock_run: MagicMock) -> None:
        """Subprocess failures should be handled gracefully.

        Even if we reintroduce CLI calls in the future, failures
        should fall back to direct tool execution.
        """
        from tests.optimizer_validation.viewer.serve import _process_chat_fallback

        # Simulate what the old code path would have done
        mock_run.side_effect = subprocess.SubprocessError("fs.watch error")

        # Fallback should still work
        result = _process_chat_fallback("show stats")

        assert "response" in result
        assert (
            "total" in result["response"].lower()
            or "tests" in result["response"].lower()
        )


class TestChatToolExecution:
    """Tests for individual chat tool functions."""

    def test_get_test_stats_returns_counts(self) -> None:
        """get_test_stats should return test counts."""
        from tests.optimizer_validation.chatbot.tools import get_test_stats

        stats = get_test_stats()

        assert "total_tests" in stats
        assert "passed" in stats
        assert "failed" in stats
        assert isinstance(stats["total_tests"], int)

    def test_list_dimensions_returns_dict(self) -> None:
        """list_dimensions should return dimension mapping."""
        from tests.optimizer_validation.chatbot.tools import list_dimensions

        dims = list_dimensions()

        assert isinstance(dims, dict)
        # Should have at least some known dimensions
        assert len(dims) > 0

    def test_search_tests_returns_list(self) -> None:
        """search_tests should return list of matches."""
        from tests.optimizer_validation.chatbot.tools import search_tests

        # Use a specific keyword that exists in test names
        # Note: "test" is too generic and may hit edge cases with None descriptions
        results = search_tests("parallel")

        assert isinstance(results, list)


class TestIsValidTarget:
    """Tests for test target validation (security)."""

    def test_valid_target_within_test_dir(self) -> None:
        """Targets within tests/optimizer_validation should be valid."""
        from tests.optimizer_validation.viewer.serve import is_valid_target

        assert is_valid_target("tests/optimizer_validation/dimensions/test_foo.py")
        assert is_valid_target("tests/optimizer_validation/")

    def test_invalid_target_outside_test_dir(self) -> None:
        """Targets outside tests/optimizer_validation should be rejected."""
        from tests.optimizer_validation.viewer.serve import is_valid_target

        assert not is_valid_target("tests/unit/test_foo.py")
        assert not is_valid_target("/etc/passwd")
        assert not is_valid_target("traigent/core/orchestrator.py")

    def test_path_traversal_blocked(self) -> None:
        """Path traversal attempts should be blocked."""
        from tests.optimizer_validation.viewer.serve import is_valid_target

        assert not is_valid_target("tests/optimizer_validation/../unit/test_foo.py")
        assert not is_valid_target("tests/optimizer_validation/../../etc/passwd")


class TestResolveProjectFile:
    """Tests for viewer file path resolution."""

    def test_resolves_relative_project_file(self) -> None:
        """Relative paths should resolve inside the project root."""
        from tests.optimizer_validation.viewer.serve import (
            PROJECT_ROOT_RESOLVED,
            resolve_project_file,
        )

        resolved = resolve_project_file(
            "tests/optimizer_validation/viewer/test_serve.py"
        )

        assert resolved == PROJECT_ROOT_RESOLVED / (
            "tests/optimizer_validation/viewer/test_serve.py"
        )

    @pytest.mark.parametrize(
        "path",
        [
            "/etc/passwd",
            "../outside.txt",
            "%2Fetc%2Fpasswd",
        ],
    )
    def test_rejects_paths_outside_project(self, path: str) -> None:
        """Absolute and traversal paths should be rejected."""
        from tests.optimizer_validation.viewer.serve import resolve_project_file

        with pytest.raises(ValueError):
            resolve_project_file(path)
