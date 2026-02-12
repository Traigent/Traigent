"""Comprehensive unit tests for user prompt formatting helpers.

Tests cover:
- _print_box function (bordered box printing)
- print_budget_prompt (budget limit prompts with overage)
- print_vendor_error_prompt (vendor error prompts)
- print_network_error_prompt (network error prompts)
- Constants verification (PROMPT_WIDTH, _OPTION_STOP)
"""

from __future__ import annotations

import pytest

from traigent.utils.user_prompts import (
    _OPTION_STOP,
    PROMPT_WIDTH,
    _print_box,
    print_budget_prompt,
    print_network_error_prompt,
    print_vendor_error_prompt,
)


@pytest.mark.unit
class TestConstants:
    """Tests for module-level constants."""

    def test_prompt_width_value(self):
        """Test PROMPT_WIDTH has expected value."""
        assert PROMPT_WIDTH == 63
        assert isinstance(PROMPT_WIDTH, int)

    def test_option_stop_value(self):
        """Test _OPTION_STOP has expected format."""
        assert _OPTION_STOP == "  [2] STOP optimization"
        assert "[2]" in _OPTION_STOP
        assert "STOP" in _OPTION_STOP


@pytest.mark.unit
class TestPrintBox:
    """Tests for _print_box function."""

    def test_print_empty_box(self, capsys):
        """Test printing box with no lines."""
        _print_box([])
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "=" * PROMPT_WIDTH
        assert lines[1] == "=" * PROMPT_WIDTH

    def test_print_box_single_line(self, capsys):
        """Test printing box with single line."""
        _print_box(["Test message"])
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert len(lines) == 3
        assert lines[0] == "=" * PROMPT_WIDTH
        assert lines[1] == "Test message"
        assert lines[2] == "=" * PROMPT_WIDTH

    def test_print_box_multiple_lines(self, capsys):
        """Test printing box with multiple lines."""
        _print_box(["Line 1", "Line 2", "Line 3"])
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert len(lines) == 5
        assert lines[0] == "=" * PROMPT_WIDTH
        assert lines[1] == "Line 1"
        assert lines[2] == "Line 2"
        assert lines[3] == "Line 3"
        assert lines[4] == "=" * PROMPT_WIDTH

    def test_print_box_border_length(self, capsys):
        """Test box borders match PROMPT_WIDTH."""
        _print_box(["Test"])
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert len(lines[0]) == PROMPT_WIDTH
        assert len(lines[-1]) == PROMPT_WIDTH


@pytest.mark.unit
class TestPrintBudgetPrompt:
    """Tests for print_budget_prompt function."""

    def test_budget_prompt_at_limit(self, capsys):
        """Test budget prompt when exactly at limit."""
        print_budget_prompt(current_limit=10.0, spent=10.0)
        captured = capsys.readouterr()

        assert "BUDGET LIMIT REACHED" in captured.out
        assert "Spent / Limit: $10.00 / $10.00" in captured.out
        assert "[1] Raise limit and CONTINUE optimization" in captured.out
        assert "[2] STOP optimization" in captured.out

    def test_budget_prompt_under_limit(self, capsys):
        """Test budget prompt when under limit (edge case)."""
        print_budget_prompt(current_limit=10.0, spent=9.50)
        captured = capsys.readouterr()

        assert "BUDGET LIMIT REACHED" in captured.out
        assert "Spent / Limit: $9.50 / $10.00" in captured.out
        assert "[1] Raise limit and CONTINUE optimization" in captured.out
        assert "[2] STOP optimization" in captured.out

    def test_budget_prompt_with_overage(self, capsys):
        """Test budget prompt shows overage when spent > limit."""
        print_budget_prompt(current_limit=10.0, spent=12.50)
        captured = capsys.readouterr()

        assert "BUDGET LIMIT REACHED" in captured.out
        assert "Spent / Limit: $12.50 / $10.00" in captured.out
        assert "Over limit by: $2.50" in captured.out
        assert "Minimum add to continue: > $2.50" in captured.out
        assert "[1] Raise limit and CONTINUE optimization" in captured.out
        assert "[2] STOP optimization" in captured.out

    def test_budget_prompt_large_overage(self, capsys):
        """Test budget prompt with large overage."""
        print_budget_prompt(current_limit=5.0, spent=20.0)
        captured = capsys.readouterr()

        assert "Spent / Limit: $20.00 / $5.00" in captured.out
        assert "Over limit by: $15.00" in captured.out
        assert "Minimum add to continue: > $15.00" in captured.out

    def test_budget_prompt_small_amounts(self, capsys):
        """Test budget prompt with small dollar amounts."""
        print_budget_prompt(current_limit=0.50, spent=0.75)
        captured = capsys.readouterr()

        assert "Spent / Limit: $0.75 / $0.50" in captured.out
        assert "Over limit by: $0.25" in captured.out

    def test_budget_prompt_zero_values(self, capsys):
        """Test budget prompt with zero values."""
        print_budget_prompt(current_limit=0.0, spent=0.0)
        captured = capsys.readouterr()

        assert "Spent / Limit: $0.00 / $0.00" in captured.out
        assert "Your optimization cost limit has been reached" in captured.out

    def test_budget_prompt_contains_borders(self, capsys):
        """Test budget prompt contains box borders."""
        print_budget_prompt(current_limit=10.0, spent=10.0)
        captured = capsys.readouterr()

        # Border should be present in output
        assert "=" * PROMPT_WIDTH in captured.out

    def test_budget_prompt_has_both_options(self, capsys):
        """Test budget prompt shows both option [1] and [2]."""
        print_budget_prompt(current_limit=10.0, spent=10.0)
        captured = capsys.readouterr()

        assert "[1]" in captured.out
        assert "[2]" in captured.out
        assert _OPTION_STOP in captured.out

    def test_budget_prompt_decimal_precision(self, capsys):
        """Test budget prompt formats decimals to 2 places."""
        print_budget_prompt(current_limit=10.123456, spent=9.876543)
        captured = capsys.readouterr()

        # Should format to exactly 2 decimal places
        assert "$10.12" in captured.out
        assert "$9.88" in captured.out


@pytest.mark.unit
class TestPrintVendorErrorPrompt:
    """Tests for print_vendor_error_prompt function."""

    def test_vendor_error_basic(self, capsys):
        """Test basic vendor error prompt."""
        print_vendor_error_prompt(
            title="API Rate Limit", explanation="Rate limit exceeded"
        )
        captured = capsys.readouterr()

        assert "VENDOR ERROR ENCOUNTERED" in captured.out
        assert "Error: API Rate Limit" in captured.out
        assert "Rate limit exceeded" in captured.out
        assert "[1] RESUME optimization" in captured.out
        assert "[2] STOP optimization" in captured.out

    def test_vendor_error_multiline_explanation(self, capsys):
        """Test vendor error prompt with multiline explanation."""
        explanation = "Line 1: First issue\nLine 2: Second issue\nLine 3: Third issue"
        print_vendor_error_prompt(title="Multiple Issues", explanation=explanation)
        captured = capsys.readouterr()

        assert "VENDOR ERROR ENCOUNTERED" in captured.out
        assert "Error: Multiple Issues" in captured.out
        assert "Line 1: First issue" in captured.out
        assert "Line 2: Second issue" in captured.out
        assert "Line 3: Third issue" in captured.out

    def test_vendor_error_empty_explanation(self, capsys):
        """Test vendor error prompt with empty explanation."""
        print_vendor_error_prompt(title="Unknown Error", explanation="")
        captured = capsys.readouterr()

        assert "VENDOR ERROR ENCOUNTERED" in captured.out
        assert "Error: Unknown Error" in captured.out
        assert "[1] RESUME optimization" in captured.out

    def test_vendor_error_long_title(self, capsys):
        """Test vendor error prompt with long error title."""
        long_title = "Very Long Error Title That Exceeds Normal Length"
        print_vendor_error_prompt(title=long_title, explanation="Description")
        captured = capsys.readouterr()

        assert "VENDOR ERROR ENCOUNTERED" in captured.out
        assert f"Error: {long_title}" in captured.out

    def test_vendor_error_contains_borders(self, capsys):
        """Test vendor error prompt contains box borders."""
        print_vendor_error_prompt(title="Test Error", explanation="Test explanation")
        captured = capsys.readouterr()

        assert "=" * PROMPT_WIDTH in captured.out

    def test_vendor_error_has_both_options(self, capsys):
        """Test vendor error prompt shows both option [1] and [2]."""
        print_vendor_error_prompt(title="Test", explanation="Test")
        captured = capsys.readouterr()

        assert "[1]" in captured.out
        assert "RESUME" in captured.out
        assert "[2]" in captured.out
        assert "STOP" in captured.out

    def test_vendor_error_explanation_indentation(self, capsys):
        """Test vendor error prompt indents explanation lines."""
        explanation = "First line\nSecond line"
        print_vendor_error_prompt(title="Test", explanation=explanation)
        captured = capsys.readouterr()

        # Each explanation line should be indented with "  "
        lines = captured.out.split("\n")
        explanation_lines = [
            line for line in lines if "First line" in line or "Second line" in line
        ]
        for line in explanation_lines:
            assert line.startswith("  ")


@pytest.mark.unit
class TestPrintNetworkErrorPrompt:
    """Tests for print_network_error_prompt function."""

    def test_network_error_basic(self, capsys):
        """Test basic network error prompt."""
        print_network_error_prompt(
            title="Connection Timeout", explanation="Network connection timed out"
        )
        captured = capsys.readouterr()

        assert "NETWORK ERROR DETECTED" in captured.out
        assert "Error: Connection Timeout" in captured.out
        assert "Network connection timed out" in captured.out
        assert "[1] RESUME (will wait for network to restore)" in captured.out
        assert "[2] STOP optimization" in captured.out

    def test_network_error_multiline_explanation(self, capsys):
        """Test network error prompt with multiline explanation."""
        explanation = "DNS resolution failed\nRetry attempts exhausted\nCheck network"
        print_network_error_prompt(title="Network Failure", explanation=explanation)
        captured = capsys.readouterr()

        assert "NETWORK ERROR DETECTED" in captured.out
        assert "Error: Network Failure" in captured.out
        assert "DNS resolution failed" in captured.out
        assert "Retry attempts exhausted" in captured.out
        assert "Check network" in captured.out

    def test_network_error_empty_explanation(self, capsys):
        """Test network error prompt with empty explanation."""
        print_network_error_prompt(title="Network Error", explanation="")
        captured = capsys.readouterr()

        assert "NETWORK ERROR DETECTED" in captured.out
        assert "Error: Network Error" in captured.out
        assert "[1] RESUME" in captured.out

    def test_network_error_long_title(self, capsys):
        """Test network error prompt with long error title."""
        long_title = "Network Connection Failed After Multiple Retry Attempts"
        print_network_error_prompt(title=long_title, explanation="Details")
        captured = capsys.readouterr()

        assert "NETWORK ERROR DETECTED" in captured.out
        assert f"Error: {long_title}" in captured.out

    def test_network_error_contains_borders(self, capsys):
        """Test network error prompt contains box borders."""
        print_network_error_prompt(title="Test Error", explanation="Test explanation")
        captured = capsys.readouterr()

        assert "=" * PROMPT_WIDTH in captured.out

    def test_network_error_has_both_options(self, capsys):
        """Test network error prompt shows both option [1] and [2]."""
        print_network_error_prompt(title="Test", explanation="Test")
        captured = capsys.readouterr()

        assert "[1]" in captured.out
        assert "RESUME" in captured.out
        assert "wait for network to restore" in captured.out
        assert "[2]" in captured.out
        assert "STOP" in captured.out

    def test_network_error_explanation_indentation(self, capsys):
        """Test network error prompt indents explanation lines."""
        explanation = "Line one\nLine two"
        print_network_error_prompt(title="Test", explanation=explanation)
        captured = capsys.readouterr()

        # Each explanation line should be indented with "  "
        lines = captured.out.split("\n")
        explanation_lines = [
            line for line in lines if "Line one" in line or "Line two" in line
        ]
        for line in explanation_lines:
            assert line.startswith("  ")

    def test_network_error_differs_from_vendor_error(self, capsys):
        """Test network error prompt differs from vendor error prompt."""
        print_network_error_prompt(title="Network", explanation="Network issue")
        network_output = capsys.readouterr().out

        print_vendor_error_prompt(title="Vendor", explanation="Vendor issue")
        vendor_output = capsys.readouterr().out

        # Should have different headers
        assert "NETWORK ERROR DETECTED" in network_output
        assert "VENDOR ERROR ENCOUNTERED" in vendor_output
        assert "NETWORK ERROR DETECTED" not in vendor_output
        assert "VENDOR ERROR ENCOUNTERED" not in network_output

        # Network error should mention "wait for network to restore"
        assert "wait for network to restore" in network_output
        assert "wait for network to restore" not in vendor_output


@pytest.mark.unit
class TestPromptFormatConsistency:
    """Tests for consistency across all prompt functions."""

    def test_all_prompts_use_same_border_width(self, capsys):
        """Test all prompts use consistent border width."""
        # Budget prompt
        print_budget_prompt(10.0, 10.0)
        budget_out = capsys.readouterr().out
        budget_border = next(
            line for line in budget_out.split("\n") if line.startswith("=")
        )

        # Vendor error prompt
        print_vendor_error_prompt("Test", "Test")
        vendor_out = capsys.readouterr().out
        vendor_border = next(
            line for line in vendor_out.split("\n") if line.startswith("=")
        )

        # Network error prompt
        print_network_error_prompt("Test", "Test")
        network_out = capsys.readouterr().out
        network_border = next(
            line for line in network_out.split("\n") if line.startswith("=")
        )

        assert len(budget_border) == PROMPT_WIDTH
        assert len(vendor_border) == PROMPT_WIDTH
        assert len(network_border) == PROMPT_WIDTH

    def test_all_prompts_have_option_stop(self, capsys):
        """Test all prompts include the STOP option."""
        print_budget_prompt(10.0, 10.0)
        assert _OPTION_STOP in capsys.readouterr().out

        print_vendor_error_prompt("Test", "Test")
        assert _OPTION_STOP in capsys.readouterr().out

        print_network_error_prompt("Test", "Test")
        assert _OPTION_STOP in capsys.readouterr().out

    def test_all_prompts_have_option_continue_resume(self, capsys):
        """Test all prompts have option [1] for continue/resume."""
        print_budget_prompt(10.0, 10.0)
        budget_out = capsys.readouterr().out
        assert "[1]" in budget_out
        assert "CONTINUE" in budget_out

        print_vendor_error_prompt("Test", "Test")
        vendor_out = capsys.readouterr().out
        assert "[1]" in vendor_out
        assert "RESUME" in vendor_out

        print_network_error_prompt("Test", "Test")
        network_out = capsys.readouterr().out
        assert "[1]" in network_out
        assert "RESUME" in network_out

    def test_all_prompts_end_with_blank_line(self, capsys):
        """Test all prompts end with a blank line."""
        print_budget_prompt(10.0, 10.0)
        budget_out = capsys.readouterr().out
        assert budget_out.endswith("\n\n")

        print_vendor_error_prompt("Test", "Test")
        vendor_out = capsys.readouterr().out
        assert vendor_out.endswith("\n\n")

        print_network_error_prompt("Test", "Test")
        network_out = capsys.readouterr().out
        assert network_out.endswith("\n\n")
