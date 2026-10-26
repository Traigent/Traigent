"""Characterization of ValidationResult.get_feedback exact output."""

from __future__ import annotations

from traigent.utils.validation import ValidationResult


def test_empty_is_success_only():
    assert ValidationResult().get_feedback() == "✅ Validation passed!"


def test_errors_with_suggestions_and_no_success_line():
    r = ValidationResult()
    r.add_error("f", "bad", suggestions=["s1", "s2"])
    r.add_error("g", "worse")
    assert r.get_feedback() == (
        "❌ Validation Errors:\n  • f: bad\n    💡 s1\n    💡 s2\n  • g: worse"
    )


def test_warnings_only_adds_no_success_line_and_no_blank_prefix():
    r = ValidationResult()
    r.add_warning("w", "careful", suggestions=["fix"])
    assert r.get_feedback() == "⚠️  Warnings:\n  • w: careful\n    💡 fix"
    assert r.get_feedback(include_warnings=False) == ""


def test_warnings_hidden_still_blocks_success_line():
    r = ValidationResult()
    r.add_warning("w", "x")
    r.suggestions.append("hint")
    assert r.get_feedback(include_warnings=False) == "💡 Suggestions:\n  • hint"


def test_errors_warnings_suggestions_blank_line_separators():
    r = ValidationResult()
    r.add_error("e", "m")
    r.add_warning("w", "n")
    r.suggestions.extend(["a", "b"])
    assert r.get_feedback() == (
        "❌ Validation Errors:\n  • e: m\n\n"
        "⚠️  Warnings:\n  • w: n\n\n"
        "💡 Suggestions:\n  • a\n  • b"
    )
    assert r.get_feedback(include_warnings=False) == (
        "❌ Validation Errors:\n  • e: m\n\n💡 Suggestions:\n  • a\n  • b"
    )


def test_suggestions_only_with_valid_result_gets_success_line():
    r = ValidationResult()
    r.suggestions.append("a")
    assert r.get_feedback() == "💡 Suggestions:\n  • a\n✅ Validation passed!"
