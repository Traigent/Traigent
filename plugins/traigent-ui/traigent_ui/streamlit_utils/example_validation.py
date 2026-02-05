"""
Example Validation Utilities for Streamlit UI.

This module provides utilities for validating and displaying
generated examples in the Streamlit interface.
"""

import json
from typing import Any, Dict, List, Tuple

import streamlit as st


def validate_examples_ui(
    examples: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Validate examples and show results in UI.

    Args:
        examples: List of examples to validate

    Returns:
        Tuple of (valid_examples, validation_errors)
    """
    valid_examples = []
    errors = []

    for i, example in enumerate(examples):
        example_errors = validate_single_example_ui(example, i + 1)

        if example_errors:
            errors.extend(example_errors)
        else:
            valid_examples.append(example)

    # Display validation results
    if errors:
        st.warning(f"⚠️ Validation found {len(errors)} issues:")
        for error in errors:
            st.text(f"• {error}")

    if valid_examples:
        st.success(f"✅ {len(valid_examples)} examples passed validation")

    return valid_examples, errors


def validate_single_example_ui(example: Dict[str, Any], example_num: int) -> List[str]:
    """
    Validate a single example and return error messages.

    Args:
        example: Example to validate
        example_num: Example number for error messages

    Returns:
        List of error messages
    """
    errors = []

    # Check required fields
    required_fields = ["input_data", "expected_output"]
    for field in required_fields:
        if field not in example:
            errors.append(f"Example {example_num}: Missing required field '{field}'")

    # Validate input_data
    if "input_data" in example:
        if not isinstance(example["input_data"], dict):
            errors.append(f"Example {example_num}: 'input_data' must be a dictionary")
        elif not example["input_data"]:
            errors.append(f"Example {example_num}: 'input_data' cannot be empty")

    # Validate expected_output
    if "expected_output" in example:
        expected_output = example["expected_output"]
        if expected_output is None or expected_output == "":
            errors.append(f"Example {example_num}: 'expected_output' cannot be empty")

    # Validate difficulty if present
    if "difficulty" in example:
        valid_difficulties = ["easy", "medium", "hard", "very_hard", "expert"]
        if example["difficulty"] not in valid_difficulties:
            errors.append(
                f"Example {example_num}: Invalid difficulty '{example['difficulty']}'. Must be one of: {', '.join(valid_difficulties)}"
            )

    return errors


def display_examples_preview(examples: List[Dict[str, Any]], max_show: int = 3):
    """
    Display a preview of generated examples.

    Args:
        examples: List of examples to display
        max_show: Maximum number of examples to show
    """
    if not examples:
        st.info("No examples to display")
        return

    st.markdown(f"### Generated Examples Preview ({len(examples)} total)")

    # Show first few examples
    for i, example in enumerate(examples[:max_show]):
        with st.expander(f"Example {i + 1}", expanded=(i == 0)):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Input:**")
                if isinstance(example.get("input_data"), dict):
                    for key, value in example["input_data"].items():
                        st.text(f"{key}: {value}")
                else:
                    st.text(str(example.get("input_data", "N/A")))

            with col2:
                st.markdown("**Expected Output:**")
                st.text(str(example.get("expected_output", "N/A")))

            # Additional metadata
            if example.get("difficulty"):
                st.text(f"**Difficulty:** {example['difficulty']}")

            if example.get("metadata", {}).get("reasoning"):
                st.text(f"**Reasoning:** {example['metadata']['reasoning']}")

    # Show count if there are more examples
    if len(examples) > max_show:
        st.info(f"... and {len(examples) - max_show} more examples")


def display_generation_summary(
    examples: List[Dict[str, Any]],
    provider: str,
    generation_time: float,
    token_usage: Dict[str, int] = None,
):
    """
    Display a summary of the generation results.

    Args:
        examples: Generated examples
        provider: LLM provider used
        generation_time: Time taken for generation
        token_usage: Optional token usage information
    """
    st.markdown("### Generation Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Examples Generated", len(examples))

    with col2:
        st.metric("Provider", provider)

    with col3:
        st.metric("Generation Time", f"{generation_time:.1f}s")

    with col4:
        if token_usage:
            total_tokens = token_usage.get("total_tokens", 0)
            st.metric("Tokens Used", f"{total_tokens:,}")
        else:
            st.metric("Tokens Used", "N/A")

    # Difficulty distribution
    if examples:
        difficulties = [ex.get("difficulty", "medium") for ex in examples]
        difficulty_counts = {}
        for diff in difficulties:
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

        if difficulty_counts:
            st.markdown("**Difficulty Distribution:**")
            for diff, count in sorted(difficulty_counts.items()):
                percentage = (count / len(examples)) * 100
                st.text(f"• {diff.title()}: {count} examples ({percentage:.1f}%)")


def show_save_confirmation(examples: List[Dict[str, Any]], problem_name: str) -> bool:
    """
    Show save confirmation dialog and return user choice.

    Args:
        examples: Examples to save
        problem_name: Name of the problem

    Returns:
        True if user confirms save, False otherwise
    """
    st.markdown("### Save Generated Examples")

    # Summary
    st.info(f"Ready to add {len(examples)} new examples to problem '{problem_name}'")

    # Warning about file modification
    st.warning(
        "⚠️ This will modify the problem file. A backup will be created automatically."
    )

    # Show what will be saved
    with st.expander("Preview examples to be saved"):
        display_examples_preview(examples, max_show=5)

    # Confirmation buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("💾 Save Examples", type="primary"):
            return True

    with col2:
        if st.button("❌ Cancel"):
            return False

    return None  # No decision made yet


def display_save_result(success: bool, message: str, examples_count: int = 0):
    """
    Display the result of saving examples.

    Args:
        success: Whether save was successful
        message: Result message
        examples_count: Number of examples saved
    """
    if success:
        st.success(f"✅ {message}")
        if examples_count > 0:
            st.balloons()  # Celebrate success!
            st.info(
                f"🎉 Successfully added {examples_count} new examples! "
                "The problem is now ready for optimization with more diverse data."
            )
    else:
        st.error(f"❌ {message}")
        st.info(
            "💡 **Troubleshooting tips:**\n"
            "- Check that the problem file is not read-only\n"
            "- Ensure you have write permissions\n"
            "- Try refreshing the page and trying again"
        )


def export_examples_json(
    examples: List[Dict[str, Any]], filename: str = "generated_examples.json"
):
    """
    Provide download button for examples as JSON.

    Args:
        examples: Examples to export
        filename: Name for the download file
    """
    if examples:
        json_data = json.dumps(examples, indent=2)

        st.download_button(
            label="📥 Download Examples as JSON",
            data=json_data,
            file_name=filename,
            mime="application/json",
            help="Download the generated examples as a JSON file for backup or external use",
        )
