"""
Enhanced Example Generation with LLM Integration.

This module provides functions to generate additional examples for existing problems
using various LLM providers with progress tracking and validation.
"""

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from traigent.utils.secure_path import safe_read_text, safe_write_text, validate_path
from .llm_providers import LLMProviderManager
from .prompt_builder import build_prompt_for_problem


async def generate_examples_for_problem(
    problem_name: str,
    count: int,
    llm_provider: str,
    custom_instructions: str = "",
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[bool, List[Dict[str, Any]], str]:
    """
    Generate additional examples for an existing problem.

    Args:
        problem_name: Name of the problem to extend
        count: Number of examples to generate (max 100)
        llm_provider: Name of the LLM provider to use
        custom_instructions: Additional instructions for generation
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (success, generated_examples, error_message)
    """
    try:
        # Validate inputs
        if count > 100:
            return False, [], "Maximum 100 examples allowed per generation"

        if count <= 0:
            return False, [], "Count must be greater than 0"

        if progress_callback:
            progress_callback("Loading problem definition...", 0.05)

        # Load the problem
        problem_instance = await load_problem_instance(problem_name)
        if not problem_instance:
            return False, [], f"Problem '{problem_name}' not found"

        if progress_callback:
            progress_callback("Building generation prompt...", 0.15)

        # Build the generation prompt
        prompt = build_prompt_for_problem(
            problem_instance=problem_instance,
            count=count,
            custom_instructions=custom_instructions,
        )

        if progress_callback:
            progress_callback(f"Generating examples with {llm_provider}...", 0.25)

        # Generate examples using the specified provider
        provider_manager = LLMProviderManager()

        def provider_progress(message: str, progress: float):
            if progress_callback:
                # Map provider progress to our overall progress (25% to 80%)
                overall_progress = 0.25 + (progress * 0.55)
                progress_callback(message, overall_progress)

        result = await provider_manager.generate_examples(
            provider_name=llm_provider,
            prompt=prompt,
            count=count,
            progress_callback=provider_progress,
        )

        if not result.success:
            return False, [], result.error or f"Generation failed with {llm_provider}"

        if progress_callback:
            progress_callback("Validating and formatting examples...", 0.85)

        # Validate and format examples
        validated_examples = validate_generated_examples(
            result.examples, problem_instance
        )

        if not validated_examples:
            return False, [], "No valid examples generated"

        if progress_callback:
            progress_callback(
                f"Successfully generated {len(validated_examples)} examples!", 1.0
            )

        return True, validated_examples, ""

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        if progress_callback:
            progress_callback(f"Error: {error_msg}", 0.0)
        return False, [], error_msg


async def load_problem_instance(problem_name: str):
    """Load a problem instance by name."""
    try:
        # Try different import paths
        try:
            from ..langchain_problems import get_problem_class
        except ImportError:
            try:
                from langchain_problems import get_problem_class
            except ImportError:
                from examples.langchain_problems import get_problem_class

        problem_class = get_problem_class(problem_name)
        return problem_class()

    except Exception as e:
        print(f"Error loading problem {problem_name}: {e}")
        return None


def validate_generated_examples(
    examples: List[Dict[str, Any]], problem_instance
) -> List[Dict[str, Any]]:
    """
    Validate that generated examples match the expected format.

    Args:
        examples: List of generated examples
        problem_instance: The problem instance for validation

    Returns:
        List of validated examples
    """
    validated = []

    for example in examples:
        if validate_single_example(example, problem_instance):
            validated.append(example)

    return validated


def validate_single_example(example: Dict[str, Any], problem_instance) -> bool:
    """
    Validate a single example.

    Args:
        example: The example to validate
        problem_instance: The problem instance for validation

    Returns:
        True if valid, False otherwise
    """
    # Basic structure validation
    required_fields = ["input_data", "expected_output"]
    if not all(field in example for field in required_fields):
        return False

    # Check input_data format
    if not isinstance(example["input_data"], dict):
        return False

    # Check that input_data is not empty
    if not example["input_data"]:
        return False

    # Check expected_output exists and is not None/empty string
    expected_output = example["expected_output"]
    if expected_output is None or expected_output == "":
        return False

    # If problem has categories, validate against them
    if hasattr(problem_instance, "CATEGORIES"):
        categories = problem_instance.CATEGORIES
        if isinstance(expected_output, str) and expected_output not in categories:
            # For classification problems, output should be a valid category
            return False

    # Validate difficulty if present
    if "difficulty" in example:
        valid_difficulties = ["easy", "medium", "hard", "very_hard", "expert"]
        if example["difficulty"] not in valid_difficulties:
            # Try to fix common variations
            diff_lower = example["difficulty"].lower()
            if diff_lower in valid_difficulties:
                example["difficulty"] = diff_lower
            else:
                example["difficulty"] = "medium"  # Default
    else:
        example["difficulty"] = "medium"  # Add default if missing

    # Ensure metadata exists
    if "metadata" not in example:
        example["metadata"] = {}

    return True


async def save_examples_to_problem(
    problem_name: str, new_examples: List[Dict[str, Any]]
) -> Tuple[bool, str]:
    """
    Save new examples to the problem file.

    Args:
        problem_name: Name of the problem
        new_examples: List of new examples to add

    Returns:
        Tuple of (success, message)
    """
    try:
        # Find the problem file
        problem_file = find_problem_file(problem_name)
        if not problem_file:
            return False, f"Could not find problem file for '{problem_name}'"

        problem_file = validate_path(problem_file, problem_file.parent, must_exist=True)

        # Read the current file content
        content = safe_read_text(problem_file, problem_file.parent)

        # Backup the original file
        backup_file = validate_path(
            problem_file.with_suffix(".py.backup"),
            problem_file.parent,
        )
        safe_write_text(backup_file, content, problem_file.parent)

        # Parse and update the examples
        updated_content = add_examples_to_content(content, new_examples)

        # Write the updated content
        safe_write_text(problem_file, updated_content, problem_file.parent)

        return (
            True,
            f"Successfully added {len(new_examples)} examples to {problem_name}",
        )

    except Exception as e:
        return False, f"Error saving examples: {str(e)}"


def find_problem_file(problem_name: str) -> Optional[Path]:
    """Find the Python file for a given problem."""
    # Look in the langchain_problems directory
    base_dir = Path(__file__).parent.parent
    problems_dir = validate_path(base_dir / "langchain_problems", base_dir)

    # Try exact match first
    exact_file = validate_path(
        problems_dir / f"{problem_name}.py",
        problems_dir,
    )
    if exact_file.exists():
        return exact_file

    # Try looking for files containing the problem name
    for file_path in problems_dir.glob("*.py"):
        if file_path.stem == problem_name or problem_name in file_path.stem:
            return validate_path(file_path, problems_dir, must_exist=True)

    return None


def add_examples_to_content(content: str, new_examples: List[Dict[str, Any]]) -> str:
    """
    Add new examples to the problem file content.

    Args:
        content: Original file content
        new_examples: New examples to add

    Returns:
        Updated file content
    """
    # Convert new examples to the format used in problem files
    examples_code = convert_examples_to_code(new_examples)

    # Find the examples_data list in the file
    # Look for patterns like: examples_data = [ or examples_data = [
    pattern = r"(examples_data\s*=\s*\[)(.*?)(\s*\])"
    match = re.search(pattern, content, re.DOTALL)

    if match:
        # Found existing examples_data
        start_part = match.group(1)
        existing_examples = match.group(2)
        end_part = match.group(3)

        # Determine if we need a separator
        stripped_examples = existing_examples.strip()

        if not stripped_examples:
            # Empty array - add examples without leading comma
            final_content = examples_code
        else:
            # Non-empty array - need to check if we need a comma
            # Check if the existing content ends with a closing brace and optional comma
            if re.search(r"}\s*,?\s*$", stripped_examples):
                # Ends with } or }, - ensure we have exactly one comma
                if re.search(r"},\s*$", stripped_examples):
                    # Already has comma, just add newline and new examples
                    final_content = existing_examples + "\n" + examples_code
                else:
                    # No comma, add comma, newline, and new examples
                    final_content = existing_examples + ",\n" + examples_code
            else:
                # Shouldn't happen in valid Python, but handle gracefully
                final_content = existing_examples + ",\n" + examples_code

        # Reconstruct the examples_data
        new_examples_data = start_part + final_content + end_part

        # Replace in content
        updated_content = (
            content[: match.start()] + new_examples_data + content[match.end() :]
        )

        return updated_content
    else:
        # Could not find examples_data, append to end of file
        return (
            content
            + "\n\n# Additional examples added by Traigent Playground\n"
            + "# You may need to manually integrate these into your examples_data list:\n"
            + f"additional_examples = [{examples_code}]"
        )


def convert_examples_to_code(examples: List[Dict[str, Any]]) -> str:
    """Convert examples to Python code format."""
    code_parts = []

    for i, example in enumerate(examples):
        # Format the example as a Python dictionary
        example_code = "            {\n"

        # Add input_data (typically query)
        if "input_data" in example and isinstance(example["input_data"], dict):
            for key, value in example["input_data"].items():
                example_code += f'                "{key}": {repr(value)},\n'

        # Add expected_output (typically category)
        if "expected_output" in example:
            example_code += (
                f'                "category": {repr(example["expected_output"])},\n'
            )

        # Add difficulty
        if "difficulty" in example:
            example_code += (
                f'                "difficulty": {repr(example["difficulty"])},\n'
            )

        # Add reasoning from metadata
        if "metadata" in example and isinstance(example["metadata"], dict):
            reasoning = example["metadata"].get(
                "reasoning", f"Generated example {i + 1}"
            )
            example_code += f'                "reasoning": {repr(reasoning)},\n'
        else:
            example_code += (
                f'                "reasoning": "Generated example {i + 1}",\n'
            )

        # Remove trailing comma from last line
        lines = example_code.rstrip().split("\n")
        if lines[-1].endswith(","):
            lines[-1] = lines[-1][:-1]
        example_code = "\n".join(lines) + "\n"

        example_code += "            }"
        code_parts.append(example_code)

    # Join all examples with commas (no leading comma)
    if code_parts:
        return ",\n".join(code_parts)
    else:
        return ""


def get_available_providers() -> List[str]:
    """Get list of available LLM providers."""
    manager = LLMProviderManager()
    return manager.get_available_providers()
