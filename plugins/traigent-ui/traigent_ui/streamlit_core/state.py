"""
Session State Management Module
==============================

This module handles initialization and management of Streamlit session state
for the Traigent control center application.
"""

import os
from typing import Any, Dict, List

import streamlit as st


def init_session_state():
    """Initialize session state variables with default values."""
    # Navigation state
    if "navigation_mode" not in st.session_state:
        st.session_state.navigation_mode = "quick_start"

    # UI state
    if "banner_dismissed" not in st.session_state:
        st.session_state.banner_dismissed = False

    if "show_result_details" not in st.session_state:
        st.session_state.show_result_details = False

    if "selected_result_id" not in st.session_state:
        st.session_state.selected_result_id = None

    # API and authentication
    if "api_key_valid" not in st.session_state:
        st.session_state.api_key_valid = False

    # Optimization state
    if "optimization_results" not in st.session_state:
        st.session_state.optimization_results = []

    if "optimization_in_progress" not in st.session_state:
        st.session_state.optimization_in_progress = False

    if "logging_enabled" not in st.session_state:
        st.session_state.logging_enabled = True

    # Quick start template state
    if "template_index" not in st.session_state:
        st.session_state.template_index = 0

    if "selected_quick_start" not in st.session_state:
        st.session_state.selected_quick_start = None

    # Problem management state
    if "generated_problems" not in st.session_state:
        st.session_state.generated_problems = []

    if "problem_generation_in_progress" not in st.session_state:
        st.session_state.problem_generation_in_progress = False

    if "example_generation_in_progress" not in st.session_state:
        st.session_state.example_generation_in_progress = False

    # Settings and configuration
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""

    if "anthropic_api_key" not in st.session_state:
        st.session_state.anthropic_api_key = ""

    # Results browsing state
    if "results_page" not in st.session_state:
        st.session_state.results_page = 0

    # Problem creation flags
    if "start_problem_creation" not in st.session_state:
        st.session_state.start_problem_creation = False

    if "start_problem_analysis" not in st.session_state:
        st.session_state.start_problem_analysis = False

    if "start_example_generation" not in st.session_state:
        st.session_state.start_example_generation = False

    # Optimization execution flags
    if "show_cost_warning" not in st.session_state:
        st.session_state.show_cost_warning = False

    if "show_optimization_plan" not in st.session_state:
        st.session_state.show_optimization_plan = False

    if "run_optimization" not in st.session_state:
        st.session_state.run_optimization = False

    if "run_mode" not in st.session_state:
        st.session_state.run_mode = "dry"

    if "start_optimization_execution" not in st.session_state:
        st.session_state.start_optimization_execution = False


def validate_api_key() -> bool:
    """Validate the API key configuration."""
    # Check for OpenAI API key
    openai_key = st.session_state.get("openai_api_key", "") or os.getenv(
        "OPENAI_API_KEY", ""
    )

    # Check for Anthropic API key
    anthropic_key = st.session_state.get("anthropic_api_key", "") or os.getenv(
        "ANTHROPIC_API_KEY", ""
    )

    # Validate OpenAI key format
    openai_valid = bool(
        openai_key and openai_key.startswith("sk-") and len(openai_key) > 20
    )

    # Validate Anthropic key format
    anthropic_valid = bool(
        anthropic_key
        and anthropic_key.startswith("sk-ant-")
        and len(anthropic_key) > 30
    )

    # At least one API key should be valid
    has_valid_key = openai_valid or anthropic_valid

    # Update session state
    st.session_state.api_key_valid = has_valid_key

    return has_valid_key


def make_safe_filename(name: str) -> str:
    """Convert a name to a safe filename."""
    import re

    # Remove or replace unsafe characters
    safe_name = re.sub(r'[<>:"/\\|?*]', "_", name)

    # Remove leading/trailing whitespace and dots
    safe_name = safe_name.strip(" .")

    # Limit length
    if len(safe_name) > 100:
        safe_name = safe_name[:100]

    # Ensure it's not empty
    if not safe_name:
        safe_name = "unnamed"

    return safe_name


def get_optimization_config() -> Dict[str, Any]:
    """Get the current optimization configuration from session state."""
    return st.session_state.get("optimization_config", {})


def set_optimization_config(config: Dict[str, Any]):
    """Set the optimization configuration in session state."""
    st.session_state.optimization_config = config


def clear_optimization_state():
    """Clear optimization-related session state."""
    keys_to_clear = [
        "optimization_in_progress",
        "optimization_config",
        "start_problem_creation",
        "start_problem_analysis",
        "start_example_generation",
        "problem_generation_in_progress",
        "example_generation_in_progress",
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def get_navigation_mode() -> str:
    """Get the current navigation mode."""
    return st.session_state.get("navigation_mode", "quick_start")


def set_navigation_mode(mode: str):
    """Set the navigation mode."""
    st.session_state.navigation_mode = mode


def add_optimization_result(result: Dict[str, Any]):
    """Add an optimization result to the session state."""
    if "optimization_results" not in st.session_state:
        st.session_state.optimization_results = []

    st.session_state.optimization_results.append(result)


def get_optimization_results() -> List[Dict[str, Any]]:
    """Get all optimization results from session state."""
    return st.session_state.get("optimization_results", [])


def clear_optimization_results():
    """Clear all optimization results from session state."""
    st.session_state.optimization_results = []


def get_generated_problems() -> List[Dict[str, Any]]:
    """Get generated problems from session state."""
    return st.session_state.get("generated_problems", [])


def add_generated_problem(problem: Dict[str, Any]):
    """Add a generated problem to session state."""
    if "generated_problems" not in st.session_state:
        st.session_state.generated_problems = []

    st.session_state.generated_problems.append(problem)


def clear_generated_problems():
    """Clear generated problems from session state."""
    st.session_state.generated_problems = []


def is_optimization_in_progress() -> bool:
    """Check if optimization is currently in progress."""
    return st.session_state.get("optimization_in_progress", False)


def set_optimization_in_progress(in_progress: bool):
    """Set the optimization in progress flag."""
    st.session_state.optimization_in_progress = in_progress


def get_current_page() -> int:
    """Get the current page for results browsing."""
    return st.session_state.get("results_page", 0)


def set_current_page(page: int):
    """Set the current page for results browsing."""
    st.session_state.results_page = page


def reset_to_first_page():
    """Reset results browsing to the first page."""
    st.session_state.results_page = 0


def get_session_summary() -> Dict[str, Any]:
    """Get a summary of the current session state."""
    return {
        "navigation_mode": get_navigation_mode(),
        "api_key_valid": st.session_state.get("api_key_valid", False),
        "optimization_results_count": len(get_optimization_results()),
        "generated_problems_count": len(get_generated_problems()),
        "optimization_in_progress": is_optimization_in_progress(),
        "banner_dismissed": st.session_state.get("banner_dismissed", False),
    }
