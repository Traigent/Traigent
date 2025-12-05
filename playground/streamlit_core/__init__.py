"""
Streamlit Core Modules for TraiGent Control Center
=================================================

This package contains modular components for the TraiGent Streamlit application,
organized by functionality for better separation of concerns.

Modules:
- quick_start: Optimize Agent functionality
- browse_results: Browse Previous Optimizations
- custom_problems: Define New Use-Case functionality
- settings: Configuration and settings management
- components: Common UI components and utilities
- navigation: Header, navigation, and main orchestration
- state: Session state management
- optimization: Optimization execution logic
- validation: Input validation utilities
"""

__version__ = "1.0.0"

from .components import render_hero_banner, render_status_indicators

# Re-export key functions for easy imports
from .navigation import main, render_header
from .state import init_session_state, validate_api_key

__all__ = [
    "main",
    "render_header",
    "render_hero_banner",
    "init_session_state",
    "validate_api_key",
    "render_status_indicators",
]
