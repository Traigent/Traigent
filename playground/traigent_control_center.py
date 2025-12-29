#!/usr/bin/env python3
"""
Traigent Playground - Streamlit Application (Refactored)
======================================================

A comprehensive UI for defining AI problems and finding the best AI agents
to solve them using Traigent's intelligent comparison framework.

This is the main entry point for the refactored modular application.

Features:
- Natural language problem definition
- AI agent comparison and testing
- Performance visualization and analysis
- Export best agent configurations

Usage:
------
streamlit run playground/traigent_control_center.py
"""

import sys
from pathlib import Path

# Add project root and playground to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import required modules
import streamlit as st
from streamlit_core.components import render_hero_banner
from streamlit_core.navigation import (
    handle_navigation_routing,
    render_footer,
    render_header,
    render_main_navigation,
    render_settings_sidebar,
)

# Import modular components
from streamlit_core.state import init_session_state


def configure_page():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="Traigent - AI Agent Discovery",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="auto",
    )

    # Custom CSS for better styling
    st.markdown(
        """
    <style>
    .main > div {
        padding-top: 1rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        border: 1px solid #374151;
        background-color: #1f2937;
        color: #e5e7eb;
    }
    .stButton > button:hover {
        border-color: #10b981;
        background-color: #065f46;
    }
    .stButton > button[data-baseweb="button"][kind="primary"] {
        background-color: #10b981;
        border-color: #10b981;
    }
    .stSelectbox > label, .stTextInput > label, .stTextArea > label {
        color: #e5e7eb !important;
        font-weight: 500;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #10b981;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def main():
    """Main application entry point."""
    # Configure page
    configure_page()

    # Initialize session state
    init_session_state()

    # Render main layout components
    render_header()
    render_hero_banner()
    render_main_navigation()

    # Render settings sidebar
    render_settings_sidebar()

    # Handle navigation routing to different sections
    handle_navigation_routing()

    # Render footer
    render_footer()


if __name__ == "__main__":
    main()
