"""
Navigation and Header Module
===========================

This module handles the main navigation, header, and layout orchestration
for the TraiGent control center application.
"""

from typing import Dict, List

import streamlit as st

from .components import (
    render_hero_banner,
    render_navigation_tabs,
    render_status_indicators,
)
from .state import get_navigation_mode, validate_api_key


def validate_openai_api_key(api_key: str) -> Dict[str, any]:
    """Validate OpenAI API key using the OpenAI SDK."""
    if not api_key:
        return {"valid": False, "message": "API key is required", "type": "error"}

    if not api_key.startswith("sk-"):
        return {
            "valid": False,
            "message": "OpenAI API keys should start with 'sk-'",
            "type": "error",
        }

    if len(api_key) < 20:
        return {"valid": False, "message": "API key appears too short", "type": "error"}

    # Test with OpenAI SDK
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        # Make a simple API call to test the key
        client.models.list()
        return {
            "valid": True,
            "message": "API key verified with OpenAI",
            "type": "success",
        }
    except ImportError:
        # Fallback to format validation if OpenAI SDK not available
        if not api_key.replace("-", "").replace("_", "").isalnum():
            return {
                "valid": False,
                "message": "API key contains invalid characters",
                "type": "error",
            }
        return {
            "valid": True,
            "message": "API key format appears valid (install openai package for verification)",
            "type": "success",
        }
    except Exception as e:
        return {
            "valid": False,
            "message": f"API key verification failed: {str(e)}",
            "type": "error",
        }


def validate_anthropic_api_key(api_key: str) -> Dict[str, any]:
    """Validate Anthropic API key using the Anthropic SDK."""
    if not api_key:
        return {"valid": False, "message": "API key is required", "type": "error"}

    if not api_key.startswith("sk-ant-"):
        return {
            "valid": False,
            "message": "Anthropic API keys should start with 'sk-ant-'",
            "type": "error",
        }

    if len(api_key) < 30:
        return {"valid": False, "message": "API key appears too short", "type": "error"}

    # Test with Anthropic SDK
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        # Make a simple API call to test the key
        client.models.list()
        return {
            "valid": True,
            "message": "API key verified with Anthropic",
            "type": "success",
        }
    except ImportError:
        # Fallback to format validation if Anthropic SDK not available
        if not api_key.replace("-", "").replace("_", "").isalnum():
            return {
                "valid": False,
                "message": "API key contains invalid characters",
                "type": "error",
            }
        return {
            "valid": True,
            "message": "API key format appears valid (install anthropic package for verification)",
            "type": "success",
        }
    except Exception as e:
        return {
            "valid": False,
            "message": f"API key verification failed: {str(e)}",
            "type": "error",
        }


def get_navigation_options() -> List[Dict[str, str]]:
    """Get the main navigation options."""
    return [
        {
            "key": "quick_start",
            "icon": "⚡",
            "title": "Optimize Agent",
            "subtitle": "Try optimizating AI agents for popular use-cases in 30 seconds",
        },
        {
            "key": "browse_wins",
            "icon": "📚",
            "title": "Browse Prev. Optimizations",
            "subtitle": "See what worked for teams like yours",
        },
        {
            "key": "custom_test",
            "icon": "🎨",
            "title": "Define New Use-Case",
            "subtitle": "Define your exact use case",
        },
    ]


def render_header():
    """Render the main application header with status indicators."""
    # Three-column header layout
    header_col1, header_col2, header_col3 = st.columns([2, 6, 2])

    with header_col1:
        # App branding
        st.markdown(
            '<h1 style="font-size: 1.5rem; color: #10b981; font-weight: 700; margin: 0;">TraiGent</h1>',
            unsafe_allow_html=True,
        )

    with header_col2:
        # Current section title and description based on navigation mode
        current_mode = get_navigation_mode()
        nav_options = get_navigation_options()

        current_section = next(
            (opt for opt in nav_options if opt["key"] == current_mode), nav_options[0]
        )

        st.markdown(
            f'<div style="text-align: center; padding: 0.5rem 0;">'
            f'<h2 style="font-size: 1.125rem; color: #e5e7eb; font-weight: 600; margin: 0;">'
            f'{current_section["icon"]} {current_section["title"]}</h2>'
            f'<p style="font-size: 0.875rem; color: #9ca3af; margin: 0;">{current_section["subtitle"]}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with header_col3:
        # Status indicators
        api_status = (
            "⚠️" if not validate_api_key() and not st.session_state.api_key_valid else ""
        )
        live_status = ""  # No running optimizations tracked currently

        # Build status indicators safely
        status_html = render_status_indicators(api_status, live_status)

        if status_html:
            st.markdown(
                f'<div style="text-align: right; padding: 0.5rem 0;">{status_html}</div>',
                unsafe_allow_html=True,
            )

    # Separator
    st.markdown(
        "<div style='border-top: 1px solid #374151; margin: 1rem 0;'></div>",
        unsafe_allow_html=True,
    )


def render_main_navigation():
    """Render the main navigation tabs."""
    nav_options = get_navigation_options()
    current_mode = get_navigation_mode()

    render_navigation_tabs(nav_options, current_mode)


def render_footer():
    """Render the application footer."""
    st.markdown(
        """
        <div style="margin-top: 3rem; padding: 1.5rem 0; border-top: 1px solid #374151; text-align: center;">
            <p style="color: #9ca3af; font-size: 0.875rem; margin: 0;">
                🤖 TraiGent SDK - AI Agent Discovery Platform |
                <a href="https://github.com/Traigent/Traigent" style="color: #10b981;">GitHub</a> |
                <a href="mailto:support@traigent.ai" style="color: #10b981;">Support</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_settings_sidebar():
    """Render settings in the sidebar if needed."""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        # API Key Configuration
        with st.expander("🔑 API Keys", expanded=False):
            # OpenAI API Key
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.get("openai_api_key", ""),
                help="Your OpenAI API key for GPT models (starts with sk-)",
            )

            # Real-time validation for OpenAI key
            if openai_key.strip():
                openai_validation = validate_openai_api_key(openai_key)
                if openai_validation["type"] == "success":
                    st.success(f"✅ {openai_validation['message']}")
                else:
                    st.error(f"❌ {openai_validation['message']}")

            # Anthropic API Key
            anthropic_key = st.text_input(
                "Anthropic API Key",
                type="password",
                value=st.session_state.get("anthropic_api_key", ""),
                help="Your Anthropic API key for Claude models (starts with sk-ant-)",
            )

            # Real-time validation for Anthropic key
            if anthropic_key.strip():
                anthropic_validation = validate_anthropic_api_key(anthropic_key)
                if anthropic_validation["type"] == "success":
                    st.success(f"✅ {anthropic_validation['message']}")
                else:
                    st.error(f"❌ {anthropic_validation['message']}")

            # Save button with validation - disable if any entered key is invalid
            save_button_disabled = False
            if openai_key.strip() and not validate_openai_api_key(openai_key)["valid"]:
                save_button_disabled = True
            if (
                anthropic_key.strip()
                and not validate_anthropic_api_key(anthropic_key)["valid"]
            ):
                save_button_disabled = True

            # Also disable if no keys are entered at all
            if not openai_key.strip() and not anthropic_key.strip():
                save_button_disabled = True

            if st.button("💾 Save API Keys", disabled=save_button_disabled):
                # Only save non-empty keys
                if openai_key.strip():
                    st.session_state.openai_api_key = openai_key.strip()
                if anthropic_key.strip():
                    st.session_state.anthropic_api_key = anthropic_key.strip()

                # Update validation status - at least one key must be valid
                has_valid_key = False
                if openai_key.strip() and validate_openai_api_key(openai_key)["valid"]:
                    has_valid_key = True
                if (
                    anthropic_key.strip()
                    and validate_anthropic_api_key(anthropic_key)["valid"]
                ):
                    has_valid_key = True

                st.session_state.api_key_valid = has_valid_key

                if has_valid_key:
                    st.success("🔑 API keys saved successfully!")
                else:
                    st.error("❌ No valid API keys provided")
                st.rerun()

            # Show current validation status
            if not openai_key and not anthropic_key:
                st.info("💡 Add at least one API key to use live optimization")
            elif st.session_state.get("api_key_valid", False):
                st.success("🔑 Valid API key configured")
            else:
                st.warning("⚠️ No valid API keys configured")

        # Optimization Settings
        with st.expander("🔧 Optimization", expanded=False):
            logging_enabled = st.checkbox(
                "Enable Detailed Logging",
                value=st.session_state.get("logging_enabled", True),
                help="Enable detailed logging for optimization runs",
            )

            if logging_enabled != st.session_state.get("logging_enabled", True):
                st.session_state.logging_enabled = logging_enabled
                st.rerun()

        # Session Management
        with st.expander("🔄 Session", expanded=False):
            if st.button("🗑️ Clear All Data"):
                # Clear optimization results
                st.session_state.optimization_results = []
                st.session_state.generated_problems = []
                st.success("All session data cleared!")
                st.rerun()

            if st.button("🔄 Reset Session"):
                # Reset to default state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("Session reset!")
                st.rerun()


def render_page_layout():
    """Render the main page layout with header, navigation, and footer."""
    # Configure page
    st.set_page_config(
        page_title="TraiGent - AI Agent Discovery",
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

    # Render main components
    render_header()
    render_hero_banner()
    render_main_navigation()


def handle_navigation_routing():
    """Handle routing based on current navigation mode."""
    current_mode = get_navigation_mode()

    # Import here to avoid circular imports
    from .browse_results import render_browse_results_tab
    from .custom_problems import render_create_problem_section
    from .quick_start import render_optimization_tab

    # Route to appropriate section
    if current_mode == "quick_start":
        render_optimization_tab()
    elif current_mode == "browse_wins":
        render_browse_results_tab()
    elif current_mode == "custom_test":
        render_create_problem_section()
    else:
        st.error(f"Unknown navigation mode: {current_mode}")


def main():
    """Main application entry point."""
    # Initialize session state
    from .state import init_session_state

    init_session_state()

    # Render page layout
    render_page_layout()

    # Render settings sidebar
    render_settings_sidebar()

    # Handle navigation routing
    handle_navigation_routing()

    # Render footer
    render_footer()


if __name__ == "__main__":
    main()
