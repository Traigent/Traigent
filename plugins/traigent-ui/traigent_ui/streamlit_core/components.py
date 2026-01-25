"""
Common UI Components Module
==========================

This module contains reusable UI components and utilities used across
different sections of the Traigent control center.
"""

from typing import Any, Dict, List, Optional

import streamlit as st


def render_status_indicators(api_status: str = "", live_status: str = "") -> str:
    """Render status indicators for the header."""
    status_html = ""
    if api_status:
        status_html += (
            f'<span style="color: #f59e0b; font-size: 0.875rem;">{api_status}</span>'
        )
    if live_status:
        status_html += (
            f'<span style="color: #9ca3af; font-size: 0.75rem;">{live_status}</span>'
        )
    return status_html


def render_hero_banner():
    """Render the main hero banner."""
    if not st.session_state.get("banner_dismissed", False):
        with st.container():
            st.markdown(
                """
                <div style="background: linear-gradient(90deg, #064e3b 0%, #10b981 100%);
                            padding: 1.5rem; border-radius: 0.75rem; margin-bottom: 1.5rem;
                            border: 1px solid #10b981;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h1 style="color: white; font-size: 1.5rem; font-weight: 700; margin: 0 0 0.5rem 0;">
                                🤖 Traigent - AI Agent Discovery Platform
                            </h1>
                            <p style="color: #d1fae5; font-size: 1rem; margin: 0; line-height: 1.4;">
                                Find the perfect AI configuration for your specific use case.
                                Compare models, optimize costs, and improve performance - all in minutes.
                            </p>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("✕ Dismiss", key="dismiss_banner", help="Hide banner"):
                st.session_state.banner_dismissed = True
                st.rerun()


def render_navigation_tabs(nav_options: List[Dict[str, str]], current_mode: str):
    """Render the main navigation tabs."""
    # Navigation options
    cols = st.columns(len(nav_options))

    for _idx, (col, option) in enumerate(zip(cols, nav_options)):
        with col:
            # Determine if this tab is active
            is_active = st.session_state.navigation_mode == option["key"]

            # Button styling based on active state
            button_style = "primary" if is_active else "secondary"

            # Create the navigation button
            if st.button(
                f"{option['icon']} {option['title']}",
                key=f"nav_{option['key']}",
                help=option["subtitle"],
                use_container_width=True,
                type=button_style,
            ):
                st.session_state.navigation_mode = option["key"]
                st.rerun()


def render_section_header(title: str, subtitle: str = "", icon: str = ""):
    """Render a standardized section header."""
    header_text = f"{icon} {title}" if icon else title

    st.markdown(
        f'<h2 style="color: #10b981; font-size: 1.5rem; font-weight: 700; margin: 1rem 0 0.5rem 0;">{header_text}</h2>',
        unsafe_allow_html=True,
    )

    if subtitle:
        st.markdown(
            f'<p style="color: #9ca3af; font-size: 1rem; margin: 0 0 1rem 0;">{subtitle}</p>',
            unsafe_allow_html=True,
        )


def render_metric_card(title: str, value: str, icon: str = "", color: str = "#10b981"):
    """Render a metric card component."""
    return f"""
    <div style="background-color: #1f2937; border-radius: 0.5rem; padding: 1rem; text-align: center;">
        <div style="color: {color}; font-size: 1.5rem; font-weight: 600;">{icon} {value}</div>
        <div style="color: #9ca3af; font-size: 0.875rem; margin-top: 0.25rem;">{title}</div>
    </div>
    """


def render_info_card(title: str, content: str, card_type: str = "info"):
    """Render an information card with different types (info, success, warning, error)."""
    colors = {
        "info": {"bg": "#1e3a8a", "border": "#3b82f6", "text": "#dbeafe"},
        "success": {"bg": "#065f46", "border": "#10b981", "text": "#d1fae5"},
        "warning": {"bg": "#92400e", "border": "#f59e0b", "text": "#fef3c7"},
        "error": {"bg": "#7f1d1d", "border": "#ef4444", "text": "#fee2e2"},
    }

    color_scheme = colors.get(card_type, colors["info"])

    st.markdown(
        f"""
        <div style="background-color: {color_scheme['bg']};
                    border: 1px solid {color_scheme['border']};
                    border-radius: 0.5rem;
                    padding: 1rem;
                    margin: 0.5rem 0;">
            <h4 style="color: {color_scheme['text']}; margin: 0 0 0.5rem 0; font-size: 1rem;">{title}</h4>
            <p style="color: {color_scheme['text']}; margin: 0; font-size: 0.875rem;">{content}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_progress_indicator(current: int, total: int, label: str = "Progress"):
    """Render a progress indicator with label."""
    percentage = (current / total * 100) if total > 0 else 0

    st.markdown(f"**{label}**: {current}/{total} ({percentage:.1f}%)")
    st.progress(current / total if total > 0 else 0)


def render_expandable_json(data: Any, title: str = "Data", expanded: bool = False):
    """Render JSON data in an expandable section."""
    with st.expander(f"🔍 {title}", expanded=expanded):
        st.json(data)


def render_confirmation_dialog(
    message: str,
    confirm_text: str = "Confirm",
    cancel_text: str = "Cancel",
    key_prefix: str = "confirm",
) -> Optional[bool]:
    """
    Render a confirmation dialog and return the user's choice.

    Returns:
        True if confirmed, False if cancelled, None if no action taken
    """
    st.warning(message)

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button(confirm_text, key=f"{key_prefix}_yes", type="primary"):
            return True

    with col2:
        if st.button(cancel_text, key=f"{key_prefix}_no"):
            return False

    return None


def render_data_table(
    data: List[Dict[str, Any]],
    columns: List[str],
    sortable: bool = True,
    searchable: bool = True,
):
    """Render a data table with optional sorting and searching."""
    if not data:
        st.info("No data to display.")
        return

    # Search functionality
    if searchable:
        search_term = st.text_input("🔍 Search", placeholder="Type to search...")
        if search_term:
            # Filter data based on search term
            filtered_data = [
                row
                for row in data
                if any(
                    search_term.lower() in str(row.get(col, "")).lower()
                    for col in columns
                )
            ]
        else:
            filtered_data = data
    else:
        filtered_data = data

    # Display table
    if filtered_data:
        import pandas as pd

        df = pd.DataFrame(filtered_data)

        # Select only specified columns if they exist
        available_columns = [col for col in columns if col in df.columns]
        if available_columns:
            df = df[available_columns]

        st.dataframe(df, use_container_width=True)
    else:
        st.info("No results match your search.")


def render_loading_spinner(message: str = "Loading..."):
    """Render a loading spinner with message."""
    with st.spinner(message):
        pass  # The spinner will show while this context is active


def create_download_button(
    data: Any,
    filename: str,
    label: str = "Download",
    mime_type: str = "application/json",
):
    """Create a download button for data."""
    import json

    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, indent=2)
    else:
        data_str = str(data)

    return st.download_button(
        label=label, data=data_str, file_name=filename, mime=mime_type
    )


def render_toast_message(message: str, message_type: str = "info", duration: int = 3):
    """
    Render a toast-style message (using Streamlit's built-in messaging).

    Args:
        message: The message to display
        message_type: Type of message (info, success, warning, error)
        duration: Duration in seconds (not actually used in Streamlit)
    """
    if message_type == "success":
        st.success(message)
    elif message_type == "warning":
        st.warning(message)
    elif message_type == "error":
        st.error(message)
    else:
        st.info(message)


def render_sidebar_filters(filters: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Render filter controls in the sidebar.

    Args:
        filters: Dictionary defining filter controls

    Returns:
        Dictionary of selected filter values
    """
    filter_values = {}

    st.sidebar.markdown("### 🔧 Filters")

    for filter_name, filter_config in filters.items():
        filter_type = filter_config.get("type", "selectbox")

        if filter_type == "selectbox":
            value = st.sidebar.selectbox(
                filter_config["label"],
                filter_config["options"],
                index=filter_config.get("default_index", 0),
            )
        elif filter_type == "multiselect":
            value = st.sidebar.multiselect(
                filter_config["label"],
                filter_config["options"],
                default=filter_config.get("default", []),
            )
        elif filter_type == "slider":
            value = st.sidebar.slider(
                filter_config["label"],
                filter_config["min_value"],
                filter_config["max_value"],
                filter_config.get("default", filter_config["min_value"]),
            )
        elif filter_type == "text_input":
            value = st.sidebar.text_input(
                filter_config["label"], value=filter_config.get("default", "")
            )
        else:
            continue

        filter_values[filter_name] = value

    return filter_values
