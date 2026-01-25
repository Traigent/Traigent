"""
Quick Start / Optimize Agent Module
==================================

This module contains all functionality related to the Quick Start optimization workflow,
including template selection, optimization configuration, and execution.
"""

import asyncio
from typing import Any, Dict

import streamlit as st

# Import Traigent modules and utilities
try:
    # Import availability checks only
    import langchain_problems  # noqa: F401 - Import check only
    import optimization_callbacks  # noqa: F401 - Import check only
    import optimization_storage  # noqa: F401 - Import check only
    import problem_management  # noqa: F401 - Import check only

    import traigent  # noqa: F401 - Import check only
except ImportError as e:
    st.error(f"Import error: {e}")


def render_quick_start_section():
    """Render the quick start section with predefined optimization templates."""
    quick_starts = [
        {
            "title": "Customer Support Classifier",
            "description": "Route tickets to the right team",
            "current": "GPT-4 ($2.50/1K)",
            "optimized": "GPT-3.5-turbo ($0.50/1K)",
            "savings": "80%",
            "time": "2 min",
        },
        {
            "title": "Document Summarizer",
            "description": "Condense reports and articles",
            "current": "GPT-4 ($3.00/1K)",
            "optimized": "Claude-3-haiku ($0.60/1K)",
            "savings": "80%",
            "time": "3 min",
        },
        {
            "title": "Code Review Assistant",
            "description": "Automated PR feedback",
            "current": "GPT-4 ($4.00/1K)",
            "optimized": "GPT-4 (optimized) ($1.80/1K)",
            "savings": "55%",
            "time": "4 min",
        },
    ]

    for qs in quick_starts:
        with st.expander(f"**{qs['title']}** - Save {qs['savings']}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Description:** {qs['description']}")
                st.markdown(f"**Current:** {qs['current']}")
                st.markdown(f"**Optimized:** {qs['optimized']}")

            with col2:
                st.metric("Savings", qs["savings"])
                st.metric("Time to optimize", qs["time"])

                if st.button("🚀 Start This Optimization", key=f"qs_{qs['title']}"):
                    st.session_state.selected_quick_start = qs["title"]
                    st.success(f"Selected: {qs['title']}")


def get_quick_start_templates() -> Dict[str, Dict[str, Any]]:
    """Get the available quick start templates."""
    return {
        "Customer Support Classifier": {
            "description": "Route tickets to the right team efficiently",
            "models": ["gpt-3.5-turbo", "gpt-4o-mini"],
            "performance": "80% accuracy in ~3 min",
            "test_size": 20,
            "problem_name": "customer_support",
        },
        "Document Summarizer": {
            "description": "Condense reports and articles intelligently",
            "models": ["gpt-3.5-turbo", "gpt-4o-mini"],
            "performance": "75% cost reduction",
            "test_size": 15,
            "problem_name": "document_summarization",
        },
        "Code Review Assistant": {
            "description": "Automated pull request feedback",
            "models": ["gpt-4o-mini", "gpt-4o"],
            "performance": "55% time savings",
            "test_size": 10,
            "problem_name": "code_review",
        },
        "Custom Use-Cases": {
            "description": "Configure your own optimization test",
            "models": [],
            "performance": "Varies by use case",
            "test_size": 0,
            "problem_name": None,
        },
    }


def render_template_navigation() -> str:
    """Render template navigation controls and return selected template."""
    quick_start_templates = get_quick_start_templates()

    # Initialize template index in session state
    if "template_index" not in st.session_state:
        st.session_state.template_index = 0

    # Get current template
    template_list = list(quick_start_templates.keys())
    current_template_idx = st.session_state.template_index
    selected_template = template_list[current_template_idx]

    # Template navigation and card
    nav_col1, nav_col2, nav_col3 = st.columns([1, 8, 1])

    with nav_col1:
        if st.button(
            "◀",
            key="prev_template",
            disabled=current_template_idx == 0,
            help="Previous use case",
            use_container_width=True,
        ):
            st.session_state.template_index = max(0, current_template_idx - 1)
            st.rerun()

    with nav_col3:
        if st.button(
            "▶",
            key="next_template",
            disabled=current_template_idx >= len(template_list) - 1,
            help="Next use case",
            use_container_width=True,
        ):
            st.session_state.template_index = min(
                len(template_list) - 1, current_template_idx + 1
            )
            st.rerun()

    return selected_template


def render_template_card(template_name: str, template: Dict[str, Any]):
    """Render the template information card."""
    st.markdown(
        f"""
    <div style="background-color: #1f2937; border: 2px solid #10b981; border-radius: 0.75rem;
                padding: 1.5rem; margin: 0;">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.75rem;">
            <div>
                <h3 style="color: #10b981; font-size: 1.125rem; font-weight: 600; margin: 0 0 0.25rem 0;">
                    {template_name}
                </h3>
                <p style="color: #9ca3af; font-size: 0.875rem; margin: 0;">
                    {template['description']}
                </p>
            </div>
            <div style="background-color: #065f46; color: #10b981; padding: 0.25rem 0.75rem;
                        border-radius: 1rem; font-size: 0.75rem; font-weight: 500;">
                {template['performance']}
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_optimization_configuration(template_name: str, template: Dict[str, Any]):
    """Render optimization configuration options."""
    st.markdown(
        '<h3 style="font-size: 1rem; color: #10b981; font-weight: 600; margin: 1.5rem 0 0.75rem 0;">⚙️ Optimization Settings</h3>',
        unsafe_allow_html=True,
    )

    # Single row configuration layout
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Inline multiselect for models - always editable
        st.markdown("**🎯 Models to Test**")
        available_models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "claude-3-haiku"]

        # Get default models from template or use defaults
        default_models = (
            template.get("models", ["gpt-3.5-turbo", "gpt-4o-mini"])
            if template_name != "Custom Use-Cases"
            else ["gpt-3.5-turbo", "gpt-4o-mini"]
        )

        selected_models = st.multiselect(
            "Select models:",
            available_models,
            default=default_models,
            key=f"models_selection_{template_name}",
            label_visibility="collapsed",
            help="Choose which AI models to compare",
        )

    with col2:
        st.markdown("**📊 Examples per Configuration**")
        # Always editable examples per configuration
        default_examples = (
            template.get("test_size", 5) if template_name != "Custom Use-Cases" else 5
        )

        examples_per_config = st.number_input(
            "Examples per config:",
            min_value=1,
            max_value=50,
            value=default_examples,
            key=f"examples_per_config_{template_name}",
            label_visibility="collapsed",
            help="Number of test examples per model configuration",
        )

    with col3:
        st.markdown("**🔧 Max. Configurations**")
        max_configurations = st.number_input(
            "Max configs:",
            min_value=1,
            max_value=100,
            value=5,
            key="max_configurations",
            help="Maximum number of configurations to explore",
            label_visibility="collapsed",
        )

    # Advanced Settings in an expander (simplified)
    with st.expander("🔧 Advanced Options", expanded=False):
        strategy_options = {
            "🔍 Systematic Exploration": "grid",
            "🎲 Random Search": "random",
            "🧠 Smart Exploration (Conservative)": "smart_exploration_conservative",
            "🚀 Smart Exploration (Aggressive)": "smart_exploration_aggressive",
            "⚡ Adaptive Batch": "adaptive_batch",
            "🔄 Parallel Batch": "parallel_batch",
        }

        strategy_display = st.selectbox(
            "Optimization Strategy:",
            options=list(strategy_options.keys()),
            index=0,
            key="strategy_selection",
            help="Choose how to explore different model configurations",
        )

        strategy = strategy_options[strategy_display]

        # Temperature settings
        temperature_range = st.slider(
            "Temperature Range:",
            min_value=0.0,
            max_value=2.0,
            value=(0.0, 1.0),
            step=0.1,
            key="temperature_range",
            help="Temperature controls randomness in model responses",
        )

    return {
        "selected_models": selected_models,
        "examples_per_config": examples_per_config,
        "max_configurations": max_configurations,
        "strategy": strategy,
        "temperature_range": temperature_range,
    }


def render_optimization_controls(template_name: str, config: Dict[str, Any]) -> bool:
    """Render optimization action buttons and handle cost warning dialog."""
    st.markdown(
        '<h3 style="font-size: 1rem; color: #10b981; font-weight: 600; margin: 1.5rem 0 0.75rem 0;">🚀 Start Optimization</h3>',
        unsafe_allow_html=True,
    )

    # Check if cost warning should be shown
    if st.session_state.get("show_cost_warning", False):
        return render_cost_warning_dialog(config)

    # Main action buttons
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        show_plan = st.button(
            "📄 Show Optimization Plan",
            help="See what configurations will be tested",
            use_container_width=True,
            type="secondary",
        )

    with col2:
        dry_run = st.button(
            "🧪 Dry Run",
            help="Test without API calls (no costs)",
            use_container_width=True,
            type="secondary",
        )

    with col3:
        start_optimization = st.button(
            "🚀 Start",
            help="Begin real optimization with API calls",
            use_container_width=True,
            type="primary",
        )

    # Handle button actions
    if show_plan:
        st.session_state.show_optimization_plan = True
        st.rerun()

    if dry_run:
        if not config["selected_models"]:
            st.error("Please select at least one model")
        else:
            st.session_state.run_optimization = True
            st.session_state.run_mode = "dry"
            return True

    if start_optimization:
        if not config["selected_models"]:
            st.error("Please select at least one model")
        else:
            st.session_state.show_cost_warning = True
            st.rerun()

    # Show optimization plan if requested
    if st.session_state.get("show_optimization_plan", False):
        render_optimization_plan(config)

    return False


def render_cost_warning_dialog(config: Dict[str, Any]) -> bool:
    """Render the cost warning dialog and return if optimization should proceed."""
    # Calculate estimated costs
    num_models = len(config["selected_models"])
    num_temperatures = 3  # Default temperature points
    total_configs = min(num_models * num_temperatures, config["max_configurations"])
    examples_per_config = config["examples_per_config"]
    total_tests = min(total_configs * examples_per_config, 100)  # Safety limit
    estimated_cost = total_tests * 0.002  # ~$0.002 per test

    # Create a centered container for the modal
    _, modal_col, _ = st.columns([1, 2, 1])

    with modal_col:
        # Warning content
        st.markdown(
            f"""
        <div style="background-color: #1f2937; border: 2px solid #f59e0b; border-radius: 0.75rem;
                    padding: 1.5rem; margin: 1rem 0;">
            <h3 style="color: #f59e0b; font-size: 1.125rem; margin: 0 0 0.75rem 0; text-align: center;">
                ⚠️ Cost Warning
            </h3>
            <p style="color: #e5e7eb; font-size: 0.95rem; margin-bottom: 0.75rem; text-align: center;">
                This will use real API calls and incur costs.<br>
                <strong>Estimated cost: ~${estimated_cost:.3f}</strong>
            </p>
            <div style="background-color: #374151; border-radius: 0.5rem; padding: 0.75rem; margin-bottom: 1rem;">
                <p style="color: #9ca3af; font-size: 0.825rem; margin: 0; text-align: center;">
                    <strong>Safety limits applied:</strong><br>
                    max {total_configs} configs × max {examples_per_config} examples × 2 calls
                </p>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Buttons centered in the modal
        button_container = st.container()
        with button_container:
            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                if st.button("❌ Cancel", key="cancel_run", use_container_width=True):
                    st.session_state.show_cost_warning = False
                    st.rerun()

            with col3:
                if st.button(
                    "✅ Proceed",
                    key="proceed_run",
                    use_container_width=True,
                    type="primary",
                ):
                    st.session_state.show_cost_warning = False
                    st.session_state.run_optimization = True
                    st.session_state.run_mode = "full"
                    return True

    return False


def render_optimization_plan(config: Dict[str, Any]):
    """Render the optimization plan details."""
    st.info("**Optimization Plan Preview:**")

    num_models = len(config["selected_models"])
    total_configs = min(
        num_models * 3, config["max_configurations"]
    )  # 3 temperature points

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Models:** {', '.join(config['selected_models'])}")
        st.write(f"**Temperature range:** {config['temperature_range']}")
        st.write(f"**Strategy:** {config['strategy']}")

    with col2:
        st.write(f"**Configurations to explore:** {total_configs}")
        st.write(f"**Examples per config:** {config['examples_per_config']}")
        st.write(f"**Max configurations:** {config['max_configurations']}")

    if st.button("✕ Close Plan", key="close_plan"):
        st.session_state.show_optimization_plan = False
        st.rerun()


def render_optimization_tab():
    """Main function to render the complete optimization tab."""
    # Use Case Selection Header
    st.markdown(
        '<h3 style="font-size: 1rem; color: #10b981; font-weight: 600; margin: 0.5rem 0 0.75rem 0;">🎯 Choose Your Use Case</h3>',
        unsafe_allow_html=True,
    )

    # Get templates and render navigation
    quick_start_templates = get_quick_start_templates()
    selected_template = render_template_navigation()
    template = quick_start_templates[selected_template]

    # Render template card
    with st.columns([1, 8, 1])[1]:
        render_template_card(selected_template, template)

    # Render optimization configuration
    config = render_optimization_configuration(selected_template, template)

    # Render optimization controls
    start_optimization = render_optimization_controls(selected_template, config)

    # Handle optimization start
    if start_optimization or st.session_state.get("run_optimization", False):
        if selected_template == "Custom Use-Cases":
            if not config["selected_models"]:
                st.error("Please select at least one model to test.")
                return

        # Store optimization config in session state
        st.session_state.optimization_config = {
            "template_name": selected_template,
            "template": template,
            **config,
        }

        # Clear the run optimization flag
        if "run_optimization" in st.session_state:
            del st.session_state["run_optimization"]

        # Trigger optimization execution
        st.session_state.start_optimization_execution = True
        st.rerun()

    # Handle optimization execution
    if st.session_state.get("start_optimization_execution", False):
        render_optimization_execution()


# Import optimization functionality
from .optimization import (  # noqa: E402
    run_optimization,
    validate_optimization_config,
)


async def run_optimization_async(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Async function to run optimization with given configuration.
    """
    # Validate configuration first
    validation = validate_optimization_config(config)
    if not validation["valid"]:
        return {
            "success": False,
            "error": "Configuration validation failed: "
            + "; ".join(validation["errors"]),
        }

    # Extract configuration parameters
    template = config.get("template", {})
    problem_name = template.get("problem_name", "custom_problem")

    # Determine if this should be a mock run
    run_mode = st.session_state.get("run_mode", "dry")
    is_mock = run_mode == "dry"

    # Run the optimization
    result = await run_optimization(
        problem_name=problem_name,
        strategy=config.get("strategy", "grid"),
        models=config.get("selected_models", ["gpt-3.5-turbo"]),
        max_iterations=config.get("max_configurations", 5),
        subset_size=config.get(
            "examples_per_config", 5
        ),  # Use the examples_per_config from UI
        temperature_range=list(config.get("temperature_range", [0.0, 1.0])),
        dry_run=(run_mode == "dry"),
        mock=is_mock,
        progress_callback=config.get("progress_callback"),
    )

    return result


def render_optimization_execution():
    """Handle the optimization execution workflow."""
    st.markdown("### 🔄 Running Optimization")

    config = st.session_state.get("optimization_config", {})
    run_mode = st.session_state.get("run_mode", "dry")

    # Show execution status
    if run_mode == "dry":
        st.info("🧪 **Dry Run Mode** - Simulation without API calls (no costs)")
    else:
        st.warning("🚀 **Live Mode** - Running optimization simulation")

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Create a progress callback
    def progress_callback(progress: float, message: str):
        progress_bar.progress(progress)
        status_text.text(message)

    # Start the async optimization
    try:
        # Add progress callback to config
        config_with_callback = config.copy()
        config_with_callback["progress_callback"] = progress_callback

        # Run the optimization
        result = asyncio.run(run_optimization_async(config_with_callback))

        # Complete the progress bar
        progress_bar.progress(1.0)
        status_text.text("Optimization completed!")

        if result.get("success", False):
            # Clear execution flag
            st.session_state.start_optimization_execution = False

            # Store result (exclude dry runs from history as per previous requirement)
            if run_mode != "dry":
                if "optimization_results" not in st.session_state:
                    st.session_state.optimization_results = []
                st.session_state.optimization_results.append(result)

                # Try to save to storage
                try:
                    from optimization_storage import save_optimization_result

                    run_id = save_optimization_result(result)
                    result["run_id"] = run_id
                except Exception as e:
                    st.warning(f"Could not save to storage: {e}")

            # Show success message
            st.success("✅ Optimization completed successfully!")

            # Display results summary
            render_optimization_results(result)

        else:
            # Clear execution flag
            st.session_state.start_optimization_execution = False

            # Show error message
            error_msg = result.get(
                "error", "Unknown error occurred during optimization"
            )
            st.error(f"❌ Optimization failed: {error_msg}")

            # If this was due to API key issues, provide helpful guidance
            if "api key" in error_msg.lower() or "openai_api_key" in error_msg.lower():
                st.info(
                    "💡 **Tip:** Make sure you've set a valid API key in the Settings tab."
                )
            elif (
                "all" in error_msg.lower()
                and "trial" in error_msg.lower()
                and "fail" in error_msg.lower()
            ):
                st.info(
                    "💡 **Tip:** This might be due to API key issues or network problems. Check your API key in the Settings tab."
                )

            # Show debug information if available
            if "all_results" in result:
                with st.expander("🔍 Debug Information", expanded=False):
                    st.write("**All trial results:**")
                    for i, trial in enumerate(
                        result.get("all_results", [])[:5]
                    ):  # Show first 5 trials
                        st.write(f"Trial {i+1}: {trial.get('status', 'unknown')}")
                        if trial.get("error"):
                            st.write(f"  Error: {trial['error']}")

            # Provide action to go back
            if st.button("🔄 Try Again", key="try_again_after_error"):
                # Clear execution state and return to setup
                if "optimization_config" in st.session_state:
                    del st.session_state["optimization_config"]
                st.rerun()

    except Exception as e:
        st.error(f"❌ Optimization failed: {str(e)}")
        st.session_state.start_optimization_execution = False


def render_optimization_results(result: Dict[str, Any]):
    """Render optimization results summary."""
    st.markdown("### 📊 Optimization Results")

    # Key metrics
    performance = result.get("performance", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Best Model", performance.get("best_model", "N/A"))

    with col2:
        accuracy = performance.get("accuracy", 0)
        st.metric("Accuracy", f"{accuracy:.1%}")

    with col3:
        cost = performance.get("cost", 0)
        st.metric("Cost per Call", f"${cost:.4f}")

    with col4:
        configs_tested = result.get("configurations_tested", 0)
        st.metric("Configs Tested", str(configs_tested))

    # Best configuration details
    if "best_config" in performance:
        st.markdown("#### 🏆 Best Configuration")
        best_config = performance["best_config"]

        config_col1, config_col2 = st.columns(2)

        with config_col1:
            st.write(f"**Model:** {best_config.get('model', 'N/A')}")
            st.write(f"**Temperature:** {best_config.get('temperature', 'N/A')}")

        with config_col2:
            st.write(f"**Success Rate:** {performance.get('success_rate', 0):.1%}")
            st.write(f"**Latency:** {performance.get('latency', 0):.2f}s")

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Run Another Test", key="run_another"):
            # Clear execution state and return to setup
            st.session_state.start_optimization_execution = False
            if "optimization_config" in st.session_state:
                del st.session_state["optimization_config"]
            st.rerun()

    with col2:
        if st.button("📚 View All Results", key="view_all_results"):
            st.session_state.navigation_mode = "browse_wins"
            st.rerun()

    with col3:
        if st.button("📋 Export Results", key="export_results"):
            import json

            result_json = json.dumps(result, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=result_json,
                file_name=f"optimization_result_{result.get('timestamp', 'unknown')}.json",
                mime="application/json",
            )
