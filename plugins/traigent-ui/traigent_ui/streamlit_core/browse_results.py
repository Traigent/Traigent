"""
Browse Previous Optimizations Module
===================================

This module contains functionality for browsing and analyzing previous optimization results,
including results dashboard, detailed views, and historical data management.
"""

from typing import Any, Dict

import pandas as pd
import streamlit as st

# Import storage utilities
try:
    import optimization_storage  # noqa: F401 - Import check only
except ImportError as e:
    st.error(f"Import error: {e}")


def render_browse_wins_section():
    """Render the browse wins section with example success stories."""
    st.markdown("### 📚 Browse Previous Runs")

    # Categories
    category = st.selectbox(
        "Filter by industry",
        ["All", "E-commerce", "SaaS", "Healthcare", "Legal", "Education"],
    )

    wins = [
        {
            "company": "TechStartup Inc",
            "industry": "SaaS",
            "problem": "Customer churn prediction",
            "before": "GPT-4, $12K/mo",
            "after": "Fine-tuned GPT-3.5, $2.4K/mo",
            "savings": "$9,600/mo",
            "accuracy": "96% → 97%",
        },
        {
            "company": "MedTech Solutions",
            "industry": "Healthcare",
            "problem": "Medical record summarization",
            "before": "GPT-4, $18K/mo",
            "after": "Claude-3-haiku, $3.6K/mo",
            "savings": "$14,400/mo",
            "accuracy": "94% → 94%",
        },
    ]

    for win in wins:
        if category == "All" or win["industry"] == category:
            with st.expander(f"**{win['company']}** - {win['problem']}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Before:**")
                    st.write(win["before"])

                with col2:
                    st.markdown("**After:**")
                    st.write(win["after"])

                with col3:
                    st.metric("Monthly Savings", win["savings"])
                    st.metric("Accuracy", win["accuracy"])

                if st.button("📋 Use This Template", key=f"win_{win['company']}"):
                    st.session_state.selected_template = win
                    st.success(f"Template selected: {win['problem']}")


def render_optimization_history_metrics(df: pd.DataFrame):
    """Render the KPI metrics for optimization history."""
    # Calculate metrics
    total_runs = len(df)
    avg_accuracy = (
        df["performance"].apply(lambda x: x.get("accuracy", 0)).mean()
        if "performance" in df.columns
        else 0
    )
    avg_cost = (
        df["performance"].apply(lambda x: x.get("cost", 0)).mean()
        if "performance" in df.columns
        else 0
    )
    unique_problems = df["problem"].nunique() if "problem" in df else 0

    # Compact KPI metrics in one row with icons
    st.markdown(
        f"""
    <div style="background-color: #1f2937; border-radius: 0.5rem; padding: 0.75rem; margin: 0.5rem 0;">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; text-align: center;">
            <div>
                <div style="color: #10b981; font-size: 1.25rem; font-weight: 600;">🔄 {total_runs}</div>
                <div style="color: #9ca3af; font-size: 0.75rem;">Total Runs</div>
            </div>
            <div>
                <div style="color: #10b981; font-size: 1.25rem; font-weight: 600;">🎯 {avg_accuracy:.1%}</div>
                <div style="color: #9ca3af; font-size: 0.75rem;">Avg Accuracy</div>
            </div>
            <div>
                <div style="color: #10b981; font-size: 1.25rem; font-weight: 600;">💲 ${avg_cost:.4f}</div>
                <div style="color: #9ca3af; font-size: 0.75rem;">Avg Cost</div>
            </div>
            <div>
                <div style="color: #10b981; font-size: 1.25rem; font-weight: 600;">📋 {unique_problems}</div>
                <div style="color: #9ca3af; font-size: 0.75rem;">Problems Tested</div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_optimization_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render filter controls and return filtered dataframe."""
    # Quick filters
    filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 2])

    with filter_col1:
        problem_filter = st.selectbox(
            "Filter by Problem",
            ["All"] + df["problem"].unique().tolist(),
            key="problem_filter",
        )

    with filter_col2:
        strategy_filter = st.selectbox(
            "Filter by Strategy",
            (
                ["All"] + df["strategy"].unique().tolist()
                if "strategy" in df.columns
                else ["All"]
            ),
            key="strategy_filter",
        )

    with filter_col3:
        success_filter = st.selectbox(
            "Filter by Status", ["All", "Success", "Failed"], key="success_filter"
        )

    # Apply filters
    filtered_df = df.copy()

    if problem_filter != "All":
        filtered_df = filtered_df[filtered_df["problem"] == problem_filter]

    if strategy_filter != "All" and "strategy" in df.columns:
        filtered_df = filtered_df[filtered_df["strategy"] == strategy_filter]

    if success_filter != "All":
        if success_filter == "Success":
            filtered_df = filtered_df[filtered_df["success"]]
        else:
            filtered_df = filtered_df[not filtered_df["success"]]

    return filtered_df


def render_optimization_table(df: pd.DataFrame):
    """Render the paginated optimization results table."""
    if len(df) == 0:
        st.info("No results match the current filters.")
        return

    # Pagination setup
    if "results_page" not in st.session_state:
        st.session_state.results_page = 0

    rows_per_page = 5
    total_pages = (len(df) - 1) // rows_per_page + 1
    start_idx = st.session_state.results_page * rows_per_page
    end_idx = min(start_idx + rows_per_page, len(df))

    # Page subset
    page_df = df.iloc[start_idx:end_idx]

    # Table header with custom CSS Grid
    st.markdown(
        """
    <div style="background-color: #374151; padding: 0.5rem; border-radius: 0.5rem; margin-bottom: 0.5rem;">
        <div style="display: grid; grid-template-columns: 1.5fr 2fr 1.5fr 1fr 1fr 1fr 1fr; gap: 0.5rem; font-weight: 600; color: #10b981; font-size: 0.875rem;">
            <div>Problem</div>
            <div>Strategy</div>
            <div>Best Model</div>
            <div>Accuracy</div>
            <div>Cost</div>
            <div>Duration</div>
            <div>Configs</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Table rows
    for idx, row in page_df.iterrows():
        # Extract data safely
        problem = row.get("problem", "Unknown")
        strategy = row.get("strategy", "Unknown")

        # Best model from results
        best_model = "Unknown"
        accuracy = "N/A"
        cost = "N/A"

        if "performance" in row and isinstance(row["performance"], dict):
            perf = row["performance"]
            best_model = perf.get("best_model", "Unknown")
            accuracy = (
                f"{perf.get('accuracy', 0):.1%}" if perf.get("accuracy") else "N/A"
            )
            cost = f"${perf.get('cost', 0):.4f}" if perf.get("cost") else "N/A"

        duration = (
            f"{row.get('duration_minutes', 0):.1f}m"
            if row.get("duration_minutes")
            else "N/A"
        )
        configs = row.get("configurations_tested", "N/A")

        # Row background based on success
        bg_color = "#1f2937" if row.get("success", False) else "#312e2e"

        # Table row
        row_html = f"""
        <div style="background-color: {bg_color}; padding: 0.5rem; border-radius: 0.5rem; margin-bottom: 0.25rem; cursor: pointer;"
             onclick="document.getElementById('row_{idx}').click()">
            <div style="display: grid; grid-template-columns: 1.5fr 2fr 1.5fr 1fr 1fr 1fr 1fr; gap: 0.5rem; font-size: 0.8rem; align-items: center;">
                <div style="color: #e5e7eb; font-weight: 500;">{problem}</div>
                <div style="color: #9ca3af;">{strategy}</div>
                <div style="color: #9ca3af;">{best_model}</div>
                <div style="color: #10b981;">{accuracy}</div>
                <div style="color: #f59e0b;">{cost}</div>
                <div style="color: #9ca3af;">{duration}</div>
                <div style="color: #9ca3af;">{configs}</div>
            </div>
        </div>
        """

        st.markdown(row_html, unsafe_allow_html=True)

        # Hidden button for row click handling
        if st.button(
            "View Details", key=f"row_{idx}", help=f"View details for {problem}"
        ):
            st.session_state.selected_result_id = row.get("run_id", idx)
            st.session_state.show_result_details = True
            st.rerun()

    # Pagination controls
    if total_pages > 1:
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

        with col1:
            if st.button(
                "⏮️", disabled=st.session_state.results_page == 0, key="first_page"
            ):
                st.session_state.results_page = 0
                st.rerun()

        with col2:
            if st.button(
                "◀️", disabled=st.session_state.results_page == 0, key="prev_page"
            ):
                st.session_state.results_page -= 1
                st.rerun()

        with col3:
            st.markdown(
                f"<div style='text-align: center; padding: 0.25rem;'>Page {st.session_state.results_page + 1} of {total_pages}</div>",
                unsafe_allow_html=True,
            )

        with col4:
            if st.button(
                "▶️",
                disabled=st.session_state.results_page >= total_pages - 1,
                key="next_page",
            ):
                st.session_state.results_page += 1
                st.rerun()

        with col5:
            if st.button(
                "⏭️",
                disabled=st.session_state.results_page >= total_pages - 1,
                key="last_page",
            ):
                st.session_state.results_page = total_pages - 1
                st.rerun()


def render_browse_results_tab():
    """Main function to render the Browse Results tab."""
    st.markdown("### 📊 Browse Previous Optimizations")

    # Load results from session state
    if "optimization_results" not in st.session_state:
        st.session_state.optimization_results = []

    results = st.session_state.optimization_results

    if not results:
        st.info(
            "No optimization results found. Run some optimizations to see results here!"
        )
        return

    # Show basic stats
    df = pd.DataFrame(results)
    total_runs = len(df)
    avg_accuracy = (
        df["performance"].apply(lambda x: x.get("accuracy", 0)).mean()
        if "performance" in df.columns
        else 0
    )
    avg_cost = (
        df["performance"].apply(lambda x: x.get("cost", 0)).mean()
        if "performance" in df.columns
        else 0
    )
    unique_problems = df["problem"].nunique() if "problem" in df else 0

    # KPI metrics with improved styling
    st.markdown(
        f"""
    <div style="background-color: #1f2937; border-radius: 0.5rem; padding: 0.75rem; margin: 0.5rem 0;">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; text-align: center;">
            <div>
                <div style="color: #10b981; font-size: 1.25rem; font-weight: 600;">🔄 {total_runs}</div>
                <div style="color: #9ca3af; font-size: 0.75rem;">Total Runs</div>
            </div>
            <div>
                <div style="color: #10b981; font-size: 1.25rem; font-weight: 600;">🎯 {avg_accuracy:.1%}</div>
                <div style="color: #9ca3af; font-size: 0.75rem;">Avg Accuracy</div>
            </div>
            <div>
                <div style="color: #10b981; font-size: 1.25rem; font-weight: 600;">💲 ${avg_cost:.4f}</div>
                <div style="color: #9ca3af; font-size: 0.75rem;">Avg Cost</div>
            </div>
            <div>
                <div style="color: #10b981; font-size: 1.25rem; font-weight: 600;">📋 {unique_problems}</div>
                <div style="color: #9ca3af; font-size: 0.75rem;">Problems Tested</div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Results table header
    st.markdown("#### Optimization History")

    # Table header
    col1, col2, col3, col4, col5, col6 = st.columns([2.5, 1.5, 1.5, 1, 1, 0.5])
    with col1:
        st.markdown("**Problem**")
    with col2:
        st.markdown("**Strategy**")
    with col3:
        st.markdown("**Best Model**")
    with col4:
        st.markdown("**Accuracy**")
    with col5:
        st.markdown("**Cost**")
    with col6:
        st.markdown("**Action**")

    st.markdown("---")

    # Results rows
    for idx, result in enumerate(results):
        col1, col2, col3, col4, col5, col6 = st.columns([2.5, 1.5, 1.5, 1, 1, 0.5])

        with col1:
            problem_name = result.get("problem", "Unknown")
            st.markdown(
                f"**{problem_name[:25]}{'...' if len(problem_name) > 25 else ''}**"
            )
            timestamp = result.get("timestamp", "")
            if timestamp:
                try:
                    from datetime import datetime

                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    st.caption(dt.strftime("%Y-%m-%d %H:%M"))
                except Exception:
                    st.caption(timestamp[:16])

        with col2:
            strategy = result.get("strategy", "Unknown")
            st.markdown(f"{strategy[:12]}{'...' if len(strategy) > 12 else ''}")

        with col3:
            perf = result.get("performance", {})
            best_model = perf.get("best_model", "N/A")
            st.markdown(f"{best_model[:12]}{'...' if len(best_model) > 12 else ''}")

        with col4:
            accuracy = perf.get("accuracy", 0)
            st.markdown(f"{accuracy:.1%}")

        with col5:
            cost = perf.get("cost", 0)
            st.markdown(f"${cost:.4f}")

        with col6:
            if st.button("📊", key=f"view_{idx}", help="View Details"):
                st.session_state.selected_result = result
                st.session_state.show_result_details = True
                st.rerun()

        # Subtle separator
        if idx < len(results) - 1:
            st.markdown(
                "<hr style='margin: 0.25rem 0; border: 0.5px solid #4b5563; opacity: 0.3;'>",
                unsafe_allow_html=True,
            )

    # Show detailed view if selected
    if st.session_state.get("show_result_details", False):
        render_detailed_result_view(st.session_state.get("selected_result"))


def render_detailed_result_view(result: Dict[str, Any]):
    """Render detailed view of a single optimization result."""
    if not result:
        return

    st.markdown("---")

    # Hero section
    st.markdown(
        """
    <div style="background-color: #065f46; border: 2px solid #10b981; border-radius: 0.75rem;
                padding: 2rem; margin: 1rem 0;">
        <h2 style="color: #ffffff; font-size: 1.875rem; font-weight: 700; margin: 0 0 1.5rem 0;">
            🏆 Optimization Details
        </h2>
    """,
        unsafe_allow_html=True,
    )

    # Close button
    if st.button("❌ Close Details", key="close_details"):
        st.session_state.show_result_details = False
        st.rerun()

    # Basic info
    performance = result.get("performance", {})
    best_config = performance.get("best_config", {})

    # Recommended Configuration Box
    st.markdown(
        f"""
        <div style="background-color: #1f2937; border: 2px solid #10b981; border-radius: 0.5rem;
                    padding: 1.5rem; margin: 0 0 1rem 0;">
            <h3 style="color: #10b981; font-size: 1.25rem; font-weight: 600; margin: 0 0 1rem 0;">
                Recommended Configuration
            </h3>
            <p style="color: #ffffff; font-size: 1.125rem; font-weight: 600; margin: 0.5rem 0;">
                Model: {performance.get("best_model", "Unknown")}
            </p>
            <div style="margin: 1rem 0;">
                <p style="color: #10b981; font-size: 1rem; margin: 0.25rem 0;">
                    🎯 {performance.get("accuracy", 0):.1%} accuracy achieved
                </p>
                <p style="color: #10b981; font-size: 1rem; margin: 0.25rem 0;">
                    💰 ${performance.get("cost", 0):.4f} cost per call
                </p>
                <p style="color: #10b981; font-size: 1rem; margin: 0.25rem 0;">
                    ⚡ {performance.get("latency", 1.0):.1f}s response time
                </p>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Key insights section
    st.markdown(
        """
    <div style="margin: 2rem 0;">
        <h3 style="color: #10b981; font-size: 1.25rem; font-weight: 600; margin: 0 0 0.5rem 0;">
            📊 Key Insights from This Optimization
        </h3>
        <div style="border-top: 2px solid #374151; margin: 0.5rem 0 1rem 0;"></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Performance insights
    st.markdown(
        f"""
    <div style="background-color: #1f2937; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem;">
        <h4 style="color: #10b981; font-size: 1rem; font-weight: 600; margin: 0 0 0.5rem 0;">
            ✅ Performance Insights
        </h4>
        <ul style="color: #e5e7eb; font-size: 0.875rem; margin: 0.5rem 0 0 1.25rem;">
            <li>Best accuracy achieved: {performance.get("accuracy", 0):.1%}</li>
            <li>Most cost-effective model: {performance.get("best_model", "N/A")}</li>
            <li>Optimal temperature: {best_config.get("temperature", "N/A")}</li>
            <li>Configurations tested: {result.get("configurations_tested", 0)}</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Execution details for advanced users
    with st.expander("🔧 Advanced Details", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Optimization Settings**")
            st.write(f"Problem: {result.get('problem', 'Unknown')}")
            st.write(f"Strategy: {result.get('strategy', 'Unknown')}")
            st.write(f"Duration: {result.get('duration_minutes', 0):.1f} minutes")
            st.write(f"Timestamp: {result.get('timestamp', 'Unknown')}")

        with col2:
            st.markdown("**Best Configuration**")
            if best_config:
                for key, value in best_config.items():
                    st.write(f"{key}: {value}")
            else:
                st.write("No configuration details available")

    # All trials for data scientists
    all_results = result.get("all_results", [])
    if all_results:
        with st.expander("📋 All Trial Details (Data Scientists)", expanded=False):
            st.markdown("**Individual Trial Results**")

            # Create a clean dataframe for display
            trial_data = []
            for i, trial in enumerate(all_results):
                trial_data.append(
                    {
                        "Trial": i + 1,
                        "Model": trial.get("config", {}).get("model", "N/A"),
                        "Temperature": trial.get("config", {}).get(
                            "temperature", "N/A"
                        ),
                        "Status": trial.get("status", "Unknown"),
                        "Accuracy": (
                            trial.get("metrics", {}).get("accuracy", 0)
                            if trial.get("metrics")
                            else 0
                        ),
                        "Duration": trial.get("duration", 0),
                    }
                )

            trials_df = pd.DataFrame(trial_data)
            st.dataframe(trials_df, use_container_width=True)

            # Show trial statistics
            st.markdown("**Trial Statistics**")
            col1, col2, col3 = st.columns(3)

            with col1:
                completed_trials = len(
                    [t for t in all_results if t.get("status") == "completed"]
                )
                st.metric("Completed Trials", f"{completed_trials}/{len(all_results)}")

            with col2:
                if trials_df["Accuracy"].sum() > 0:
                    avg_accuracy = trials_df["Accuracy"].mean()
                    st.metric("Average Accuracy", f"{avg_accuracy:.1%}")
                else:
                    st.metric("Average Accuracy", "N/A")

            with col3:
                avg_duration = trials_df["Duration"].mean()
                st.metric("Avg Trial Duration", f"{avg_duration:.1f}s")

    # Export options
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📁 Export Full Results", key="export_detailed"):
            import json

            result_json = json.dumps(result, indent=2, default=str)
            from datetime import datetime

            st.download_button(
                label="Download JSON",
                data=result_json,
                file_name=f"optimization_result_{result.get('problem', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

    with col2:
        if st.button("📊 Export Summary Report", key="export_summary"):
            # Create a summary report
            summary = f"""
# Optimization Report

**Problem:** {result.get('problem', 'Unknown')}
**Strategy:** {result.get('strategy', 'Unknown')}
**Date:** {result.get('timestamp', 'Unknown')}

## Results
- **Best Model:** {performance.get('best_model', 'N/A')}
- **Accuracy:** {performance.get('accuracy', 0):.1%}
- **Cost per Call:** ${performance.get('cost', 0):.4f}
- **Response Time:** {performance.get('latency', 1.0):.1f}s

## Configuration Tested
{result.get('configurations_tested', 0)} different configurations were evaluated.

## Recommendations
Use {performance.get('best_model', 'the recommended model')} with the optimal settings identified in this optimization.
"""
            from datetime import datetime

            st.download_button(
                label="Download Report",
                data=summary,
                file_name=f"optimization_summary_{result.get('problem', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )


def render_optimization_details(result: Dict[str, Any]):
    """Render detailed view of a specific optimization result."""
    st.markdown("### 🔍 Optimization Details")

    # Back button
    if st.button("← Back to Results", key="back_to_results"):
        st.session_state.show_result_details = False
        st.rerun()

    # Result overview
    st.markdown(f"**Problem:** {result.get('problem', 'Unknown')}")
    st.markdown(f"**Strategy:** {result.get('strategy', 'Unknown')}")
    st.markdown(f"**Status:** {'✅ Success' if result.get('success') else '❌ Failed'}")

    if "performance" in result:
        perf = result["performance"]
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Accuracy", f"{perf.get('accuracy', 0):.1%}")
        with col2:
            st.metric("Cost", f"${perf.get('cost', 0):.4f}")
        with col3:
            st.metric("Best Model", perf.get("best_model", "Unknown"))

    # Show detailed results if available
    if "detailed_results" in result:
        st.markdown("#### Detailed Results")
        st.json(result["detailed_results"])


def show_optimization_logs(result: Dict[str, Any]):
    """Show optimization logs and trial details."""
    st.markdown("### 📋 Optimization Logs")

    # Implementation would show trial-by-trial progress
    # For now, showing basic information
    if "trials" in result:
        st.markdown(f"**Total Trials:** {len(result['trials'])}")

        # Show trial summary
        trials_df = pd.DataFrame(result["trials"])
        if not trials_df.empty:
            st.dataframe(trials_df)
    else:
        st.info("No detailed trial logs available for this optimization.")
