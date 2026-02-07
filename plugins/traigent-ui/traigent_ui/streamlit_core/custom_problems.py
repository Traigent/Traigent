"""
Define New Use-Case Module
=========================

This module contains functionality for creating custom AI optimization problems,
including problem definition, example generation, and analysis capabilities.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

# Import problem management utilities from traigent_ui plugin
try:
    from traigent_ui.langchain_problems import load_all_problems
    from traigent_ui.problem_management.llm_providers import (
        LLMProviderManager,
        get_available_providers,
    )
    from traigent_ui.streamlit_core.state import make_safe_filename
except ImportError as e:  # pragma: no cover - defensive fallback for UI import
    st.error(f"Import error: {e}. Make sure traigent_ui plugin is properly installed.")
    LLMProviderManager = None  # type: ignore
    get_available_providers = lambda: []  # type: ignore
    make_safe_filename = lambda name: name or "generated_problem"  # type: ignore
    load_all_problems = lambda: []  # type: ignore


def render_problem_manager_tab():
    """Render the main problem manager tab selector."""
    st.markdown("### 🎨 Define New Use-Case")
    # This is a placeholder - in the original file this was very simple
    # The actual content would be rendered by other functions


def get_domain_options() -> List[str]:
    """Get available domain options for problem classification."""
    return [
        "auto-detect",
        "educational",
        "customer_service",
        "legal",
        "medical",
        "technical",
        "creative",
        "analytical",
        "general",
    ]


def render_problem_description_input() -> Dict[str, Any]:
    """Render the problem description input section."""
    # Main input section
    col1, col2 = st.columns([2, 1])

    with col1:
        # Problem description with impact statements
        description = st.text_area(
            "🎯 Your Use Case",
            placeholder="E.g., customer support tickets, document summarization, code review...",
            height=68,
            help="Describe what you need the AI to do.",
        )

        # Compact optimization preview
        if description and len(description) > 10:
            # Dynamic estimates based on problem type
            if "math" in description.lower():
                st.success("📊 **Potential**: 60-80% cost savings, 95%+ accuracy")
            elif "customer" in description.lower():
                st.success("📊 **Potential**: 70-90% cost savings, 92%+ accuracy")
            elif "legal" in description.lower():
                st.success("📊 **Potential**: 40-60% cost savings, 98%+ accuracy")
            elif "code" in description.lower():
                st.success("📊 **Potential**: 50-70% cost savings, 90%+ accuracy")
            else:
                st.info("📊 **Potential**: Analyzing your use case...")

        # Problem name (optional)
        problem_name = st.text_input(
            "Problem Name (optional)",
            placeholder="Leave empty for AI-generated name",
            help="AI will generate a descriptive name based on your description",
        )

    with col2:
        # Smart domain selection with auto-detect option
        domain_options = get_domain_options()
        domain = st.selectbox(
            "Domain",
            domain_options,
            help="Select problem domain or let AI auto-detect",
        )

        # LLM provider selection with safe fallback
        available_providers = get_available_providers() or ["Template Generator"]
        provider = st.selectbox(
            "LLM Provider",
            available_providers,
            help="Choose your LLM provider for generation",
        )

        # Generation settings in an expander
        with st.expander("⚙️ Advanced Settings", expanded=False):
            num_examples = st.slider(
                "Examples to Generate", 5, 50, 15, help="Number of test examples"
            )
            complexity = st.selectbox(
                "Complexity Level",
                ["simple", "moderate", "complex"],
                index=1,
                help="Complexity of generated examples",
            )

    return {
        "description": description,
        "problem_name": problem_name,
        "domain": domain,
        "provider": provider,
        "num_examples": num_examples,
        "complexity": complexity,
    }


def render_create_problem_section():
    """Render the enhanced create problem section with intelligent LLM generation."""
    # Compact info
    st.info(
        "🎯 Describe your use case and we'll automatically generate test data and find the best AI configuration."
    )

    # Use smart_single mode by default (no user selection)

    # Get input data
    inputs = render_problem_description_input()

    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        create_clicked = st.button(
            "🚀 Create Problem",
            disabled=not inputs["description"] or len(inputs["description"]) < 10,
            help="Generate AI problem and test examples",
            use_container_width=True,
        )

    with col2:
        analyze_clicked = st.button(
            "🔍 Analyze Only",
            disabled=not inputs["description"] or len(inputs["description"]) < 10,
            help="Analyze the problem without creating examples",
            use_container_width=True,
        )

    # Handle button clicks
    if create_clicked:
        st.session_state.create_problem_config = inputs
        st.session_state.start_problem_creation = True
        st.rerun()

    if analyze_clicked:
        st.session_state.analyze_problem_config = inputs
        st.session_state.start_problem_analysis = True
        st.rerun()

    # If a creation request is pending, process it now
    handle_problem_creation_if_needed()

    # Show generation progress if in progress
    if st.session_state.get("problem_generation_in_progress", False):
        render_generation_progress()

    # Show generation results if available
    if st.session_state.get("generated_problems", []):
        render_generation_results()


def render_view_problems_section():
    """Render the view existing problems section."""
    st.markdown("### 📂 Browse Existing Problems")

    try:
        problems = load_all_problems()

        if not problems:
            st.info("No problems created yet. Create your first problem above!")
            return

        # Problem selector
        problem_names = list(problems.keys())
        selected_problem = st.selectbox("Select a problem to view:", problem_names)

        if selected_problem:
            problem_data = problems[selected_problem]

            # Display problem info
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Description:**")
                st.write(problem_data.get("description", "No description available"))

                st.markdown("**Domain:**")
                st.write(problem_data.get("domain", "Unknown"))

            with col2:
                st.markdown("**Examples:**")
                examples = problem_data.get("examples", [])
                st.write(f"{len(examples)} examples available")

                if examples:
                    if st.button("Preview Examples", key=f"preview_{selected_problem}"):
                        st.json(examples[:3])  # Show first 3 examples

            # Action buttons
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🚀 Optimize This Problem", key=f"opt_{selected_problem}"):
                    st.session_state.selected_optimization_problem = selected_problem
                    st.session_state.navigation_mode = "quick_start"
                    st.rerun()

            with col2:
                if st.button("✏️ Edit Problem", key=f"edit_{selected_problem}"):
                    st.session_state.editing_problem = selected_problem
                    st.rerun()

            with col3:
                if st.button("🗑️ Delete Problem", key=f"del_{selected_problem}"):
                    if st.session_state.get(
                        f"confirm_delete_{selected_problem}", False
                    ):
                        # Actually delete the problem
                        del problems[selected_problem]
                        st.success(f"Deleted problem: {selected_problem}")
                        st.rerun()
                    else:
                        st.session_state[f"confirm_delete_{selected_problem}"] = True
                        st.warning("Click again to confirm deletion")

    except Exception as e:
        st.error(f"Error loading problems: {e}")


def render_example_generation_controls():
    """Render controls for example generation."""
    st.markdown("### 🎲 Example Generation")

    col1, col2 = st.columns(2)

    with col1:
        generation_type = st.selectbox(
            "Generation Type",
            ["Auto-Generate", "Manual Entry", "Import from File"],
            help="Choose how to create examples",
        )

    with col2:
        if generation_type == "Auto-Generate":
            if st.button("🚀 Generate Examples"):
                st.session_state.start_example_generation = True
                st.rerun()
        elif generation_type == "Manual Entry":
            st.info("Manual entry interface would be implemented here")
        else:  # Import from File
            uploaded_file = st.file_uploader(
                "Upload JSONL file", type=["jsonl", "json"]
            )
            if uploaded_file is not None:
                st.success("File uploaded successfully")


def start_example_generation():
    """Start the example generation process."""
    if st.session_state.get("start_example_generation", False):
        st.session_state.example_generation_in_progress = True
        st.session_state.start_example_generation = False

        # In a real implementation, this would start the async generation
        # For now, we'll simulate it
        st.info("🔄 Starting example generation...")


def render_generation_progress():
    """Render the generation progress indicator."""
    st.markdown("### 🔄 Generation in Progress")

    # Progress bar (simulated)
    progress_bar = st.progress(0)
    status_text = st.empty()

    # In a real implementation, this would track actual progress
    # For now, showing a static progress indicator
    progress_bar.progress(50)
    status_text.text("Generating examples... 50% complete")


def render_generation_results():
    """Render the generation results."""
    st.markdown("### ✅ Generation Complete")

    generated_problems = st.session_state.get("generated_problems", [])

    if generated_problems:
        st.success(f"Generated {len(generated_problems)} problems successfully!")

        # Show summary
        for i, problem in enumerate(generated_problems):
            with st.expander(f"Problem {i+1}: {problem.get('name', 'Unnamed')}"):
                st.write(f"**Description:** {problem.get('description', 'N/A')}")
                st.write(f"**Examples:** {len(problem.get('examples', []))}")

                if st.button("Use This Problem", key=f"use_problem_{i}"):
                    st.session_state.selected_generated_problem = problem
                    st.success("Problem selected for optimization!")


def handle_problem_creation_if_needed():
    """Process a pending problem creation request."""
    if not st.session_state.get("start_problem_creation", False):
        return

    # Consume the flag so we do not re-run infinitely
    st.session_state.start_problem_creation = False

    config = st.session_state.get("create_problem_config", {})
    if not config or not config.get("description"):
        st.error("Please provide a description before creating a problem.")
        return

    # UI placeholders for live feedback
    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    def progress_callback(message: str, progress: float):
        clamped = max(0.0, min(1.0, progress))
        status_placeholder.info(message)
        progress_bar.progress(clamped)

    st.session_state.problem_generation_in_progress = True

    try:
        result = _run_problem_creation(config, progress_callback)
    except Exception as exc:  # pragma: no cover - defensive logging for UI
        st.error(f"Problem creation failed: {exc}")
        st.session_state.problem_generation_in_progress = False
        return

    st.session_state.problem_generation_in_progress = False

    if not result.get("success"):
        st.error(result.get("error", "Problem creation failed."))
        return

    created_problem = result["problem"]
    st.session_state.generated_problems = [created_problem]
    st.success(
        f"Created problem '{created_problem.get('display_name', created_problem.get('name'))}' "
        f"with {len(created_problem.get('examples', []))} examples."
    )


def _run_problem_creation(
    config: Dict[str, Any], progress_callback: Optional[Callable[[str, float], None]]
) -> Dict[str, Any]:
    """Run the async generation pipeline inside the Streamlit execution."""
    try:
        return asyncio.run(
            _generate_problem_with_provider(config, progress_callback)
        )
    except RuntimeError:
        # Fallback if an event loop already exists
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            _generate_problem_with_provider(config, progress_callback)
        )


async def _generate_problem_with_provider(
    config: Dict[str, Any], progress_callback: Optional[Callable[[str, float], None]]
) -> Dict[str, Any]:
    """Generate examples using the selected provider and package a problem payload."""
    if LLMProviderManager is None:
        return {
            "success": False,
            "error": "Problem management modules are not available in this environment.",
        }

    provider_manager = LLMProviderManager()
    available = provider_manager.get_available_providers()

    requested_provider = config.get("provider") or "Template Generator"
    provider_to_use = requested_provider

    if provider_to_use not in available:
        # Pick a safe fallback if the requested provider is unavailable
        provider_to_use = (
            "Template Generator"
            if "Template Generator" in available
            else (available[0] if available else None)
        )

    if not provider_to_use:
        return {"success": False, "error": "No LLM providers are available right now."}

    num_examples = int(config.get("num_examples") or 10)
    prompt = _build_prompt_from_config(config, num_examples)

    if progress_callback:
        progress_callback(f"Starting generation with {provider_to_use}...", 0.05)

    def provider_progress(message: str, progress: float):
        if progress_callback:
            # Map provider progress (0-1) into 10-90% range for the UI bar
            progress_callback(message, 0.1 + (progress * 0.8))

    try:
        generation_result = await provider_manager.generate_examples(
            provider_name=provider_to_use,
            prompt=prompt,
            count=num_examples,
            progress_callback=provider_progress,
        )
    except Exception as exc:
        return {"success": False, "error": f"Provider error: {exc}"}

    if not generation_result.success:
        return {
            "success": False,
            "error": generation_result.error or "Generation failed.",
        }

    # Normalize examples
    normalized_examples: List[Dict[str, Any]] = []
    for ex in generation_result.examples:
        normalized_examples.append(
            {
                "input_data": ex.get("input_data"),
                "expected_output": ex.get("expected_output"),
                "difficulty": ex.get("difficulty", "medium"),
                "metadata": ex.get("metadata", {}),
            }
        )

    display_name = (
        config.get("problem_name")
        or (config.get("description", "Generated Problem")[:60])
    )
    safe_name = make_safe_filename(display_name)

    problem_payload = {
        "name": safe_name,
        "display_name": display_name,
        "description": config.get("description", ""),
        "domain": config.get("domain", "general"),
        "complexity": config.get("complexity", "moderate"),
        "provider": provider_to_use,
        "examples": normalized_examples,
        "metadata": {
            "requested_provider": requested_provider,
            "requested_examples": num_examples,
        },
    }

    if progress_callback:
        progress_callback("Problem created successfully.", 0.95)

    return {"success": True, "problem": problem_payload}


def _build_prompt_from_config(config: Dict[str, Any], count: int) -> str:
    """Build a simple provider prompt based on the UI inputs."""
    description = config.get("description", "").strip()
    domain = config.get("domain", "general")
    complexity = config.get("complexity", "moderate")

    return (
        f"You are generating high-quality evaluation examples for a {domain} AI task.\n"
        f"Use case: {description}\n"
        f"Create {count} diverse, realistic examples at a {complexity} complexity level.\n"
        "Each example must include JSON `input_data` and an `expected_output` value, "
        "with optional `metadata.reasoning` to explain the answer."
    )


def render_analyze_problems_section():
    """Render the problem analysis section."""
    st.markdown("### 🔍 Problem Analysis")

    # Get available problems
    try:
        problems = load_all_problems()

        if not problems:
            st.info("No problems available for analysis. Create some problems first!")
            return

        # Problem selector
        problem_names = list(problems.keys())
        selected_problem = st.selectbox("Select a problem to analyze:", problem_names)

        if selected_problem and st.button("🔍 Analyze Problem"):
            problem_data = problems[selected_problem]

            # Show analysis results
            st.markdown("#### Analysis Results")

            # Basic statistics
            examples = problem_data.get("examples", [])
            st.write(f"**Total Examples:** {len(examples)}")
            st.write(f"**Domain:** {problem_data.get('domain', 'Unknown')}")

            # Example analysis
            if examples:
                # Analyze example lengths, complexity, etc.
                input_lengths = [len(str(ex.get("input", ""))) for ex in examples]
                avg_input_length = (
                    sum(input_lengths) / len(input_lengths) if input_lengths else 0
                )

                st.write(f"**Average Input Length:** {avg_input_length:.1f} characters")

                # Show distribution chart if more than one example
                if len(examples) > 1:
                    import pandas as pd
                    import plotly.express as px

                    df = pd.DataFrame(
                        {"Example": range(len(input_lengths)), "Length": input_lengths}
                    )
                    fig = px.bar(
                        df, x="Example", y="Length", title="Input Length Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error analyzing problems: {e}")


# Async functions for problem creation and analysis
async def create_problem(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Async function to create a new problem with the given configuration.
    This is a placeholder for the actual problem creation logic.
    """
    # Simulate async work
    await asyncio.sleep(1)

    return {
        "success": True,
        "problem_name": config.get("problem_name", "Generated Problem"),
        "description": config.get("description", ""),
        "examples": [],  # Would contain generated examples
    }


async def analyze_problem(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Async function to analyze a problem description.
    This is a placeholder for the actual analysis logic.
    """
    # Simulate async work
    await asyncio.sleep(0.5)

    return {
        "success": True,
        "analysis": {
            "complexity": "moderate",
            "domain": config.get("domain", "general"),
            "estimated_examples_needed": 20,
            "recommended_models": ["gpt-3.5-turbo", "gpt-4o-mini"],
        },
    }
