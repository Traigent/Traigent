#!/usr/bin/env python3
"""
Traigent Python Orchestrator - Arkia Sales Agent Margin Optimization

This script demonstrates using Traigent to optimize a travel sales agent
using Mastra.ai patterns. The key insight is that MEMORY CONFIGURATION
is a hidden cost driver that significantly impacts margins.

NOTE: This demo uses a MOCK implementation. No API keys are required.
      The agent simulates LLM behavior for cost-free optimization testing.

Optimization Goals:
1. Maximize margin_efficiency (conversion per dollar spent)
2. Maintain quality thresholds (relevancy, completeness, tone)
3. Find the sweet spot between model capability and cost

Key Discovery: You might find that:
- gpt-4o-mini with 5 memory turns outperforms gpt-4o with 10 memory turns
- The 'consultative' prompt style has better ROI than 'sales_aggressive'
- Reducing memory from 10 to 5 turns saves 50% on input tokens!

Prerequisites:
1. Build the JS demo: cd demos/arkia-sales-agent && npm run build
2. Install Traigent Python SDK: pip install traigent

Usage:
    python run_with_python.py

Environment Variables:
    TRAIGENT_EXECUTION_MODE: "edge_analytics" (default) or "mock"
    TRAIGENT_COST_APPROVED: Set to "true" to skip cost approval prompt
"""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

# Get the directory where this script lives (for relative paths)
SCRIPT_DIR = Path(__file__).parent.resolve()


def setup_node_path():
    """Ensure Node.js is in PATH for subprocess calls.

    When using nvm, the Node.js binaries aren't in the default PATH.
    This function finds and adds the appropriate Node version to PATH.
    """
    # Check if node is already available
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            return  # Node already in PATH
    except FileNotFoundError:
        pass  # Need to find node

    # Try common nvm locations
    home = Path.home()
    nvm_versions = home / ".nvm" / "versions" / "node"

    if nvm_versions.exists():
        # Find the highest version >= 22 (for Mastra compatibility)
        versions = []
        for v in nvm_versions.iterdir():
            if v.is_dir() and v.name.startswith("v"):
                try:
                    major = int(v.name[1:].split(".")[0])
                    versions.append((major, v))
                except ValueError:
                    continue

        # Sort by major version descending, prefer >= 22
        versions.sort(key=lambda x: x[0], reverse=True)

        for major, version_path in versions:
            bin_path = version_path / "bin"
            if bin_path.exists() and (bin_path / "node").exists():
                os.environ["PATH"] = f"{bin_path}:{os.environ.get('PATH', '')}"
                print(f"[✓] Added Node.js v{version_path.name[1:]} to PATH")
                return

    print("WARNING: Could not find Node.js. Ensure 'node' and 'npx' are in PATH.")


def check_node_version():
    """Check Node.js version. Mastra requires Node >= 22 for REAL_MODE."""
    real_mode = os.getenv("REAL_MODE", "").lower() == "true"

    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        version_str = result.stdout.strip().lstrip("v")
        major = int(version_str.split(".")[0])

        if real_mode and major < 22:
            print("=" * 70)
            print("WARNING: REAL_MODE requires Node.js >= 22 (Mastra dependency)")
            print("=" * 70)
            print(f"Current Node version: v{version_str}")
            print("\nTo fix, run:")
            print("  nvm use 22")
            print("  # Then re-run this script")
            print("=" * 70)
            print("\nContinuing anyway - Mastra calls may fail...\n")
        elif real_mode:
            print(f"[✓] Node.js v{version_str} detected (>= 22 required for REAL_MODE)")
        else:
            print(f"[✓] Node.js v{version_str} detected")
    except Exception as e:
        if real_mode:
            print(f"WARNING: Could not check Node version: {e}")
            print("REAL_MODE requires Node.js >= 22 for Mastra integration.")

# Try to import traigent - first from pip, then from local SDK
try:
    import traigent
except ImportError:
    # Fall back to local SDK if installed in development mode
    sdk_path = SCRIPT_DIR.parent.parent.parent / "Traigent"
    if sdk_path.exists():
        sys.path.insert(0, str(sdk_path))
        import traigent
    else:
        print("ERROR: Could not import traigent.")
        print("Install with: pip install traigent")
        print(f"Or ensure SDK exists at: {sdk_path}")
        sys.exit(1)


def verify_build():
    """Verify that the JS module has been built."""
    js_module = SCRIPT_DIR / "dist" / "trial.js"
    if not js_module.exists():
        print("=" * 70)
        print("ERROR: JS module not built!")
        print("=" * 70)
        print(f"\nExpected file: {js_module}")
        print("\nTo fix, run:")
        print(f"  cd {SCRIPT_DIR}")
        print("  npm install")
        print("  npm run build")
        print("=" * 70)
        sys.exit(1)
    return str(js_module)


def verify_dataset():
    """Verify that the dataset file exists."""
    dataset_path = SCRIPT_DIR / "dataset.jsonl"
    if not dataset_path.exists():
        print("=" * 70)
        print("ERROR: Dataset not found!")
        print("=" * 70)
        print(f"\nExpected file: {dataset_path}")
        print("=" * 70)
        sys.exit(1)
    return str(dataset_path)


async def main():
    """Run optimization with parallel JS trial execution."""
    start_time = time.time()

    # Verify prerequisites
    setup_node_path()  # Ensure Node.js is in PATH for subprocesses
    check_node_version()
    js_module_path = verify_build()
    dataset_path = verify_dataset()

    # Initialize Traigent SDK
    traigent.initialize(
        execution_mode=os.getenv("TRAIGENT_EXECUTION_MODE", "edge_analytics"),
    )

    # Configuration space size for reference
    config_space_size = 5 * 4 * 3 * 4 * 4  # models * temps * prompts * memory * tools
    max_trials = 24
    coverage_pct = (max_trials / config_space_size) * 100

    # Define the optimization with margin-focused objectives
    @traigent.optimize(
        # Execution configuration - use Node.js runtime with parallel workers
        execution={
            "runtime": "node",
            "js_module": js_module_path,
            "js_function": "runTrial",
            "js_timeout": 60.0,  # 60 second timeout per trial
            "js_parallel_workers": 4,  # Run 4 Node.js processes in parallel
        },
        # Config injection mode - must be 'parameter' for Node.js runtime
        injection_mode="parameter",
        # Dataset for sampling (indices sent to JS runtime)
        eval_dataset=dataset_path,
        # Configuration space - THE KEY TO MARGIN OPTIMIZATION
        configuration_space={
            # Model selection: cost vs capability tradeoff
            # Includes both OpenAI and Groq models via LiteLLM
            "model": [
                # OpenAI models
                "gpt-3.5-turbo",
                "gpt-4o-mini",
                "gpt-4o",
                # Groq models - great for margin optimization!
                "groq/llama-3.3-70b-versatile",  # High quality + cheap + fast
                "groq/llama-3.1-8b-instant",      # Ultra cheap + ultra fast
            ],
            # Temperature: affects consistency
            "temperature": [0.0, 0.3, 0.5, 0.7],
            # Prompt style: affects conversion rate
            "system_prompt": ["sales_aggressive", "consultative", "informative"],
            # Memory turns: HIDDEN COST DRIVER!
            # More turns = more context = higher quality BUT more tokens = higher cost
            "memory_turns": [2, 5, 10, 15],
            # Tool set: affects capability AND cost (tool descriptions = tokens)
            # More tools = better conversion but higher token cost
            "tool_set": ["minimal", "standard", "enhanced", "full"],
        },
        # Objectives to optimize (Mastra-compatible + business metrics)
        objectives=[
            "margin_efficiency",  # PRIMARY: conversion / cost (higher is better)
            "conversion_score",   # Business metric
            "cost",               # Minimize for margins
        ],
        # Runtime configuration
        max_trials=max_trials,  # Explore the configuration space
        plateau_window=5,  # Stop if no improvement for 5 trials
    )
    def arkia_sales_agent(customer_query: str, config: dict = None) -> str:
        """
        Arkia travel sales agent.

        This function signature defines the agent interface. The actual
        implementation runs in the JS module (trial.ts -> agent.ts).

        Note: This is a MOCK implementation that simulates LLM behavior.
        No actual API calls are made. See mastra-agent.ts for real patterns.

        Args:
            customer_query: Customer's travel-related question
            config: Configuration dict injected by Traigent (injection_mode="parameter")

        Returns:
            Agent response (sales pitch, information, support, etc.)
        """
        pass

    # Print banner
    real_mode = os.getenv("REAL_MODE", "").lower() == "true"
    mode_label = "REAL MODE - Mastra LLM calls" if real_mode else "MOCK MODE - No API costs"
    print("\n" + "=" * 70)
    print("  ARKIA SALES AGENT - MARGIN OPTIMIZATION")
    print(f"  Powered by Traigent ({mode_label})")
    print("=" * 70)
    print(f"\nJS Module: {js_module_path}")
    print(f"Dataset: {dataset_path}")
    print("Parallel Workers: 4")
    print(f"Max Trials: {max_trials} (covers {coverage_pct:.1f}% of {config_space_size} configs)")
    print("\n" + "-" * 70)
    print("CONFIGURATION SPACE:")
    print("-" * 70)
    print("  OpenAI Models: gpt-3.5-turbo, gpt-4o-mini, gpt-4o")
    print("  Groq Models:   groq/llama-3.3-70b-versatile, groq/llama-3.1-8b-instant")
    print("  Temperature:   0.0, 0.3, 0.5, 0.7")
    print("  Prompt Style:  sales_aggressive, consultative, informative")
    print("  Memory Turns:  2, 5, 10, 15  <-- HIDDEN COST DRIVER!")
    print("  Tool Set:      minimal, standard, enhanced, full  <-- COST vs CAPABILITY!")
    print("-" * 70)
    print("\nOPTIMIZATION OBJECTIVES:")
    print("  1. margin_efficiency (conversion per dollar - MAXIMIZE)")
    print("  2. conversion_score (sales success rate)")
    print("  3. cost (token spending - MINIMIZE)")
    print("-" * 70)
    print("\nEstimated runtime: ~5 minutes (mock mode)")

    # Run optimization
    print("\nStarting optimization...\n")

    try:
        result = await arkia_sales_agent.optimize()

        elapsed = time.time() - start_time

        # Print results
        print("\n" + "=" * 70)
        print("  OPTIMIZATION COMPLETE - SUCCESS")
        print("=" * 70)

        if result.best_config:
            print("\n[BEST CONFIGURATION]")
            print(f"  Model:         {result.best_config.get('model')}")
            print(f"  Temperature:   {result.best_config.get('temperature')}")
            print(f"  Prompt Style:  {result.best_config.get('system_prompt')}")
            print(f"  Memory Turns:  {result.best_config.get('memory_turns')}")
            print(f"  Tool Set:      {result.best_config.get('tool_set')}")

            # Highlight memory optimization insight
            default_memory = 10
            optimal_memory = result.best_config.get('memory_turns', default_memory)
            if optimal_memory < default_memory:
                savings = ((default_memory - optimal_memory) / default_memory) * 100
                print(f"\n  [INSIGHT] Reducing memory from {default_memory} to {optimal_memory} turns")
                print(f"            saves ~{savings:.0f}% on input tokens!")

            # Highlight tool set optimization insight
            default_tools = "full"
            optimal_tools = result.best_config.get('tool_set', default_tools)
            tool_token_savings = {"minimal": 700, "standard": 500, "enhanced": 300, "full": 0}
            if optimal_tools != default_tools:
                token_savings = tool_token_savings.get(optimal_tools, 0)
                print(f"\n  [INSIGHT] Using '{optimal_tools}' instead of '{default_tools}' tool set")
                print(f"            saves ~{token_savings} tokens per request!")

        if result.best_metrics:
            print("\n[BEST METRICS]")
            for name, value in result.best_metrics.items():
                if isinstance(value, (int, float)):
                    if name == "cost" or name == "cost_per_conversation":
                        print(f"  {name}: ${value:.6f}")
                    elif name in ["conversion_score", "relevancy", "completeness", "tone_consistency"]:
                        print(f"  {name}: {value * 100:.1f}%")
                    elif name == "margin_efficiency":
                        print(f"  {name}: {value:.2f} (higher is better)")
                    else:
                        print(f"  {name}: {value:.2f}")
                else:
                    print(f"  {name}: {value}")

        print("\n[SUMMARY]")
        # Handle different attribute names across SDK versions
        trials = getattr(result, 'trials_completed', None) or getattr(result, 'num_trials', None) or 'N/A'
        stop_reason = getattr(result, 'stop_reason', None) or getattr(result, 'termination_reason', 'completed')
        print(f"  Trials Completed: {trials}")
        print(f"  Stop Reason: {stop_reason}")
        print(f"  Total Time: {elapsed:.1f}s")

        if hasattr(result, "total_cost") and result.total_cost:
            print(f"  Optimization Cost: ${result.total_cost:.6f}")
        else:
            print("  Optimization Cost: $0.00 (mock mode)")

        # ROI insight
        print("\n[ROI PROJECTION]")
        print("  If you process 10,000 conversations/month:")
        if result.best_metrics:
            cost = result.best_metrics.get("cost_per_conversation", 0.001)
            conv = result.best_metrics.get("conversion_score", 0.5)
            monthly_cost = cost * 10000
            monthly_sales = conv * 10000
            print(f"    Estimated LLM cost: ${monthly_cost:.2f}/month")
            print(f"    Estimated conversions: {monthly_sales:.0f}/month")
            print(f"    Cost per conversion: ${monthly_cost / max(monthly_sales, 1):.4f}")

    except FileNotFoundError as e:
        print(f"\nERROR: File not found: {e}")
        print("Make sure to build the JS module first:")
        print(f"  cd {SCRIPT_DIR} && npm run build")
        sys.exit(1)

    except Exception as e:
        print(f"\nERROR: Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  Demo complete! Review results to optimize your margins.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
