#!/usr/bin/env python3
"""Benchmark Loader - Downloads and integrates real benchmark datasets.

This module provides functions to download and convert real benchmark
datasets into the JSONL format used by this study.

Available Benchmarks:
- SWE-bench Verified: Real software engineering tasks from GitHub issues
- Aider Polyglot: 225 Exercism coding exercises across 6 languages
- IFEval: Instruction following evaluation from Google Research

Usage:
    # List available benchmarks
    python use-cases/gpt-4.1-study/datasets/benchmark_loader.py --list

    # Download all available benchmarks
    python use-cases/gpt-4.1-study/datasets/benchmark_loader.py --all

    # Download specific benchmark
    python use-cases/gpt-4.1-study/datasets/benchmark_loader.py --swebench
    python use-cases/gpt-4.1-study/datasets/benchmark_loader.py --ifeval
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

DATASETS_DIR = Path(__file__).parent

# Error message constants
ERR_DATASETS_NOT_INSTALLED = "Error: 'datasets' package not installed."
ERR_DATASETS_INSTALL_HINT = "Run: pip install datasets"
ERR_GIT_NOT_FOUND = "Error: git not found. Please install git."
GIT_DEPTH_FLAG = "--depth=1"


def download_swebench_verified(
    output_file: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Download SWE-bench Verified dataset from HuggingFace.

    Args:
        output_file: Optional path to save JSONL file
        limit: Maximum number of samples to download (default 50)

    Returns:
        List of task dictionaries in study format
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print(ERR_DATASETS_NOT_INSTALLED)
        print(ERR_DATASETS_INSTALL_HINT)
        return []

    print(f"Downloading SWE-bench Verified (limit={limit})...")

    try:
        dataset = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
    except Exception as e:
        print(f"Error downloading SWE-bench: {e}")
        return []

    tasks = []
    for i, item in enumerate(dataset):
        if i >= limit:
            break

        task = {
            "input": {
                "task_id": item.get("instance_id", f"swebench_{i}"),
                "task_type": "bug_fix",
                "specification": item.get("problem_statement", ""),
                "original_code": None,  # Full repo context not included
                "repo": item.get("repo", ""),
                "base_commit": item.get("base_commit", ""),
            },
            "output": {
                "expected_patch": item.get("patch", ""),
                "test_patch": item.get("test_patch", ""),
            },
            "metadata": {
                "source": "SWE-bench Verified",
                "difficulty": item.get("difficulty", "unknown"),
            },
        }
        tasks.append(task)

    if output_file:
        output_path = DATASETS_DIR / output_file
        with open(output_path, "w") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")
        print(f"Saved {len(tasks)} tasks to {output_path}")

    return tasks


def download_ifeval(
    output_file: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Download IFEval dataset from Google Research.

    IFEval tests instruction following with verifiable constraints like:
    - Output length requirements
    - Format requirements (JSON, bullets, etc.)
    - Keyword inclusion/exclusion

    Args:
        output_file: Optional path to save JSONL file
        limit: Maximum number of samples to download

    Returns:
        List of task dictionaries in study format
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print(ERR_DATASETS_NOT_INSTALLED)
        print(ERR_DATASETS_INSTALL_HINT)
        return []

    print(f"Downloading IFEval (limit={limit})...")

    try:
        # IFEval is available on HuggingFace
        dataset = load_dataset("google/IFEval", split="train")
    except Exception as e:
        print(f"Error downloading IFEval: {e}")
        return []

    tasks = []
    for i, item in enumerate(dataset):
        if i >= limit:
            break

        # Map IFEval instruction types to our categories
        instruction_types = item.get("instruction_id_list", [])
        category = _map_ifeval_category(instruction_types)

        task = {
            "input": {
                "task_id": f"ifeval_{i}",
                "category": category,
                "prompt": item.get("prompt", ""),
                "instructions": instruction_types,
            },
            "output": {
                "expected_format": None,  # IFEval uses programmatic verification
                "constraints": item.get("kwargs", {}),
            },
            "metadata": {
                "source": "IFEval (Google Research)",
                "instruction_types": instruction_types,
            },
        }
        tasks.append(task)

    if output_file:
        output_path = DATASETS_DIR / output_file
        with open(output_path, "w") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")
        print(f"Saved {len(tasks)} tasks to {output_path}")

    return tasks


def _map_ifeval_category(instruction_types: list[str]) -> str:
    """Map IFEval instruction types to our study categories."""
    type_str = " ".join(instruction_types).lower()

    if "format" in type_str or "json" in type_str or "xml" in type_str:
        return "format_following"
    elif "not" in type_str or "avoid" in type_str or "exclude" in type_str:
        return "negative_instructions"
    elif "first" in type_str or "then" in type_str or "order" in type_str:
        return "ordered_instructions"
    elif "include" in type_str or "contain" in type_str or "must" in type_str:
        return "content_requirements"
    elif "sort" in type_str or "rank" in type_str or "order" in type_str:
        return "ranking"
    else:
        return "other"


def download_babilong(
    output_file: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Download BABILong dataset for long-context multi-needle evaluation.

    BABILong (NeurIPS 2024) extends bAbI tasks to test long-context
    understanding with sequences up to 1M tokens.

    Args:
        output_file: Optional path to save JSONL file
        limit: Maximum number of samples to download

    Returns:
        List of task dictionaries in study format
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print(ERR_DATASETS_NOT_INSTALLED)
        print(ERR_DATASETS_INSTALL_HINT)
        return []

    print(f"Downloading BABILong (limit={limit})...")

    try:
        # BABILong is available on HuggingFace
        dataset = load_dataset("booydar/babilong", "qa1", split="test")
    except Exception as e:
        print(f"Error downloading BABILong: {e}")
        print("Try: pip install datasets")
        return []

    tasks = []
    for i, item in enumerate(dataset):
        if i >= limit:
            break

        task = {
            "input": {
                "task_id": f"babilong_{i}",
                "category": "multi_needle",
                "context": item.get("input", ""),
                "question": item.get("question", ""),
            },
            "output": {
                "expected_answer": item.get("answer", ""),
            },
            "metadata": {
                "source": "BABILong (NeurIPS 2024)",
                "task_type": item.get("task", "qa1"),
            },
        }
        tasks.append(task)

    if output_file:
        output_path = DATASETS_DIR / output_file
        with open(output_path, "w") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")
        print(f"Saved {len(tasks)} tasks to {output_path}")

    return tasks


def download_longbench(
    output_file: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Download LongBench v2 dataset for long-context evaluation.

    LongBench (ACL 2025) tests long-context understanding with
    contexts up to 2M tokens across multiple task types.

    Args:
        output_file: Optional path to save JSONL file
        limit: Maximum number of samples to download

    Returns:
        List of task dictionaries in study format
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print(ERR_DATASETS_NOT_INSTALLED)
        print(ERR_DATASETS_INSTALL_HINT)
        return []

    print(f"Downloading LongBench v2 (limit={limit})...")

    try:
        # LongBench v2 on HuggingFace
        dataset = load_dataset("THUDM/LongBench-v2", split="test")
    except Exception as e:
        print(f"Error downloading LongBench: {e}")
        print("Trying original LongBench...")
        try:
            dataset = load_dataset("THUDM/LongBench", "qasper", split="test")
        except Exception as e2:
            print(f"Error downloading LongBench: {e2}")
            return []

    tasks = []
    for i, item in enumerate(dataset):
        if i >= limit:
            break

        task = {
            "input": {
                "task_id": f"longbench_{i}",
                "category": "long_context",
                "context": item.get("context", item.get("input", "")),
                "question": item.get("question", item.get("input", "")),
            },
            "output": {
                "expected_answer": item.get("answers", item.get("answer", "")),
            },
            "metadata": {
                "source": "LongBench v2 (THUDM)",
                "length": item.get("length", "unknown"),
            },
        }
        tasks.append(task)

    if output_file:
        output_path = DATASETS_DIR / output_file
        with open(output_path, "w") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")
        print(f"Saved {len(tasks)} tasks to {output_path}")

    return tasks


def clone_multihop_rag(target_dir: str | None = None) -> bool:
    """Clone the MultiHop-RAG benchmark repository.

    MultiHop-RAG (COLM 2024) tests multi-hop reasoning across
    documents - alternative to OpenAI's Graphwalks.

    Args:
        target_dir: Directory to clone into (default: datasets/multihop-rag/)

    Returns:
        True if successful, False otherwise
    """
    target = Path(target_dir) if target_dir else DATASETS_DIR / "multihop-rag"

    if target.exists():
        print(f"MultiHop-RAG already exists at {target}")
        return True

    print(f"Cloning MultiHop-RAG benchmark to {target}...")

    try:
        subprocess.run(
            [
                "git",
                "clone",
                GIT_DEPTH_FLAG,
                "https://github.com/yixuantt/MultiHop-RAG",
                str(target),
            ],
            check=True,
            capture_output=True,
        )
        print(f"Successfully cloned to {target}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        return False
    except FileNotFoundError:
        print(ERR_GIT_NOT_FOUND)
        return False


def download_multi_if(
    output_file: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Download Multi-IF dataset for instruction following evaluation.

    Multi-IF (Facebook Research) tests multi-turn instruction following
    - alternative to Scale's MultiChallenge.

    Args:
        output_file: Optional path to save JSONL file
        limit: Maximum number of samples to download

    Returns:
        List of task dictionaries in study format
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print(ERR_DATASETS_NOT_INSTALLED)
        print(ERR_DATASETS_INSTALL_HINT)
        return []

    print(f"Downloading Multi-IF (limit={limit})...")

    try:
        # Try HuggingFace first
        dataset = load_dataset("facebook/multi-if", split="test")
    except Exception:
        print("Multi-IF not on HuggingFace. Trying alternative source...")
        try:
            # Try IFEval as fallback (similar task)
            dataset = load_dataset("google/IFEval", split="train")
            print("Using IFEval as Multi-IF alternative")
        except Exception as e:
            print(f"Error downloading Multi-IF alternative: {e}")
            return []

    tasks = []
    for i, item in enumerate(dataset):
        if i >= limit:
            break

        task = {
            "input": {
                "task_id": f"multi_if_{i}",
                "category": "multi_turn_if",
                "prompt": item.get("prompt", item.get("input", "")),
                "instructions": item.get("instruction_id_list", []),
            },
            "output": {
                "constraints": item.get("kwargs", {}),
            },
            "metadata": {
                "source": "Multi-IF (Facebook Research)",
            },
        }
        tasks.append(task)

    if output_file:
        output_path = DATASETS_DIR / output_file
        with open(output_path, "w") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")
        print(f"Saved {len(tasks)} tasks to {output_path}")

    return tasks


def clone_toolbench(target_dir: str | None = None) -> bool:
    """Clone the ToolBench benchmark repository.

    ToolBench (ICLR 2024) provides 16K+ real APIs for testing
    function calling - alternative to ComplexFuncBench.

    Args:
        target_dir: Directory to clone into (default: datasets/toolbench/)

    Returns:
        True if successful, False otherwise
    """
    target = Path(target_dir) if target_dir else DATASETS_DIR / "toolbench"

    if target.exists():
        print(f"ToolBench already exists at {target}")
        return True

    print(f"Cloning ToolBench to {target}...")
    print("Note: ToolBench is large. This may take a while...")

    try:
        subprocess.run(
            [
                "git",
                "clone",
                GIT_DEPTH_FLAG,
                "https://github.com/OpenBMB/ToolBench",
                str(target),
            ],
            check=True,
            capture_output=True,
        )
        print(f"Successfully cloned to {target}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        return False
    except FileNotFoundError:
        print(ERR_GIT_NOT_FOUND)
        return False


def clone_aider_benchmark(target_dir: str | None = None) -> bool:
    """Clone the Aider polyglot benchmark repository.

    The Aider benchmark uses Exercism exercises and requires the full
    repository to run properly.

    Args:
        target_dir: Directory to clone into (default: datasets/aider/)

    Returns:
        True if successful, False otherwise
    """
    target = Path(target_dir) if target_dir else DATASETS_DIR / "aider"

    if target.exists():
        print(f"Aider benchmark already exists at {target}")
        return True

    print(f"Cloning Aider polyglot benchmark to {target}...")

    try:
        subprocess.run(
            [
                "git",
                "clone",
                GIT_DEPTH_FLAG,
                "https://github.com/Aider-AI/polyglot-benchmark",
                str(target),
            ],
            check=True,
            capture_output=True,
        )
        print(f"Successfully cloned to {target}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        return False
    except FileNotFoundError:
        print(ERR_GIT_NOT_FOUND)
        return False


def list_available_benchmarks() -> None:
    """Print information about available benchmarks."""
    print("\n=== Available Benchmarks (Original) ===\n")

    print("1. SWE-bench Verified")
    print("   Source: HuggingFace (SWE-bench/SWE-bench_Verified)")
    print("   Tasks: 500 verified software engineering problems")
    print("   Command: --swebench\n")

    print("2. IFEval")
    print("   Source: Google Research")
    print("   Tasks: ~500 instruction following tests with verifiable constraints")
    print("   Command: --ifeval\n")

    print("3. Aider Polyglot")
    print("   Source: GitHub (Aider-AI/polyglot-benchmark)")
    print("   Tasks: 225 Exercism coding exercises")
    print("   Command: --aider\n")

    print("=== Open-Source Alternatives ===\n")

    print("4. BABILong (alternative to OpenAI-MRCR)")
    print("   Source: HuggingFace (booydar/babilong)")
    print("   Tasks: Long-context QA up to 1M tokens")
    print("   Paper: NeurIPS 2024")
    print("   Command: --babilong\n")

    print("5. LongBench v2 (alternative to OpenAI-MRCR)")
    print("   Source: HuggingFace (THUDM/LongBench-v2)")
    print("   Tasks: Long-context understanding up to 2M tokens")
    print("   Paper: ACL 2025")
    print("   Command: --longbench\n")

    print("6. MultiHop-RAG (alternative to Graphwalks)")
    print("   Source: GitHub (yixuantt/MultiHop-RAG)")
    print("   Tasks: Multi-hop reasoning across documents")
    print("   Paper: COLM 2024")
    print("   Command: --multihop-rag\n")

    print("7. Multi-IF (alternative to MultiChallenge)")
    print("   Source: Facebook Research")
    print("   Tasks: Multi-turn instruction following")
    print("   Command: --multi-if\n")

    print("8. ToolBench (alternative to ComplexFuncBench)")
    print("   Source: GitHub (OpenBMB/ToolBench)")
    print("   Tasks: 16K+ real APIs for function calling")
    print("   Paper: ICLR 2024")
    print("   Command: --toolbench\n")

    print("=== NOT Available (Proprietary/Unreleased) ===\n")

    print("- OpenAI-MRCR (github.com/openai/mrcr - 404)")
    print("- Graphwalks (github.com/openai/graphwalks - 404)")
    print("- MultiChallenge (Scale AI proprietary)")
    print("- ComplexFuncBench (not public)")


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Download and integrate real benchmark datasets"
    )

    # Download all
    parser.add_argument(
        "--all", action="store_true", help="Download all available benchmarks"
    )

    # Original benchmarks
    parser.add_argument(
        "--swebench", action="store_true", help="Download SWE-bench Verified"
    )
    parser.add_argument("--ifeval", action="store_true", help="Download IFEval")
    parser.add_argument(
        "--aider", action="store_true", help="Clone Aider polyglot benchmark"
    )

    # Open-source alternatives
    parser.add_argument(
        "--babilong", action="store_true", help="Download BABILong (MRCR alternative)"
    )
    parser.add_argument(
        "--longbench", action="store_true", help="Download LongBench v2 (MRCR alt)"
    )
    parser.add_argument(
        "--multihop-rag",
        action="store_true",
        help="Clone MultiHop-RAG (Graphwalks alt)",
    )
    parser.add_argument(
        "--multi-if",
        action="store_true",
        help="Download Multi-IF (MultiChallenge alt)",
    )
    parser.add_argument(
        "--toolbench",
        action="store_true",
        help="Clone ToolBench (ComplexFuncBench alt)",
    )

    # Other options
    parser.add_argument("--list", action="store_true", help="List available benchmarks")
    parser.add_argument(
        "--limit", type=int, default=50, help="Max samples per benchmark"
    )

    return parser


def _run_downloads(args: argparse.Namespace) -> None:
    """Execute benchmark downloads based on args."""
    download_all = args.all

    # Define benchmarks: (flag_name, download_func, output_file or None for clone)
    hf_benchmarks = [
        ("swebench", download_swebench_verified, "swebench_verified.jsonl"),
        ("ifeval", download_ifeval, "ifeval_tasks.jsonl"),
        ("babilong", download_babilong, "babilong_tasks.jsonl"),
        ("longbench", download_longbench, "longbench_tasks.jsonl"),
        ("multi_if", download_multi_if, "multi_if_tasks.jsonl"),
    ]

    clone_benchmarks = [
        ("aider", clone_aider_benchmark),
        ("multihop_rag", clone_multihop_rag),
        ("toolbench", clone_toolbench),
    ]

    # Run HuggingFace downloads
    for flag_name, download_func, output_file in hf_benchmarks:
        if download_all or getattr(args, flag_name, False):
            download_func(output_file, limit=args.limit)

    # Run git clones
    for flag_name, clone_func in clone_benchmarks:
        if download_all or getattr(args, flag_name, False):
            clone_func()


def main() -> None:
    """Main entry point for benchmark loader."""
    parser = _create_argument_parser()
    args = parser.parse_args()

    if args.list:
        list_available_benchmarks()
        return

    # Check if any benchmark flag is set
    benchmark_flags = [
        args.all,
        args.swebench,
        args.ifeval,
        args.aider,
        args.babilong,
        args.longbench,
        getattr(args, "multihop_rag", False),
        getattr(args, "multi_if", False),
        args.toolbench,
    ]

    if not any(benchmark_flags):
        parser.print_help()
        print("\nUse --list to see available benchmarks")
        return

    _run_downloads(args)
    print("\nDone! Benchmark data is now available.")


if __name__ == "__main__":
    main()
