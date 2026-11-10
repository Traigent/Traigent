#!/usr/bin/env python3
"""
P0: Structured Output Engineering with Traigent
===============================================

Demonstrates optimizing structured data extraction from text.
Traigent finds the best output format and parsing strategy.

Run: TRAIGENT_MOCK_LLM=true python main.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import traigent
except ImportError:  # pragma: no cover - support IDE execution paths
    import importlib

    _sdk = os.environ.get("TRAIGENT_SDK_PATH")
    if _sdk:
        sys.path.insert(0, _sdk)
    else:
        module_path = Path(__file__).resolve()
        for depth in (2, 3):
            try:
                sys.path.append(str(module_path.parents[depth]))
            except IndexError:
                continue
    traigent = importlib.import_module("traigent")

from traigent.core.objectives import create_default_objectives

# Mock mode for demo
os.environ["TRAIGENT_MOCK_LLM"] = "true"


def _write_eval_dataset(samples: list[dict]) -> str:
    """Write the demo dataset next to this script and return its path.

    The SDK only loads datasets under the current directory or
    TRAIGENT_DATASET_ROOT, so the system temp dir is not usable. Default the
    root to this script's directory so the example runs from any cwd.
    """
    # Write under the effective dataset root: the configured one, else this
    # script's directory (which is then made the root).
    configured = os.environ.get("TRAIGENT_DATASET_ROOT")
    root = Path(configured) if configured else Path(__file__).resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRAIGENT_DATASET_ROOT", str(root))
    path = root / f"_demo_{Path(__file__).resolve().parent.name}_eval_dataset.jsonl"
    text = "".join(json.dumps(s) + "\n" for s in samples)
    # Never clobber a file this example did not create: a rerun regenerates
    # identical content, anything else at this path belongs to someone else.
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise SystemExit(
            f"Refusing to overwrite {path}: it exists and was not written by this "
            "example. Delete it or point TRAIGENT_DATASET_ROOT at another directory."
        )
    path.write_text(text, encoding="utf-8")
    return str(path)


def create_extraction_dataset() -> str:
    """Create dataset for entity extraction."""
    samples = [
        {
            "input": {
                "text": "Meeting with John Smith (CEO of Acme Corp) next Tuesday at 3pm"
            },
            "output": json.dumps(
                {"person": "John Smith", "company": "Acme Corp", "time": "Tuesday 3pm"}
            ),
        },
        {
            "input": {
                "text": "Sarah Johnson from TechStart will present the Q3 results"
            },
            "output": json.dumps(
                {
                    "person": "Sarah Johnson",
                    "company": "TechStart",
                    "event": "Q3 results",
                }
            ),
        },
        {
            "input": {"text": "Contact Mike Davis at GlobalTech about the partnership"},
            "output": json.dumps(
                {
                    "person": "Mike Davis",
                    "company": "GlobalTech",
                    "topic": "partnership",
                }
            ),
        },
    ]

    return _write_eval_dataset(samples)


PERSON_PATTERNS = {
    "John Smith": "John Smith",
    "Sarah Johnson": "Sarah Johnson",
    "Mike Davis": "Mike Davis",
}

COMPANY_PATTERNS = {
    "Acme Corp": "Acme Corp",
    "TechStart": "TechStart",
    "GlobalTech": "GlobalTech",
}


def _match_pattern(text: str, patterns: dict[str, str]) -> str | None:
    for key, value in patterns.items():
        if key in text:
            return value
    return None


def _format_entities(entities: dict[str, str], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(entities)
    if output_format == "xml":
        xml_parts = (
            ["<entities>"]
            + [f"  <{k}>{v}</{k}>" for k, v in entities.items()]
            + ["</entities>"]
        )
        return "\n".join(xml_parts)
    md_parts = ["## Extracted Entities"] + [
        f"- **{k}**: {v}" for k, v in entities.items()
    ]
    return "\n".join(md_parts)


@traigent.optimize(
    eval_dataset=create_extraction_dataset(),
    objectives=create_default_objectives(
        ["accuracy", "parsing_success"],
        orientations={"parsing_success": "maximize"},
    ),
    configuration_space={
        "output_format": ["json", "xml", "markdown"],
        "validation": ["strict", "lenient"],
        "extraction_style": ["verbose", "concise"],
    },
    offline=True,
)
def extract_entities(
    text: str,
    output_format: str = "json",
    validation: str = "strict",
    extraction_style: str = "concise",
) -> str:
    """Extract structured entities from text."""

    entities: dict[str, str] = {}
    person = _match_pattern(text, PERSON_PATTERNS)
    if person:
        entities["person"] = person
    company = _match_pattern(text, COMPANY_PATTERNS)
    if company:
        entities["company"] = company
    return _format_entities(entities, output_format)


async def main() -> None:
    """Demonstrate structured output optimization."""
    print("📋 Structured Output Optimization with Traigent")
    print("=" * 50)

    test_text = "Meeting with John Smith (CEO of Acme Corp) next Tuesday at 3pm"

    # Baseline extraction
    print("\n🔍 Baseline Extraction (JSON, strict):")
    baseline = extract_entities(test_text, output_format="json", validation="strict")
    print(f"Output: {baseline}")

    # Run optimization
    print("\n🚀 Running Traigent optimization...")
    print("Testing 12 configurations (3 formats × 2 validation × 2 styles)")

    optimization = await extract_entities.optimize(max_trials=12, timeout=60)

    # Show results
    print("\n✨ Optimization Results:")
    print("Best configuration found:")
    print(f"  • Format: {optimization.best_config['output_format']}")
    print(f"  • Validation: {optimization.best_config['validation']}")
    print(f"  • Style: {optimization.best_config['extraction_style']}")
    print(f"  • Accuracy: {optimization.best_score:.1%}")

    # Apply best configuration
    optimized = extract_entities(test_text, **optimization.best_config)
    print("\nOptimized output:")
    print(optimized)

    print("\n💡 Key Benefits:")
    print("  • Automatically finds best output format")
    print("  • Optimizes parsing reliability")
    print("  • Balances accuracy vs. token usage")
    print("  • Data-driven configuration selection")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
