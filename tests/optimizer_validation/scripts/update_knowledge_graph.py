#!/usr/bin/env python3
"""Update knowledge graph after test runs.

This script updates the knowledge graph from test evidence files
and exports to both JSON and optional Turtle formats.

Usage:
    # Update from evidence directory
    python tests/optimizer_validation/scripts/update_knowledge_graph.py

    # Update from specific evidence directory
    python tests/optimizer_validation/scripts/update_knowledge_graph.py evidence/

    # Rebuild from test files
    python tests/optimizer_validation/scripts/update_knowledge_graph.py --rebuild
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def find_evidence_files(evidence_dir: Path) -> list[Path]:
    """Find all evidence JSON files."""
    files = []
    for pattern in ["*.json", "**/*.json"]:
        files.extend(evidence_dir.glob(pattern))
    return [f for f in files if f.is_file()]


def update_from_evidence(
    evidence_dir: Path,
    output_json: Path,
    output_turtle: Path | None = None,
) -> int:
    """Update knowledge graph from evidence files.

    Args:
        evidence_dir: Directory containing evidence JSON files
        output_json: Path for JSON output
        output_turtle: Optional path for Turtle output

    Returns:
        Number of tests updated
    """
    from tests.optimizer_validation.viewer.knowledge_graph import TestKnowledgeGraph

    # Load existing graph if present
    if output_json.exists():
        kg = TestKnowledgeGraph.load(str(output_json))
        print(f"Loaded existing graph with {len(kg.tests)} tests")
    else:
        kg = TestKnowledgeGraph()

    # Find and process evidence files
    evidence_files = find_evidence_files(evidence_dir)
    print(f"Found {len(evidence_files)} evidence files")

    updated = 0
    for evidence_file in evidence_files:
        try:
            with open(evidence_file) as f:
                evidence = json.load(f)

            # Extract test info from evidence
            test_id = evidence.get("test_id", evidence_file.stem)
            scenario = evidence.get("scenario", {})
            result = evidence.get("result", {})
            validation = evidence.get("validation", {})

            # Update or add test
            test_data = {
                "name": scenario.get("name", test_id),
                "description": scenario.get("description", ""),
                "test_file": evidence.get("test_file", ""),
                "class": evidence.get("test_class", ""),
                "method": evidence.get("test_method", ""),
                "dimensions": evidence.get("dimensions", {}),
                "expected_outcome": scenario.get("expected", {}).get(
                    "outcome", "success"
                ),
                "intent": evidence.get("intent", ""),
                "markers": evidence.get("markers", []),
                "params": scenario.get("params", {}),
                "result": {
                    "status": "PASS" if validation.get("passed") else "FAIL",
                    "trial_count": result.get("trial_count", 0),
                    "best_score": result.get("best_score"),
                    "stop_reason": result.get("stop_reason"),
                    "duration": result.get("duration"),
                    "error_type": result.get("error_type"),
                    "actual_outcome": result.get("outcome"),
                },
                "gist": evidence.get("gist", {}),
            }

            kg._tests_dict[test_id] = test_data
            updated += 1

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not process {evidence_file}: {e}")
            continue

    # Export JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(kg.to_json(), f, indent=2)
    print(f"Exported JSON to {output_json}")

    # Export Turtle if requested
    if output_turtle:
        try:
            kg.build_rdf_graph()
            kg.export_turtle(output_turtle)
            print(f"Exported Turtle to {output_turtle}")
        except Exception as e:
            print(f"Warning: Could not export Turtle: {e}")

    return updated


def rebuild_from_tests(
    test_dir: Path,
    output_json: Path,
    output_turtle: Path | None = None,
) -> int:
    """Rebuild knowledge graph from test files.

    Args:
        test_dir: Directory containing test files
        output_json: Path for JSON output
        output_turtle: Optional path for Turtle output

    Returns:
        Number of tests found
    """
    from tests.optimizer_validation.viewer.knowledge_graph import TestKnowledgeGraph

    kg = TestKnowledgeGraph()
    kg.load_tests(test_dir)

    test_count = len(kg._get_test_list())
    print(f"Found {test_count} tests in {test_dir}")

    # Export JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(kg.to_json(), f, indent=2)
    print(f"Exported JSON to {output_json}")

    # Export Turtle if requested
    if output_turtle:
        try:
            kg.build_rdf_graph()
            kg.export_turtle(output_turtle)
            print(f"Exported Turtle to {output_turtle}")
        except Exception as e:
            print(f"Warning: Could not export Turtle: {e}")

    return test_count


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update test knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "evidence_dir",
        type=Path,
        nargs="?",
        default=Path("evidence"),
        help="Directory containing evidence JSON files (default: evidence/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("tests/optimizer_validation/viewer/graph_data.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--turtle",
        "-t",
        type=Path,
        help="Optional Turtle output file path",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild graph from test files instead of evidence",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("tests/optimizer_validation"),
        help="Test directory for --rebuild (default: tests/optimizer_validation)",
    )

    args = parser.parse_args()

    if args.rebuild:
        count = rebuild_from_tests(args.test_dir, args.output, args.turtle)
        print(f"Rebuilt graph with {count} tests")
    else:
        if not args.evidence_dir.exists():
            print(f"Evidence directory not found: {args.evidence_dir}")
            print("Use --rebuild to build from test files instead")
            sys.exit(1)

        count = update_from_evidence(args.evidence_dir, args.output, args.turtle)
        print(f"Updated {count} tests in knowledge graph")


if __name__ == "__main__":
    main()
