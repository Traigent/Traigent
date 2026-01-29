#!/usr/bin/env python3
"""
TVL 0.9 Tutorial: Exploring the TVAR Type System

This script demonstrates how to work with TVL 0.9's typed TVARs programmatically.
You'll learn how to:
- Load and inspect TVAR definitions
- Understand domain specifications
- Work with different TVAR types

Run with (from repo root): .venv/bin/python examples/tvl/tutorials/02_typed_tvars/explore_tvars.py
"""

from pathlib import Path

from traigent.tvl import (
    TVarDecl,
    load_tvl_spec,
    normalize_tvar_type,
    parse_domain_spec,
)

SPEC_PATH = Path(__file__).parent / "advanced_tvars.tvl.yml"


def explore_tvar(tvar: TVarDecl) -> None:
    """Print detailed information about a TVAR."""
    print(f"\n  {tvar.name}:")
    print(f"    Type: {tvar.type} (raw: {tvar.raw_type})")
    print(f"    Default: {tvar.default}")
    if tvar.unit:
        print(f"    Unit: {tvar.unit}")

    # Explore the domain specification
    domain = tvar.domain
    print(f"    Domain kind: {domain.kind}")

    if domain.kind == "enum":
        print(f"    Values: {domain.values}")
    elif domain.kind == "range":
        print(f"    Range: {domain.range}")
        if domain.resolution:
            print(f"    Resolution: {domain.resolution}")
    elif domain.kind == "registry":
        print(f"    Registry: {domain.registry}")
        if domain.filter:
            print(f"    Filter: {domain.filter}")


def demonstrate_type_normalization():
    """Show how raw type strings are normalized."""
    print("\n" + "=" * 60)
    print("Type Normalization Examples")
    print("=" * 60)

    examples = [
        "bool",
        "boolean",
        "int",
        "integer",
        "float",
        "continuous",
        "number",
        "enum[str]",
        "enum[int]",
        "tuple[int, float]",
        "callable[ScorerProto]",
    ]

    for raw_type in examples:
        normalized = normalize_tvar_type(raw_type)
        print(f"  {raw_type:25} -> {normalized}")


def demonstrate_domain_parsing():
    """Show how domain specifications are parsed."""
    print("\n" + "=" * 60)
    print("Domain Parsing Examples")
    print("=" * 60)

    # Enum domain from list
    domain1 = parse_domain_spec("model", "enum", ["gpt-4", "claude-3"])
    print(f"\n  List domain: {domain1}")

    # Range domain
    domain2 = parse_domain_spec("temperature", "float", {"range": [0.0, 2.0]})
    print(f"  Range domain: {domain2}")

    # Range with resolution
    domain3 = parse_domain_spec(
        "temp_discrete", "float", {"range": [0.0, 1.0], "resolution": 0.1}
    )
    print(f"  Range + resolution: {domain3}")

    # Bool domain (implicit)
    domain4 = parse_domain_spec("flag", "bool", None)
    print(f"  Bool domain (implicit): {domain4}")

    # Registry domain
    domain5 = parse_domain_spec(
        "scorer", "callable", {"registry": "scorers", "filter": "version >= 2"}
    )
    print(f"  Registry domain: {domain5}")


def main():
    """Main tutorial demonstration."""
    print("=" * 60)
    print("TVL 0.9 Tutorial: TVAR Type System")
    print("=" * 60)

    # Load the spec
    print("\n1. Loading TVL Spec...")
    spec = load_tvl_spec(spec_path=SPEC_PATH)

    if not spec.tvars:
        print("   No TVARs found in spec!")
        return

    # Group TVARs by type
    print(f"\n2. Inspecting {len(spec.tvars)} TVARs by type:")

    tvars_by_type: dict[str, list[TVarDecl]] = {}
    for tvar in spec.tvars:
        ttype = tvar.type
        if ttype not in tvars_by_type:
            tvars_by_type[ttype] = []
        tvars_by_type[ttype].append(tvar)

    for ttype, tvars in sorted(tvars_by_type.items()):
        print(f"\n  === {ttype.upper()} TVARs ({len(tvars)}) ===")
        for tvar in tvars:
            explore_tvar(tvar)

    # Show configuration space conversion
    print("\n\n3. Configuration Space (for optimizer):")
    for name, space_entry in spec.configuration_space.items():
        print(f"  {name}: {space_entry}")

    # Demonstrate type normalization
    demonstrate_type_normalization()

    # Demonstrate domain parsing
    demonstrate_domain_parsing()

    print("\n" + "=" * 60)
    print("Tutorial complete! Next: 03_multi_objective")
    print("=" * 60)


if __name__ == "__main__":
    main()
