#!/usr/bin/env python3
"""Example: Exploration Space Definition with Traigent.

This example demonstrates how to define and manipulate configuration
spaces for Haystack pipeline optimization.

Coverage: Epic 2 (Configuration Space & TVL)
"""

from __future__ import annotations


def example_manual_space_creation():
    """Demonstrate manual exploration space creation."""
    print("=" * 60)
    print("Example 1: Manual Exploration Space Creation")
    print("=" * 60)

    from traigent.integrations.haystack import (
        TVAR,
        CategoricalConstraint,
        ExplorationSpace,
        NumericalConstraint,
    )

    # Create TVARs manually
    model_tvar = TVAR(
        name="model",
        scope="generator",
        python_type="str",
        default_value="gpt-4o",
        constraint=CategoricalConstraint(
            choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        ),
    )

    temperature_tvar = TVAR(
        name="temperature",
        scope="generator",
        python_type="float",
        default_value=0.7,
        constraint=NumericalConstraint(min=0.0, max=2.0),
    )

    top_k_tvar = TVAR(
        name="top_k",
        scope="retriever",
        python_type="int",
        default_value=10,
        constraint=NumericalConstraint(min=1, max=50, step=1),
    )

    # Create exploration space
    space = ExplorationSpace(
        tvars={
            "generator.model": model_tvar,
            "generator.temperature": temperature_tvar,
            "retriever.top_k": top_k_tvar,
        }
    )

    print(f"\nCreated space with {len(space.tvars)} parameters:")
    for name, tvar in space.tvars.items():
        print(f"  - {name}: {tvar.constraint}")

    print("\n")


def example_constraint_types():
    """Demonstrate different constraint types."""
    print("=" * 60)
    print("Example 2: Constraint Types")
    print("=" * 60)

    from traigent.integrations.haystack import (
        TVAR,
        CategoricalConstraint,
        ConditionalConstraint,
        ExplorationSpace,
        NumericalConstraint,
    )

    # Categorical constraint
    categorical = CategoricalConstraint(choices=["option_a", "option_b", "option_c"])
    print(f"Categorical: {categorical}")

    # Numerical constraint (continuous)
    continuous = NumericalConstraint(min=0.0, max=1.0)
    print(f"Continuous: {continuous}")

    # Numerical constraint with log scale
    log_scale = NumericalConstraint(min=1e-5, max=1e-1, log_scale=True)
    print(f"Log scale: {log_scale}")

    # Numerical constraint with step
    stepped = NumericalConstraint(min=0, max=100, step=10)
    print(f"Stepped: {stepped}")

    # Conditional constraint (depends on parent value)
    conditional = ConditionalConstraint(
        parent_qualified_name="generator.model",
        conditions={
            "gpt-4o": NumericalConstraint(min=100, max=8192),
            "gpt-4o-mini": NumericalConstraint(min=100, max=4096),
        },
        default_constraint=NumericalConstraint(min=100, max=2048),
    )
    print(f"Conditional: {conditional}")

    print("\n")


def example_space_operations():
    """Demonstrate exploration space operations."""
    print("=" * 60)
    print("Example 3: Space Operations")
    print("=" * 60)

    from traigent.integrations.haystack import (
        TVAR,
        CategoricalConstraint,
        ExplorationSpace,
        NumericalConstraint,
    )

    # Create initial space
    space = ExplorationSpace(
        tvars={
            "generator.model": TVAR(
                name="model",
                scope="generator",
                python_type="str",
                default_value="gpt-4o",
                constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
            ),
            "generator.temperature": TVAR(
                name="temperature",
                scope="generator",
                python_type="float",
                default_value=0.7,
                constraint=NumericalConstraint(min=0.0, max=2.0),
            ),
        }
    )

    # Get TVAR by qualified name
    temp_tvar = space.get_tvar_by_qualified_name("generator.temperature")
    print(f"Retrieved TVAR: {temp_tvar.qualified_name}")
    print(f"  Constraint: {temp_tvar.constraint}")

    # Get tunable vs fixed TVARs
    tunable = space.tunable_tvars
    fixed = space.fixed_tvars
    print(f"\nTunable TVARs: {len(tunable)}")
    print(f"Fixed TVARs: {len(fixed)}")

    # Get scopes (component namespaces)
    scopes = space.scope_names
    print(f"\nScopes: {scopes}")

    # Check if configuration is valid
    test_config = {"generator.model": "gpt-4o", "generator.temperature": 0.5}
    is_valid, errors = space.validate_config(test_config)
    print(f"\nConfiguration valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")

    print("\n")


def example_tvl_export_import():
    """Demonstrate TVL export and import."""
    print("=" * 60)
    print("Example 4: TVL Export and Import")
    print("=" * 60)

    import tempfile

    from traigent.integrations.haystack import (
        TVAR,
        CategoricalConstraint,
        ExplorationSpace,
        NumericalConstraint,
    )

    # Create space
    original = ExplorationSpace(
        tvars={
            "generator.model": TVAR(
                name="model",
                scope="generator",
                python_type="str",
                default_value="gpt-4o",
                constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
            ),
            "generator.temperature": TVAR(
                name="temperature",
                scope="generator",
                python_type="float",
                default_value=0.7,
                constraint=NumericalConstraint(min=0.0, max=2.0, log_scale=False),
            ),
            "retriever.top_k": TVAR(
                name="top_k",
                scope="retriever",
                python_type="int",
                default_value=10,
                constraint=NumericalConstraint(min=1, max=50, step=1),
            ),
        }
    )

    # Export to TVL
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        tvl_path = f.name
        original.to_tvl(tvl_path, description="RAG pipeline optimization space")
        print(f"Exported to: {tvl_path}")

        # Read and display the TVL content
        f.flush()
        with open(tvl_path) as fp:
            print("\nTVL content:")
            print(fp.read())

    # Import from TVL
    reloaded = ExplorationSpace.from_tvl_spec(tvl_path)
    print(f"Reloaded space with {len(reloaded.tvars)} parameters")

    # Verify round-trip
    for name in original.tvars:
        orig_tvar = original.get_tvar_by_qualified_name(name)
        reload_tvar = reloaded.get_tvar_by_qualified_name(name)
        if orig_tvar and reload_tvar:
            same_type = isinstance(reload_tvar.constraint, type(orig_tvar.constraint))
            print(f"  {name}: constraint preserved = {same_type}")

    print("\n")


if __name__ == "__main__":
    example_manual_space_creation()
    example_constraint_types()
    example_space_operations()
    example_tvl_export_import()

    print("All exploration space examples completed successfully!")
