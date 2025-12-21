#!/usr/bin/env python3
"""Example: TVL Round-Trip with Advanced Features.

This example demonstrates TVL (Tunable Variable Language) export and
import with advanced features like log_scale, step, and conditionals.

Coverage: Epic 3, Story 3.6 (Extend spec_loader for TVL round-trip)
"""

from __future__ import annotations

import tempfile


def example_basic_tvl():
    """Demonstrate basic TVL export and import."""
    print("=" * 60)
    print("Example 1: Basic TVL Round-Trip")
    print("=" * 60)

    from traigent.integrations.haystack import (
        TVAR,
        CategoricalConstraint,
        ExplorationSpace,
        NumericalConstraint,
    )

    # Create exploration space
    space = ExplorationSpace(
        tvars={
            "generator.model": TVAR(
                name="model",
                scope="generator",
                python_type="str",
                default_value="gpt-4o",
                constraint=CategoricalConstraint(
                    choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
                ),
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

    # Export to TVL
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        tvl_path = f.name
        space.to_tvl(tvl_path, description="Basic RAG pipeline")

        # Show exported content
        with open(tvl_path) as fp:
            content = fp.read()
            print(f"\nExported TVL:\n{content}")

    # Import back
    reloaded = ExplorationSpace.from_tvl_spec(tvl_path)

    print(f"Reloaded {len(reloaded.tvars)} parameters:")
    for name, tvar in reloaded.tvars.items():
        print(f"  - {name}: {type(tvar.constraint).__name__}")

    print("\n")


def example_log_scale():
    """Demonstrate log_scale parameter handling."""
    print("=" * 60)
    print("Example 2: Log Scale Parameters")
    print("=" * 60)

    from traigent.integrations.haystack import (
        TVAR,
        ExplorationSpace,
        NumericalConstraint,
    )

    # Learning rate with log scale (common in ML)
    space = ExplorationSpace(
        tvars={
            "optimizer.learning_rate": TVAR(
                name="learning_rate",
                scope="optimizer",
                python_type="float",
                default_value=0.001,
                constraint=NumericalConstraint(
                    min=1e-5,
                    max=1e-1,
                    log_scale=True,  # Sample on log scale
                ),
            ),
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        tvl_path = f.name
        space.to_tvl(tvl_path, include_metadata=False)

        with open(tvl_path) as fp:
            print(f"\nTVL with log_scale:\n{fp.read()}")

    # Verify round-trip preserves log_scale
    reloaded = ExplorationSpace.from_tvl_spec(tvl_path)
    tvar = reloaded.get_tvar_by_qualified_name("optimizer.learning_rate")

    print(f"Round-trip check:")
    print(f"  log_scale preserved: {tvar.constraint.log_scale}")
    print(f"  min: {tvar.constraint.min}")
    print(f"  max: {tvar.constraint.max}")

    print("\n")


def example_step_parameter():
    """Demonstrate step parameter for discrete numerical values."""
    print("=" * 60)
    print("Example 3: Step Parameters")
    print("=" * 60)

    from traigent.integrations.haystack import (
        TVAR,
        ExplorationSpace,
        NumericalConstraint,
    )

    # Batch size with step (powers of 2 are common)
    space = ExplorationSpace(
        tvars={
            "training.batch_size": TVAR(
                name="batch_size",
                scope="training",
                python_type="int",
                default_value=32,
                constraint=NumericalConstraint(
                    min=16,
                    max=256,
                    step=16,  # Values: 16, 32, 48, ..., 256
                ),
            ),
            "training.epochs": TVAR(
                name="epochs",
                scope="training",
                python_type="int",
                default_value=10,
                constraint=NumericalConstraint(
                    min=1,
                    max=100,
                    step=1,
                ),
            ),
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        tvl_path = f.name
        space.to_tvl(tvl_path, include_metadata=False)

        with open(tvl_path) as fp:
            print(f"\nTVL with step:\n{fp.read()}")

    # Verify round-trip
    reloaded = ExplorationSpace.from_tvl_spec(tvl_path)
    batch_tvar = reloaded.get_tvar_by_qualified_name("training.batch_size")

    print(f"Round-trip check for batch_size:")
    print(f"  step preserved: {batch_tvar.constraint.step}")
    print(f"  range: [{batch_tvar.constraint.min}, {batch_tvar.constraint.max}]")

    print("\n")


def example_conditional_parameters():
    """Demonstrate conditional parameters that depend on parent values."""
    print("=" * 60)
    print("Example 4: Conditional Parameters")
    print("=" * 60)

    from traigent.integrations.haystack import (
        TVAR,
        CategoricalConstraint,
        ConditionalConstraint,
        ExplorationSpace,
        NumericalConstraint,
    )

    # Model-dependent max_tokens limits
    space = ExplorationSpace(
        tvars={
            "generator.model": TVAR(
                name="model",
                scope="generator",
                python_type="str",
                default_value="gpt-4o",
                constraint=CategoricalConstraint(
                    choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
                ),
            ),
            "generator.max_tokens": TVAR(
                name="max_tokens",
                scope="generator",
                python_type="int",
                default_value=1024,
                constraint=ConditionalConstraint(
                    parent_qualified_name="generator.model",
                    conditions={
                        "gpt-4o": NumericalConstraint(min=100, max=8192),
                        "gpt-4o-mini": NumericalConstraint(min=100, max=4096),
                        "gpt-3.5-turbo": NumericalConstraint(min=100, max=4096),
                    },
                    default_constraint=NumericalConstraint(min=100, max=2048),
                ),
            ),
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        tvl_path = f.name
        space.to_tvl(tvl_path, include_metadata=False)

        with open(tvl_path) as fp:
            print(f"\nTVL with conditional:\n{fp.read()}")

    # Verify round-trip
    reloaded = ExplorationSpace.from_tvl_spec(tvl_path)
    max_tokens = reloaded.get_tvar_by_qualified_name("generator.max_tokens")

    print(f"Round-trip check for conditional:")
    print(f"  Type: {type(max_tokens.constraint).__name__}")
    print(f"  Parent: {max_tokens.constraint.parent_qualified_name}")
    print(f"  Conditions: {list(max_tokens.constraint.conditions.keys())}")

    # Show valid max_tokens ranges for each model
    print("\nMax tokens ranges by model:")
    for model in ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]:
        constraint = max_tokens.constraint.conditions.get(model)
        print(f"  {model}: max_tokens in [{constraint.min}, {constraint.max}]")

    print("\n")


def example_load_from_yaml():
    """Demonstrate loading TVL from YAML string."""
    print("=" * 60)
    print("Example 5: Load from YAML String")
    print("=" * 60)

    from traigent.integrations.haystack import ExplorationSpace
    from traigent.tvl import NumericalDomain, load_tvl_spec

    # Define TVL as YAML string
    tvl_content = """
configuration_space:
  learning_rate:
    type: continuous
    range: [0.0001, 0.1]
    log_scale: true
    default: 0.01

  batch_size:
    type: integer
    range: [16, 128]
    step: 16
    default: 32

  optimizer:
    type: categorical
    values: ["adam", "sgd", "adamw"]
    default: "adam"
"""

    # Write to temp file and load
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(tvl_content)
        f.flush()
        tvl_path = f.name

    # Load using spec_loader
    spec = load_tvl_spec(spec_path=tvl_path)

    print(f"\nParsed TVL spec:")
    for name, domain in spec.configuration_space.items():
        if isinstance(domain, NumericalDomain):
            print(
                f"  {name}: NumericalDomain(min={domain.min}, max={domain.max}, "
                f"log_scale={domain.log_scale}, step={domain.step})"
            )
        elif isinstance(domain, tuple):
            print(f"  {name}: range {domain}")
        elif isinstance(domain, list):
            print(f"  {name}: choices {domain}")

    # Create ExplorationSpace
    space = ExplorationSpace.from_tvl_spec(tvl_path)
    print(f"\nCreated ExplorationSpace with {len(space.tvars)} parameters")

    print("\n")


def example_full_pipeline_spec():
    """Demonstrate a complete pipeline optimization specification."""
    print("=" * 60)
    print("Example 6: Full Pipeline Specification")
    print("=" * 60)

    from traigent.integrations.haystack import (
        TVAR,
        CategoricalConstraint,
        ConditionalConstraint,
        ExplorationSpace,
        NumericalConstraint,
    )

    # Complete RAG pipeline optimization space
    space = ExplorationSpace(
        tvars={
            # Retriever parameters
            "retriever.top_k": TVAR(
                name="top_k",
                scope="retriever",
                python_type="int",
                default_value=10,
                constraint=NumericalConstraint(min=1, max=50, step=1),
            ),
            # Generator model selection
            "generator.model": TVAR(
                name="model",
                scope="generator",
                python_type="str",
                default_value="gpt-4o",
                constraint=CategoricalConstraint(
                    choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
                ),
            ),
            # Temperature (all models)
            "generator.temperature": TVAR(
                name="temperature",
                scope="generator",
                python_type="float",
                default_value=0.7,
                constraint=NumericalConstraint(min=0.0, max=2.0),
            ),
            # Max tokens (model-dependent)
            "generator.max_tokens": TVAR(
                name="max_tokens",
                scope="generator",
                python_type="int",
                default_value=1024,
                constraint=ConditionalConstraint(
                    parent_qualified_name="generator.model",
                    conditions={
                        "gpt-4o": NumericalConstraint(min=100, max=8192),
                        "gpt-4o-mini": NumericalConstraint(min=100, max=4096),
                        "gpt-3.5-turbo": NumericalConstraint(min=100, max=4096),
                    },
                ),
            ),
        }
    )

    # Export with full metadata
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        tvl_path = f.name
        space.to_tvl(
            tvl_path,
            description="RAG pipeline optimization for Q&A task",
            include_metadata=True,
        )

        with open(tvl_path) as fp:
            print(f"\nComplete TVL specification:\n{fp.read()}")

    # Show default configuration and validate
    print("Default configuration values:")
    for name, tvar in sorted(space.tvars.items()):
        print(f"  {name}: {tvar.default_value}")

    # Validate a sample configuration
    test_config = {
        "retriever.top_k": 15,
        "generator.model": "gpt-4o",
        "generator.temperature": 0.8,
        "generator.max_tokens": 2048,
    }
    is_valid, errors = space.validate_config(test_config)
    print(f"\nTest config valid: {is_valid}")
    if errors:
        print(f"  Errors: {errors}")

    print("\n")


if __name__ == "__main__":
    example_basic_tvl()
    example_log_scale()
    example_step_parameter()
    example_conditional_parameters()
    example_load_from_yaml()
    example_full_pipeline_spec()

    print("All TVL round-trip examples completed successfully!")
