"""End-to-end tests for TVL example specifications.

This module validates that all TVL example specs in the examples/tvl/ directory
can be loaded successfully without errors. This serves as E2E validation that
the TVL 0.9 implementation handles all documented example scenarios correctly.
"""

from pathlib import Path

import pytest

from traigent.tvl.spec_loader import TVLSpecArtifact, load_tvl_spec
from traigent.utils.exceptions import TVLValidationError

# Discover all TVL example specs
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples" / "tvl"
TVL_SPECS = list(EXAMPLES_DIR.glob("**/*.tvl.yml"))


def _get_spec_id(spec_path: Path) -> str:
    """Generate a readable test ID from spec path."""
    relative = spec_path.relative_to(EXAMPLES_DIR)
    return str(relative).replace("/", "__").replace(".tvl.yml", "")


class TestTVLExamplesLoad:
    """Test that all TVL example specs can be loaded."""

    @pytest.mark.parametrize(
        "spec_path",
        TVL_SPECS,
        ids=[_get_spec_id(p) for p in TVL_SPECS],
    )
    def test_load_spec(self, spec_path: Path) -> None:
        """Each TVL spec file should load without errors."""
        artifact = load_tvl_spec(spec_path=spec_path)

        # Verify basic artifact structure
        assert isinstance(artifact, TVLSpecArtifact)
        assert artifact.path == spec_path
        assert isinstance(artifact.configuration_space, dict)
        assert isinstance(artifact.metadata, dict)

    @pytest.mark.parametrize(
        "spec_path",
        TVL_SPECS,
        ids=[_get_spec_id(p) for p in TVL_SPECS],
    )
    def test_spec_has_configuration_space(self, spec_path: Path) -> None:
        """Each spec should define a non-empty configuration space."""
        # Skip base specs that are meant for inheritance (they define constraints/evaluators
        # but child specs provide the configuration space)
        if "base_" in spec_path.name or spec_path.name.startswith("base"):
            pytest.skip(
                f"Base spec {spec_path.name} is meant for inheritance, no config space expected"
            )

        artifact = load_tvl_spec(spec_path=spec_path)

        # All examples should have at least one tunable parameter
        assert (
            len(artifact.configuration_space) > 0
        ), f"Spec {spec_path.name} has no configuration space parameters"


class TestTVLExamplesContent:
    """Test that TVL examples have expected content structure."""

    @pytest.mark.parametrize(
        "spec_path",
        TVL_SPECS,
        ids=[_get_spec_id(p) for p in TVL_SPECS],
    )
    def test_metadata_present(self, spec_path: Path) -> None:
        """Each spec should have metadata section."""
        artifact = load_tvl_spec(spec_path=spec_path)

        # Check for basic metadata
        assert artifact.metadata.get("spec_path") is not None

    @pytest.mark.parametrize(
        "spec_path",
        TVL_SPECS,
        ids=[_get_spec_id(p) for p in TVL_SPECS],
    )
    def test_constraints_compile(self, spec_path: Path) -> None:
        """Constraints in each spec should compile successfully."""
        # Load with constraint validation enabled (default)
        artifact = load_tvl_spec(spec_path=spec_path, validate_constraints=True)

        # All compiled constraints should be callable
        for constraint in artifact.constraints:
            assert callable(constraint)


class TestTVL09Features:
    """Test TVL 0.9 specific features in example specs."""

    def _find_specs_with_feature(self, feature_check) -> list[Path]:
        """Find specs that have a specific feature."""
        matching = []
        for spec_path in TVL_SPECS:
            try:
                artifact = load_tvl_spec(spec_path=spec_path)
                if feature_check(artifact):
                    matching.append(spec_path)
            except TVLValidationError:
                pass
        return matching

    def test_tvars_examples_exist(self) -> None:
        """At least one example uses TVL 0.9 tvars format."""
        specs_with_tvars = self._find_specs_with_feature(
            lambda a: a.tvars is not None and len(a.tvars) > 0
        )
        assert len(specs_with_tvars) > 0, "No examples use TVL 0.9 tvars format"

    def test_banded_objectives_examples_exist(self) -> None:
        """At least one example uses banded objectives."""
        specs_with_bands = self._find_specs_with_feature(
            lambda a: (
                a.objective_schema is not None
                and any(
                    obj.orientation == "band" for obj in a.objective_schema.objectives
                )
            )
        )
        assert len(specs_with_bands) > 0, "No examples use banded objectives"

    def test_promotion_policy_examples_exist(self) -> None:
        """At least one example defines a promotion policy."""
        specs_with_policy = self._find_specs_with_feature(
            lambda a: a.promotion_policy is not None
        )
        assert len(specs_with_policy) > 0, "No examples define a promotion policy"

    def test_structural_constraints_examples_exist(self) -> None:
        """At least one example uses structural constraints."""
        specs_with_structural = self._find_specs_with_feature(
            lambda a: len(a.constraints) > 0
        )
        assert len(specs_with_structural) > 0, "No examples use structural constraints"


class TestSpecificExamples:
    """Test specific example scenarios in detail."""

    def test_hello_tvl_loads(self) -> None:
        """The hello_tvl example should load correctly."""
        spec_path = EXAMPLES_DIR / "hello_tvl" / "hello_tvl.tvl.yml"
        if not spec_path.exists():
            pytest.skip("hello_tvl example not found")

        artifact = load_tvl_spec(spec_path=spec_path)
        assert "model" in artifact.configuration_space

    def test_banded_objectives_example_loads(self) -> None:
        """The banded_objectives example should load with multi-objective config."""
        spec_path = EXAMPLES_DIR / "banded_objectives" / "banded.tvl.yml"
        if not spec_path.exists():
            pytest.skip("banded_objectives example not found")

        artifact = load_tvl_spec(spec_path=spec_path)

        # Should have objective schema with multiple objectives
        assert artifact.objective_schema is not None
        assert (
            len(artifact.objective_schema.objectives) >= 2
        ), "Expected at least two objectives"

    def test_promotion_policy_example_loads(self) -> None:
        """The promotion_policy example should load with policy configuration."""
        spec_path = EXAMPLES_DIR / "promotion_policy" / "promotion.tvl.yml"
        if not spec_path.exists():
            pytest.skip("promotion_policy example not found")

        artifact = load_tvl_spec(spec_path=spec_path)

        # Should have promotion policy defined
        assert artifact.promotion_policy is not None

    def test_constraints_example_loads(self) -> None:
        """The constraints_units example should load with structural constraints."""
        spec_path = EXAMPLES_DIR / "constraints_units" / "constraints.tvl.yml"
        if not spec_path.exists():
            pytest.skip("constraints_units example not found")

        artifact = load_tvl_spec(spec_path=spec_path)

        # Should have constraints compiled
        assert len(artifact.constraints) > 0

    def test_environment_overlays_example_loads(self) -> None:
        """The environment_overlays example should load with different environments."""
        spec_path = EXAMPLES_DIR / "environment_overlays" / "overlays.tvl.yml"
        if not spec_path.exists():
            pytest.skip("environment_overlays example not found")

        # Load without environment
        artifact_base = load_tvl_spec(spec_path=spec_path)
        assert artifact_base.environment is None

        # Load with 'development' environment if it exists
        try:
            artifact_dev = load_tvl_spec(spec_path=spec_path, environment="development")
            assert artifact_dev.environment == "development"
        except TVLValidationError:
            # Environment might not be defined
            pass

    def test_banded_tost_complete_example_loads(self) -> None:
        """The banded_tost_complete example should demonstrate full banded syntax."""
        spec_path = (
            EXAMPLES_DIR
            / "tutorials"
            / "05_statistical_testing"
            / "banded_tost_complete.tvl.yml"
        )
        if not spec_path.exists():
            pytest.skip("banded_tost_complete example not found")

        artifact = load_tvl_spec(spec_path=spec_path)

        # Should have banded objectives
        assert artifact.objective_schema is not None
        band_objectives = [
            obj
            for obj in artifact.objective_schema.objectives
            if obj.orientation == "band"
        ]
        assert len(band_objectives) >= 2, "Expected at least two banded objectives"

        # Check band configurations
        for band_obj in band_objectives:
            assert band_obj.band is not None
            assert band_obj.band_test == "TOST"
            assert band_obj.band_alpha is not None


class TestTVLExamplesRuntimeOverrides:
    """Test that runtime_overrides() method works for all specs."""

    @pytest.mark.parametrize(
        "spec_path",
        TVL_SPECS,
        ids=[_get_spec_id(p) for p in TVL_SPECS],
    )
    def test_runtime_overrides_returns_dict(self, spec_path: Path) -> None:
        """runtime_overrides() should return a dict for all specs."""
        artifact = load_tvl_spec(spec_path=spec_path)
        overrides = artifact.runtime_overrides()

        assert isinstance(overrides, dict)


# Marker for integration tests
pytestmark = pytest.mark.integration
