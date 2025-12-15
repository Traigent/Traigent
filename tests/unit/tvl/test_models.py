"""Unit tests for TVL 0.9 data models."""

import pytest

from traigent.tvl.models import (
    BandTarget,
    ChanceConstraint,
    DerivedConstraint,
    DomainSpec,
    PromotionPolicy,
    RegistryResolver,
    StructuralConstraint,
    TVarDecl,
    normalize_tvar_type,
    parse_domain_spec,
)


class TestDomainSpec:
    """Tests for DomainSpec data class."""

    def test_enum_domain_requires_values(self) -> None:
        """Enum domains must have non-empty values."""
        with pytest.raises(ValueError, match="requires non-empty values"):
            DomainSpec(kind="enum", values=None)

        with pytest.raises(ValueError, match="requires non-empty values"):
            DomainSpec(kind="enum", values=[])

    def test_enum_domain_valid(self) -> None:
        """Valid enum domain is created successfully."""
        domain = DomainSpec(kind="enum", values=["a", "b", "c"])
        assert domain.kind == "enum"
        assert domain.values == ["a", "b", "c"]

    def test_range_domain_requires_range(self) -> None:
        """Range domains must have a range tuple."""
        with pytest.raises(ValueError, match="requires a range tuple"):
            DomainSpec(kind="range", range=None)

    def test_range_domain_valid(self) -> None:
        """Valid range domain is created successfully."""
        domain = DomainSpec(kind="range", range=(0.0, 1.0), resolution=0.1)
        assert domain.kind == "range"
        assert domain.range == (0.0, 1.0)
        assert domain.resolution == 0.1

    def test_registry_domain_requires_registry(self) -> None:
        """Registry domains must have a registry identifier."""
        with pytest.raises(ValueError, match="requires a registry identifier"):
            DomainSpec(kind="registry", registry=None)

    def test_to_configuration_space_entry_enum(self) -> None:
        """Enum domain converts to list of values."""
        domain = DomainSpec(kind="enum", values=[1, 2, 3])
        assert domain.to_configuration_space_entry() == [1, 2, 3]

    def test_to_configuration_space_entry_range(self) -> None:
        """Range domain converts to (min, max) tuple."""
        domain = DomainSpec(kind="range", range=(0.0, 10.0))
        assert domain.to_configuration_space_entry() == (0.0, 10.0)


class TestTVarDecl:
    """Tests for TVarDecl data class."""

    def test_tvar_creation(self) -> None:
        """TVAR can be created with all fields."""
        domain = DomainSpec(kind="enum", values=["gpt-4", "gpt-3.5"])
        tvar = TVarDecl(
            name="model",
            type="enum",
            raw_type="enum[str]",
            domain=domain,
            default="gpt-4",
            unit=None,
        )
        assert tvar.name == "model"
        assert tvar.type == "enum"
        assert tvar.raw_type == "enum[str]"
        assert tvar.default == "gpt-4"

    def test_to_configuration_space_entry(self) -> None:
        """TVAR converts to configuration space format."""
        domain = DomainSpec(kind="range", range=(0.0, 2.0))
        tvar = TVarDecl(
            name="temperature",
            type="float",
            raw_type="float",
            domain=domain,
        )
        name, spec = tvar.to_configuration_space_entry()
        assert name == "temperature"
        assert spec == (0.0, 2.0)


class TestBandTarget:
    """Tests for BandTarget data class."""

    def test_band_from_interval(self) -> None:
        """Band can be created from low/high interval."""
        band = BandTarget(low=0.8, high=0.95)
        assert band.low == 0.8
        assert band.high == 0.95

    def test_band_from_center_tol(self) -> None:
        """Band can be created from center/tolerance."""
        band = BandTarget(center=0.9, tol=0.05)
        assert abs(band.low - 0.85) < 1e-10
        assert abs(band.high - 0.95) < 1e-10

    def test_band_requires_valid_spec(self) -> None:
        """Band requires either (low, high) or (center, tol)."""
        with pytest.raises(ValueError, match="requires either"):
            BandTarget()

    def test_band_invalid_bounds(self) -> None:
        """Band low must be less than high."""
        with pytest.raises(ValueError, match="must be less than high"):
            BandTarget(low=0.95, high=0.8)

    def test_contains(self) -> None:
        """contains() returns True if value is within band."""
        band = BandTarget(low=0.8, high=0.95)
        assert band.contains(0.85) is True
        assert band.contains(0.8) is True
        assert band.contains(0.95) is True
        assert band.contains(0.79) is False
        assert band.contains(0.96) is False

    def test_deviation(self) -> None:
        """deviation() returns distance to nearest bound."""
        band = BandTarget(low=0.8, high=0.95)
        assert abs(band.deviation(0.85) - 0.0) < 1e-10  # Inside
        assert abs(band.deviation(0.75) - 0.05) < 1e-10  # Below
        assert abs(band.deviation(1.0) - 0.05) < 1e-10  # Above

    def test_from_dict_list(self) -> None:
        """BandTarget.from_dict handles list format."""
        band = BandTarget.from_dict([0.8, 0.95])
        assert band.low == 0.8
        assert band.high == 0.95

    def test_from_dict_center_tol(self) -> None:
        """BandTarget.from_dict handles center/tol dict."""
        band = BandTarget.from_dict({"center": 0.9, "tol": 0.05})
        assert abs(band.low - 0.85) < 1e-10
        assert abs(band.high - 0.95) < 1e-10


class TestChanceConstraint:
    """Tests for ChanceConstraint data class."""

    def test_valid_constraint(self) -> None:
        """Valid chance constraint is created."""
        cc = ChanceConstraint(name="accuracy", threshold=0.9, confidence=0.95)
        assert cc.name == "accuracy"
        assert cc.threshold == 0.9
        assert cc.confidence == 0.95

    def test_invalid_confidence(self) -> None:
        """Confidence must be in (0, 1]."""
        with pytest.raises(ValueError, match="must be in"):
            ChanceConstraint(name="accuracy", threshold=0.9, confidence=0.0)

        with pytest.raises(ValueError, match="must be in"):
            ChanceConstraint(name="accuracy", threshold=0.9, confidence=1.5)

    def test_from_dict(self) -> None:
        """ChanceConstraint.from_dict works correctly."""
        cc = ChanceConstraint.from_dict(
            {"name": "latency", "threshold": 100.0, "confidence": 0.9}
        )
        assert cc.name == "latency"
        assert cc.threshold == 100.0
        assert cc.confidence == 0.9


class TestPromotionPolicy:
    """Tests for PromotionPolicy data class."""

    def test_default_policy(self) -> None:
        """Default promotion policy has sensible defaults."""
        policy = PromotionPolicy()
        assert policy.dominance == "epsilon_pareto"
        assert policy.alpha == 0.05
        assert policy.adjust == "none"
        assert policy.min_effect == {}

    def test_invalid_alpha(self) -> None:
        """Alpha must be in (0, 1)."""
        with pytest.raises(ValueError, match="must be in"):
            PromotionPolicy(alpha=0.0)

        with pytest.raises(ValueError, match="must be in"):
            PromotionPolicy(alpha=1.0)

    def test_negative_epsilon(self) -> None:
        """min_effect values must be non-negative."""
        with pytest.raises(ValueError, match="must be non-negative"):
            PromotionPolicy(min_effect={"accuracy": -0.01})

    def test_get_epsilon(self) -> None:
        """get_epsilon returns configured or default value."""
        policy = PromotionPolicy(min_effect={"accuracy": 0.01, "latency": 5.0})
        assert policy.get_epsilon("accuracy") == 0.01
        assert policy.get_epsilon("latency") == 5.0
        assert policy.get_epsilon("unknown") == 0.0
        assert policy.get_epsilon("unknown", default=0.1) == 0.1

    def test_from_dict(self) -> None:
        """PromotionPolicy.from_dict works correctly."""
        policy = PromotionPolicy.from_dict(
            {
                "dominance": "epsilon_pareto",
                "alpha": 0.10,
                "min_effect": {"accuracy": 0.02},
                "adjust": "BH",
                "chance_constraints": [
                    {"name": "accuracy", "threshold": 0.85, "confidence": 0.95}
                ],
            }
        )
        assert policy.alpha == 0.10
        assert policy.min_effect == {"accuracy": 0.02}
        assert policy.adjust == "BH"
        assert len(policy.chance_constraints) == 1


class TestStructuralConstraint:
    """Tests for StructuralConstraint data class."""

    def test_standalone_expr(self) -> None:
        """Standalone expression constraint."""
        constraint = StructuralConstraint(expr="max_tokens >= 256")
        assert constraint.expr == "max_tokens >= 256"
        assert constraint.when is None
        assert constraint.then is None

    def test_implication(self) -> None:
        """When/then implication constraint."""
        constraint = StructuralConstraint(
            when="model = 'gpt-4'", then="temperature <= 0.7"
        )
        assert constraint.when == "model = 'gpt-4'"
        assert constraint.then == "temperature <= 0.7"

    def test_requires_valid_form(self) -> None:
        """Constraint requires either expr or when/then."""
        with pytest.raises(ValueError, match="requires either"):
            StructuralConstraint()

    def test_cannot_have_both(self) -> None:
        """Constraint cannot have both expr and when/then."""
        with pytest.raises(ValueError, match="cannot have both"):
            StructuralConstraint(
                expr="x > 0", when="model = 'gpt-4'", then="temp <= 0.7"
            )

    def test_to_rule_expression_standalone(self) -> None:
        """Standalone expr converts directly."""
        constraint = StructuralConstraint(expr="x > 0")
        assert constraint.to_rule_expression() == "x > 0"

    def test_to_rule_expression_implication(self) -> None:
        """Implication converts to disjunction."""
        constraint = StructuralConstraint(when="a", then="b")
        assert constraint.to_rule_expression() == "not (a) or (b)"


class TestNormalizeTvarType:
    """Tests for normalize_tvar_type function."""

    def test_basic_types(self) -> None:
        """Basic types are normalized correctly."""
        assert normalize_tvar_type("bool") == "bool"
        assert normalize_tvar_type("boolean") == "bool"
        assert normalize_tvar_type("int") == "int"
        assert normalize_tvar_type("integer") == "int"
        assert normalize_tvar_type("float") == "float"
        assert normalize_tvar_type("number") == "float"
        assert normalize_tvar_type("continuous") == "float"

    def test_enum_types(self) -> None:
        """Enum types with subtypes are normalized."""
        assert normalize_tvar_type("enum[str]") == "enum"
        assert normalize_tvar_type("ENUM[int]") == "enum"

    def test_tuple_types(self) -> None:
        """Tuple types are normalized."""
        assert normalize_tvar_type("tuple[int, float]") == "tuple"

    def test_callable_types(self) -> None:
        """Callable types are normalized."""
        assert normalize_tvar_type("callable[ScorerProto]") == "callable"

    def test_unknown_types(self) -> None:
        """Unknown types return None."""
        assert normalize_tvar_type("unknown") is None
        assert normalize_tvar_type("complex") is None


class TestParseDomainSpec:
    """Tests for parse_domain_spec function."""

    def test_bool_implicit_domain(self) -> None:
        """Bool type has implicit [True, False] domain."""
        domain = parse_domain_spec("flag", "bool", None)
        assert domain.kind == "enum"
        assert domain.values == [True, False]

    def test_list_domain(self) -> None:
        """List is converted to enum domain."""
        domain = parse_domain_spec("model", "enum", ["gpt-4", "gpt-3.5"])
        assert domain.kind == "enum"
        assert domain.values == ["gpt-4", "gpt-3.5"]

    def test_range_domain(self) -> None:
        """Range dict is converted to range domain."""
        domain = parse_domain_spec("temp", "float", {"range": [0.0, 2.0]})
        assert domain.kind == "range"
        assert domain.range == (0.0, 2.0)

    def test_range_with_resolution(self) -> None:
        """Range with resolution is preserved."""
        domain = parse_domain_spec(
            "temp", "float", {"range": [0.0, 2.0], "resolution": 0.1}
        )
        assert domain.kind == "range"
        assert domain.resolution == 0.1

    def test_set_domain(self) -> None:
        """Set wrapper is converted to set domain."""
        domain = parse_domain_spec("values", "enum", {"set": [1, 2, 3]})
        assert domain.kind == "set"
        assert domain.values == [1, 2, 3]

    def test_registry_domain(self) -> None:
        """Registry reference is converted to registry domain."""
        domain = parse_domain_spec(
            "scorer", "callable", {"registry": "scorers", "filter": "version >= 2"}
        )
        assert domain.kind == "registry"
        assert domain.registry == "scorers"
        assert domain.filter == "version >= 2"

    def test_invalid_domain(self) -> None:
        """Invalid domain raises ValueError."""
        with pytest.raises(ValueError, match="invalid domain specification"):
            parse_domain_spec("x", "int", 42)


class TestDerivedConstraint:
    """Tests for DerivedConstraint data class."""

    def test_valid_constraint(self) -> None:
        """Valid derived constraint is created."""
        constraint = DerivedConstraint(require="env.budget >= 1000")
        assert constraint.require == "env.budget >= 1000"
        assert constraint.index == 0
        assert constraint.description is None

    def test_constraint_with_description(self) -> None:
        """Derived constraint with description."""
        constraint = DerivedConstraint(
            require="env.price_per_token * max_tokens <= env.budget",
            index=5,
            description="Budget constraint",
        )
        assert constraint.require == "env.price_per_token * max_tokens <= env.budget"
        assert constraint.index == 5
        assert constraint.description == "Budget constraint"

    def test_empty_require_raises(self) -> None:
        """Empty require expression raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DerivedConstraint(require="")

        with pytest.raises(ValueError, match="cannot be empty"):
            DerivedConstraint(require="   ")

    def test_from_dict(self) -> None:
        """DerivedConstraint.from_dict works correctly."""
        constraint = DerivedConstraint.from_dict(
            {"require": "env.quota > 0", "description": "Quota available"},
            index=3,
        )
        assert constraint.require == "env.quota > 0"
        assert constraint.index == 3
        assert constraint.description == "Quota available"

    def test_from_dict_missing_require(self) -> None:
        """DerivedConstraint.from_dict raises on missing require."""
        with pytest.raises(ValueError, match="requires a 'require' string"):
            DerivedConstraint.from_dict({}, index=0)


class TestRegistryResolver:
    """Tests for RegistryResolver protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """RegistryResolver is a runtime-checkable protocol."""

        # Create a minimal implementation
        class MockResolver:
            def resolve(
                self,
                registry_id: str,
                filter_expr: str | None = None,
                version: str | None = None,
            ) -> list:
                return ["model_a", "model_b"]

        resolver = MockResolver()
        assert isinstance(resolver, RegistryResolver)

    def test_non_conforming_class_fails_check(self) -> None:
        """Non-conforming class fails isinstance check."""

        class NotAResolver:
            def do_something(self) -> None:
                """Intentionally empty - this class doesn't implement resolve()."""

        not_resolver = NotAResolver()
        assert not isinstance(not_resolver, RegistryResolver)

    def test_resolver_implementation(self) -> None:
        """Test a complete resolver implementation."""

        class TestResolver:
            def __init__(self, data: dict[str, list]):
                self._data = data

            def resolve(
                self,
                registry_id: str,
                filter_expr: str | None = None,
                version: str | None = None,
            ) -> list:
                values = self._data.get(registry_id, [])
                if version:
                    values = [v for v in values if version in str(v)]
                return values

        resolver = TestResolver(
            {
                "models": ["gpt-4", "gpt-3.5", "claude-3"],
                "scorers": ["accuracy_v1", "accuracy_v2", "f1_v1"],
            }
        )

        assert isinstance(resolver, RegistryResolver)
        assert resolver.resolve("models") == ["gpt-4", "gpt-3.5", "claude-3"]
        assert resolver.resolve("models", version="gpt") == ["gpt-4", "gpt-3.5"]
        assert resolver.resolve("unknown") == []
