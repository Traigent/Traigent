"""#3194: ConfigSpace normalises constraints and validates its inputs."""

from __future__ import annotations

import pytest

from traigent.api.config_space import ConfigSpace, ConfigSpaceInputError
from traigent.api.constraints import Constraint, implies
from traigent.api.parameter_ranges import Choices, IntRange, Range
from traigent.utils.exceptions import ValidationError
from traigent_validation.validators import PythonConstraintValidator


def _k() -> IntRange:
    return IntRange(0, 10, name="k")


def test_bare_condition_validates():
    k = _k()
    cs = ConfigSpace.from_decorator_args(
        configuration_space={"k": k}, constraints=[k.gte(3)]
    )
    assert isinstance(cs.constraints[0], Constraint)
    assert cs.validate({"k": 5}).is_valid
    result = cs.validate({"k": 1})
    assert not result.is_valid and len(result.violations) == 1


def test_bare_and_condition_validates():
    k = _k()
    cs = ConfigSpace(tvars={"k": k}, constraints=[k.gte(3) & k.lte(8)])
    assert cs.validate({"k": 5}).is_valid
    assert not cs.validate({"k": 9}).is_valid


def test_bare_condition_check_satisfiability():
    k = _k()
    sat = ConfigSpace(tvars={"k": k}, constraints=[k.gte(3)]).check_satisfiability()
    assert sat.status.name == "SAT"
    unsat = ConfigSpace(
        tvars={"k": k}, constraints=[k.gte(10) & k.lte(2)]
    ).check_satisfiability()
    assert unsat.status.name == "UNSAT"


def test_control_implies_still_false():
    k = _k()
    m = Choices(["a", "b"], name="m")
    cs = ConfigSpace(
        tvars={"k": k, "m": m},
        constraints=[implies(m.equals("a"), k.gte(3))],
    )
    assert cs.validate({"k": 1, "m": "a"}).is_valid is False
    assert cs.validate({"k": 5, "m": "a"}).is_valid is True


def test_deep_chain_does_not_crash():
    k = _k()
    expr = k.gte(3)
    for _ in range(3000):
        expr = expr & k.gte(3)
    cs = ConfigSpace(tvars={"k": k}, constraints=[expr])
    result = cs.validate({"k": 5})  # must return, not raise AttributeError
    assert result.checked_count == 1


def test_validator_handler_survives_object_without_id():
    class Boom:
        def evaluate(self, config, var_names):
            raise RuntimeError("original problem")

    result = PythonConstraintValidator().validate_config({"k": 1}, [Boom()], {})
    assert not result.is_valid
    assert "original problem" in result.violations[0].message


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"tvars": None}, "tvars must be a mapping"),
        ({"tvars": {"a": 5}}, r"tvars\['a'\] must be a parameter range"),
        ({"tvars": {"a": Range(0.0, 1.0, name="b")}}, r"tvars\['a'\] has name 'b'"),
        ({"tvars": {1: Range(0.0, 1.0)}}, "keys must be strings"),
    ],
)
def test_bad_tvars_rejected(kwargs, match):
    with pytest.raises(ConfigSpaceInputError, match=match):
        ConfigSpace(**kwargs)


def test_duplicate_tvar_names_rejected():
    # same own name under the key matching only one of them is a mismatch;
    # two entries claiming one name is caught as a duplicate or mismatch.
    z = Range(0.0, 1.0, name="z")
    with pytest.raises(ConfigSpaceInputError):
        ConfigSpace(tvars={"z": z, "y": Range(0.0, 1.0, name="z")})


@pytest.mark.parametrize("bad", [5, None, "k >= 3", object()])
def test_bad_constraint_items_rejected(bad):
    k = _k()
    with pytest.raises(ConfigSpaceInputError, match=r"constraints\[1\]"):
        ConfigSpace(tvars={"k": k}, constraints=[k.gte(3), bad])


def test_bad_constraints_container_rejected():
    with pytest.raises(ConfigSpaceInputError, match="constraints must be a sequence"):
        ConfigSpace(tvars={"k": _k()}, constraints=5)  # type: ignore[arg-type]


def test_error_is_sdk_validation_error_and_legacy_types():
    err = None
    try:
        ConfigSpace(tvars=None)  # type: ignore[arg-type]
    except ConfigSpaceInputError as exc:
        err = exc
    assert isinstance(err, ValidationError)
    assert isinstance(err, TypeError) and isinstance(err, ValueError)


def test_error_does_not_echo_values():
    canary = "zq-acme-private-canary-91"
    with pytest.raises(ConfigSpaceInputError) as ei:
        ConfigSpace(tvars={"a": Range(0.0, 1.0, name="a")}, constraints=[canary])
    assert canary not in str(ei.value)
    with pytest.raises(ConfigSpaceInputError) as ei:
        ConfigSpace(tvars={"a": canary})  # type: ignore[dict-item]
    assert canary not in str(ei.value)


def test_decorator_path_keeps_key_authoritative_over_tvar_name():
    temp = Range(0.0, 2.0, name="wrong_name")
    cs = ConfigSpace.from_decorator_args(configuration_space={"temperature": temp})
    assert list(cs.tvars) == ["temperature"]


def test_tvars_type_checked_even_when_knobs_given():
    with pytest.raises(ConfigSpaceInputError, match="tvars must be a mapping"):
        ConfigSpace(tvars=42, knobs={})  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [5, 0, ""])
def test_from_decorator_args_rejects_malformed_constraints(bad):
    with pytest.raises(ConfigSpaceInputError, match="constraints must be a sequence"):
        ConfigSpace.from_decorator_args(
            configuration_space={"k": _k()},
            constraints=bad,  # type: ignore[arg-type]
        )
