"""Tests for surfacing the single-value configuration-space warning (issue #2021).

``Validators.validate_configuration_space`` already flags a parameter with a
single value ("no optimization possible"), but the ``@traigent.optimize``
decorator only called ``raise_if_invalid()``, which raises on errors and drops
warnings on the floor. The declaration therefore looked accepted while the run
could not possibly compare anything.
"""

import warnings

import pytest

from traigent.api.decorators import optimize
from traigent.utils.exceptions import TraigentWarning

_SINGLE_VALUE_MESSAGE = "Single value in list - no optimization possible"


def test_single_value_config_space_warns_on_decorator():
    """A one-value knob must reach the user at decoration time."""
    with pytest.warns(TraigentWarning, match=_SINGLE_VALUE_MESSAGE) as record:

        @optimize(
            configuration_space={"temperature": [0.7]},
            objectives=["accuracy"],
            algorithm="grid",
        )
        def generate(question):
            return "x"

    messages = [str(w.message) for w in record]
    assert any("configuration_space.temperature" in m for m in messages)
    assert any("Add more values or remove this parameter" in m for m in messages)


def test_single_value_config_space_warns_once():
    """Internal re-validation must not repeat the same warning at declaration."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")

        @optimize(
            configuration_space={"temperature": [0.7]},
            objectives=["accuracy"],
            algorithm="grid",
        )
        def generate(question):
            return "x"

    single_value = [w for w in record if _SINGLE_VALUE_MESSAGE in str(w.message)]
    assert len(single_value) == 1


def test_real_config_space_does_not_warn():
    """A knob with two or more values is a real search - stay quiet."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")

        @optimize(
            configuration_space={"temperature": [0.0, 0.7]},
            objectives=["accuracy"],
            algorithm="grid",
        )
        def generate(question):
            return "x"

    assert [w for w in record if _SINGLE_VALUE_MESSAGE in str(w.message)] == []
