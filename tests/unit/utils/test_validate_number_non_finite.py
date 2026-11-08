"""validate_number and TraigentConfig reject NaN, inf and bool."""

from __future__ import annotations

import pytest

from traigent.config.types import TraigentConfig
from traigent.utils.validation import Validators

NAN = float("nan")
INF = float("inf")

# --- validate_number / TraigentConfig ---------------------------------------


@pytest.mark.parametrize("bad", [NAN, INF, -INF, True, False])
def test_validate_number_rejects_nan_inf_bool(bad):
    result = Validators.validate_number(bad, "x", 0.0, 1.0)
    assert not result.is_valid
    assert "x" in str(result.errors[0].field)


def test_validate_number_unbounded_still_rejects_nan_bool():
    assert not Validators.validate_number(NAN, "x", min_value=0.0).is_valid
    assert not Validators.validate_number(INF, "x", min_value=0.0).is_valid
    assert not Validators.validate_number(True, "x", 0, 1).is_valid


@pytest.mark.parametrize("value", [0, 0.5, 1, 1.0])
def test_validate_number_accepts_finite(value):
    assert Validators.validate_number(value, "x", 0.0, 1.0).is_valid


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": NAN},
        {"top_p": NAN},
        {"temperature": True},
        {"frequency_penalty": NAN},
        {"presence_penalty": NAN},
        {"top_p": True},
    ],
)
def test_traigent_config_rejects_nan_and_bool_sampling(kwargs):
    with pytest.raises(Exception) as exc:
        TraigentConfig(**kwargs)
    assert next(iter(kwargs)) in str(exc.value)
