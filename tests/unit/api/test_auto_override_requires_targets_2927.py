"""auto_override_frameworks=True without framework_targets overrides nothing (#2927)."""

import pytest

from traigent.api.decorators import optimize
from traigent.utils.exceptions import TraigentWarning

_CFG = {"temperature": [0.1, 0.2]}


def test_auto_override_without_targets_warns() -> None:
    with pytest.warns(TraigentWarning, match="framework_targets"):

        @optimize(
            configuration_space=_CFG,
            objectives=["accuracy"],
            auto_override_frameworks=True,
        )
        def f(q):
            return q


def test_auto_override_with_targets_is_silent(recwarn) -> None:
    @optimize(
        configuration_space=_CFG,
        objectives=["accuracy"],
        auto_override_frameworks=True,
        framework_targets=["openai.OpenAI"],
    )
    def g(q):
        return q

    assert not [w for w in recwarn if "framework_targets" in str(w.message)]
