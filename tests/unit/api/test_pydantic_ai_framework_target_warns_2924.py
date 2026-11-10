"""framework_targets=['pydantic_ai.Agent'] warns that tuned values are not injected (#2924)."""

from __future__ import annotations

import pytest

from traigent.api.decorators import optimize
from traigent.utils.exceptions import TraigentWarning

_CFG = {"temperature": [0.1, 0.2]}


def test_pydantic_ai_agent_target_warns_that_nothing_is_injected() -> None:
    with pytest.warns(TraigentWarning, match="pydantic_ai") as record:

        @optimize(
            configuration_space=_CFG,
            objectives=["accuracy"],
            auto_override_frameworks=True,
            framework_targets=["pydantic_ai.Agent"],
        )
        def f(q):
            return q

    messages = [str(w.message) for w in record if "pydantic_ai" in str(w.message)]
    assert len(messages) == 1
    message = messages[0]
    assert "not injected" in message
    assert "model_settings" in message
    assert "traigent.get_trial_config()" in message


def test_other_framework_targets_do_not_get_the_pydantic_ai_warning(recwarn) -> None:
    @optimize(
        configuration_space=_CFG,
        objectives=["accuracy"],
        auto_override_frameworks=True,
        framework_targets=["openai.OpenAI"],
    )
    def g(q):
        return q

    assert not [w for w in recwarn if "pydantic_ai" in str(w.message)]


def test_pydantic_ai_warning_points_at_the_user_decorator() -> None:
    """The warning is attributed to the user's @optimize line, not an SDK frame."""
    from pathlib import Path

    with pytest.warns(TraigentWarning, match="pydantic_ai") as record:

        @optimize(
            configuration_space=_CFG,
            objectives=["accuracy"],
            auto_override_frameworks=True,
            framework_targets=["pydantic_ai.Agent"],
        )
        def h(q):
            return q

    hits = [w for w in record if "pydantic_ai" in str(w.message)]
    assert [Path(w.filename).resolve() for w in hits] == [Path(__file__).resolve()]
