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


def test_auto_override_warning_points_at_the_user_decorator_2927() -> None:
    """The warning is attributed to the user's @optimize line, not an SDK frame."""
    import warnings
    from pathlib import Path

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")

        @optimize(
            configuration_space=_CFG,
            objectives=["accuracy"],
            auto_override_frameworks=True,
        )
        def h(q):
            return q

    hits = [w for w in recorded if "framework_targets" in str(w.message)]
    assert hits, "expected the framework_targets warning"
    assert all(Path(w.filename).resolve() == Path(__file__).resolve() for w in hits), [
        w.filename for w in hits
    ]
