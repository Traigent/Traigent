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


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda f: optimize(
                configuration_space=_CFG,
                injection=__import__(
                    "traigent.api.decorators", fromlist=["InjectionOptions"]
                ).InjectionOptions(
                    auto_override_frameworks=True, framework_targets=["OpenAI"]
                ),
            )(f),
            id="InjectionOptions",
        ),
        pytest.param(
            lambda f: optimize(
                configuration_space=_CFG,
                injection={
                    "auto_override_frameworks": True,
                    "framework_targets": ["OpenAI"],
                },
            )(f),
            id="injection-dict",
        ),
        pytest.param(
            lambda f: __import__(
                "traigent.core.optimized_function", fromlist=["OptimizedFunction"]
            ).OptimizedFunction(
                f,
                configuration_space=_CFG,
                auto_override_frameworks=True,
                framework_targets=["OpenAI"],
            ),
            id="OptimizedFunction",
        ),
    ],
)
def test_bare_framework_target_is_a_configuration_error_on_every_route(
    build,
) -> None:
    """A bare name raises ConfigurationError at construction on every route.

    The flat ``framework_targets=`` keyword was checked at decoration, but the
    InjectionOptions, injection-dict and direct-constructor routes were not, so
    the error surfaced at run time as an OptimizationError.
    """
    from traigent.utils.exceptions import ConfigurationError

    def f(q):
        return q

    with pytest.raises(ConfigurationError, match="dotted 'module.ClassName'"):
        build(f)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"auto_override_frameworks": True}, "framework_targets"),
        (
            {
                "auto_override_frameworks": True,
                "framework_targets": ["pydantic_ai.Agent"],
            },
            "pydantic_ai",
        ),
    ],
    ids=["no-targets", "pydantic-ai"],
)
def test_framework_warnings_point_at_a_direct_constructor_call(kwargs, match) -> None:
    """Built directly, OptimizedFunction warns at the caller's line, not sys:1."""
    import warnings
    from pathlib import Path

    from traigent.core.optimized_function import OptimizedFunction

    def f(q):
        return q

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        OptimizedFunction(f, configuration_space=_CFG, **kwargs)

    hits = [
        w
        for w in record
        if match in str(w.message) and issubclass(w.category, TraigentWarning)
    ]
    assert hits
    assert {Path(w.filename).resolve() for w in hits} == {Path(__file__).resolve()}
