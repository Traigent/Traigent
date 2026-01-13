"""Pytest configuration for doctests in the traigent package.

This conftest.py provides the necessary namespace fixtures to make
doctest examples executable. The doctests serve as both documentation
and executable tests.
"""

import pytest


@pytest.fixture(autouse=True)
def add_doctest_namespace(doctest_namespace):
    """Add traigent to the doctest namespace for all doctests.

    This fixture makes doctests that use `traigent.xxx()` syntax work
    by importing the traigent module into the doctest namespace.
    """
    import traigent
    from traigent import (
        Choices,
        ConfigSpace,
        IntRange,
        LogRange,
        Range,
        TraigentConfig,
        configure,
        get_config,
        get_current_config,
        get_optimization_insights,
        get_trial_config,
        get_version_info,
        initialize,
        optimize,
        override_config,
        set_strategy,
    )
    from traigent.api.constraints import (
        AndCondition,
        BoolExpr,
        Condition,
        Constraint,
        NotCondition,
        OrCondition,
        WhenBuilder,
        implies,
        require,
        when,
    )

    # Add main module
    doctest_namespace["traigent"] = traigent

    # Add commonly used classes and functions
    doctest_namespace["Range"] = Range
    doctest_namespace["IntRange"] = IntRange
    doctest_namespace["LogRange"] = LogRange
    doctest_namespace["Choices"] = Choices
    doctest_namespace["ConfigSpace"] = ConfigSpace
    doctest_namespace["TraigentConfig"] = TraigentConfig

    # Add constraint classes
    doctest_namespace["Constraint"] = Constraint
    doctest_namespace["Condition"] = Condition
    doctest_namespace["AndCondition"] = AndCondition
    doctest_namespace["OrCondition"] = OrCondition
    doctest_namespace["NotCondition"] = NotCondition
    doctest_namespace["BoolExpr"] = BoolExpr
    doctest_namespace["WhenBuilder"] = WhenBuilder
    doctest_namespace["when"] = when
    doctest_namespace["require"] = require
    doctest_namespace["implies"] = implies

    # Add API functions
    doctest_namespace["optimize"] = optimize
    doctest_namespace["configure"] = configure
    doctest_namespace["initialize"] = initialize
    doctest_namespace["get_config"] = get_config
    doctest_namespace["get_current_config"] = get_current_config
    doctest_namespace["get_trial_config"] = get_trial_config
    doctest_namespace["get_version_info"] = get_version_info
    doctest_namespace["get_optimization_insights"] = get_optimization_insights
    doctest_namespace["override_config"] = override_config
    doctest_namespace["set_strategy"] = set_strategy
