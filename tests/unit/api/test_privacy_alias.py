from __future__ import annotations

import pytest

from traigent.api.decorators import optimize
from traigent.utils.exceptions import ConfigurationError


def test_decorator_privacy_alias_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TRAIGENT_ALLOW_LEGACY_CLOUD_EXECUTION_MODE", "1")

    with pytest.raises(ConfigurationError, match="fails closed"):

        @optimize(
            configuration_space={"x": [1]},
            objectives=["accuracy"],
            execution_mode="privacy",
        )
        def f(v: int) -> int:
            return v
