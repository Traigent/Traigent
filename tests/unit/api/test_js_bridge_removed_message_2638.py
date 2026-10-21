"""The JS bridge removal error must not name an unpublished npm package (#2638)."""

import pytest

import traigent


def test_js_bridge_error_does_not_name_unpublished_npm_package() -> None:
    with pytest.raises(TypeError) as excinfo:

        @traigent.optimize(
            configuration_space={"m": ["a"]}, execution={"runtime": "node"}
        )
        def f(x):  # pragma: no cover - decoration itself raises
            return x

    message = str(excinfo.value)
    assert "traigent-js" not in message
    assert "not published to npm yet" in message
    assert "docs/guides/js-bridge.md" in message
