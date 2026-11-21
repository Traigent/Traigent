"""route_shape: the one heuristic that turns a path into a loggable shape."""

from __future__ import annotations

import pytest

from traigent.egress.log import route_shape


@pytest.mark.parametrize(
    ("path", "shape"),
    [
        ("/api/v1/projects", "/api/v1/projects"),
        ("/api/v1beta/projects/", "/api/v1beta/projects/"),
        (
            "/api/v2/sessions/3f2b8c1e-aaaa-bbbb-cccc-123456789abc/trials",
            "/api/v2/sessions/{}/trials",
        ),
        ("/api/v1/projects/12345", "/api/v1/projects/{}"),
        ("/api/v1/projects/Alice_has_diabetes", "/api/v1/projects/{}"),
        ("/api/v1/projects/MixedCase", "/api/v1/projects/{}"),
        ("/api/v1/projects/abc123", "/api/v1/projects/{}"),
        ("/api/v1/projects/a" + "b" * 80, "/api/v1/projects/{}"),
        ("/api/v1/projects/%61lice", "/api/v1/projects/{}"),
        ("/api/best-config/head", "/api/best-config/head"),
        ("/api/v1/things?token=abc#frag", "/api/v1/things"),
        ("/", "/"),
        ("", "/"),
        ("/v1/x_y/v10alpha", "/v1/x_y/v10alpha"),
        ("/vX/1v", "/{}/{}"),
    ],
)
def test_route_shape(path, shape):
    assert route_shape(path) == shape
