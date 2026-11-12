"""The best-config publish projector is default-deny (#3100)."""

from __future__ import annotations

import pytest

from traigent.utils import optimization_only_egress as oo

_SPEC = {
    "schema_version": "traigent.best_config.v1",
    "config_id": "cfg-1",
    "config": {"temperature": 0.2},
}


def test_unknown_top_level_members_are_dropped() -> None:
    out = oo.project_best_config_payload(
        {"spec": dict(_SPEC), "environment": "prod", "note": "free text here"}
    )
    assert set(out) == {"spec", "environment"}


def test_free_text_forward_compat_is_refused() -> None:
    spec = {**_SPEC, "forward_compat": "some free text with spaces"}
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_best_config_payload({"spec": spec})


@pytest.mark.parametrize("value", [1, "true", ["x"], {"a": 1}])
def test_non_boolean_forward_compat_is_refused(value: object) -> None:
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_best_config_payload({"spec": {**_SPEC, "forward_compat": value}})


@pytest.mark.parametrize("value", [True, False])
def test_boolean_forward_compat_passes(value: bool) -> None:
    out = oo.project_best_config_payload({"spec": {**_SPEC, "forward_compat": value}})
    assert out["spec"]["forward_compat"] is value
