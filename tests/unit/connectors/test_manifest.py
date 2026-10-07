from traigent.connectors.manifest import (
    GuaranteeState,
    load_manifest,
    resolve_guarantee,
)


def test_guarantee_resolver_returns_degraded_or_unavailable_with_reason():
    manifest = load_manifest(
        {
            "schema_version": "1",
            "connector": "dummy",
            "version": "1.0",
            "operations": {
                "write": {
                    "idempotency": "emulated",
                    "conditional_update": "read_check",
                    "limits": {"max_page": 10},
                }
            },
        }
    )
    degraded = resolve_guarantee(
        manifest,
        "write",
        {"idempotency": "native", "conditional_update": "none"},
        "1.0",
        {"conditional_update": "atomic"},
    )
    unavailable = resolve_guarantee(
        manifest, "write", {"idempotency": "none"}, "2.0", {"idempotency": "native"}
    )
    permission_denied = resolve_guarantee(
        manifest,
        "write",
        {},
        "1.0",
    )
    malformed_probe = resolve_guarantee(
        manifest,
        "write",
        {"idempotency": "native", "conditional_update": "native"},
        "1.0",
    )
    assert degraded.state is GuaranteeState.DEGRADED
    assert degraded.reason == "permission_limited"
    assert unavailable.state is GuaranteeState.UNAVAILABLE
    assert unavailable.reason == "version_mismatch"
    assert permission_denied.state is GuaranteeState.UNAVAILABLE
    assert permission_denied.reason == "permission_limited"
    assert permission_denied.guarantees == {
        "idempotency": "none",
        "conditional_update": "none",
    }
    assert malformed_probe.guarantees == {
        "idempotency": "emulated",
        "conditional_update": "none",
    }
    assert set(GuaranteeState) == {
        GuaranteeState.AVAILABLE,
        GuaranteeState.DEGRADED,
        GuaranteeState.UNAVAILABLE,
    }


def test_manifest_is_closed_and_rejects_unknown_fields():
    try:
        load_manifest(
            {
                "schema_version": "1",
                "connector": "dummy",
                "version": "1",
                "surprise": True,
            }
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unknown manifest key accepted")
