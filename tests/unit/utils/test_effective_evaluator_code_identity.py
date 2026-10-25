"""``effective_evaluator_code_identity``: identify the scorer that actually runs.

Separate from legacy fp1 (which must stay byte-identical, pinned by a golden).
"""

from __future__ import annotations

import hashlib
import json
import re

from traigent.utils import artifact_fingerprints as af
from traigent.utils.artifact_fingerprints import (
    compute_evaluator_fingerprint,
    effective_evaluator_code_identity,
)

WIRE = re.compile(r"^sha256:[0-9a-f]{64}$")


def score_a(output, expected):
    return 1.0 if output == expected else 0.0


def score_b(output, expected):
    return 0.5


def score_c(output, expected):
    return 0.25


def score_c_edited(output, expected):
    return 0.26


def custom_one(func, config, example):
    return None


class _Instance:
    def __call__(self, output, expected):
        return 1.0


def _digest(**kwargs):
    ident = effective_evaluator_code_identity(**kwargs)
    assert ident["origin"] == "code", ident
    assert WIRE.match(ident["digest"])
    return ident["digest"]


def test_custom_alone_is_code() -> None:
    assert _digest(custom_evaluator=custom_one, objectives=["accuracy"])


def test_custom_plus_hybrid_is_code() -> None:
    ident = effective_evaluator_code_identity(
        custom_evaluator=custom_one,
        objectives=["accuracy"],
        external={"kind": "hybrid_api", "endpoint": "https://x"},
    )
    assert ident["origin"] == "code"
    assert ident["digest"] == _digest(custom_evaluator=custom_one, objectives=[])


def test_scoring_function_plus_hybrid_is_external_without_digest() -> None:
    ident = effective_evaluator_code_identity(
        scoring_function=score_a,
        objectives=["accuracy"],
        external={"kind": "hybrid_api", "endpoint": "https://x"},
    )
    assert ident == {"origin": "external", "digest": None}


def test_metric_functions_plus_hybrid_is_external() -> None:
    ident = effective_evaluator_code_identity(
        metric_functions={"accuracy": score_a},
        objectives=["accuracy"],
        external={"kind": "hybrid_api"},
    )
    assert ident == {"origin": "external", "digest": None}


def test_external_service_evaluator_plus_scoring_function_is_external() -> None:
    ident = effective_evaluator_code_identity(
        scoring_function=score_a, objectives=["accuracy"], external=object()
    )
    assert ident["origin"] == "external" and ident["digest"] is None


def test_scoring_function_alone_is_code() -> None:
    assert _digest(scoring_function=score_a, objectives=["accuracy"])


def test_metrics_only_is_builtin_without_digest() -> None:
    ident = effective_evaluator_code_identity(objectives=["accuracy", "cost"])
    assert ident == {"origin": "builtin", "digest": None}
    # the fp1 "none" sentinel is never emitted
    assert hashlib.sha256(b"none").hexdigest() not in repr(ident)


def test_unreadable_instance_is_unavailable() -> None:
    ident = effective_evaluator_code_identity(
        scoring_function=_Instance(), objectives=["accuracy"]
    )
    assert ident == {"origin": "unavailable", "digest": None}
    ident = effective_evaluator_code_identity(
        custom_evaluator=len, objectives=["accuracy"]
    )
    assert ident["origin"] == "unavailable"


def test_scoring_function_shadowed_by_metric_functions_names_the_running_code() -> None:
    combined = _digest(
        scoring_function=score_a,
        metric_functions={"accuracy": score_b},
        objectives=["accuracy"],
    )
    only_b = _digest(metric_functions={"accuracy": score_b}, objectives=["accuracy"])
    only_a = _digest(scoring_function=score_a, objectives=["accuracy"])
    assert combined == only_b
    assert combined != only_a


def test_combined_map_changes_when_other_member_changes() -> None:
    base = {"scoring_function": score_a, "objectives": ["accuracy"]}
    c = _digest(metric_functions={"latency": score_c}, **base)
    c2 = _digest(metric_functions={"latency": score_c_edited}, **base)
    assert c != c2
    assert c != _digest(**base)


def test_swapping_function_to_metric_binding_changes_digest() -> None:
    one = _digest(
        metric_functions={"accuracy": score_a, "latency": score_b},
        objectives=["accuracy", "latency"],
    )
    two = _digest(
        metric_functions={"accuracy": score_b, "latency": score_a},
        objectives=["accuracy", "latency"],
    )
    assert one != two


def test_dict_order_does_not_change_digest() -> None:
    one = _digest(
        metric_functions={"accuracy": score_a, "latency": score_b}, objectives=[]
    )
    two = _digest(
        metric_functions={"latency": score_b, "accuracy": score_a}, objectives=[]
    )
    assert one == two


def test_runtime_override_changes_digest() -> None:
    assert _digest(custom_evaluator=custom_one, objectives=[]) != _digest(
        scoring_function=score_a, objectives=["accuracy"]
    )


def test_digest_is_domain_separated_from_fp1() -> None:
    import inspect

    fp1 = compute_evaluator_fingerprint(scoring_function=score_a)
    assert fp1 is not None
    ident = _digest(scoring_function=score_a, objectives=["accuracy"])
    assert ident != "sha256:" + fp1.removeprefix("fp1:")
    raw = hashlib.sha256(inspect.getsource(score_a).encode()).hexdigest()
    assert ident != f"sha256:{raw}"


def test_fp1_output_is_byte_identical_golden() -> None:
    """fp1 must not move: it keys history (artifact_versions.evaluator)."""
    import inspect

    src = inspect.getsource(score_a)
    expected = "fp1:" + hashlib.sha256(src.encode("utf-8")).hexdigest()
    assert compute_evaluator_fingerprint(scoring_function=score_a) == expected
    assert compute_evaluator_fingerprint(custom_evaluator=score_a) == expected
    sentinel = "fp1:140bedbf9c3f6d56a9846d2ba7088798683f4da0c248231336e6a05679e4fdfe"
    assert compute_evaluator_fingerprint() == sentinel
    joined = inspect.getsource(score_a) + inspect.getsource(score_b)
    assert (
        compute_evaluator_fingerprint(metric_functions={"b": score_b, "a": score_a})
        == "fp1:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()
    )


def test_helper_calls_pipeline_build_metric_functions(monkeypatch) -> None:
    import traigent.core.optimization_pipeline as pipeline

    seen: list[tuple] = []
    real = pipeline.build_metric_functions

    def spy(metric_functions, scoring_function, objectives):
        seen.append((metric_functions, scoring_function, list(objectives)))
        return real(metric_functions, scoring_function, objectives)

    monkeypatch.setattr(pipeline, "build_metric_functions", spy)
    _digest(scoring_function=score_a, objectives=["accuracy"])
    assert seen == [(None, score_a, ["accuracy"])]


def test_helper_reads_no_instance_state() -> None:
    import inspect

    source = inspect.getsource(af.effective_evaluator_code_identity)
    assert "self." not in source
    assert "custom_evaluator" in source  # parameter only


def _hand_digest(kind: str, members: list[list[str]]) -> str:
    """Independent ecd1 preimage, written out literally from the spec."""
    preimage = json.dumps(
        {"v": "ecd1", "kind": kind, "members": members},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(preimage).hexdigest()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_ecd1_custom_vector_matches_hand_built_preimage() -> None:
    import inspect

    expected = _hand_digest("custom", [["", _sha(inspect.getsource(custom_one))]])
    assert _digest(custom_evaluator=custom_one, objectives=[]) == expected


def test_ecd1_metric_map_vector_matches_hand_built_preimage() -> None:
    import inspect

    expected = _hand_digest(
        "metric_map",
        [
            ["accuracy", _sha(inspect.getsource(score_a))],
            ["latency", _sha(inspect.getsource(score_b))],
        ],
    )
    got = _digest(
        metric_functions={"latency": score_b, "accuracy": score_a},
        objectives=["accuracy", "latency"],
    )
    assert got == expected


def test_ecd1_fixed_literal_vector() -> None:
    """A pinned literal: any change to the preimage form changes this."""
    members = [["accuracy", "11" * 32]]
    assert _hand_digest("metric_map", members) == (
        "sha256:"
        + hashlib.sha256(
            b'{"kind":"metric_map","members":[["accuracy","'
            + b"11" * 32
            + b'"]],"v":"ecd1"}'
        ).hexdigest()
    )
