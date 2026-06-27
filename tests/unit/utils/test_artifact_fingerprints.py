from __future__ import annotations

import hashlib
import inspect
import json

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils import artifact_fingerprints as af


def _example(input_data, expected_output):
    return EvaluationExample(input_data=input_data, expected_output=expected_output)


def _fp_text(value: str) -> str:
    return f"fp1:{hashlib.sha256(value.encode()).hexdigest()}"


def _fp_json(value) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"fp1:{hashlib.sha256(canonical).hexdigest()}"


def test_dataset_fingerprint_is_order_independent_and_changes_on_content() -> None:
    first = _example({"question": "a"}, "answer-a")
    second = _example({"question": "b"}, "answer-b")

    original = af.compute_dataset_fingerprint([first, second])
    reordered = af.compute_dataset_fingerprint([second, first])
    edited = af.compute_dataset_fingerprint([first, _example({"question": "b"}, "x")])
    added = af.compute_dataset_fingerprint(
        [first, second, _example({"question": "c"}, "answer-c")]
    )

    assert original is not None
    assert original == reordered
    assert original != edited
    assert original != added
    assert af.compute_dataset_fingerprint([]) is None
    assert af.compute_dataset_fingerprint(None) is None


def test_agent_fingerprint_uses_source() -> None:
    def sample_agent(prompt: str) -> str:
        return prompt.upper()

    expected = _fp_text(inspect.getsource(sample_agent))

    assert af.compute_agent_fingerprint(sample_agent) == expected


def test_agent_fingerprint_falls_back_to_identity_and_meta(monkeypatch) -> None:
    def sample_agent(prompt: str) -> str:
        return prompt

    monkeypatch.setattr(
        af.inspect, "getsource", lambda _func: (_ for _ in ()).throw(OSError)
    )

    payload = af.build_artifact_fingerprints(
        dataset=[_example({"question": "a"}, "answer-a")],
        func=sample_agent,
        configuration_space={"temperature": [0.1]},
    )

    assert payload["artifact_fingerprints"]["agent"] == _fp_text(
        f"{sample_agent.__module__}:{sample_agent.__qualname__}"
    )
    assert payload["fingerprint_meta"]["source_available"] is False


def test_evaluator_fingerprint_variants() -> None:
    def custom_evaluator() -> float:
        return 1.0

    def scoring_function() -> float:
        return 0.5

    def metric_a() -> float:
        return 0.1

    def metric_b() -> float:
        return 0.2

    assert af.compute_evaluator_fingerprint(custom_evaluator=custom_evaluator) == (
        _fp_text(inspect.getsource(custom_evaluator))
    )
    assert af.compute_evaluator_fingerprint(scoring_function=scoring_function) == (
        _fp_text(inspect.getsource(scoring_function))
    )
    metric_fp = af.compute_evaluator_fingerprint(
        metric_functions={"b": metric_b, "a": metric_a}
    )
    assert metric_fp == _fp_text(
        inspect.getsource(metric_a) + inspect.getsource(metric_b)
    )
    assert af.compute_evaluator_fingerprint(
        external={
            "kind": "hybrid_api",
            "endpoint": "https://user:secret@example.test:8443/evaluate?token=x#frag",
        }
    ) == _fp_text("hybrid_api:https://example.test:8443/evaluate")
    assert af.compute_evaluator_fingerprint() == _fp_text("none")


def test_config_space_fingerprint_uses_canonical_json() -> None:
    first = {"model": ["b", "a"], "temperature": {"low": 0, "high": 1}}
    second = {"temperature": {"high": 1, "low": 0}, "model": ["b", "a"]}

    assert af.compute_config_space_fingerprint(
        first
    ) == af.compute_config_space_fingerprint(second)
    assert af.compute_config_space_fingerprint(first) == _fp_json(first)
    assert af.compute_config_space_fingerprint(None) is None


def test_build_payload_counts_examples_and_wire_helpers_filter_shape() -> None:
    def sample_agent(prompt: str) -> str:
        return prompt

    dataset = Dataset(
        examples=[
            _example({"question": "a"}, "answer-a"),
            _example({"question": "b"}, "answer-b"),
        ]
    )

    payload = af.build_artifact_fingerprints(
        dataset=dataset,
        func=sample_agent,
        configuration_space={"temperature": [0.1, 0.9]},
    )

    assert set(payload["artifact_fingerprints"]) == {
        "dataset",
        "agent",
        "evaluator",
        "config_space",
    }
    assert payload["fingerprint_meta"] == {
        "algorithm": "fp1",
        "dataset_example_count": 2,
        "source_available": True,
    }

    wire = af.artifact_fingerprints_to_wire(
        {
            **payload["artifact_fingerprints"],
            "dataset": "raw content",
            "extra": "fp1:" + ("a" * 64),
        }
    )
    assert wire == {
        "dataset": None,
        "agent": payload["artifact_fingerprints"]["agent"],
        "evaluator": payload["artifact_fingerprints"]["evaluator"],
        "config_space": payload["artifact_fingerprints"]["config_space"],
    }
    assert af.fingerprint_meta_to_wire(
        {
            "algorithm": "ignored",
            "dataset_example_count": 2,
            "source_available": True,
            "raw": "SECRET",
        }
    ) == {
        "algorithm": "fp1",
        "dataset_example_count": 2,
        "source_available": True,
    }


def test_wire_helpers_reject_malformed_fp1_values() -> None:
    assert af.artifact_fingerprints_to_wire(
        {
            "dataset": "fp1:" + ("a" * 64),
            "agent": "fp1:CANARY_SOURCE_i9j0",
            "evaluator": "fp1:" + ("B" * 64),
            "config_space": "fp1:" + ("0" * 63),
        }
    ) == {
        "dataset": "fp1:" + ("a" * 64),
        "agent": None,
        "evaluator": None,
        "config_space": None,
    }


def test_build_payload_returns_none_for_hostile_user_objects() -> None:
    class BoolRaisesCallable:
        def __bool__(self) -> bool:
            raise RuntimeError("truthiness should not escape")

        def __call__(self) -> None:
            return None

    class StrRaises:
        def __str__(self) -> str:
            raise RuntimeError("stringification should not escape")

    payload = af.build_artifact_fingerprints(
        dataset=[_example({"question": StrRaises()}, "answer-a")],
        func=BoolRaisesCallable(),
        configuration_space={"bad": StrRaises()},
    )

    assert payload["artifact_fingerprints"]["dataset"] is None
    assert payload["artifact_fingerprints"]["agent"] is None
    assert payload["artifact_fingerprints"]["config_space"] is None
    assert payload["fingerprint_meta"]["dataset_example_count"] == 0
