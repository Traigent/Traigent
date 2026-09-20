# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Traigent-Commercial
# Copyright (c) 2024-2026 Traigent Ltd. Dual-licensed: AGPL-3.0 or commercial.
"""``_coerce_bundle`` must accept a valid bundle from a *reloaded* module.

``importlib.reload(traigent.api.decorators)`` re-executes the module body in the
SAME module object, minting brand-new class objects and rebinding the module
attributes. Every caller that already did ``from traigent.api.decorators import
ExecutionOptions`` keeps the pre-reload class, so ``isinstance(value, model_cls)``
inside ``_coerce_bundle`` is ``False`` for a perfectly valid instance and the
caller gets the nonsensical ``TypeError: execution must be a dict or
ExecutionOptions, got ExecutionOptions``. A notebook running ``%autoreload``, a
plugin that reloads SDK modules, and the "reimport the module to pick up the
env-var change" pattern all reach it.

The tolerance is deliberately narrow — same ``__qualname__``, same module
basename, still a pydantic model, and the values must re-validate against the
live class. A ``str``, a ``list``, a duck-typed stand-in and an unrelated model
must keep raising ``TypeError``.

The rebuild *derives* the new object's private state instead of transplanting
it: the recorded deprecated inputs (``ExecutionOptions._legacy_options``) go back
into the payload and the live wrap validator fills the stash itself, so an
invalid legacy value is rejected rather than smuggled, and the two instances
share no mutable state.
"""

from __future__ import annotations

import re
from importlib import reload
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError, create_model

import traigent.api.decorators as decorators_module
from traigent.api.decorators import (
    EvaluationOptions,
    ExecutionOptions,
    _coerce_bundle,
    optimize,
)
from traigent.core.optimized_function import OptimizedFunction


@pytest.fixture
def twin_classes():
    """Yield ``(pre_reload_cls, live_cls)`` for the real reload mechanism.

    Uses an actual ``reload()`` rather than a hand-built look-alike so the test
    pins the mechanism a user actually hits, not an imitation of it. The module
    ``__dict__`` is restored afterwards (a second ``reload()`` would only mint a
    third generation) so the pollution cannot escape this test.
    """
    saved = dict(decorators_module.__dict__)
    pre_reload_execution = decorators_module.ExecutionOptions
    try:
        reload(decorators_module)
        live_execution = decorators_module.ExecutionOptions
        assert pre_reload_execution is not live_execution, (
            "reload() did not mint a new class object; this test no longer "
            "exercises the duplicated-class-identity path"
        )
        yield pre_reload_execution, live_execution
    finally:
        decorators_module.__dict__.clear()
        decorators_module.__dict__.update(saved)


class TestReloadedClassIsAccepted:
    """A same-named twin instance is a valid bundle, not a ``TypeError``."""

    def test_the_two_classes_are_genuinely_distinct_objects(self, twin_classes):
        pre_reload_cls, live_cls = twin_classes
        assert pre_reload_cls is not live_cls
        assert id(pre_reload_cls) != id(live_cls)
        # Same name, same defining module -- indistinguishable in the message.
        assert pre_reload_cls.__qualname__ == live_cls.__qualname__
        assert pre_reload_cls.__module__ == live_cls.__module__
        # ...and that is exactly why isinstance fails.
        assert not isinstance(pre_reload_cls(), live_cls)

    def test_coerce_bundle_accepts_the_pre_reload_instance(self, twin_classes):
        pre_reload_cls, live_cls = twin_classes
        value = pre_reload_cls(require_run_id=True, algorithm="grid")

        coerced = _coerce_bundle(value, live_cls, "execution")

        assert isinstance(coerced, live_cls)
        assert coerced.require_run_id is True
        assert coerced.algorithm == "grid"

    def test_public_decorator_plumbs_a_pre_reload_bundle(self, twin_classes):
        """The user-facing symptom: ``@optimize(execution=<valid object>)``."""
        pre_reload_cls, _live_cls = twin_classes

        @decorators_module.optimize(
            configuration_space={"model": ["cheap", "strong"]},
            objectives=["accuracy"],
            execution=pre_reload_cls(require_run_id=True),
        )
        def answer(text: str, model: str = "cheap") -> str:
            return f"{text} ({model})"

        assert isinstance(answer, OptimizedFunction)
        assert answer.require_run_id is True


class TestLegacyPrivateStateIsDerivedNotTransplanted:
    """The deprecated inputs are replayed through the live validator."""

    def test_private_legacy_option_stash_survives_the_rebuild(self, twin_classes):
        """``_legacy_options`` is a ``PrivateAttr``; a naive round-trip drops it."""
        pre_reload_cls, live_cls = twin_classes
        value = pre_reload_cls(execution_mode="cloud")
        assert value.legacy_option_values == {"execution_mode": "cloud"}

        coerced = _coerce_bundle(value, live_cls, "execution")

        assert isinstance(coerced, live_cls)
        assert coerced.legacy_option_values == {"execution_mode": "cloud"}

    def test_the_two_objects_do_not_share_the_legacy_stash(self, twin_classes):
        """Rebuilt state is a distinct dict, not a reference to the old one."""
        pre_reload_cls, live_cls = twin_classes
        value = pre_reload_cls(execution_mode="cloud")

        coerced = _coerce_bundle(value, live_cls, "execution")

        assert coerced._legacy_options is not value._legacy_options
        # Mutating the rebuilt object must not reach back into the original...
        coerced._legacy_options["execution_mode"] = "local"
        coerced._legacy_options["privacy_enabled"] = True
        assert value.legacy_option_values == {"execution_mode": "cloud"}
        # ...nor the other way around.
        value._legacy_options["cloud_fallback_policy"] = "never"
        assert "cloud_fallback_policy" not in coerced.legacy_option_values

    def test_an_invalid_legacy_value_is_rejected_not_smuggled(self, twin_classes):
        """A twin whose stash holds a value the live model rejects must raise.

        The stash is written directly here on purpose: that is what a twin built
        by a *different* version of the class -- one whose legacy validation
        differed -- looks like to the live model. The coercion step must not be
        the thing that lets such a value past ``HybridAPIOptions``'
        ``batch_size >= 1``.
        """
        pre_reload_cls, live_cls = twin_classes
        value = pre_reload_cls()
        value._legacy_options["hybrid_api_batch_size"] = 0

        with pytest.raises(TypeError, match="do not validate against this one"):
            _coerce_bundle(value, live_cls, "execution")

    def test_the_same_invalid_legacy_value_is_rejected_on_construction(self):
        """Control: the rejection above is the live validator, not a new rule."""
        with pytest.raises(ValidationError):
            ExecutionOptions.model_validate({"hybrid_api_batch_size": 0})

    def test_a_valid_legacy_hybrid_value_still_round_trips(self, twin_classes):
        pre_reload_cls, live_cls = twin_classes
        value = pre_reload_cls(hybrid_api_batch_size=4)

        coerced = _coerce_bundle(value, live_cls, "execution")

        assert coerced.legacy_option_values == {"hybrid_api_batch_size": 4}


class TestStructurallyWrongTwinStillRejected:
    """Name and module matching alone is not enough to trust the object."""

    def test_twin_with_an_unknown_field_raises(self):
        """``extra="forbid"`` must still bite after the identity check passes."""
        impostor_cls = create_model(
            "ExecutionOptions",
            __module__="traigent.api.decorators",
            not_a_real_execution_field=(str, "nope"),
        )
        impostor = impostor_cls()
        # It clears the identity gate exactly as a reload twin would...
        assert impostor_cls.__qualname__ == ExecutionOptions.__qualname__
        assert impostor_cls.__module__ == ExecutionOptions.__module__
        assert impostor_cls is not ExecutionOptions
        # ...and is still rejected, because the values do not validate.
        with pytest.raises(TypeError, match="do not validate against this one"):
            _coerce_bundle(impostor, ExecutionOptions, "execution")

    def test_a_locally_defined_same_named_class_is_not_a_twin(self):
        """A nested class's ``__qualname__`` carries ``<locals>`` -- not a twin."""

        class ExecutionOptions(BaseModel):  # noqa: N801 - deliberate same name
            require_run_id: bool = True

        assert "<locals>" in ExecutionOptions.__qualname__
        with pytest.raises(TypeError, match="must be a dict or ExecutionOptions"):
            _coerce_bundle(
                ExecutionOptions(), decorators_module.ExecutionOptions, "execution"
            )


class TestGenuinelyWrongTypesStillRaise:
    """The public decorator boundary must not become duck-typed."""

    @pytest.mark.parametrize(
        "value",
        [
            "ExecutionOptions",
            ["require_run_id"],
            ("require_run_id",),
            42,
            True,
            object(),
        ],
        ids=["str", "list", "tuple", "int", "bool", "object"],
    )
    def test_non_model_values_raise_type_error(self, value: Any):
        with pytest.raises(TypeError, match="must be a dict or ExecutionOptions"):
            _coerce_bundle(value, ExecutionOptions, "execution")

    def test_an_unrelated_pydantic_model_raises(self):
        class SomethingElse(BaseModel):
            require_run_id: bool = True

        with pytest.raises(TypeError, match="must be a dict or ExecutionOptions"):
            _coerce_bundle(SomethingElse(), ExecutionOptions, "execution")

    def test_a_different_traigent_bundle_raises(self):
        """``EvaluationOptions`` is a sibling bundle, not an ``ExecutionOptions``."""
        with pytest.raises(TypeError, match="must be a dict or ExecutionOptions"):
            _coerce_bundle(EvaluationOptions(), ExecutionOptions, "execution")

    def test_a_duck_typed_stand_in_raises(self):
        """Has the fields and a ``model_dump``, is not a pydantic model."""

        class QuacksLikeExecutionOptions:
            __qualname__ = "ExecutionOptions"
            __module__ = "traigent.api.decorators"
            require_run_id = True

            def model_dump(self) -> dict[str, Any]:
                return {"require_run_id": True}

        with pytest.raises(TypeError, match="must be a dict or ExecutionOptions"):
            _coerce_bundle(QuacksLikeExecutionOptions(), ExecutionOptions, "execution")

    def test_the_public_decorator_still_rejects_a_string(self):
        with pytest.raises(TypeError, match="must be a dict or ExecutionOptions"):

            @optimize(
                configuration_space={"model": ["cheap", "strong"]},
                objectives=["accuracy"],
                execution="require_run_id=True",  # type: ignore[arg-type]
            )
            def answer(text: str, model: str = "cheap") -> str:
                return text


class TestHappyPathsUnchanged:
    """The tolerant branch must not disturb the two documented forms."""

    def test_none_is_still_none(self):
        assert _coerce_bundle(None, ExecutionOptions, "execution") is None

    def test_a_real_instance_is_returned_by_identity(self):
        value = ExecutionOptions(require_run_id=True)
        assert _coerce_bundle(value, ExecutionOptions, "execution") is value

    def test_the_dict_form_still_validates(self):
        coerced = _coerce_bundle(
            {"require_run_id": True}, ExecutionOptions, "execution"
        )
        assert isinstance(coerced, ExecutionOptions)
        assert coerced.require_run_id is True

    def test_the_dict_form_still_rejects_an_unknown_key(self):
        with pytest.raises(Exception, match=re.compile("not_a_field", re.I)):
            _coerce_bundle({"not_a_field": 1}, ExecutionOptions, "execution")
