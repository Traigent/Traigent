# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Traigent-Commercial
# Copyright (c) 2024-2026 Traigent Ltd. Dual-licensed: AGPL-3.0 or commercial.
"""``EvaluationOptions.task_type`` must survive every hop to the session request.

The hint travels decorator -> OptimizedFunction -> (InteractiveOptimizer | orchestrator
-> BackendSessionManager -> BackendIntegratedClient -> SessionOperations) ->
SessionCreationRequest. Each hop is a place it can be silently dropped -- exactly how
``agent_key`` once vanished (see test_session_narrative_wire). These tests pin each hop.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from tests.unit.cloud.test_session_creation_warm_start import CapturingFakeClient
from traigent.api.decorators import EvaluationOptions, optimize
from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.models import SessionCreationResponse
from traigent.cloud.session_operations import SessionOperations
from traigent.optimizers.interactive_optimizer import (
    InteractiveOptimizer,
    RemoteGuidanceService,
)

pytestmark = pytest.mark.backend_online


class TestDecoratorHop:
    """The decorator hop is exercised through BOTH the dict and object forms.

    This class used to avoid the object form, on the stated grounds that "CI's module
    graph resolves ``EvaluationOptions`` to a second same-named class". That
    diagnosis was wrong about the cause. There is only one
    ``traigent.api.decorators`` in ``sys.modules`` and one definition of the class;
    the second class object came from an ``importlib.reload`` of that module inside
    another test file, which re-executes the module body in the SAME module object
    and mints brand-new class objects while every module that imported the name at
    collection time keeps the pre-reload one. It looked like a CI-only property
    because xdist decides whether the reloading file lands in this worker.

    Both halves are fixed now, so the object form is a fair test again: those
    reloads are gone (PR #2357), and ``_coerce_bundle`` accepts a same-named twin
    and re-validates it against the live class
    (``tests/unit/api/test_coerce_bundle_class_identity.py``). Keep the dict form
    too -- it is the documented path and takes a different route through
    ``model_validate``.
    """

    def test_task_type_reaches_the_optimized_function(self):
        @optimize(
            evaluation={"task_type": " multiple_choice "},
            configuration_space={"temperature": [0.1, 0.9]},
        )
        def answer(question: str) -> str:
            return question

        assert answer.task_type == "multiple_choice"

    def test_dict_form_and_declared_default_agree(self):
        @optimize(
            evaluation={"task_type": "text2sql"},
            configuration_space={"temperature": [0.1, 0.9]},
        )
        def answer(question: str) -> str:
            return question

        assert answer.task_type == "text2sql"

    def test_object_form_reaches_the_optimized_function(self):
        """The bundle-object form, which this class used to avoid."""

        @optimize(
            evaluation=EvaluationOptions(task_type="multiple_choice"),
            configuration_space={"temperature": [0.1, 0.9]},
        )
        def answer(question: str) -> str:
            return question

        assert answer.task_type == "multiple_choice"

    def test_object_and_dict_forms_agree(self):
        @optimize(
            evaluation=EvaluationOptions(task_type="text2sql"),
            configuration_space={"temperature": [0.1, 0.9]},
        )
        def via_object(question: str) -> str:
            return question

        @optimize(
            evaluation={"task_type": "text2sql"},
            configuration_space={"temperature": [0.1, 0.9]},
        )
        def via_dict(question: str) -> str:
            return question

        assert via_object.task_type == via_dict.task_type == "text2sql"


class TestEvaluationOptionsModel:
    """The field is declared on the bundle model, with the right default."""

    def test_model_carries_the_field(self):
        options = optimize.__globals__["EvaluationOptions"](task_type="multiple_choice")
        assert options.task_type == "multiple_choice"

    def test_model_default_is_none(self):
        assert optimize.__globals__["EvaluationOptions"]().task_type is None

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_hint_is_none_not_an_empty_token(self, blank):
        @optimize(
            evaluation={"task_type": blank},
            configuration_space={"temperature": [0.1, 0.9]},
        )
        def answer(question: str) -> str:
            return question

        assert answer.task_type is None

    def test_default_is_none(self):
        @optimize(configuration_space={"temperature": [0.1, 0.9]})
        def answer(question: str) -> str:
            return question

        assert answer.task_type is None


class TestInteractiveOptimizerHop:
    @pytest.mark.asyncio
    async def test_task_type_is_on_the_session_request(self):
        service = Mock(spec=RemoteGuidanceService)
        service.create_session = AsyncMock(
            return_value=SessionCreationResponse(
                session_id="s-1", status="active", optimization_strategy={}
            )
        )
        optimizer = InteractiveOptimizer(
            config_space={"temperature": (0.0, 1.0)},
            objectives=["accuracy"],
            remote_service=service,
            dataset_metadata={"size": 8},
            task_type="multiple_choice",
        )
        await optimizer.initialize_session(function_name="mcq_agent", max_trials=2)
        request = service.create_session.call_args[0][0]
        assert request.task_type == "multiple_choice"

    @pytest.mark.asyncio
    async def test_absent_hint_stays_none(self):
        service = Mock(spec=RemoteGuidanceService)
        service.create_session = AsyncMock(
            return_value=SessionCreationResponse(
                session_id="s-1", status="active", optimization_strategy={}
            )
        )
        optimizer = InteractiveOptimizer(
            config_space={"temperature": (0.0, 1.0)},
            objectives=["accuracy"],
            remote_service=service,
        )
        await optimizer.initialize_session(function_name="mcq_agent", max_trials=2)
        assert service.create_session.call_args[0][0].task_type is None


class TestOrchestratorPathHops:
    def test_session_operations_threads_task_type(self):
        client = CapturingFakeClient()
        ops = SessionOperations(cast(Any, client))
        ops.create_session(
            "my_func",
            {"model": ["a", "b"]},
            metadata={"max_trials": 5, "dataset_size": 10, "evaluation_set": "test"},
            task_type="multiple_choice",
        )
        assert client.captured_session_request is not None
        assert client.captured_session_request.task_type == "multiple_choice"

    def test_session_operations_default_is_none(self):
        client = CapturingFakeClient()
        ops = SessionOperations(cast(Any, client))
        ops.create_session(
            "my_func",
            {"model": ["a", "b"]},
            metadata={"max_trials": 5, "dataset_size": 10, "evaluation_set": "test"},
        )
        assert client.captured_session_request is not None
        assert client.captured_session_request.task_type is None

    def test_backend_client_forwards_task_type_as_a_keyword(self):
        """Positional forwarding is the drift risk here (see the signature comment)."""
        client = object.__new__(BackendIntegratedClient)
        client._session_ops = MagicMock()
        client.create_session("f", {"x": [1]}, task_type="text2sql")
        assert (
            client._session_ops.create_session.call_args.kwargs["task_type"]
            == "text2sql"
        )
