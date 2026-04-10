"""Sync-friendly evaluation operations client."""

from __future__ import annotations

import json
import time
from typing import Any, Literal, cast, overload
from urllib import error, request
from urllib.parse import urlencode

from traigent.evaluation.config import EvaluationConfig
from traigent.evaluation.dtos import (
    AnnotationQueueDTO,
    AnnotationQueueItemDTO,
    AnnotationQueueItemListResponse,
    AnnotationQueueItemStatus,
    AnnotationQueueListResponse,
    AnnotationQueueStatus,
    BackfillResultDTO,
    EvaluationTargetRefDTO,
    EvaluationTargetType,
    EvaluatorDefinitionDTO,
    EvaluatorListResponse,
    EvaluatorRunDTO,
    EvaluatorRunListResponse,
    EvaluatorRunStatus,
    JudgeConfigDTO,
    ScoreRecordDTO,
    ScoreRecordListResponse,
)
from traigent.utils.exceptions import (
    AuthenticationError,
    ClientError,
    TraigentConnectionError,
)


class EvaluationClient:
    """Client for typed measures, evaluator definitions, runs, and scores.

    The client is intentionally lightweight and does not coordinate mutable state
    across threads. Treat instances as thread-local or create separate clients
    per worker when using the SDK from multi-threaded code.
    """

    def __init__(
        self,
        config: EvaluationConfig | None = None,
        *,
        request_sender=None,
    ) -> None:
        self.config = config or EvaluationConfig()
        self._request_sender_override = request_sender

    def list_evaluators(
        self,
        *,
        page: int = 1,
        per_page: int = 20,
        search: str | None = None,
        measure_id: str | None = None,
        target_type: EvaluationTargetType | str | None = None,
        is_active: bool | None = None,
    ) -> EvaluatorListResponse:
        if isinstance(target_type, str):
            target_type = EvaluationTargetType(target_type)
        path = self._build_query_path(
            "/evaluators",
            page=page,
            per_page=per_page,
            search=search,
            measure_id=measure_id,
            target_type=target_type.value if target_type else None,
            is_active=str(is_active).lower() if is_active is not None else None,
        )
        payload = self._request_json("GET", path)
        return EvaluatorListResponse.from_dict(
            self._unwrap_data(payload, "evaluator list")
        )

    def get_evaluator(self, evaluator_id: str) -> EvaluatorDefinitionDTO:
        payload = self._request_json("GET", f"/evaluators/{evaluator_id}")
        return EvaluatorDefinitionDTO.from_dict(
            self._unwrap_data(payload, "evaluator detail")
        )

    def create_evaluator(
        self,
        *,
        name: str,
        measure_id: str,
        target_type: EvaluationTargetType | str,
        judge_config: JudgeConfigDTO | dict[str, Any],
        description: str | None = None,
        sampling_rate: float = 1.0,
        target_filters: dict[str, Any] | None = None,
        is_active: bool = True,
    ) -> EvaluatorDefinitionDTO:
        if isinstance(target_type, str):
            target_type = EvaluationTargetType(target_type)
        if isinstance(judge_config, dict):
            judge_config = JudgeConfigDTO.from_dict(judge_config)
        payload = self._request_json(
            "POST",
            "/evaluators",
            {
                "name": name,
                "description": description,
                "measure_id": measure_id,
                "target_type": target_type.value,
                "judge_config": judge_config.to_dict(),
                "sampling_rate": sampling_rate,
                "target_filters": dict(target_filters or {}),
                "is_active": is_active,
            },
        )
        return EvaluatorDefinitionDTO.from_dict(
            self._unwrap_data(payload, "evaluator create")
        )

    def update_evaluator(
        self,
        evaluator_id: str,
        **fields: Any,
    ) -> EvaluatorDefinitionDTO:
        payload = dict(fields)
        if "target_type" in payload and isinstance(
            payload["target_type"], EvaluationTargetType
        ):
            payload["target_type"] = payload["target_type"].value
        if "judge_config" in payload:
            judge_config = payload["judge_config"]
            if isinstance(judge_config, JudgeConfigDTO):
                payload["judge_config"] = judge_config.to_dict()
        response = self._request_json("PATCH", f"/evaluators/{evaluator_id}", payload)
        return EvaluatorDefinitionDTO.from_dict(
            self._unwrap_data(response, "evaluator update")
        )

    def execute_evaluator(
        self,
        evaluator_id: str,
        *,
        target: EvaluationTargetRefDTO | dict[str, Any],
        override_judge_config: JudgeConfigDTO | dict[str, Any] | None = None,
    ) -> EvaluatorRunDTO:
        if isinstance(target, dict):
            target = EvaluationTargetRefDTO.from_dict(target)
        payload = target.to_dict()
        if override_judge_config is not None:
            if isinstance(override_judge_config, dict):
                override_judge_config = JudgeConfigDTO.from_dict(override_judge_config)
            payload["override_judge_config"] = override_judge_config.to_dict()
        response = self._request_json(
            "POST",
            f"/evaluators/{evaluator_id}/execute",
            payload,
        )
        return EvaluatorRunDTO.from_dict(
            self._unwrap_data(response, "evaluator execute")
        )

    def backfill_evaluator(
        self,
        evaluator_id: str,
        *,
        target_type: EvaluationTargetType | str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 25,
        skip_existing_scores: bool = True,
    ) -> BackfillResultDTO:
        if isinstance(target_type, str):
            target_type = EvaluationTargetType(target_type)
        payload: dict[str, Any] = {
            "filters": dict(filters or {}),
            "limit": limit,
            "skip_existing_scores": skip_existing_scores,
        }
        if target_type is not None:
            payload["target_type"] = target_type.value
        response = self._request_json(
            "POST",
            f"/evaluators/{evaluator_id}/backfill",
            payload,
        )
        return BackfillResultDTO.from_dict(
            self._unwrap_data(response, "evaluator backfill")
        )

    def list_evaluator_runs(
        self,
        *,
        page: int = 1,
        per_page: int = 20,
        evaluator_id: str | None = None,
        target_type: EvaluationTargetType | str | None = None,
        target_id: str | None = None,
        status: EvaluatorRunStatus | str | None = None,
    ) -> EvaluatorRunListResponse:
        if isinstance(target_type, str):
            target_type = EvaluationTargetType(target_type)
        if isinstance(status, str):
            status = EvaluatorRunStatus(status)
        path = self._build_query_path(
            "/evaluator-runs",
            page=page,
            per_page=per_page,
            evaluator_id=evaluator_id,
            target_type=target_type.value if target_type else None,
            target_id=target_id,
            status=status.value if status else None,
        )
        payload = self._request_json("GET", path)
        return EvaluatorRunListResponse.from_dict(
            self._unwrap_data(payload, "evaluator run list")
        )

    def get_evaluator_run(self, run_id: str) -> EvaluatorRunDTO:
        payload = self._request_json("GET", f"/evaluator-runs/{run_id}")
        return EvaluatorRunDTO.from_dict(
            self._unwrap_data(payload, "evaluator run detail")
        )

    def retry_evaluator_run(
        self,
        run_id: str,
        *,
        override_judge_config: JudgeConfigDTO | dict[str, Any] | None = None,
    ) -> EvaluatorRunDTO:
        payload: dict[str, Any] = {}
        if override_judge_config is not None:
            if isinstance(override_judge_config, dict):
                override_judge_config = JudgeConfigDTO.from_dict(override_judge_config)
            payload["override_judge_config"] = override_judge_config.to_dict()
        response = self._request_json(
            "POST", f"/evaluator-runs/{run_id}/retry", payload
        )
        return EvaluatorRunDTO.from_dict(
            self._unwrap_data(response, "evaluator run retry")
        )

    def wait_for_evaluator_run(
        self,
        run_id: str,
        *,
        max_attempts: int = 60,
        interval_seconds: float = 2.0,
    ) -> EvaluatorRunDTO:
        for attempt in range(max_attempts):
            run = self.get_evaluator_run(run_id)
            if run.status in {EvaluatorRunStatus.COMPLETED, EvaluatorRunStatus.FAILED}:
                return run
            if attempt < max_attempts - 1:
                time.sleep(interval_seconds)
        raise ClientError(f"Timed out waiting for evaluator run '{run_id}' to complete")

    def create_score(
        self,
        *,
        measure_id: str,
        target: EvaluationTargetRefDTO | dict[str, Any],
        numeric_value: float | None = None,
        categorical_value: str | None = None,
        boolean_value: bool | None = None,
        comment: str | None = None,
        correction_output: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> ScoreRecordDTO:
        if isinstance(target, dict):
            target = EvaluationTargetRefDTO.from_dict(target)
        payload = {
            "measure_id": measure_id,
            **target.to_dict(),
            "numeric_value": numeric_value,
            "categorical_value": categorical_value,
            "boolean_value": boolean_value,
            "comment": comment,
            "correction_output": correction_output,
            "metadata": dict(metadata or {}),
        }
        response = self._request_json("POST", "/scores", payload)
        return ScoreRecordDTO.from_dict(self._unwrap_data(response, "score create"))

    def list_scores(
        self,
        *,
        target: EvaluationTargetRefDTO | dict[str, Any],
        measure_id: str | None = None,
        page: int = 1,
        per_page: int = 20,
    ) -> ScoreRecordListResponse:
        if isinstance(target, dict):
            target = EvaluationTargetRefDTO.from_dict(target)
        path = self._build_query_path(
            "/scores",
            page=page,
            per_page=per_page,
            measure_id=measure_id,
            target_type=target.target_type.value,
            target_id=target.target_id,
        )
        payload = self._request_json("GET", path)
        return ScoreRecordListResponse.from_dict(
            self._unwrap_data(payload, "score list")
        )

    def list_annotation_queues(
        self,
        *,
        page: int = 1,
        per_page: int = 20,
        search: str | None = None,
        target_type: EvaluationTargetType | str | None = None,
        status: AnnotationQueueStatus | str | None = None,
    ) -> AnnotationQueueListResponse:
        if isinstance(target_type, str):
            target_type = EvaluationTargetType(target_type)
        if isinstance(status, str):
            status = AnnotationQueueStatus(status)
        path = self._build_query_path(
            "/annotation-queues",
            page=page,
            per_page=per_page,
            search=search,
            target_type=target_type.value if target_type else None,
            status=status.value if status else None,
        )
        payload = self._request_json("GET", path)
        return AnnotationQueueListResponse.from_dict(
            self._unwrap_data(payload, "annotation queue list")
        )

    def get_annotation_queue(self, queue_id: str) -> AnnotationQueueDTO:
        payload = self._request_json("GET", f"/annotation-queues/{queue_id}")
        return AnnotationQueueDTO.from_dict(
            self._unwrap_data(payload, "annotation queue detail")
        )

    def create_annotation_queue(
        self,
        *,
        name: str,
        target_type: EvaluationTargetType | str,
        measure_ids: list[str],
        description: str | None = None,
        status: AnnotationQueueStatus | str = AnnotationQueueStatus.ACTIVE,
    ) -> AnnotationQueueDTO:
        if isinstance(target_type, str):
            target_type = EvaluationTargetType(target_type)
        if isinstance(status, str):
            status = AnnotationQueueStatus(status)
        payload = self._request_json(
            "POST",
            "/annotation-queues",
            {
                "name": name,
                "description": description,
                "target_type": target_type.value,
                "measure_ids": list(measure_ids),
                "status": status.value,
            },
        )
        return AnnotationQueueDTO.from_dict(
            self._unwrap_data(payload, "annotation queue create")
        )

    def update_annotation_queue(
        self, queue_id: str, **fields: Any
    ) -> AnnotationQueueDTO:
        payload = dict(fields)
        if "status" in payload and isinstance(payload["status"], AnnotationQueueStatus):
            payload["status"] = payload["status"].value
        response = self._request_json(
            "PATCH", f"/annotation-queues/{queue_id}", payload
        )
        return AnnotationQueueDTO.from_dict(
            self._unwrap_data(response, "annotation queue update")
        )

    def add_annotation_queue_items(
        self,
        queue_id: str,
        *,
        targets: list[EvaluationTargetRefDTO | dict[str, Any]],
        assigned_user_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_targets = [
            (
                item.to_dict()
                if isinstance(item, EvaluationTargetRefDTO)
                else EvaluationTargetRefDTO.from_dict(item).to_dict()
            )
            for item in targets
        ]
        return self._unwrap_data(
            self._request_json(
                "POST",
                f"/annotation-queues/{queue_id}/items",
                {
                    "targets": normalized_targets,
                    "assigned_user_id": assigned_user_id,
                },
            ),
            "annotation queue item create",
        )

    def list_annotation_queue_items(
        self,
        queue_id: str,
        *,
        page: int = 1,
        per_page: int = 20,
        status: AnnotationQueueItemStatus | str | None = None,
        assigned_user_id: str | None = None,
        target_id: str | None = None,
    ) -> AnnotationQueueItemListResponse:
        if isinstance(status, str):
            status = AnnotationQueueItemStatus(status)
        path = self._build_query_path(
            f"/annotation-queues/{queue_id}/items",
            page=page,
            per_page=per_page,
            status=status.value if status else None,
            assigned_user_id=assigned_user_id,
            target_id=target_id,
        )
        payload = self._request_json("GET", path)
        return AnnotationQueueItemListResponse.from_dict(
            self._unwrap_data(payload, "annotation queue item list")
        )

    def get_next_annotation_queue_item(
        self,
        queue_id: str,
        *,
        assigned_user_id: str | None = None,
    ) -> AnnotationQueueItemDTO | None:
        path = self._build_query_path(
            f"/annotation-queues/{queue_id}/next",
            assigned_user_id=assigned_user_id,
        )
        payload = self._request_json("GET", path)
        data = self._unwrap_data(payload, "annotation queue next item", allow_none=True)
        if data is None:
            return None
        return AnnotationQueueItemDTO.from_dict(data)

    def update_annotation_queue_item(
        self, item_id: str, **fields: Any
    ) -> AnnotationQueueItemDTO:
        payload = dict(fields)
        if "status" in payload and isinstance(
            payload["status"], AnnotationQueueItemStatus
        ):
            payload["status"] = payload["status"].value
        response = self._request_json(
            "PATCH", f"/annotation-queues/items/{item_id}", payload
        )
        return AnnotationQueueItemDTO.from_dict(
            self._unwrap_data(response, "annotation queue item update")
        )

    def complete_annotation_queue_item(
        self,
        item_id: str,
        *,
        scores: list[dict[str, Any]],
        note: str | None = None,
    ) -> dict[str, Any]:
        response = self._request_json(
            "POST",
            f"/annotation-queues/items/{item_id}/complete",
            {
                "scores": scores,
                "note": note,
            },
        )
        return self._unwrap_data(response, "annotation queue completion")

    def create_typed_measure(
        self,
        measure_data: dict[str, Any],
    ) -> dict[str, Any]:
        return self._request_measure_json("POST", "", measure_data)

    def update_typed_measure(
        self,
        measure_id: str,
        measure_data: dict[str, Any],
    ) -> dict[str, Any]:
        return self._request_measure_json("PUT", f"/{measure_id}", measure_data)

    def judge_config_from_benchmark_payload(
        self,
        payload: dict[str, Any] | None,
    ) -> JudgeConfigDTO | None:
        return JudgeConfigDTO.from_benchmark_payload(payload)

    def judge_config_to_benchmark_payload(
        self,
        judge_config: JudgeConfigDTO | dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(judge_config, dict):
            judge_config = JudgeConfigDTO.from_dict(judge_config)
        return judge_config.to_benchmark_payload()

    def _request_measure_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        full_path = f"{self.config.measures_api_path}{path}"
        if self._request_sender_override is not None:
            return cast(
                dict[str, Any],
                self._request_sender_override(method, full_path, payload),
            )
        return self._request_json_sync(method, full_path, payload)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        full_path = f"{self.config.api_path}{path}"
        if self._request_sender_override is not None:
            return cast(
                dict[str, Any],
                self._request_sender_override(method, full_path, payload),
            )
        return self._request_json_sync(method, full_path, payload)

    def _request_json_sync(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        encoded_payload = None
        if payload is not None:
            encoded_payload = json.dumps(payload).encode("utf-8")

        http_request = request.Request(
            f"{self.config.backend_origin}{path}",
            data=encoded_payload,
            headers=self.config.build_headers(),
            method=method,
        )
        try:
            with request.urlopen(  # nosec B310 - backend_origin is caller-configured API endpoint
                http_request, timeout=self.config.request_timeout
            ) as response:
                status_code = getattr(response, "status", 200)
                body = response.read().decode("utf-8") if response else ""
                parsed = json.loads(body) if body else {}
                if status_code >= 400:
                    raise ClientError(
                        f"Evaluation request failed with status {status_code}",
                        status_code=status_code,
                        details={"body": body},
                    )
                return parsed
        except error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8") if exc.fp else ""
            finally:
                exc.close()
            if exc.code in {401, 403}:
                raise AuthenticationError(
                    f"Evaluation request rejected with status {exc.code}"
                ) from exc
            raise ClientError(
                f"Evaluation request failed with status {exc.code}",
                status_code=exc.code,
                details={"body": body},
            ) from exc
        except error.URLError as exc:
            raise TraigentConnectionError(
                f"Failed to connect to evaluation backend at {self.config.backend_origin}"
            ) from exc

    @overload
    def _unwrap_data(
        self,
        payload: dict[str, Any],
        label: str,
        *,
        allow_none: Literal[False] = False,
    ) -> dict[str, Any]: ...

    @overload
    def _unwrap_data(
        self,
        payload: dict[str, Any],
        label: str,
        *,
        allow_none: Literal[True],
    ) -> dict[str, Any] | None: ...

    def _unwrap_data(
        self,
        payload: dict[str, Any],
        label: str,
        *,
        allow_none: bool = False,
    ) -> dict[str, Any] | None:
        data = payload.get("data")
        if data is None and allow_none:
            return None
        if not isinstance(data, dict):
            raise ClientError(f"Unexpected response structure for {label}")
        return data

    def _build_query_path(self, base_path: str, **params: Any) -> str:
        query = urlencode(
            {
                key: str(value)
                for key, value in params.items()
                if value is not None and value != ""
            }
        )
        return f"{base_path}?{query}" if query else base_path
