"""Sync-friendly client for core metrics and fine-tuning export."""

from __future__ import annotations

import json
from typing import Any, cast
from urllib import error, request
from urllib.parse import quote, unquote, urlencode

from traigent.core_metrics.config import CoreMetricsConfig
from traigent.core_metrics.dtos import (
    CoreExperimentTrendDTO,
    CoreMetricsOverviewDTO,
    FineTuningExportDTO,
    FineTuningManifestDTO,
    ProjectAnalyticsSummaryDTO,
    ProjectAnalyticsTrendDTO,
    ProjectEvaluatorQualityDashboardDTO,
    ProjectExportJobListDTO,
    ProjectExportJobResponseDTO,
    ProjectMeasureDistributionDTO,
    ProjectOptimizationOverviewDashboardDTO,
    ProjectPricingCatalogDTO,
    ProjectUsageDashboardDTO,
)
from traigent.utils.exceptions import (
    AuthenticationError,
    ClientError,
    TraigentConnectionError,
)

DEFAULT_EXPORT_FILENAME = "fine-tuning.jsonl"
DEFAULT_EXPORT_CONTENT_TYPE = "application/x-ndjson"


class CoreMetricsClient:
    """Client for tenant-scoped core metrics and fine-tuning export."""

    def __init__(
        self,
        config: CoreMetricsConfig | None = None,
        *,
        request_sender=None,
    ) -> None:
        self.config = config or CoreMetricsConfig()
        self._request_sender_override = request_sender

    def get_core_metrics_overview(self) -> CoreMetricsOverviewDTO:
        payload = self._request_json("GET", "/core-metrics/overview")
        return CoreMetricsOverviewDTO.from_dict(
            self._unwrap_data(payload, "core metrics overview")
        )

    def get_analytics_summary(self, *, days: int = 30) -> ProjectAnalyticsSummaryDTO:
        payload = self._request_json(
            "GET",
            self._build_query_path("/analytics/summary", days=days),
        )
        return ProjectAnalyticsSummaryDTO.from_dict(
            self._unwrap_data(payload, "analytics summary")
        )

    def get_pricing_catalog(self) -> ProjectPricingCatalogDTO:
        payload = self._request_json(
            "GET",
            "/analytics/pricing-catalog",
        )
        return ProjectPricingCatalogDTO.from_dict(
            self._unwrap_data(payload, "pricing catalog")
        )

    def get_optimization_overview_dashboard(
        self,
        *,
        days: int = 30,
        limit: int = 5,
    ) -> ProjectOptimizationOverviewDashboardDTO:
        payload = self._request_json(
            "GET",
            self._build_query_path(
                "/analytics/dashboards/optimization-overview",
                days=days,
                limit=limit,
            ),
        )
        return ProjectOptimizationOverviewDashboardDTO.from_dict(
            self._unwrap_data(payload, "optimization overview dashboard")
        )

    def get_evaluator_quality_dashboard(
        self,
        *,
        days: int = 30,
        limit: int = 5,
    ) -> ProjectEvaluatorQualityDashboardDTO:
        payload = self._request_json(
            "GET",
            self._build_query_path(
                "/analytics/dashboards/evaluator-quality",
                days=days,
                limit=limit,
            ),
        )
        return ProjectEvaluatorQualityDashboardDTO.from_dict(
            self._unwrap_data(payload, "evaluator quality dashboard")
        )

    def get_project_usage_dashboard(
        self,
        *,
        days: int = 30,
        limit: int = 5,
    ) -> ProjectUsageDashboardDTO:
        payload = self._request_json(
            "GET",
            self._build_query_path(
                "/analytics/dashboards/project-usage",
                days=days,
                limit=limit,
            ),
        )
        return ProjectUsageDashboardDTO.from_dict(
            self._unwrap_data(payload, "project usage dashboard")
        )

    def get_run_volume_trend(
        self,
        *,
        experiment_id: str | None = None,
        days: int = 30,
        bucket: str | None = None,
    ) -> ProjectAnalyticsTrendDTO:
        payload = self._request_json(
            "GET",
            self._build_query_path(
                "/analytics/trends/run-volume",
                experiment_id=experiment_id,
                days=days,
                bucket=bucket,
            ),
        )
        return ProjectAnalyticsTrendDTO.from_dict(
            self._unwrap_data(payload, "run-volume trend")
        )

    def get_measure_distribution(
        self,
        measure_key: str,
        *,
        experiment_id: str | None = None,
        bins: int = 10,
    ) -> ProjectMeasureDistributionDTO:
        payload = self._request_json(
            "GET",
            self._build_query_path(
                f"/analytics/distributions/measures/{quote(measure_key, safe='')}",
                experiment_id=experiment_id,
                bins=bins,
            ),
        )
        return ProjectMeasureDistributionDTO.from_dict(
            self._unwrap_data(payload, "measure distribution")
        )

    def get_experiment_trend(self, experiment_id: str) -> CoreExperimentTrendDTO:
        payload = self._request_json(
            "GET", f"/core-metrics/experiments/{quote(experiment_id, safe='')}/trend"
        )
        return CoreExperimentTrendDTO.from_dict(
            self._unwrap_data(payload, "core experiment trend")
        )

    def export_fine_tuning_jsonl(
        self,
        *,
        experiment_id: str | None = None,
        experiment_run_id: str | None = None,
        limit: int = 1000,
    ) -> FineTuningExportDTO:
        path = self._build_query_path(
            "/core-exports/fine-tuning.jsonl",
            experiment_id=experiment_id,
            experiment_run_id=experiment_run_id,
            limit=limit,
        )
        body, headers = self._request_text("GET", path)
        return FineTuningExportDTO(
            content=body,
            filename=self._parse_filename(
                headers.get("content-disposition") or headers.get("Content-Disposition")
            ),
            content_type=headers.get("content-type")
            or headers.get("Content-Type")
            or DEFAULT_EXPORT_CONTENT_TYPE,
        )

    def export_fine_tuning_manifest(
        self,
        *,
        experiment_id: str | None = None,
        experiment_run_id: str | None = None,
        limit: int = 1000,
        include_content: bool = False,
    ) -> FineTuningManifestDTO:
        payload = self._request_json(
            "GET",
            self._build_query_path(
                "/analytics/exports/fine-tuning.manifest",
                experiment_id=experiment_id,
                experiment_run_id=experiment_run_id,
                limit=limit,
                include_content=str(include_content).lower(),
            ),
        )
        return FineTuningManifestDTO.from_dict(
            self._unwrap_data(payload, "fine-tuning manifest")
        )

    def list_export_jobs(
        self,
        *,
        page: int = 1,
        per_page: int = 20,
    ) -> ProjectExportJobListDTO:
        payload = self._request_json(
            "GET",
            self._build_query_path(
                "/analytics/export-jobs",
                page=page,
                per_page=per_page,
            ),
        )
        return ProjectExportJobListDTO.from_dict(
            self._unwrap_data(payload, "export jobs")
        )

    def get_export_job(self, job_id: str) -> ProjectExportJobResponseDTO:
        payload = self._request_json(
            "GET",
            f"/analytics/export-jobs/{quote(job_id, safe='')}",
        )
        return ProjectExportJobResponseDTO.from_dict(
            self._unwrap_data(payload, "export job")
        )

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._request_sender_override is not None:
            response = self._request_sender_override(method, path, payload, "json")
            if not isinstance(response, dict):
                raise ClientError(
                    "Custom request sender must return a dict for JSON requests"
                )
            return cast(dict[str, Any], response)
        return self._request_json_sync(method, path, payload)

    def _request_text(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, str]]:
        if self._request_sender_override is not None:
            response = self._request_sender_override(method, path, payload, "text")
            if (
                not isinstance(response, tuple)
                or len(response) != 2
                or not isinstance(response[0], str)
                or not isinstance(response[1], dict)
            ):
                raise ClientError(
                    "Custom request sender must return a (str, dict[str, str]) tuple for text requests"
                )
            return cast(tuple[str, dict[str, str]], response)
        return self._request_text_sync(method, path, payload)

    def _request_json_sync(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        encoded_payload = None
        if payload is not None:
            encoded_payload = json.dumps(payload).encode("utf-8")

        http_request = request.Request(
            f"{self.config.backend_origin}{self.config.api_path}{path}",
            data=encoded_payload,
            headers=self.config.build_headers(),
            method=method,
        )
        try:
            with request.urlopen(  # nosec B310 - caller-configured backend endpoint
                http_request, timeout=self.config.request_timeout
            ) as response:
                status_code = getattr(response, "status", 200)
                body = response.read().decode("utf-8") if response else ""
                parsed = json.loads(body) if body else {}
                if status_code >= 400:
                    raise ClientError(
                        f"Core metrics request failed with status {status_code}",
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
                    f"Core metrics request rejected with status {exc.code}"
                ) from exc
            raise ClientError(
                f"Core metrics request failed with status {exc.code}",
                status_code=exc.code,
                details={"body": body},
            ) from exc
        except error.URLError as exc:
            raise TraigentConnectionError(
                f"Failed to connect to core metrics backend at {self.config.backend_origin}"
            ) from exc

    def _request_text_sync(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, str]]:
        encoded_payload = None
        if payload is not None:
            encoded_payload = json.dumps(payload).encode("utf-8")

        http_request = request.Request(
            f"{self.config.backend_origin}{self.config.api_path}{path}",
            data=encoded_payload,
            headers=self.config.build_headers(),
            method=method,
        )
        try:
            with request.urlopen(  # nosec B310 - caller-configured backend endpoint
                http_request, timeout=self.config.request_timeout
            ) as response:
                status_code = getattr(response, "status", 200)
                body = response.read().decode("utf-8") if response else ""
                headers = dict(getattr(response, "headers", {}) or {})
                if status_code >= 400:
                    raise ClientError(
                        f"Core metrics request failed with status {status_code}",
                        status_code=status_code,
                        details={"body": body},
                    )
                return body, headers
        except error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8") if exc.fp else ""
            finally:
                exc.close()
            if exc.code in {401, 403}:
                raise AuthenticationError(
                    f"Core metrics request rejected with status {exc.code}"
                ) from exc
            raise ClientError(
                f"Core metrics request failed with status {exc.code}",
                status_code=exc.code,
                details={"body": body},
            ) from exc
        except error.URLError as exc:
            raise TraigentConnectionError(
                f"Failed to connect to core metrics backend at {self.config.backend_origin}"
            ) from exc

    @staticmethod
    def _unwrap_data(payload: dict[str, Any], label: str) -> dict[str, Any]:
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ClientError(f"Unexpected response structure for {label}")
        return data

    @staticmethod
    def _build_query_path(path: str, **params: Any) -> str:
        serialized = urlencode(
            {
                key: value
                for key, value in params.items()
                if value is not None and value != ""
            }
        )
        return f"{path}?{serialized}" if serialized else path

    @staticmethod
    def _parse_filename(content_disposition: str | None) -> str:
        if not content_disposition:
            return DEFAULT_EXPORT_FILENAME
        if "filename*=UTF-8''" in content_disposition:
            return unquote(
                content_disposition.split("filename*=UTF-8''", 1)[1].split(";", 1)[0]
            )
        if "filename=" in content_disposition:
            return content_disposition.split("filename=", 1)[1].strip().strip('"')
        return DEFAULT_EXPORT_FILENAME
