"""Traigent backend and reserved cloud integration module.

This module provides backend integration for portal-tracked SDK runs. Remote
cloud execution is reserved for a future release and fails closed today.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

from importlib import import_module

_TRAIGENT_CLOUD_DTOS = "traigent.cloud.dtos"
_TRAIGENT_CLOUD_SUBSET_SELECTION = "traigent.cloud.subset_selection"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "TraigentCloudClient": ("traigent.cloud.client", "TraigentCloudClient"),
    "TraigentCloudService": ("traigent.cloud.service", "TraigentCloudService"),
    "TraigentCloudRemoteGuidanceAdapter": (
        "traigent.cloud.remote_guidance",
        "TraigentCloudRemoteGuidanceAdapter",
    ),
    "AuthManager": ("traigent.cloud.auth", "AuthManager"),
    "APIKey": ("traigent.cloud.auth", "APIKey"),
    "BillingManager": ("traigent.cloud.billing", "BillingManager"),
    "QuotaExceededErrorDTO": (_TRAIGENT_CLOUD_DTOS, "QuotaExceededErrorDTO"),
    "WalletInsufficientBalanceErrorDTO": (
        _TRAIGENT_CLOUD_DTOS,
        "WalletInsufficientBalanceErrorDTO",
    ),
    "WalletTopUpPackDTO": (_TRAIGENT_CLOUD_DTOS, "WalletTopUpPackDTO"),
    "WalletTopUpPacksResponseDTO": (
        _TRAIGENT_CLOUD_DTOS,
        "WalletTopUpPacksResponseDTO",
    ),
    "ExperimentGroupSourceExperimentDTO": (
        _TRAIGENT_CLOUD_DTOS,
        "ExperimentGroupSourceExperimentDTO",
    ),
    "GroupedConfigurationRunRowDTO": (
        _TRAIGENT_CLOUD_DTOS,
        "GroupedConfigurationRunRowDTO",
    ),
    "RunGroupRow": ("traigent.cloud.run_group_table", "RunGroupRow"),
    "RunGroupTable": ("traigent.cloud.run_group_table", "RunGroupTable"),
    "ExperimentGroupOverviewDTO": (
        _TRAIGENT_CLOUD_DTOS,
        "ExperimentGroupOverviewDTO",
    ),
    "ExperimentGroupDetailDTO": (
        _TRAIGENT_CLOUD_DTOS,
        "ExperimentGroupDetailDTO",
    ),
    "ExperimentGroupStatusSummaryDTO": (
        _TRAIGENT_CLOUD_DTOS,
        "ExperimentGroupStatusSummaryDTO",
    ),
    "ExperimentGroupsPageDTO": (
        _TRAIGENT_CLOUD_DTOS,
        "ExperimentGroupsPageDTO",
    ),
    "GroupedConfigurationRunsPageDTO": (
        _TRAIGENT_CLOUD_DTOS,
        "GroupedConfigurationRunsPageDTO",
    ),
    "UsageTracker": ("traigent.cloud.billing", "UsageTracker"),
    "SmartSubsetSelector": (_TRAIGENT_CLOUD_SUBSET_SELECTION, "SmartSubsetSelector"),
    "DiverseSampling": (_TRAIGENT_CLOUD_SUBSET_SELECTION, "DiverseSampling"),
    "RepresentativeSampling": (
        _TRAIGENT_CLOUD_SUBSET_SELECTION,
        "RepresentativeSampling",
    ),
    "HighConfidenceSampling": (
        _TRAIGENT_CLOUD_SUBSET_SELECTION,
        "HighConfidenceSampling",
    ),
}

__all__ = [
    "TraigentCloudClient",
    "TraigentCloudRemoteGuidanceAdapter",
    "TraigentCloudService",
    "AuthManager",
    "APIKey",
    "BillingManager",
    "QuotaExceededErrorDTO",
    "WalletInsufficientBalanceErrorDTO",
    "WalletTopUpPackDTO",
    "WalletTopUpPacksResponseDTO",
    "ExperimentGroupSourceExperimentDTO",
    "GroupedConfigurationRunRowDTO",
    "ExperimentGroupOverviewDTO",
    "ExperimentGroupDetailDTO",
    "ExperimentGroupStatusSummaryDTO",
    "ExperimentGroupsPageDTO",
    "GroupedConfigurationRunsPageDTO",
    "RunGroupRow",
    "RunGroupTable",
    "UsageTracker",
    "SmartSubsetSelector",
    "DiverseSampling",
    "RepresentativeSampling",
    "HighConfidenceSampling",
]


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
