"""Traigent backend and reserved cloud integration module.

This module provides backend integration for portal-tracked SDK runs. Remote
cloud execution is reserved for a future release and fails closed today.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

from importlib import import_module

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
    "QuotaExceededErrorDTO": ("traigent.cloud.dtos", "QuotaExceededErrorDTO"),
    "WalletInsufficientBalanceErrorDTO": (
        "traigent.cloud.dtos",
        "WalletInsufficientBalanceErrorDTO",
    ),
    "WalletTopUpPackDTO": ("traigent.cloud.dtos", "WalletTopUpPackDTO"),
    "WalletTopUpPacksResponseDTO": (
        "traigent.cloud.dtos",
        "WalletTopUpPacksResponseDTO",
    ),
    "ExperimentGroupSourceExperimentDTO": (
        "traigent.cloud.dtos",
        "ExperimentGroupSourceExperimentDTO",
    ),
    "GroupedConfigurationRunRowDTO": (
        "traigent.cloud.dtos",
        "GroupedConfigurationRunRowDTO",
    ),
    "ExperimentGroupOverviewDTO": (
        "traigent.cloud.dtos",
        "ExperimentGroupOverviewDTO",
    ),
    "ExperimentGroupDetailDTO": (
        "traigent.cloud.dtos",
        "ExperimentGroupDetailDTO",
    ),
    "ExperimentGroupStatusSummaryDTO": (
        "traigent.cloud.dtos",
        "ExperimentGroupStatusSummaryDTO",
    ),
    "ExperimentGroupsPageDTO": (
        "traigent.cloud.dtos",
        "ExperimentGroupsPageDTO",
    ),
    "GroupedConfigurationRunsPageDTO": (
        "traigent.cloud.dtos",
        "GroupedConfigurationRunsPageDTO",
    ),
    "UsageTracker": ("traigent.cloud.billing", "UsageTracker"),
    "SmartSubsetSelector": ("traigent.cloud.subset_selection", "SmartSubsetSelector"),
    "DiverseSampling": ("traigent.cloud.subset_selection", "DiverseSampling"),
    "RepresentativeSampling": (
        "traigent.cloud.subset_selection",
        "RepresentativeSampling",
    ),
    "HighConfidenceSampling": (
        "traigent.cloud.subset_selection",
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
