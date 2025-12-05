"""Validation tests for TraiGentCloudClient."""

import pytest

from traigent.cloud.client import TraiGentCloudClient
from traigent.utils.exceptions import ValidationError as ValidationException


def test_cloud_client_rejects_blank_api_key():
    with pytest.raises(ValidationException):
        TraiGentCloudClient(api_key="   ")


def test_cloud_client_rejects_invalid_base_url():
    with pytest.raises(ValidationException):
        TraiGentCloudClient(api_key="test-key", base_url="://invalid")
