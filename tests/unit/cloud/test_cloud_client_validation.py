"""Validation tests for TraigentCloudClient."""

import pytest

from traigent.cloud.client import TraigentCloudClient
from traigent.utils.exceptions import ValidationError as ValidationException

# SDK #2033: opt into the connected/backend code paths (see pyproject markers).
pytestmark = pytest.mark.backend_online


def test_cloud_client_rejects_blank_api_key():
    with pytest.raises(ValidationException):
        TraigentCloudClient(api_key="   ")


def test_cloud_client_rejects_invalid_base_url():
    with pytest.raises(ValidationException):
        TraigentCloudClient(api_key="test-key", base_url="://invalid")
