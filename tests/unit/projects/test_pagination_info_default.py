from __future__ import annotations

import pytest

from traigent.observability.dtos import PaginationInfo as ObservabilityPaginationInfo
from traigent.projects.dtos import PaginationInfo as ProjectsPaginationInfo
from traigent.prompts.dtos import PaginationInfo as PromptsPaginationInfo


@pytest.mark.unit
@pytest.mark.parametrize(
    "pagination_info_cls",
    [
        ProjectsPaginationInfo,
        PromptsPaginationInfo,
        ObservabilityPaginationInfo,
    ],
)
def test_pagination_info_missing_per_page_defaults_to_twenty(
    pagination_info_cls: type[
        ProjectsPaginationInfo | PromptsPaginationInfo | ObservabilityPaginationInfo
    ],
) -> None:
    assert pagination_info_cls.from_dict({}).per_page == 20
