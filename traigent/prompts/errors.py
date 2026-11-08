"""Errors raised by the prompt management client.

The backend reports prompt failures as ``application/problem+json`` documents
with a stable ``code``. :func:`error_from_problem` turns one into the matching
exception below. Every class derives from
:class:`~traigent.utils.exceptions.ClientError`, so ``except ClientError``
keeps catching all of them.

A ``404`` is deliberately ambiguous: the project, the prompt or the version may
be missing, or the caller may not be allowed to see them. The backend does not
say which, and neither does :class:`PromptNotFoundError`.
"""

from __future__ import annotations

from typing import Any

from traigent.utils.exceptions import ClientError


class PromptApiError(ClientError):
    """A prompt request was rejected with a problem document.

    Attributes:
        code: Stable machine-readable error code (for example ``not_found``).
        status_code: HTTP status of the response.
        problem_id: Identifier of this failure, useful when contacting support.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        problem_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            details={"code": code, "problem_id": problem_id},
        )
        self.code = code
        self.problem_id = problem_id


class PromptNotFoundError(PromptApiError):
    """Not found, or not visible to the caller (``404 not_found``)."""


class PromptInvalidRequestError(PromptApiError):
    """The request was malformed.

    Covers ``invalid_page_size``, ``invalid_cursor`` and ``validation_failed``.
    """


class PromptConflictError(PromptApiError):
    """The prompt or version already exists, or changed concurrently.

    Covers ``prompt_already_exists`` and ``prompt_version_conflict``.
    """


class PromptNotPermittedError(PromptApiError):
    """The plan or policy does not allow the operation.

    Covers ``feature_not_entitled`` and ``provider_not_allowed``.
    """


class PromptPaymentRequiredError(PromptApiError):
    """The account balance does not allow the operation.

    Covers ``insufficient_balance`` and ``spend_blocked``.
    """


class PromptUpstreamError(PromptApiError):
    """The model call made by the playground failed (``upstream_failed``)."""


_BY_CODE: dict[str, type[PromptApiError]] = {
    "not_found": PromptNotFoundError,
    "invalid_page_size": PromptInvalidRequestError,
    "invalid_cursor": PromptInvalidRequestError,
    "validation_failed": PromptInvalidRequestError,
    "prompt_already_exists": PromptConflictError,
    "prompt_version_conflict": PromptConflictError,
    "feature_not_entitled": PromptNotPermittedError,
    "provider_not_allowed": PromptNotPermittedError,
    "insufficient_balance": PromptPaymentRequiredError,
    "spend_blocked": PromptPaymentRequiredError,
    "upstream_failed": PromptUpstreamError,
}

_BY_STATUS: dict[int, type[PromptApiError]] = {
    400: PromptInvalidRequestError,
    402: PromptPaymentRequiredError,
    403: PromptNotPermittedError,
    404: PromptNotFoundError,
    409: PromptConflictError,
    422: PromptInvalidRequestError,
    502: PromptUpstreamError,
}


def error_from_problem(status: int, problem: dict[str, Any]) -> PromptApiError:
    """Build the exception for a problem document.

    The matching is by ``code`` first and by HTTP status when the code is
    unknown, so a code added by a newer backend still maps to a sensible class.
    Only the document's own fixed text is used in the message.
    """
    raw_code = problem.get("code")
    code = raw_code if isinstance(raw_code, str) else None
    raw_id = problem.get("problem_id")
    problem_id = raw_id if isinstance(raw_id, str) else None
    cls = _BY_CODE.get(code or "") or _BY_STATUS.get(status) or PromptApiError
    title = problem.get("title")
    summary = title if isinstance(title, str) and title else "Prompt request failed"
    detail = problem.get("detail")
    message = f"{summary} (status {status}, code {code or 'unknown'})"
    if isinstance(detail, str) and detail:
        message = f"{message}: {detail}"
    return cls(message, code=code, status_code=status, problem_id=problem_id)
