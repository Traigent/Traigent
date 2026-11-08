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

import re
from typing import Any

from traigent.utils.exceptions import ClientError

#: Problem codes the prompts routes return. Any other code is treated as unknown.
KNOWN_PROBLEM_CODES: frozenset[str] = frozenset(
    {
        "not_found",
        "invalid_page_size",
        "invalid_cursor",
        "validation_failed",
        "prompt_already_exists",
        "prompt_version_conflict",
        "feature_not_entitled",
        "provider_not_allowed",
        "insufficient_balance",
        "spend_blocked",
        "upstream_failed",
    }
)

_PROBLEM_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


class PromptApiError(ClientError):
    """A prompt request was rejected with a problem document.

    Only the closed ``code``, a pattern-checked ``problem_id`` and the HTTP
    status are kept. The server's ``title`` and ``detail`` text is never copied
    onto the exception, and the message is a fixed string per class.

    Attributes:
        code: A known machine-readable error code (for example ``not_found``),
            or ``None`` when the server sent an unknown one.
        status_code: HTTP status of the response.
        problem_id: Identifier of this failure, useful when contacting support,
            or ``None`` when it did not look like an identifier.
    """

    default_message = "Prompt request failed"

    def __init__(
        self,
        *,
        code: str | None = None,
        status_code: int | None = None,
        problem_id: str | None = None,
    ) -> None:
        safe_code = code if code in KNOWN_PROBLEM_CODES else None
        safe_id = (
            problem_id
            if isinstance(problem_id, str) and _PROBLEM_ID_PATTERN.fullmatch(problem_id)
            else None
        )
        super().__init__(
            self.default_message,
            status_code=status_code,
            details={"code": safe_code, "problem_id": safe_id},
        )
        self.code = safe_code
        self.problem_id = safe_id


class PromptNotFoundError(PromptApiError):
    """Not found, or not visible to the caller (``404 not_found``)."""

    default_message = "Prompt resource not found"


class PromptInvalidRequestError(PromptApiError):
    """The request was malformed.

    Covers ``invalid_page_size``, ``invalid_cursor`` and ``validation_failed``.
    """

    default_message = "Prompt request was rejected as invalid"


class PromptConflictError(PromptApiError):
    """The prompt or version already exists, or changed concurrently.

    Covers ``prompt_already_exists`` and ``prompt_version_conflict``.
    """

    default_message = "Prompt request conflicts with existing state"


class PromptNotPermittedError(PromptApiError):
    """The plan or policy does not allow the operation.

    Covers ``feature_not_entitled`` and ``provider_not_allowed``.
    """

    default_message = "Prompt operation is not permitted"


class PromptPaymentRequiredError(PromptApiError):
    """The account balance does not allow the operation.

    Covers ``insufficient_balance`` and ``spend_blocked``.
    """

    default_message = "Prompt operation blocked by balance or spend limits"


class PromptUpstreamError(PromptApiError):
    """The model call made by the playground failed (``upstream_failed``)."""

    default_message = "Prompt upstream model call failed"


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


class PromptPaginationError(ClientError):
    """Cursor iteration cannot make progress.

    Raised when the backend repeats a cursor already used, or reports
    ``has_more`` without a ``next_cursor``. Iteration stops rather than looping
    or silently dropping the remaining pages.
    """


def error_from_problem(status: int, problem: dict[str, Any]) -> PromptApiError:
    """Build the exception for a problem document.

    The class is chosen by ``code`` only. A code outside the known list gives
    the generic :class:`PromptApiError`, whatever the HTTP status. Nothing the
    server wrote as free text (``title``, ``detail``) is kept.
    """
    raw_code = problem.get("code")
    code = raw_code if isinstance(raw_code, str) else None
    raw_id = problem.get("problem_id")
    problem_id = raw_id if isinstance(raw_id, str) else None
    cls = _BY_CODE.get(code or "") or PromptApiError
    return cls(code=code, status_code=status, problem_id=problem_id)
