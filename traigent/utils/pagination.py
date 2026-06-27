"""Generic auto-pagination helper for Traigent list endpoints.

Every paginated ``list_*`` response in the SDK carries a ``pagination``
field (a :class:`~traigent.prompts.dtos.PaginationInfo`) with ``has_next``
and ``page`` attributes.  This module provides a single private helper that
any client module can use to implement transparent ``iter_*`` companions
without duplicating the while-loop boilerplate across every caller.

Usage (within a client)::

    from traigent.utils.pagination import iter_pages

    def iter_evaluators(self, *, per_page: int = 100, **filters):
        yield from iter_pages(self.list_evaluators, per_page=per_page, **filters)

The helper is intentionally *not* exported from the public ``traigent``
top-level namespace — it is a client-implementation detail.
"""

from __future__ import annotations

from typing import Any, TypeVar
from collections.abc import Callable, Generator

# The concrete item type yielded by the iterator (covariant).
_T = TypeVar("_T")


def iter_pages(
    list_fn: Callable[..., Any],
    *positional: Any,
    per_page: int = 100,
    **filters: Any,
) -> Generator[Any, None, None]:
    """Iterate transparently over all pages of a paginated list endpoint.

    Parameters
    ----------
    list_fn:
        A bound ``list_*`` method whose first keyword arguments are ``page``
        and ``per_page`` and whose return value has ``.items`` (list) and
        ``.pagination.has_next`` (bool).
    *positional:
        Positional arguments forwarded verbatim to *list_fn* on every call
        (e.g. the ``queue_id`` required by
        :meth:`~traigent.evaluation.client.EvaluationClient.list_annotation_queue_items`).
    per_page:
        Number of items to request per page.  Defaults to 100 to minimise
        round-trips while staying within typical server limits.
    **filters:
        Keyword filter arguments forwarded verbatim to *list_fn* on every
        call (e.g. ``search=``, ``measure_id=``, ``status=`` …).

    Yields
    ------
    object
        Every item across all pages, in order.

    Notes
    -----
    * The loop exits as soon as ``response.pagination.has_next`` is
      ``False`` — the same signal the caller would use when hand-rolling.
    * An infinite-loop guard: if the page number returned by the server does
      *not* advance between two consecutive fetches the loop aborts.  This
      handles pathological backends that always return ``has_next=True``.
    * An empty first page (zero items) terminates immediately.
    """
    page = 1
    seen_pages: set[int] = set()

    while True:
        response = list_fn(*positional, page=page, per_page=per_page, **filters)

        pagination = response.pagination

        # Infinite-loop guard: abort if the server returns a page we have
        # already processed.  This handles pathological backends that always
        # return ``has_next=True`` but never advance the page number.  We
        # check *before* yielding so duplicate items are never emitted.
        if pagination.page in seen_pages:
            return

        seen_pages.add(pagination.page)
        yield from response.items

        if not pagination.has_next:
            return

        page = pagination.page + 1
