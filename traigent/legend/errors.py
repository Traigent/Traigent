"""Legend errors. Messages are fixed text: never a name, path or customer content."""

from __future__ import annotations


class LegendError(ValueError):
    """A Legend operation was refused locally or by the transport.

    ``code`` is a closed token (a ``LegendErrorCode`` value where one applies,
    else a local ``local_*`` token). The message never echoes caller input.
    """

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)
