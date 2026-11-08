"""Every non-deprecated lazy root export is declared in ``traigent.__all__``."""

import traigent


def test_undeclared_lazy_exports_are_exactly_the_deprecated_aliases() -> None:
    undeclared = {n for n in traigent._LAZY_EXPORTS if n not in traigent.__all__}
    assert undeclared == set(traigent._DEPRECATED_LAZY_EXPORTS)


def test_vendor_pause_error_is_public_root_export() -> None:
    assert "VendorPauseError" in traigent.__all__
    from traigent.utils.exceptions import VendorPauseError

    assert traigent.VendorPauseError is VendorPauseError
