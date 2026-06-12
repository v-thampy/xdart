"""Placeholder for texture analysis (pole figures, ODF).

This module is a reserved namespace. None of the planned functionality
is implemented yet:

* ``chi_series``  — azimuthal intensity profiles vs χ for a peak
* ``pole_figure`` — pole figure generation from multi-orientation scans
* ``odf``         — orientation distribution function estimation

If you need these today, use an external tool (e.g. MTEX, pyEBSDIndex)
and open an issue if you would like to see them natively supported.
"""
from __future__ import annotations


def _not_implemented(*_args, **_kwargs):
    raise NotImplementedError(
        "Texture analysis is not yet implemented in xrd_tools. "
        "See xrd_tools/analysis/texture.py for the planned API."
    )


# Planned public API — all currently raise.
chi_series = _not_implemented
pole_figure = _not_implemented
odf = _not_implemented


__all__: list[str] = []
