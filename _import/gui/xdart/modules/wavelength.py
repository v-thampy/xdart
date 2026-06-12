# -*- coding: utf-8 -*-
"""Shared wavelength helpers.

``LiveScan.mg_args`` historically defaults to ``{"wavelength": 1e-10}``,
which is 1.0 Angstrom in pyFAI's metre convention. That value is only a
constructor sentinel, not a measured calibration wavelength, so display and
headless adapters must treat it as unknown unless a real integrator or
persisted NeXus wavelength supplies a value.
"""

from __future__ import annotations

DEFAULT_WAVELENGTH_SENTINEL_M = 1.0e-10
_SENTINEL_ATOL_M = 1.0e-14


def is_default_wavelength_sentinel_m(value) -> bool:
    """Whether *value* is the historical ``1e-10`` metre placeholder."""
    try:
        wl = float(value)
    except (TypeError, ValueError):
        return False
    return abs(wl - DEFAULT_WAVELENGTH_SENTINEL_M) <= _SENTINEL_ATOL_M


def normalize_wavelength_m(value, *, allow_default_sentinel: bool = False) -> float | None:
    """Return a real positive wavelength in metres, or ``None``.

    Rejects non-numeric, non-positive, and (by default) the historical
    placeholder value.  Pass ``allow_default_sentinel=True`` only for an
    authoritative source such as a persisted NeXus ``wavelength_A`` field, where
    1.0 Angstrom can be a real beam wavelength rather than a constructor
    default.
    """
    try:
        wl = float(value)
    except (TypeError, ValueError):
        return None
    if wl <= 0:
        return None
    if not allow_default_sentinel and is_default_wavelength_sentinel_m(wl):
        return None
    return wl


def wavelength_m_to_angstrom(value, *, allow_default_sentinel: bool = False) -> float | None:
    wl = normalize_wavelength_m(
        value,
        allow_default_sentinel=allow_default_sentinel,
    )
    return None if wl is None else wl * 1e10


def wavelength_angstrom_to_m(value) -> float | None:
    try:
        wl_a = float(value)
    except (TypeError, ValueError):
        return None
    if wl_a <= 0:
        return None
    return wl_a * 1e-10
