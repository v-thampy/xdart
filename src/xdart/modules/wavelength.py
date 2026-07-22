# -*- coding: utf-8 -*-
"""Compatibility re-export shim (X1 Slice 3a0).

The generic wavelength helpers moved to :mod:`xrd_tools.core.energy` — the one
public headless wavelength vocabulary (R3-P1) — so the headless projection and
the GUI share a single definition.  This module re-exports them unchanged for
the remaining xdart importers during the X1 tranche; new/modified code should
import ``xrd_tools.core.energy`` directly.
"""

from __future__ import annotations

from xrd_tools.core.energy import (  # noqa: F401
    DEFAULT_WAVELENGTH_SENTINEL_M,
    is_default_wavelength_sentinel_m,
    normalize_wavelength_m,
    wavelength_angstrom_to_m,
    wavelength_m_to_angstrom,
)

__all__ = [
    "DEFAULT_WAVELENGTH_SENTINEL_M",
    "is_default_wavelength_sentinel_m",
    "normalize_wavelength_m",
    "wavelength_angstrom_to_m",
    "wavelength_m_to_angstrom",
]
