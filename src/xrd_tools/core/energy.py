"""X-ray energy ↔ wavelength conversion — one canonical place.

**The canonical energy source for an experiment is the calibration wavelength**
(it persists under ``/entry/diffractometer`` and feeds the pyFAI integrators).
``RSMPlan.energy`` and ``GICorrectionStack.energy_eV`` are conveniences that must
be **consistent** with it — :func:`check_energy_consistency` warns on divergence
so a single GUI energy widget (bound to the calibration wavelength) can never
silently disagree with the persisted file.

The conversion was previously duplicated (``12398/λ`` in the RSM pipeline vs
``xrayutilities.en2lam`` in the GI corrections); this is the single definition.
"""
from __future__ import annotations

import logging
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

#: h·c in eV·m (CODATA 2018): E[eV] = _HC_EV_M / λ[m]  (≡ 12398.42 eV·Å).
_HC_EV_M = 1.239841984e-6

__all__ = [
    "energy_eV_to_wavelength_m",
    "wavelength_m_to_energy_eV",
    "check_energy_consistency",
    # X1 Slice 3a0 (R3-P1): the one headless wavelength vocabulary.
    "WavelengthUnit",
    "canonical_wavelength_m",
    "DEFAULT_WAVELENGTH_SENTINEL_M",
    "is_default_wavelength_sentinel_m",
    "normalize_wavelength_m",
    "wavelength_m_to_angstrom",
    "wavelength_angstrom_to_m",
]


class WavelengthUnit(str, Enum):
    """Explicit wavelength unit declaration (X1 R3-P1).

    A wavelength value is evidence only when its unit is DECLARED by its
    source — the NeXus container descriptor reports angstroms, PONI/run values
    are metres, and an unqualified number has no enforceable unit.  Consumers
    must never infer a unit from value magnitude.
    """

    METRE = "m"
    ANGSTROM = "angstrom"


def canonical_wavelength_m(value, unit: "WavelengthUnit | None") -> float | None:
    """Canonicalize an explicitly unit-declared wavelength to metres.

    ``None`` when the unit is undeclared (no evidence — never magnitude
    inference), or the value is non-numeric / non-finite / non-positive.
    Sentinel rejection is provenance-sensitive and does NOT apply here: an
    explicitly declared ``1.0 angstrom`` (or ``1e-10 m``) source is valid
    physical evidence even though it equals the historical untrusted
    constructor placeholder (see :func:`normalize_wavelength_m`).
    """
    if unit is None:
        return None
    try:
        wl = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(wl) or wl <= 0.0:
        return None
    if unit is WavelengthUnit.ANGSTROM:
        return wl * 1e-10
    return wl


# ── historical default-sentinel handling (formerly in xdart) ────────────────
#
# ``LiveScan.mg_args`` historically defaults to ``{"wavelength": 1e-10}``,
# which is 1.0 Angstrom in pyFAI's metre convention.  That value is only a
# constructor sentinel, not a measured calibration wavelength, so display and
# headless adapters must treat it as unknown unless a real integrator or
# persisted NeXus wavelength supplies a value.

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


def energy_eV_to_wavelength_m(energy_eV: float) -> float:
    """X-ray energy (eV) → wavelength (m)."""
    return _HC_EV_M / float(energy_eV)


def wavelength_m_to_energy_eV(wavelength_m: float) -> float:
    """X-ray wavelength (m) → energy (eV)."""
    return _HC_EV_M / float(wavelength_m)


def check_energy_consistency(
    energy_a_eV: float | None,
    energy_b_eV: float | None,
    *,
    what_a: str,
    what_b: str,
    rtol: float = 1.0e-3,
) -> None:
    """Warn if two energy sources disagree by more than ``rtol`` (default 0.1%).

    No-op when either is ``None``/non-finite.  Used at the run sites to surface a
    GI/correction energy that diverges from the canonical calibration energy
    (instead of silently using inconsistent optical constants / q-conversion).
    """
    if energy_a_eV is None or energy_b_eV is None:
        return
    a, b = float(energy_a_eV), float(energy_b_eV)
    if not (np.isfinite(a) and np.isfinite(b)):
        return
    if abs(a - b) > rtol * max(abs(a), abs(b), 1.0):
        logger.warning(
            "X-ray energy mismatch: %s=%.1f eV vs %s=%.1f eV (>%.2f%%). The "
            "calibration wavelength is the canonical source — make them agree.",
            what_a, a, what_b, b, rtol * 100.0)
