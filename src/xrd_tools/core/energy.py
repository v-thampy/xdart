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

import numpy as np

logger = logging.getLogger(__name__)

#: h·c in eV·m (CODATA 2018): E[eV] = _HC_EV_M / λ[m]  (≡ 12398.42 eV·Å).
_HC_EV_M = 1.239841984e-6

__all__ = [
    "energy_eV_to_wavelength_m",
    "wavelength_m_to_energy_eV",
    "check_energy_consistency",
]


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
