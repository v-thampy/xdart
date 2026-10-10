"""X-ray energy ↔ wavelength conversion — one canonical place.

New acquisition plans select valid scan metadata before the calibration wavelength.
Persisted results keep their recorded wavelength. Existing fitted-energy offsets
are separate from the acquisition energy.

**Legacy calibration consumers use the calibration wavelength**
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
    "xray_energy_eV",
    "xray_energy_keV",
    "calibration_recorded_energy_eV",
    "check_energy_consistency",
    "resolve_acquisition_energy",
    "metadata_energy_values",
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
    must never infer a wavelength's unit from its magnitude (an X-ray energy's
    scale, by contrast, is decided by its size: :func:`xray_energy_eV`).
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


#: A recorded X-ray energy up to this is in keV, any larger one in eV (maintainer,
#: 2026-10-10: "For energies, the rule should be 0-150 read are keV, above 150 in eV.
#: This accounts for soft xrays as well").  No unit is needed, and a mislabel cannot
#: mislead it.
_XRAY_KEV_MAX = 150.0


def _xray_energy_scaled(value, *, to_eV: bool):
    energy = np.asarray(value, dtype=float)
    positive = np.isfinite(energy) & (energy > 0)
    in_keV = positive & (energy <= _XRAY_KEV_MAX)
    if to_eV:
        result = np.where(in_keV, energy * 1000.0, energy)
    else:
        result = np.where(positive & ~in_keV, energy / 1000.0, energy)
    return float(result) if result.ndim == 0 else result


def xray_energy_eV(value):
    """A recorded X-ray energy (a number or an array) in eV, its scale decided by its size.

    Up to 150 it is keV; a larger value is already eV and is returned exactly.  A value
    that is not finite and positive is returned unchanged: its consumer decides what is
    impossible.
    """
    return _xray_energy_scaled(value, to_eV=True)


def xray_energy_keV(value):
    """As :func:`xray_energy_eV`, in keV: a recorded value above 150 is eV."""
    return _xray_energy_scaled(value, to_eV=False)


def _recorded_energy_eV(value):
    """A recorded energy in eV by its size; anything that is not a number is passed on unchanged,
    for :func:`resolve_acquisition_energy` to count as no energy."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return xray_energy_eV(number)


def metadata_energy_values(row, *, energy_is_eV=False, wavelength=None, wavelength_unit=None) -> tuple:
    """Read raw energy by size, canonical eV unchanged, or a declared wavelength, in eV.

    Missing evidence returns no values. An impossible energy (zero, negative or not a
    number) counts as none, so a recorded wavelength applies instead; invalid wavelength
    evidence refuses, and a wavelength's unit is never inferred from its size.
    """
    names = {str(key).casefold(): value for key, value in row.items()}
    if 'energy' in names:
        value = names['energy'] if energy_is_eV else _recorded_energy_eV(names['energy'])
        energy = _possible_energy(value)
        if energy is not None:
            return (energy,)
    values = [(names[name], unit) for name, unit in
              (('wavelength_m', WavelengthUnit.METRE),
               ('wavelength_a', WavelengthUnit.ANGSTROM)) if name in names]
    if not values and wavelength is not None and wavelength_unit is not None:
        values = [(wavelength, wavelength_unit)]
    energies = []
    for value, unit in values:
        metres = canonical_wavelength_m(value, unit)
        if isinstance(value, (bool, np.bool_)) or metres is None:
            raise ValueError('metadata energy from wavelength must be finite and positive')
        energies.append(wavelength_m_to_energy_eV(metres))
    return tuple(energies)


def _possible_energy(value) -> float | None:
    """A recorded energy that is a finite positive number, else None: zero, negative or not a number is none."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def resolve_acquisition_energy(calibration_wavelength_m, metadata_energy_eV=(), *,
                               fitted_source_energy_eV=None, allow_energy_scan=False) -> dict:
    """Freeze the acquisition energy and its explicit fitted correction, without I/O.

    A recorded energy that is zero, negative or not a number counts as none (maintainer,
    2026-10-09): with no valid value the calibration fallback applies, and a glitched frame
    in a valid series takes the series' energy. Valid values must be common within the
    existing 0.1 % source-energy tolerance, unless the caller reduces each frame at its own
    energy (``allow_energy_scan``): a varying series is then an energy scan, and
    ``frame_energies_eV`` holds each value's effective energy in order, a glitched one
    taking the nearest earlier valid frame's (the first valid one's when it leads). The first
    valid value is the representative energy. The calibration itself remains immutable.
    """
    recorded = [_possible_energy(value) for value in metadata_energy_eV]
    valid = [value for value in recorded if value is not None]
    first = valid[0] if valid else None
    scan = any(abs(value - first) > .001 * first for value in valid)
    if scan and not allow_energy_scan:
        raise ValueError("metadata energy must be common across selected frames")
    wavelength = canonical_wavelength_m(calibration_wavelength_m, WavelengthUnit.METRE)
    calibration = None if wavelength is None else wavelength_m_to_energy_eV(wavelength)
    if fitted_source_energy_eV is not None and calibration is None:
        raise ValueError('a fitted energy offset requires a positive calibration wavelength')
    offset = 0. if fitted_source_energy_eV is None else calibration - float(fitted_source_energy_eV)
    base = first if first is not None else (calibration if fitted_source_energy_eV is None
                                   else float(fitted_source_energy_eV))
    effective = None if base is None else base + offset
    if effective is not None and (not np.isfinite(effective) or effective <= 0):
        raise ValueError("effective acquisition energy must be finite and positive")
    reference = calibration if fitted_source_energy_eV is None else float(fitted_source_energy_eV)
    disagreement = bool(first is not None and reference is not None and abs(base - reference) > .001 * reference)
    # Retain the exact accepted wavelength for the fallback and unchanged energy.
    # A hc/(hc/wavelength) round trip can otherwise move it by one ulp.
    if first is None or effective == calibration:
        effective, selected_wavelength = calibration, wavelength
    else:
        selected_wavelength = None if effective is None else energy_eV_to_wavelength_m(effective)
    selection = {"source": "metadata" if first is not None else "calibration", "source_energy_eV": base,
                 "calibration_energy_eV": calibration, "fitted_energy_offset_eV": offset,
                 "energy_eV": effective,
                 "wavelength_m": selected_wavelength,
                 "disagreement": disagreement}
    if scan:
        frames, previous = [], first
        for value in recorded:
            previous = previous if value is None else value
            energy = previous + offset
            if not np.isfinite(energy) or energy <= 0:
                raise ValueError("effective acquisition energy must be finite and positive")
            frames.append(energy)
        selection["frame_energies_eV"] = tuple(frames)
    return selection


def calibration_recorded_energy_eV(record) -> float | None:
    """A goniometer calibration's recorded energy (``source_energy_eV``: kept when its energy was
    refined, and what the scans' energies are checked against), or None when it has none.

    Refused unless an actual finite positive number within 5 % of the calibration's own energy (its
    wavelength's): a NaN or infinite reference would pass any floating comparison (Codex re-review
    R2-F1).
    """
    import math
    if not isinstance(record, dict) or "source_energy_eV" not in record:
        return None
    value, wavelength = record["source_energy_eV"], record.get("wavelength")
    for number in (value, wavelength):
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise ValueError("the calibration's recorded energy (source_energy_eV) and its wavelength must be "
                             "finite positive numbers")
    energy = wavelength_m_to_energy_eV(wavelength)
    if not abs(value - energy) < .05 * energy:
        raise ValueError(f"the calibration's recorded energy {value:g} eV is not within 5 % of its own "
                         f"{energy:.1f} eV")
    return float(value)


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
