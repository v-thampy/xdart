# -*- coding: utf-8 -*-
"""Pure, Qt-free detector raw-dtype saturation-ceiling policy.

Single source of truth for "the max intensity used by Mask Saturated" shown as
the GUI's manual-threshold default (LV-UI-11).  Saturated-pixel masking itself
is dtype-derived at reduction time (:func:`xrd_tools.core.invalid.
integer_saturation_ceiling` — ``np.iinfo(frame.dtype).max``); pyFAI detector
models carry no bit-depth fact, so this module maps the detector FAMILY named
in the PONI to the raw dtype that family's frames arrive in, and returns that
dtype's integer ceiling.

The table lists only families whose raw dtype this codebase has verified
against real data; an unknown or absent detector returns ``None`` — the GUI
shows a blank default rather than a guessed number.  Extend the table with the
family's raw dtype when a new detector is validated, never with a hand-derived
physical well depth (the masking keys off the FILE dtype, not the sensor).
"""

from __future__ import annotations

import numpy as np

#: Detector family (normalized-name prefix) -> raw frame dtype, as written by
#: the beamline file formats this codebase has verified: Eiger/Eiger2 HDF5 is
#: uint32 (its saturation sentinel is the uint32 max, 4294967295); Rayonix TIFF
#: and Perkin-Elmer are uint16 (ceiling 65535).
DETECTOR_FAMILY_RAW_DTYPES: dict[str, str] = {
    "eiger": "uint32",
    "rayonix": "uint16",
    "perkin": "uint16",
}


def _normalized(detector: object) -> str:
    """Lowercase alphanumeric form of a detector name (``Eiger2 CdTe 1M`` ->
    ``eiger2cdte1m``), so family prefixes match across pyFAI aliases."""
    text = "" if detector is None else str(detector)
    return "".join(ch for ch in text.lower() if ch.isalnum())


def detector_saturation_ceiling(detector: object) -> float | None:
    """The raw-frame integer saturation ceiling for a named detector, or
    ``None`` when the family (and therefore the raw dtype) is not known."""
    name = _normalized(detector)
    if not name:
        return None
    for family, dtype in DETECTOR_FAMILY_RAW_DTYPES.items():
        if name.startswith(family):
            return float(np.iinfo(np.dtype(dtype)).max)
    return None


__all__ = [
    "DETECTOR_FAMILY_RAW_DTYPES",
    "detector_saturation_ceiling",
]
