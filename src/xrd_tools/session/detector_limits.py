# -*- coding: utf-8 -*-
"""Pure, Qt-free detector-family DISPLAY-DEFAULT saturation ceiling.

SCOPE (review 2026-08-04): this is a beamline-scoped GUI seed, NOT an
acquisition-dtype oracle.  A PONI names a detector family but carries no
bit-depth fact, and one family can legitimately deliver more than one raw
dtype (the supported Eiger path preserves uint16 AND uint32 frames —
``xdart.modules.ewald.frame``).  The value produced here therefore only SEEDS
the manual-threshold inputs (LV-UI-11, maintainer-confirmed 0..detector-max
defaults); saturated-pixel masking itself always derives its ceiling from the
ACQUIRED frame's own dtype at reduction time
(:func:`xrd_tools.core.invalid.integer_saturation_ceiling` —
``np.iinfo(frame.dtype).max``) and is never governed by this table.

The table lists the typical raw stream of the families this deployment has
verified (SSRL beamlines); an unknown or absent detector returns ``None`` and
the GUI shows a blank default rather than a guessed number.  A family default
that overshoots a narrower acquisition (uint32 ceiling seeded for a uint16
Eiger stream) is benign as a manual bound — a max above the data range clips
nothing.  Extend the table with a family's verified typical stream dtype,
never with a hand-derived physical well depth.
"""

from __future__ import annotations

import numpy as np

#: Detector family (normalized-name prefix) -> TYPICAL raw stream dtype for
#: the deployments this codebase has verified: Eiger/Eiger2 HDF5 streams here
#: are uint32 (the stream carrying the 4294967295 saturation sentinel);
#: Rayonix TIFF and Perkin-Elmer are uint16 (ceiling 65535).  Display-default
#: scope only — see the module docstring.
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
    """The DISPLAY-DEFAULT ceiling for a named detector's family, or ``None``
    when the family is not in the table.

    A family does NOT determine an acquisition's raw dtype (the supported
    Eiger path delivers uint16 and uint32 alike); this value is the family's
    typical raw-stream ceiling, used only to seed the GUI's manual-threshold
    inputs.  Reduction-time saturation masking keys off the acquired frame's
    own dtype and never reads this table."""
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
