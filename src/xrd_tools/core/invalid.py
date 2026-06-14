"""Detector invalid-pixel policy (headless, numpy-only).

The reduction-relevant half of what was an xdart-GUI-only policy (R3-C): the
dtype-derived saturation ceiling and the fraction-guarded saturation mask now
live here so headless ``xrd_tools`` callers can exclude the same dead/overflowed
detector pixels the GUI does — instead of every consumer re-deriving it.

Two ceilings, two policies:

* the **uint32 dead/hot dummy** (``UINT32_CEILING`` = 4294967295, e.g. Eiger
  masters): unambiguous — never a real photon count — so a caller masks it
  ALWAYS, alongside non-finite values.  Not gated by anything here.
* the **detector saturation ceiling** (``np.iinfo(dtype).max`` — 65535 for
  uint16, 255 for uint8, …): AMBIGUOUS — both the max real count and a common
  overflow sentinel — so it is masked only when a whole module sits there
  (:func:`saturation_pixels`'s ``min_fraction`` guard) and only when the caller
  opts in.

Deliberately dtype-derived and **never hardcodes 65535**: a float frame whose
integer dtype was lost upstream returns ``None`` from
:func:`integer_saturation_ceiling`, leaving the fallback (if any) to the caller
— the GUI keeps its legacy 65535 fallback in xdart, out of core.
"""

from __future__ import annotations

import numpy as np

#: uint32 max — the unambiguous dead/hot-pixel dummy (Eiger masters etc.).
UINT32_CEILING = 4294967295.0

__all__ = ["UINT32_CEILING", "integer_saturation_ceiling", "saturation_pixels"]


def integer_saturation_ceiling(arr) -> float | None:
    """The saturation ceiling implied by an array's integer dtype
    (``np.iinfo(dtype).max`` — 65535 for uint16, 255 for uint8; numpy has no
    12-bit type so 4095 never arises), learned from the detector bit depth
    rather than assuming 16-bit.

    Returns ``None`` when ``arr`` is already float — the original integer dtype
    was lost upstream (e.g. after a threshold/background float conversion) — so
    the caller chooses any fallback.  Core never hardcodes 65535.
    """
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.integer):
        return float(np.iinfo(a.dtype).max)
    return None


def saturation_pixels(values, *, ceiling, min_fraction: float = 1e-4) -> np.ndarray:
    """Boolean mask (same shape as ``values``) of the ambiguous detector-
    saturation pixels: values exactly at ``ceiling``, but ONLY when more than
    ``min_fraction`` of the frame sits there — a dead/overflowed module, not a
    handful of genuinely-saturated Bragg pixels.

    Returns an all-``False`` mask (never raises) when ``ceiling`` is ``None``,
    the frame is empty, or ``ceiling >= UINT32_CEILING`` (that ceiling is the
    unambiguous dead sentinel — a caller masks it always, not through this
    opt-in gate).  Equality with a finite ceiling already excludes NaN/inf, so
    no separate finite guard is needed.
    """
    a = np.asarray(values)
    out = np.zeros(a.shape, dtype=bool)
    if ceiling is None or a.size == 0 or float(ceiling) >= UINT32_CEILING:
        return out
    sat = (a == float(ceiling))
    if sat.any() and sat.sum() / a.size > float(min_fraction):
        return sat
    return out
