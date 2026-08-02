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

__all__ = [
    "UINT32_CEILING",
    "combine_detector_masks",
    "detector_value_mask",
    "integer_saturation_ceiling",
    "saturation_pixels",
]


def combine_detector_masks(
    static_mask,
    frame_mask,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    """Resolve and union static and frame-local detector masks exactly once."""
    from xrd_tools.core.scan import MaskSpec

    def resolved(mask, name: str) -> np.ndarray | None:
        if mask is None:
            return None
        value = (
            mask.to_bool(image_shape)
            if isinstance(mask, MaskSpec)
            else np.asarray(mask, dtype=bool)
        )
        if value.shape != image_shape:
            raise ValueError(
                f"{name} shape {value.shape} does not match "
                f"image shape {image_shape}"
            )
        return value

    static = resolved(static_mask, "static detector mask")
    local = resolved(frame_mask, "frame detector mask")
    if static is None:
        return local
    if local is None:
        return static
    return static | local


def detector_value_mask(
    mask,
    raw_image,
    *,
    enabled: bool,
) -> np.ndarray | None:
    """Union the accepted detector-value policy with an existing static mask.

    The operator toggle controls value masking only.  An existing static mask
    remains authoritative when value masking is disabled.  Enabled masking
    adds negative values, the unambiguous UINT32 dummy, and the existing
    fraction-guarded native integer ceiling.
    """
    if not enabled:
        return mask
    raw = np.asarray(raw_image)
    bad = (raw < 0) | (raw >= UINT32_CEILING)
    bad |= saturation_pixels(
        raw,
        ceiling=integer_saturation_ceiling(raw),
    )
    if mask is None:
        return bad if bad.any() else None
    static = np.asarray(mask, dtype=bool)
    if static.shape != raw.shape:
        raise ValueError(
            f"detector mask shape {static.shape} does not match "
            f"image shape {raw.shape}"
        )
    return static if not bad.any() else (static | bad)


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
