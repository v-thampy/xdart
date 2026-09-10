"""Detector invalid-pixel policy (headless, numpy-only).

Mask Saturated excludes every native integer ceiling pixel, including isolated
ones, alongside negative detector values and the uint32 invalid sentinel.
The caller's toggle controls all finite value exclusions; static masks remain
independent. Active manual intensity thresholds take precedence at the run-policy
boundary, so they do not require an additional saturation pass.

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
    adds negative values, the UINT32 dummy, and every native integer ceiling.
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


def saturation_pixels(values, *, ceiling) -> np.ndarray:
    """Select every pixel at the native saturation ceiling, without a count gate."""
    a = np.asarray(values)
    if ceiling is None:
        return np.zeros(a.shape, dtype=bool)
    return a == float(ceiling)
