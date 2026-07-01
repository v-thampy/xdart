"""Rectangular ROI spec + reducer math (headless, numpy-only).

The geometry/reducer half of the ROI-stats feature (see
``docs/design/design_roi_stats_plotting_jun2026.md`` §3 and
``design_scan_plotter_metadata_roi_jun2026.md``).  A :class:`RoiSpec` is a
rectangle in RAW detector-pixel coordinates; :func:`roi_reduce` reduces it over
VALID pixels (excluding masked/dead/saturated, via :mod:`xrd_tools.core.invalid`
so it matches the reducer/display — R3-C).  The scan-level driver
``run_roi_stats`` (in ``analysis/plans.py``) builds the per-frame series on top.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xrd_tools.core.invalid import (
    UINT32_CEILING,
    integer_saturation_ceiling,
    saturation_pixels,
)

_REDUCERS = ("mean", "sum", "max", "min", "std")


@dataclass(frozen=True, slots=True)
class RoiSpec:
    """A rectangle on the RAW frame, in detector-pixel coords.

    ``center_x`` / ``width_x`` are along the COLUMN axis, ``center_y`` /
    ``width_y`` along the ROW axis (image shape is ``(rows, cols)``)."""

    center_x: float
    center_y: float
    width_x: float
    width_y: float
    name: str = ""

    @classmethod
    def full_frame(cls, name: str = "full") -> "RoiSpec":
        """An ROI that clamps to the whole detector (the default ROI)."""
        return cls(center_x=0.0, center_y=0.0, width_x=1e12, width_y=1e12, name=name)

    def pixel_slice(self, image_shape) -> "tuple[slice, slice]":
        """``(row_slice, col_slice)`` clamped to the image bounds; at least 1 px
        in each axis where the image allows."""
        n_rows, n_cols = int(image_shape[0]), int(image_shape[1])
        r0 = int(round(self.center_y - self.width_y / 2.0))
        r1 = int(round(self.center_y + self.width_y / 2.0))
        c0 = int(round(self.center_x - self.width_x / 2.0))
        c1 = int(round(self.center_x + self.width_x / 2.0))
        r0 = max(0, min(r0, n_rows))
        r1 = max(0, min(r1, n_rows))
        c0 = max(0, min(c0, n_cols))
        c1 = max(0, min(c1, n_cols))
        if r1 <= r0:
            r1 = min(r0 + 1, n_rows)
        if c1 <= c0:
            c1 = min(c0 + 1, n_cols)
        return slice(r0, r1), slice(c0, c1)


def invalid_pixel_mask(image, *, mask_saturation: bool = False) -> np.ndarray:
    """Boolean mask (True = exclude) of invalid pixels — the SAME policy the
    reducer applies (R3-C): non-finite + the unambiguous uint32 dead/hot dummy
    ALWAYS; the dtype-derived saturation ceiling only when ``mask_saturation``
    AND a whole module sits there (the fraction guard in
    :func:`xrd_tools.core.invalid.saturation_pixels`)."""
    a = np.asarray(image)
    bad = ~np.isfinite(a)
    bad |= (a == UINT32_CEILING)
    if mask_saturation:
        bad |= saturation_pixels(a, ceiling=integer_saturation_ceiling(a))
    return bad


def roi_reduce(image, roi: RoiSpec, *, mask=None, reducer: str = "mean"):
    """Reduce ``roi`` over the VALID pixels of ``image``.

    ``mask`` (True = invalid) is excluded alongside any non-finite values.
    Returns ``(value, n_valid)``; ``value`` is NaN when no valid pixels remain.
    """
    if reducer not in _REDUCERS:
        raise ValueError(f"reducer must be one of {_REDUCERS}; got {reducer!r}")
    rs, cs = roi.pixel_slice(np.asarray(image).shape)
    patch = np.asarray(image[rs, cs], dtype=float)
    if mask is not None:
        patch = np.where(np.asarray(mask[rs, cs], dtype=bool), np.nan, patch)
    valid = np.isfinite(patch)
    n = int(valid.sum())
    if n == 0:
        return float("nan"), 0
    if reducer == "sum":
        v = np.nansum(patch)
    elif reducer == "mean":
        v = np.nanmean(patch)
    elif reducer == "max":
        v = np.nanmax(patch)
    elif reducer == "min":
        v = np.nanmin(patch)
    else:  # std
        v = np.nanstd(patch)
    return float(v), n
