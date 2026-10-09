"""Detector invalid-pixel policy (headless, numpy-only).

Mask Saturated excludes every native integer ceiling pixel, including isolated
ones, alongside negative detector values and the uint32 invalid sentinel.
The caller's toggle controls all finite value exclusions; static masks remain
independent. Calibration is the one exception (maintainer, 2026-10-08): its
automatic ring search and its staged frames' mask (pyFAI-calib2's, predicted
rings, orientation checks) leave the uint32 invalid sentinel out whatever its
threshold (``stitch_autocalibration._calibration_pixels``,
``stitch_calibration.calibration_frame_mask``). Active manual intensity thresholds take precedence at the run-policy
boundary, so they do not require an additional saturation pass.

Deliberately dtype-derived and **never hardcodes 65535**: a float frame whose
integer dtype was lost upstream returns ``None`` from
:func:`integer_saturation_ceiling`, leaving the fallback (if any) to the caller
— the GUI keeps its legacy 65535 fallback in xdart, out of core.

The automatic ceiling (decision 39, :func:`resolve_detector_saturation`) is the
value the file records, else the detector's documented counter limit, else the
data type's maximum. A caller passes a resolved non-dtype value as
``saturation`` and Auto then also leaves out ``raw >= saturation``; the native
exclusions above stay, so a uint16 65535 stays out under a higher cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

#: uint32 max — the unambiguous dead/hot-pixel dummy (Eiger masters etc.).
UINT32_CEILING = 4294967295.0

#: The PILATUS2/PILATUS3 20-bit counter's largest count: the documented
#: saturation of PILATUS frames whose file records none (the 17-2 ``.raw``).
#: A header's own ``Count_cutoff`` comes first: after count-rate correction it
#: can be higher (DECTRIS: 1,208,517 on a PILATUS3, 1,280,469 on a PILATUS2).
PILATUS_COUNTER_LIMIT = 1_048_575

#: Where an automatic ceiling came from, in the order decision 39 applies them.
SATURATION_SOURCES = ("recorded", "detector", "dtype")

__all__ = [
    "PILATUS_COUNTER_LIMIT",
    "SATURATION_SOURCES",
    "UINT32_CEILING",
    "DetectorSaturation",
    "combine_detector_masks",
    "detector_value_mask",
    "documented_saturation",
    "integer_saturation_ceiling",
    "resolve_detector_saturation",
    "saturation_count",
    "saturation_pixels",
]


@dataclass(frozen=True, slots=True)
class DetectorSaturation:
    """An automatic ceiling (decision 39): Auto leaves out pixels at or above ``value``."""

    value: int
    source: str

    def __post_init__(self) -> None:
        if saturation_count(self.value) != self.value or type(self.value) is not int:
            raise ValueError("a saturation value is a positive whole count")
        if self.source not in SATURATION_SOURCES:
            raise ValueError(f"saturation source must be one of {SATURATION_SOURCES}")


def saturation_count(value) -> int | None:
    """``value`` as a positive whole count, else ``None``.

    A recorded value that is missing, not a scalar, not finite, not a whole
    number or not positive is ignored, and the next rule applies.
    """
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        array = np.asarray(value)
    except Exception:  # noqa: BLE001 - any unreadable value falls to the next rule
        return None
    if array.shape != ():
        return None
    if array.dtype.kind in "iu":
        number = int(array)
    elif array.dtype.kind == "f" and math.isfinite(float(array)) and float(array).is_integer():
        number = int(float(array))
    else:
        return None
    return number if 0 < number < 2**63 else None


def documented_saturation(detector: object) -> int | None:
    """The documented counter limit for a detector name: PILATUS2/3 only.

    Names are matched lowercase and alphanumeric (``Pilatus 300kw`` ->
    ``pilatus300kw``). PILATUS4 (``pilatus4...``), whose counter this code does
    not know, and every other detector return ``None``.
    """
    name = "".join(ch for ch in ("" if detector is None else str(detector)).lower() if ch.isalnum())
    if name.startswith("pilatus") and not name.startswith("pilatus4"):
        return PILATUS_COUNTER_LIMIT
    return None


def resolve_detector_saturation(
    *,
    recorded: object = None,
    detector: object = None,
    dtype: object = None,
) -> DetectorSaturation | None:
    """The automatic ceiling, in decision 39's order.

    First the value the file records (a PILATUS header's ``Count_cutoff``, a
    NeXus ``saturation_value``); else the detector's documented counter limit,
    for integer data; else the integer data type's maximum. A value at or above
    that maximum changes nothing (the maximum is always left out), so the data
    type's rule stands. ``None`` when none is known (float data with nothing
    recorded).
    """
    kind = None if dtype is None else np.dtype(dtype).kind
    ceiling = None if dtype is None else _integer_dtype_saturation_ceiling(np.dtype(dtype))
    native = None if ceiling is None else DetectorSaturation(int(ceiling), "dtype")
    for value, source in ((saturation_count(recorded), "recorded"),
                          (documented_saturation(detector) if kind in {None, "i", "u"} else None, "detector")):
        if value is not None:
            return native if native is not None and value >= native.value else DetectorSaturation(value, source)
    return native


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
    saturation: int | None = None,
) -> np.ndarray | None:
    """Union the accepted detector-value policy with an existing static mask.

    The operator toggle controls value masking only.  An existing static mask
    remains authoritative when value masking is disabled.  Enabled masking
    adds negative values, the UINT32 dummy, and every native integer ceiling,
    and with a resolved ``saturation`` (not the dtype rule) every native value
    at or above it.
    """
    if saturation is not None and (type(saturation) is not int or saturation <= 0):
        raise TypeError("saturation must be a positive whole count or absent")
    if not enabled:
        return mask
    raw = np.asarray(raw_image)
    bad = (raw < 0) | (raw >= UINT32_CEILING)
    bad |= saturation_pixels(
        raw,
        ceiling=integer_saturation_ceiling(raw),
    )
    if saturation is not None:
        bad |= raw >= saturation
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
    return _integer_dtype_saturation_ceiling(np.asarray(arr).dtype)


@lru_cache(maxsize=32)
def _integer_dtype_saturation_ceiling(dtype) -> float | None:
    """Resolve type metadata once, never from a frame's observed maximum."""
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return None


def saturation_pixels(values, *, ceiling) -> np.ndarray:
    """Select every pixel at the native saturation ceiling, without a count gate."""
    a = np.asarray(values)
    if ceiling is None:
        return np.zeros(a.shape, dtype=bool)
    return a == float(ceiling)
