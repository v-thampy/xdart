"""Custom Slice: an RSM volume cut on any plane, such as HH–L (AUD-9).

The maintainer, 2026-10-01: "The ability to only slices which are not just along h, k or l. So say HH and l",
as "extra tools that pop up separate windows". The window is a shared GUI component; this module is its headless
half (hard invariant 7: no Qt).

The plane is ``origin + t·u + s·v`` in the volume's own coordinates (r.l.u. in HKL, Å⁻¹ in Q). Each sample is
trilinear in the gridded volume, the scheme of ``scipy.ndimage.map_coordinates(order=1)``, computed by gathering
each sample's eight corner voxels, so no volume-sized temporary is made. It is NaN-aware: a sample is the weighted
mean of its finite corners when they carry at least half its weight (``_MIN_COVERAGE``), else NaN, so an empty
voxel keeps its size: it neither spreads nor is filled by its neighbours beyond half a voxel. Samples outside the
volume's bin-centre extent are NaN (no extrapolation). A fractional index within ``_SNAP`` of a whole number snaps
to it, so a plane through voxel centres reproduces those voxels exactly, float32 axes included.

The slice is interpolated from the gridded volume, not re-gridded from the frames; ``CustomSlice.interpolated``
says so, and the window shows ``CustomSlice.note``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


#: A slice above this many samples is refused (a 2000 × 2000 plane is 16 MB of float32).
MAX_SLICE_POINTS = 4_000_000
_SNAP = 1e-6  # index units: a float32 axis' rounding (about 1e-7 of a voxel) still snaps
_MIN_COVERAGE = 0.5
_CHUNK_POINTS = 1 << 16
_NOTE = "interpolated from the gridded volume (trilinear), not re-gridded from the frames"


class CustomSliceRefused(ValueError):
    """A plane Custom Slice does not cut, with the reason the window shows."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class CustomSlice:
    """One plane of an RSM volume: ``values[i, j]`` at ``origin + t[i]·u + s[j]·v``."""

    t: np.ndarray
    s: np.ndarray
    values: np.ndarray
    t_label: str
    s_label: str
    projection_t: np.ndarray
    projection_s: np.ndarray
    origin: tuple[float, float, float]
    u: tuple[float, float, float]
    v: tuple[float, float, float]
    interpolated: bool = True
    note: str = _NOTE


def direction_label(direction, symbols=("H", "K", "L"), unit="r.l.u.") -> str:
    """A direction's axis label: an axis's own symbol for a unit vector along it, "HH" for (1, 1, 0) in HKL
    (the H = K direction), else the generic ``t·[a b c]``; with the frame's unit."""
    vector = tuple(float(item) for item in direction)
    if sorted(vector) == [0.0, 0.0, 1.0]:
        name = symbols[vector.index(1.0)]
    elif tuple(symbols) == ("H", "K", "L") and vector == (1.0, 1.0, 0.0):
        name = "HH"
    else:
        name = "t·[" + " ".join(f"{item:g}" for item in vector) + "]"
    return f"{name} ({unit})" if unit else name


def sample_volume(values, axes, points) -> np.ndarray:
    """The volume at ``points`` (shape ``(..., 3)``, the volume's coordinates) by NaN-aware trilinear
    interpolation, as float64; NaN outside the bin-centre extent or where the finite corners carry less than
    ``_MIN_COVERAGE`` of the weight."""
    volume = np.asarray(values)
    centres = _checked_axes(volume, axes)
    where = np.asarray(points, dtype=np.float64)
    if where.shape[-1:] != (3,):
        raise CustomSliceRefused("CUSTOM_SLICE_SHAPE_INVALID", "each point needs three coordinates")
    flat = where.reshape(-1, 3)
    result = np.empty(flat.shape[0])
    for begin in range(0, flat.shape[0], _CHUNK_POINTS):  # bounded working set at any slice size
        result[begin:begin + _CHUNK_POINTS] = _sample_chunk(volume, centres, flat[begin:begin + _CHUNK_POINTS])
    return result.reshape(where.shape[:-1])


def _sample_chunk(volume: np.ndarray, centres, flat: np.ndarray) -> np.ndarray:
    inside = np.ones(flat.shape[0], dtype=bool)
    low_index, fraction = [], []
    for k, centre in enumerate(centres):
        coordinate = flat[:, k]
        inside &= np.isfinite(coordinate) & (coordinate >= centre[0]) & (coordinate <= centre[-1])
        index = np.interp(np.where(inside, coordinate, centre[0]), centre, np.arange(centre.size, dtype=np.float64))
        nearest = np.rint(index)
        index = np.where(np.abs(index - nearest) <= _SNAP, nearest, index)
        lower = np.clip(np.floor(index), 0, max(centre.size - 2, 0)).astype(np.intp)
        low_index.append(lower)
        fraction.append(np.clip(index - lower, 0.0, 1.0) if centre.size > 1 else np.zeros_like(index))
    weighted = np.zeros(flat.shape[0])
    weight_finite = np.zeros(flat.shape[0])
    complete = np.ones(flat.shape[0], dtype=bool)
    for corner in range(8):
        offsets = ((corner >> 2) & 1, (corner >> 1) & 1, corner & 1)
        weight = np.ones(flat.shape[0])
        index = []
        for k, offset in enumerate(offsets):
            size = centres[k].size
            index.append(np.minimum(low_index[k] + offset, size - 1))
            weight = weight * (fraction[k] if offset else 1.0 - fraction[k])
        corner_values = np.asarray(volume[index[0], index[1], index[2]], dtype=np.float64)
        finite = np.isfinite(corner_values)
        counted = weight > 0.0
        complete &= finite | ~counted
        weighted += np.where(finite & counted, weight * np.where(finite, corner_values, 0.0), 0.0)
        weight_finite += np.where(finite & counted, weight, 0.0)
    result = np.full(flat.shape[0], np.nan)
    result[complete] = weighted[complete]  # every weighted corner finite: the plain trilinear sum
    partial = ~complete & (weight_finite >= _MIN_COVERAGE)
    result[partial] = weighted[partial] / weight_finite[partial]
    result[~inside] = np.nan
    return result


def custom_slice(values, axes, origin=None, u=(1.0, 1.0, 0.0), v=(0.0, 0.0, 1.0), u_range=None, v_range=None,
                 points=None, *, symbols=("H", "K", "L"), unit="r.l.u.") -> CustomSlice:
    """The plane ``origin + t·u + s·v`` of the volume (HH–L by default: u = (1, 1, 0), v = (0, 0, 1)).

    ``origin`` defaults to the volume's centre; each range defaults to the volume's extent along its direction
    from the origin; ``points`` (t, s) defaults to the volume's longest axis in each direction. A zero or
    parallel direction, a plane that misses the volume, a bad range or more than ``MAX_SLICE_POINTS`` samples
    are refused with a reason."""
    volume = np.asarray(values)
    centres = _checked_axes(volume, axes)
    first, second = _directions(u, v)
    low = np.array([centre[0] for centre in centres])
    high = np.array([centre[-1] for centre in centres])
    centre_point = _origin(volume_centre(centres) if origin is None else origin)
    t_low, t_high = _span(first, centre_point, low, high) if u_range is None else _range(u_range)
    s_low, s_high = _span(second, centre_point, low, high) if v_range is None else _range(v_range)
    longest = max(volume.shape)
    count_t, count_s = (longest, longest) if points is None else (int(points[0]), int(points[1]))
    if count_t < 2 or count_s < 2:
        raise CustomSliceRefused("CUSTOM_SLICE_RANGE_INVALID", "a slice needs at least two points each way")
    if count_t * count_s > MAX_SLICE_POINTS:
        raise CustomSliceRefused("CUSTOM_SLICE_TOO_LARGE",
                                 f"{count_t} × {count_s} samples is above Custom Slice's {MAX_SLICE_POINTS:,}")
    t = np.linspace(t_low, t_high, count_t)
    s = np.linspace(s_low, s_high, count_s)
    # the displayed values, float32 as the viewer's; sampled a block of rows at a time, so no (t, s, 3) grid of
    # coordinates is ever whole
    shown = np.empty((count_t, count_s), dtype=np.float32)
    rows = max(1, _CHUNK_POINTS // count_s)
    for begin in range(0, count_t, rows):
        block = centre_point + t[begin:begin + rows, None, None] * first + s[None, :, None] * second
        shown[begin:begin + rows] = sample_volume(volume, centres, block)
    if not np.isfinite(shown).any():
        raise CustomSliceRefused("CUSTOM_SLICE_OUTSIDE", "the plane does not cross the volume's data")
    arrays = (t, s, shown, _finite_mean(shown, axis=1), _finite_mean(shown, axis=0))
    for array in arrays:
        array.setflags(write=False)
    return CustomSlice(
        t=t, s=s, values=shown,
        t_label=direction_label(first, symbols, unit), s_label=direction_label(second, symbols, unit),
        projection_t=arrays[3], projection_s=arrays[4],
        origin=tuple(float(item) for item in centre_point),
        u=tuple(float(item) for item in first), v=tuple(float(item) for item in second),
    )


def volume_centre(axes) -> tuple[float, float, float]:
    """The volume's centre, midway between its first and last bin centres on each axis: Custom Slice's default
    origin."""
    return tuple(float((axis[0] + axis[-1]) / 2.0) for axis in (np.asarray(item, dtype=np.float64) for item in axes))


def plane_offsets(origin, u=(1.0, 1.0, 0.0), v=(0.0, 0.0, 1.0)) -> tuple[float, float]:
    """Where the plane's own coordinates start: ``(a, b)``, with ``origin + a·u + b·v`` the plane's point nearest
    the reciprocal-space origin. ``CustomSlice.t`` and ``s`` are offsets from ``origin``; a sample lies at that
    point ``+ (t − a)·u + (s − b)·v``, so ``t − a`` and ``s − b`` are its coordinates along the two directions:
    H and K themselves on an H–K plane; L and the HH coordinate (H + K)/2 on HH–L, through any point. Zero and
    parallel directions are refused as ``custom_slice`` refuses them."""
    first, second = _directions(u, v)
    point = _origin(origin)
    gram = np.array([[first @ first, first @ second], [first @ second, second @ second]])
    a, b = np.linalg.solve(gram, -np.array([point @ first, point @ second]))
    return float(a), float(b)


def _directions(u, v) -> tuple[np.ndarray, np.ndarray]:
    first, second = _direction(u, "first"), _direction(v, "second")
    if np.linalg.norm(np.cross(first, second)) <= 1e-12 * np.linalg.norm(first) * np.linalg.norm(second):
        raise CustomSliceRefused("CUSTOM_SLICE_DIRECTIONS_PARALLEL", "the two directions are parallel")
    return first, second


def _origin(value) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise CustomSliceRefused("CUSTOM_SLICE_RANGE_INVALID", "the plane's origin needs three finite coordinates")
    return point


def _checked_axes(volume: np.ndarray, axes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if volume.ndim != 3 or len(axes) != 3:
        raise CustomSliceRefused("CUSTOM_SLICE_SHAPE_INVALID", "Custom Slice needs one 3-D volume and three axes")
    centres = tuple(np.asarray(axis, dtype=np.float64) for axis in axes)
    for size, centre in zip(volume.shape, centres):
        if centre.shape != (size,) or not np.all(np.isfinite(centre)) or (
                size > 1 and not np.all(np.diff(centre) > 0)):
            raise CustomSliceRefused("CUSTOM_SLICE_SHAPE_INVALID",
                                     "each axis must hold one increasing, finite centre per voxel")
    return centres


def _direction(value, which: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)) or not np.any(vector != 0.0):
        raise CustomSliceRefused("CUSTOM_SLICE_DIRECTION_ZERO", f"the {which} direction must be non-zero")
    return vector


def _range(value) -> tuple[float, float]:
    low, high = (float(item) for item in value)
    if not (np.isfinite(low) and np.isfinite(high) and high > low):
        raise CustomSliceRefused("CUSTOM_SLICE_RANGE_INVALID", "a range needs two finite values, low below high")
    return low, high


def _span(direction: np.ndarray, origin: np.ndarray, low: np.ndarray, high: np.ndarray) -> tuple[float, float]:
    """The coefficients along ``direction`` for which ``origin + c·direction`` stays inside the volume's box."""
    lowest, highest = -np.inf, np.inf
    for k in range(3):
        if direction[k] != 0.0:
            ends = sorted(((low[k] - origin[k]) / direction[k], (high[k] - origin[k]) / direction[k]))
            lowest, highest = max(lowest, ends[0]), min(highest, ends[1])
        elif not low[k] <= origin[k] <= high[k]:
            lowest, highest = 1.0, 0.0  # the line runs outside the box on this axis
    if not highest > lowest:
        raise CustomSliceRefused("CUSTOM_SLICE_OUTSIDE", "the plane does not cross the volume")
    return float(lowest), float(highest)


def _finite_mean(values: np.ndarray, *, axis: int) -> np.ndarray:
    """The finite mean along ``axis``, the viewer model's recipe (``rsm_viewer_model._projection_values``):
    float64 sums and counts, float32 out, NaN where nothing is finite."""
    finite = np.isfinite(values)
    sums = np.sum(np.where(finite, values, 0.0), axis=axis, dtype=np.float64)
    counts = np.sum(finite, axis=axis, dtype=np.int64)
    out = np.full(sums.shape, np.nan, dtype=np.float32)
    present = counts > 0
    out[present] = sums[present] / counts[present]
    return out
