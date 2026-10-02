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

An Auto range is the plane's own extent inside the volume's bin-centre box (a slab's, across its thickness), so
any point on the plane cuts the same samples (Codex's combined stage-2b review F1).

A Thickness above 0 averages a slab (the maintainer, 2026-10-01: "Do you want the Thickness option? Yes"): the
planes parallel to the plane across ±thickness/2 along its unit normal n = u×v/|u×v|, in the volume's own
coordinates (in HKL, r.l.u. taken as orthogonal axes: for a non-orthogonal lattice neither n nor the thickness is
the reciprocal-space one, which the window says). The planes are an odd count, the plane itself in the middle,
spaced at no more than the voxel spacing along n; each is sampled as the plane is, and its finite samples are
summed and counted a block of rows at a time, so no planes × points array is made. Each value is the finite mean
of its planes' samples and each projection the finite mean of every sample of the slab at its coordinate: an
empty voxel counts in no mean, and a slab's value is NaN only where none of its planes has data. Thickness 0 is
the plane alone, by the same code as before the slab.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np


#: A slice above this many samples is refused (a 2000 × 2000 plane is 16 MB of float32).
MAX_SLICE_POINTS = 4_000_000
#: A slab above this many samples (planes × points) is refused. The window cuts on the GUI thread, and a sample
#: costs about 0.1 µs on an M3 Max (G11 probes 1 and 3), so the largest slab takes about 2 s there, four times
#: the largest plane; its memory is the plane's (a block of rows of sums and counts at a time).
MAX_SLAB_SAMPLES = 4 * MAX_SLICE_POINTS
_SNAP = 1e-6  # index units: a float32 axis' rounding (about 1e-7 of a voxel) still snaps
_MIN_COVERAGE = 0.5
_CHUNK_POINTS = 1 << 16
_NOTE = "interpolated from the gridded volume (trilinear), not re-gridded from the frames"
_SPACING_ROUNDING = 1e-9  # voxel spacings: a half-thickness a whole number of them, up to rounding, adds no planes
_VERTEX_TOLERANCE = 1e-9  # of a face's scale: a vertex of the plane's cut this near outside another face is on it


class CustomSliceRefused(ValueError):
    """A plane Custom Slice does not cut, with the reason the window shows."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class CustomSlice:
    """One plane of an RSM volume: ``values[i, j]`` at ``origin + t[i]·u + s[j]·v``; for a ``thickness`` above 0,
    the finite mean of its ``planes`` parallel planes at ``origin + o·normal + t[i]·u + s[j]·v``, with ``o`` across
    ±thickness/2 (``normal`` the plane's unit normal u×v/|u×v|, in the volume's coordinates)."""

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
    normal: tuple[float, float, float]
    interpolated: bool = True
    note: str = _NOTE
    thickness: float = 0.0
    planes: int = 1


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
                 points=None, *, symbols=("H", "K", "L"), unit="r.l.u.", thickness=0.0) -> CustomSlice:
    """The plane ``origin + t·u + s·v`` of the volume (HH–L by default: u = (1, 1, 0), v = (0, 0, 1)).

    ``origin`` defaults to the volume's centre; each range defaults to the plane's extent inside the volume's
    bin-centre box along its direction (for a ``thickness`` above 0, the slab's), the same cut through any point on
    the plane; ``points`` (t, s) defaults to the volume's longest axis in each direction. ``thickness``
    above 0 averages the slab across ±thickness/2 along the plane's unit normal (the module's docstring). A zero
    or parallel direction, a plane that misses the volume, a bad range, more than ``MAX_SLICE_POINTS`` samples a
    plane, a negative or non-finite thickness or a slab of more than ``MAX_SLAB_SAMPLES`` are refused with a
    reason."""
    volume = np.asarray(values)
    centres = _checked_axes(volume, axes)
    first, second = _directions(u, v)
    width = _thickness(thickness)
    low = np.array([centre[0] for centre in centres])
    high = np.array([centre[-1] for centre in centres])
    centre_point = _origin(volume_centre(centres) if origin is None else origin)
    across = np.cross(first, second)
    normal = across / np.linalg.norm(across)
    extent = None  # the plane's (the slab's) extent in the box, found once for an Auto range
    if u_range is None:
        extent = _extent(centre_point, first, second, normal, width, low, high)
        t_low, t_high = extent[0]
    else:
        t_low, t_high = _range(u_range)
    if v_range is None:
        extent = extent or _extent(centre_point, first, second, normal, width, low, high)
        s_low, s_high = extent[1]
    else:
        s_low, s_high = _range(v_range)
    longest = max(volume.shape)
    count_t, count_s = (longest, longest) if points is None else (int(points[0]), int(points[1]))
    if count_t < 2 or count_s < 2:
        raise CustomSliceRefused("CUSTOM_SLICE_RANGE_INVALID", "a slice needs at least two points each way")
    if count_t * count_s > MAX_SLICE_POINTS:
        raise CustomSliceRefused("CUSTOM_SLICE_TOO_LARGE",
                                 f"{count_t} × {count_s} samples is above Custom Slice's {MAX_SLICE_POINTS:,}")
    planes = _slab_planes(centres, normal, width, count_t * count_s)
    t = np.linspace(t_low, t_high, count_t)
    s = np.linspace(s_low, s_high, count_s)
    if width == 0.0:
        # the displayed values, float32 as the viewer's; sampled a block of rows at a time, so no (t, s, 3) grid
        # of coordinates is ever whole
        shown = np.empty((count_t, count_s), dtype=np.float32)
        rows = max(1, _CHUNK_POINTS // count_s)
        for begin in range(0, count_t, rows):
            block = centre_point + t[begin:begin + rows, None, None] * first + s[None, :, None] * second
            shown[begin:begin + rows] = sample_volume(volume, centres, block)
        projection_t, projection_s = _finite_mean(shown, axis=1), _finite_mean(shown, axis=0)
        note = _NOTE
    else:
        offsets = np.linspace(-width / 2.0, width / 2.0, planes)
        shown, projection_t, projection_s = _slab(volume, centres, centre_point, first, second, normal, offsets,
                                                  t, s)
        note = f"{_NOTE}; averaged over {planes} planes, ±{width / 2.0:g} along [{_vector_text(across)}]"
    if not np.isfinite(shown).any():
        raise CustomSliceRefused("CUSTOM_SLICE_OUTSIDE", "the plane does not cross the volume's data")
    arrays = (t, s, shown, projection_t, projection_s)
    for array in arrays:
        array.setflags(write=False)
    return CustomSlice(
        t=t, s=s, values=shown,
        t_label=direction_label(first, symbols, unit), s_label=direction_label(second, symbols, unit),
        projection_t=arrays[3], projection_s=arrays[4],
        origin=tuple(float(item) for item in centre_point),
        u=tuple(float(item) for item in first), v=tuple(float(item) for item in second),
        normal=tuple(float(item) for item in normal), note=note, thickness=width, planes=planes,
    )


def _slab(volume, centres, centre_point, first, second, normal, offsets, t, s):
    """A slab's finite means, float32 out: each (t, s) over its planes, and each t (each s) over every sample of
    the slab at it. Each plane is sampled as the plane alone is; float64 sums and counts are kept a block of rows
    at a time, so neither a planes × points array nor a grid of coordinates is ever whole."""
    count_t, count_s = t.size, s.size
    shown = np.empty((count_t, count_s), dtype=np.float32)
    sums_t, counts_t = np.zeros(count_t), np.zeros(count_t, dtype=np.int64)
    sums_s, counts_s = np.zeros(count_s), np.zeros(count_s, dtype=np.int64)
    rows = max(1, _CHUNK_POINTS // count_s)
    for begin in range(0, count_t, rows):
        sums = np.zeros((min(rows, count_t - begin), count_s))
        counts = np.zeros(sums.shape, dtype=np.int64)
        for offset in offsets:
            plane = centre_point + offset * normal
            block = plane + t[begin:begin + rows, None, None] * first + s[None, :, None] * second
            sampled = sample_volume(volume, centres, block)
            finite = np.isfinite(sampled)
            sums += np.where(finite, sampled, 0.0)
            counts += finite
        sums_t[begin:begin + rows], counts_t[begin:begin + rows] = sums.sum(axis=1), counts.sum(axis=1)
        sums_s += sums.sum(axis=0)
        counts_s += counts.sum(axis=0)
        shown[begin:begin + rows] = _mean(sums, counts)
    return shown, _mean(sums_t, counts_t), _mean(sums_s, counts_s)


def _mean(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Sums over counts as float32, NaN where nothing was counted."""
    out = np.full(sums.shape, np.nan, dtype=np.float32)
    present = counts > 0
    out[present] = sums[present] / counts[present]
    return out


def _slab_planes(centres, normal: np.ndarray, width: float, samples: int) -> int:
    """How many planes a slab of ``width`` samples: 1 for none; else the least odd count whose spacing,
    width / (count − 1), is no more than the voxel spacing along ``normal`` (the least step_k / |n_k|: one voxel
    along the axis the normal crosses fastest). Refused above ``MAX_SLAB_SAMPLES`` planes × ``samples``."""
    if width == 0.0:
        return 1
    spacing = min((float(np.min(np.diff(centre))) / abs(component)
                   for centre, component in zip(centres, normal) if component != 0.0 and centre.size > 1),
                  default=math.inf)
    halves = width / 2.0 / spacing  # voxel spacings in half the slab
    if not halves < MAX_SLAB_SAMPLES:
        raise CustomSliceRefused("CUSTOM_SLICE_TOO_LARGE", f"a slab {width:g} thick needs more than "
                                 f"{MAX_SLAB_SAMPLES:,} planes: a thinner slab")
    planes = 2 * max(1, math.ceil(halves - _SPACING_ROUNDING)) + 1
    if planes * samples > MAX_SLAB_SAMPLES:
        raise CustomSliceRefused("CUSTOM_SLICE_TOO_LARGE", f"{planes} planes × {samples:,} samples is above Custom "
                                 f"Slice's {MAX_SLAB_SAMPLES:,}: a thinner slab, or fewer points")
    return planes


def _thickness(value) -> float:
    try:
        width = float(value)
    except (TypeError, ValueError):
        width = math.nan
    if not (math.isfinite(width) and width >= 0.0):
        raise CustomSliceRefused("CUSTOM_SLICE_THICKNESS_INVALID", "the thickness must be a finite number, 0 or more")
    return width + 0.0  # -0.0 is no thickness


def _vector_text(vector) -> str:
    """A direction as the note writes it, "1 −1 0": a true minus, no negative zero."""
    return " ".join(f"{float(item) + 0.0:g}".replace("-", "−") for item in vector)


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


def _extent(origin, first, second, normal, width, low, high) -> tuple[tuple[float, float], tuple[float, float]]:
    """The Auto ranges: the least and greatest t and s of the points ``origin + t·u + s·v`` (for a ``width``
    above 0, ``+ w·normal`` with |w| ≤ width/2: the slab) inside the box ``low``…``high``, the bin centres'
    extent; refused when there is none. That is the plane's (the slab's) cut through the box, the same cut
    through any point on the plane (Codex's combined stage-2b review F1: the lines through the point, which the
    ranges were before, can miss the box where the plane crosses it).

    The extremes are at vertices, where the plane meets two of the box's faces (the slab, three of the box's and
    its own two faces): each is solved and kept when inside the other faces. An axis only one direction moves
    bounds that direction alone, ``(face − origin) / direction``, as the line through the point did, and no
    vertex passes that bound; so where each axis moves with one direction at most (H–K, HH–L) and the point's
    lines cross the box, the ranges are those lines' (before F1), to the last bit. Where an axis moves with both
    directions, the plane's extent can be wider than the lines' through the point, and is the same through any
    point on the plane."""
    columns = (first, second, normal) if width else (first, second)
    rows = []  # per moved axis (and the slab): its coefficients, its two faces less the origin, its scale
    bounds = [[-math.inf, math.inf], [-math.inf, math.inf]]  # what an axis moved by t (by s) alone allows
    for k in range(3):
        coefficients = tuple(float(column[k]) for column in columns)
        moved = [index for index, value in enumerate(coefficients) if value != 0.0]
        if not moved:  # an axis the plane does not move: the point must be inside the box on it
            if not low[k] <= origin[k] <= high[k]:
                raise CustomSliceRefused("CUSTOM_SLICE_OUTSIDE", "the plane does not cross the volume")
            continue
        faces = (low[k] - origin[k], high[k] - origin[k])
        rows.append((coefficients, faces, abs(low[k]) + abs(high[k]) + abs(origin[k])))
        if len(moved) == 1 and moved[0] < 2:
            index = moved[0]
            ends = sorted((faces[0] / coefficients[index], faces[1] / coefficients[index]))
            bounds[index] = [max(bounds[index][0], ends[0]), min(bounds[index][1], ends[1])]
    if width:
        rows.append(((0.0, 0.0, 1.0), (-width / 2.0, width / 2.0), width))
    found = [[math.inf, -math.inf], [math.inf, -math.inf]]
    for chosen in itertools.combinations(range(len(rows)), len(columns)):
        for sides in itertools.product((0, 1), repeat=len(columns)):
            vertex = _solve([rows[r][0] for r in chosen], [rows[r][1][side] for r, side in zip(chosen, sides)])
            if vertex is None or not all(_inside_faces(*rows[r], vertex) for r in range(len(rows))
                                         if r not in chosen):
                continue
            for index in (0, 1):
                found[index] = [min(found[index][0], vertex[index]), max(found[index][1], vertex[index])]
    extent = []
    for (lowest, highest), (floor, ceiling) in zip(found, bounds):
        lowest, highest = max(lowest, floor), min(highest, ceiling)
        if not highest > lowest:
            raise CustomSliceRefused("CUSTOM_SLICE_OUTSIDE", "the plane does not cross the volume")
        extent.append((float(lowest), float(highest)))
    return extent[0], extent[1]


def _solve(rows, values) -> list[float] | None:
    """``rows · x = values`` for two or three unknowns, by Gaussian elimination with partial pivoting in floats;
    None when singular or not finite. A row that moves one unknown leaves the others' rows as they are, so where
    every row does (an axis-aligned vertex) each unknown is its face's one division, as ``_extent``'s bounds are."""
    size = len(values)
    matrix = [[float(item) for item in row] + [float(value)] for row, value in zip(rows, values)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(matrix[row][column]))
        if matrix[pivot][column] == 0.0:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        for row in range(column + 1, size):
            factor = matrix[row][column] / matrix[column][column]
            if factor != 0.0:
                matrix[row] = [item - factor * lead for item, lead in zip(matrix[row], matrix[column])]
    solution = [0.0] * size
    for row in reversed(range(size)):
        known = sum(matrix[row][j] * solution[j] for j in range(row + 1, size))
        solution[row] = (matrix[row][size] - known) / matrix[row][row]
    return solution if all(math.isfinite(item) for item in solution) else None


def _inside_faces(coefficients, faces, scale, vertex) -> bool:
    """Whether a vertex is between one axis's (or the slab's) two faces, up to ``_VERTEX_TOLERANCE`` of its
    scale: solving rounds, and a plane through a box corner meets three faces there."""
    value = sum(coefficient * item for coefficient, item in zip(coefficients, vertex))
    slack = _VERTEX_TOLERANCE * (scale + abs(value))
    return faces[0] - slack <= value <= faces[1] + slack


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
