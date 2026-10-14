"""Angular support kernel: how much of a profile's χ window the detector reaches, per radial bin.

The primitive A(2θ) is the measure, in degrees, of the union of every unmasked pixel's footprint on the cone of
scattering angle 2θ, clipped to the χ interval the reducer actually admitted. Each pose is represented by the
boundary of its unmasked region (pixel edges between an unmasked pixel and a masked or absent one), mapped through
pyFAI's own forward geometry into (2θ, φ) and subdivided until each piece spans at most 0.25° of φ. On a radial line
the boundary crossings, oriented so the region lies on a known side, delimit the covered φ intervals exactly; a
ray along the χ seam anchors each line's count, so rings wholly inside or outside the region need no extra test.
Union over poses (Stitch) is the set where the summed count is positive, so overlaps and repeated poses count once.
For the grazing-incidence χ_GI convention the region is split where q_horz = 0, keeping the missing wedge empty.

The geometry stage (boundaries in 2θ and φ) is wavelength-independent and cacheable; sampling evaluates a uniform
table of radial lines in 2θ (``oversample`` lines per smallest native bin) and integrates it over each native bin in
the native measure (sin θ for q, 2θ for 2θ): the bin mean. Continuous geometric support deliberately does not
reproduce a reducer's pixel-granular admission at a χ window edge (pyFAI's split methods admit whole pixels); that
difference is documented rather than hidden.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from xrd_tools.core.angular_support import (
    AngularSupportProfile,
    SUPPORT_CONVENTIONS,
    SupportOrigin,
    SupportReason,
)

__all__ = [
    "AdmittedWindow",
    "DEFAULT_RECORD_ANGULAR_SUPPORT",
    "AngularSupportPolicy",
    "AngularSupportRequest",
    "RunSupportCapture",
    "StitchSupportAccumulator",
    "SupportCancelled",
    "SupportFootprint",
    "SupportLineTable",
    "XuSupportAccumulator",
    "SupportPose",
    "SUPPORTED_UNITS",
    "admitted_window",
    "angular_support",
    "edges_from_centres",
    "footprint_from_corners",
    "support_footprint",
    "support_line_table",
    "xu_corner_grid",
]

#: Whether new results record angular support unless a setting says otherwise (decision 6): the one default every
#: owner follows (AngularSupportPolicy, Int's run option and Stitch's row).
DEFAULT_RECORD_ANGULAR_SUPPORT = True
#: Native radial units a support profile can describe.
SUPPORTED_UNITS = ("q_A^-1", "q_nm^-1", "2th_deg", "2th_rad")
#: Reducer window rules (see :func:`admitted_window`).
REDUCERS = ("int_standard", "int_gi_total_q", "stitch_multigeometry", "stitch_xu")
_SUB_EDGE_PHI = math.radians(0.25)
_SINGULAR_STEP = math.pi / 4
#: Reachable support the reducer did not admit, as a share of the requested window, above which support refuses.
_TRUNCATION_SHARE = 0.0025
_PI = math.pi
_TWO_PI = 2 * math.pi


class SupportCancelled(Exception):
    """The caller's cancellation, raised from inside a support computation."""


_Cancelled = SupportCancelled


def _check(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise _Cancelled


# --------------------------------------------------------------------------------------------- policy and window

@dataclass(frozen=True, slots=True)
class AngularSupportPolicy:
    """Run-level support capture policy (D3).

    ``max_geometry_keys`` bounds how many distinct detector geometries (pose, geometric mask, χ_GI angles) one Run
    may compute; beyond it support is unavailable with ``over_budget`` while intensity processing is unchanged.
    ``None`` leaves it unbounded. ``enabled=False`` (“Record angular coverage” off) computes nothing: every eligible
    profile carries ``recording_disabled`` and only that reason is saved.
    """

    enabled: bool = DEFAULT_RECORD_ANGULAR_SUPPORT
    max_geometry_keys: int | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("angular support policy 'enabled' must be a bool")
        if self.max_geometry_keys is not None and (
                type(self.max_geometry_keys) is not int or self.max_geometry_keys < 0):
            raise ValueError("max_geometry_keys must be None or a nonnegative integer")


@dataclass(frozen=True, slots=True)
class AdmittedWindow:
    """A profile's requested χ window and the χ intervals its reducer actually admitted.

    Degrees, in the reducer's raw frame on [−180°, 180°). ``denominator_deg`` is the requested width (360 for no
    window). ``lost_deg`` lists requested directions the reducer did not admit. A refused window carries ``reason``.
    """

    requested_deg: tuple[float, float] | None
    denominator_deg: float
    admitted_deg: tuple[tuple[float, float], ...]
    lost_deg: tuple[tuple[float, float], ...] = ()
    reason: SupportReason | None = None
    detail: str = ""


_FULL = ((-180.0, 180.0),)


def _circle(lo: float, hi: float) -> tuple[tuple[float, float], ...]:
    """The directions of [lo, hi] (degrees) as intervals on [−180, 180)."""
    if hi - lo >= 360.0:
        return _FULL
    start = (lo + 180.0) % 360.0 - 180.0
    stop = start + (hi - lo)
    if stop <= 180.0:
        return ((start, stop),)
    return ((start, 180.0), (-180.0, stop - 360.0))


def _subtract(whole, part) -> tuple[tuple[float, float], ...]:
    pieces = list(whole)
    for lo, hi in part:
        nxt = []
        for a, b in pieces:
            if hi <= a or lo >= b:
                nxt.append((a, b))
                continue
            if lo > a:
                nxt.append((a, lo))
            if hi < b:
                nxt.append((hi, b))
        pieces = nxt
    return tuple((a, b) for a, b in pieces if b - a > 1e-12)


def _pyfai_admitted(lo: float, hi: float, chi_disc_at_pi: bool) -> tuple[tuple[float, float], ...]:
    """pyFAI's own normalize_azimuth_range, clipped to the χ the pixels can have, as degrees on [−180, 180)."""
    from pyFAI.geometry import Geometry
    geometry = Geometry()
    if chi_disc_at_pi:
        geometry.setChiDiscAtPi()
    else:
        geometry.setChiDiscAtZero()
    a, b = geometry.normalize_azimuth_range((lo, hi))
    domain = (-_PI, _PI) if chi_disc_at_pi else (0.0, _TWO_PI)
    a, b = max(a, domain[0]), min(b, domain[1])
    if b <= a:
        return ()
    a_deg, b_deg = math.degrees(a), math.degrees(b)
    if chi_disc_at_pi:
        return ((a_deg, b_deg),)
    out = []
    for x, y in ((a_deg, min(b_deg, 180.0)), (max(a_deg, 180.0) - 360.0, b_deg - 360.0)):
        if y > x:
            out.append((x, y))
    return tuple(out)


def admitted_window(requested_deg: tuple[float, float] | None, *, reducer: str, offset_deg: float = 0.0,
                    chi_disc_at_pi: bool = True,
                    scouted_deg: tuple[float, float] | None = None) -> AdmittedWindow:
    """The χ directions ``reducer`` admits for ``requested_deg``, by that reducer's own rule.

    * ``int_standard``: Int's offset subtraction, then pyFAI's normalize_azimuth_range with ``chi_disc_at_pi``.
    * ``int_gi_total_q``: no window is the full-turn fast path; a window is χ_GI on the polar path, refused
      outside [−180°, 180°] (pyFAI wraps it there rather than clamping).
    * ``stitch_multigeometry``: MultiGeometry's normalisation (χ discontinuity at π); without a window the pixel-centre
      χ extent ``scouted_deg`` it actually passes.
    * ``stitch_xu``: XU's closed centre-χ window, never wrapped.

    Equal or reversed endpoints are degenerate. The denominator is always the requested width (360 without a window).
    """
    if reducer not in REDUCERS:
        raise ValueError(f"unknown reducer window rule {reducer!r}")
    if requested_deg is None:
        if reducer == "stitch_multigeometry" and scouted_deg is not None:
            lo, hi = (float(value) for value in scouted_deg)
            admitted = _circle(lo, hi) if hi > lo else ()
            return AdmittedWindow(None, 360.0, admitted, _subtract(_FULL, admitted))
        return AdmittedWindow(None, 360.0, _FULL)
    lo, hi = (float(value) for value in requested_deg)
    requested = (lo, hi)
    width = hi - lo
    if not (math.isfinite(lo) and math.isfinite(hi)) or width <= 0.0:
        return AdmittedWindow(requested, 360.0, (), reason=SupportReason.DEGENERATE_WINDOW,
                              detail=f"{lo:g}° to {hi:g}°")
    denominator = min(width, 360.0)
    if reducer == "int_gi_total_q":
        if lo < -180.0 or hi > 180.0:
            return AdmittedWindow(requested, denominator, (), reason=SupportReason.WINDOW_OUTSIDE_DOMAIN,
                                  detail=f"{lo:g}° to {hi:g}°")
        return AdmittedWindow(requested, denominator, ((lo, hi),))
    if reducer == "stitch_xu":
        raw = (lo, hi)
        admitted = ((max(lo, -180.0), min(hi, 180.0)),) if min(hi, 180.0) > max(lo, -180.0) else ()
    else:
        shift = float(offset_deg) if reducer == "int_standard" else 0.0
        raw = (lo - shift, hi - shift)
        admitted = _pyfai_admitted(*raw, chi_disc_at_pi if reducer == "int_standard" else True)
    lost = _subtract(_circle(*raw), admitted)
    return AdmittedWindow(requested, denominator, admitted, lost)


# --------------------------------------------------------------------------------------------- poses and requests

def _mask_digest(mask: np.ndarray | None) -> str:
    if mask is None:
        return "none"
    return hashlib.sha256(np.packbits(mask, axis=None).tobytes() + repr(mask.shape).encode()).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class SupportPose:
    """One detector pose: detached pyFAI geometry, the geometric mask its reducer applied, and χ_GI angles.

    ``geometry`` is ``AzimuthalIntegrator.get_config()``; ``mask`` marks masked pixels (True) and is copied read-only
    unless it already is a read-only bool array; ``gi`` is (incident °, tilt °, EXIF sample orientation) for χ_GI.
    ``key`` identifies equal poses (geometry, mask content, angles) for deduplication and caching.
    """

    geometry: Mapping[str, Any]
    mask: np.ndarray | None = None
    gi: tuple[float, float, int] | None = None
    mask_digest: str | None = None
    key: str = field(init=False)

    def __post_init__(self) -> None:
        geometry = json.loads(json.dumps(dict(self.geometry), sort_keys=True, default=_json_default))
        geometry.pop("wavelength", None)  # footprints are wavelength-independent; an energy scan is one pose
        object.__setattr__(self, "geometry", geometry)
        mask = self.mask
        if mask is not None:
            if not (isinstance(mask, np.ndarray) and mask.dtype == np.dtype(bool) and not mask.flags.writeable
                    and mask.flags.c_contiguous):
                mask = np.array(mask, dtype=bool, order="C", copy=True)
                mask.setflags(write=False)
            if mask.ndim != 2:
                raise ValueError("a support pose mask must be 2-D")
            object.__setattr__(self, "mask", mask)
        if self.gi is not None:
            incident, tilt, orientation = self.gi
            gi = (float(incident), float(tilt), int(orientation))
            if not (math.isfinite(gi[0]) and math.isfinite(gi[1])) or gi[2] not in range(1, 9):
                raise ValueError("χ_GI angles must be finite and the sample orientation 1-8")
            object.__setattr__(self, "gi", gi)
        digest = self.mask_digest or _mask_digest(mask)
        object.__setattr__(self, "mask_digest", digest)
        identity = json.dumps({"geometry": geometry, "mask": digest, "gi": self.gi}, sort_keys=True)
        object.__setattr__(self, "key", hashlib.sha256(identity.encode()).hexdigest())

    @classmethod
    def from_integrator(cls, ai: Any, mask: np.ndarray | None = None, *, gi=None,
                        mask_digest: str | None = None) -> "SupportPose":
        return cls(dict(ai.get_config()), mask, gi, mask_digest)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


@dataclass(frozen=True, slots=True, eq=False)
class AngularSupportRequest:
    """Everything one profile's support needs, detached: poses, convention, native bins, energy and window."""

    poses: tuple[SupportPose, ...]
    convention: str
    radial_unit: str
    bin_edges: np.ndarray
    wavelength_m: float | None
    window: AdmittedWindow
    origin: SupportOrigin = SupportOrigin.RUN
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.poses or any(type(pose) is not SupportPose for pose in self.poses):
            raise TypeError("an angular support request needs at least one SupportPose")
        object.__setattr__(self, "poses", tuple(self.poses))
        if self.convention not in SUPPORT_CONVENTIONS:
            raise ValueError(f"unknown angular support convention {self.convention!r}")
        if any((pose.gi is None) == (self.convention == "gi_chi") for pose in self.poses):
            raise ValueError("χ_GI support needs every pose's GI angles; detector χ support takes none")
        edges = np.array(self.bin_edges, dtype=np.float64, copy=True)
        if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError("bin edges must be a finite increasing 1-D array")
        uniform = np.linspace(edges[0], edges[-1], edges.size)
        if np.max(np.abs(edges - uniform)) > 1e-9 * max(abs(edges[-1] - edges[0]), 1e-300):
            raise ValueError("bin edges must be uniform (a reducer's linear grid)")
        edges.setflags(write=False)
        object.__setattr__(self, "bin_edges", edges)
        if self.wavelength_m is not None:
            wavelength = float(self.wavelength_m)
            if not (math.isfinite(wavelength) and wavelength > 0):
                raise ValueError("wavelength must be finite and positive")
            object.__setattr__(self, "wavelength_m", wavelength)


# --------------------------------------------------------------------------------------------- geometry stage

@dataclass(frozen=True, slots=True, eq=False)
class SupportFootprint:
    """One pose's wavelength-independent boundary in (2θ, φ), or the reason it cannot have one.

    Sub-edges run from (t0, p0) to (t1, p1) (radians; φ continuous along each sub-edge) and carry the side of the
    region as ``sign``: a line crossing an outward sub-edge enters the region when sweeping φ upward iff sign > 0.
    ``seam_t``/``seam_sign`` are crossings of the χ seam ray (φ = ±π); ``pole_inside`` says whether the beam itself
    falls on an unmasked pixel. χ_GI footprints never reach the seam.
    """

    key: str
    convention: str
    t0: np.ndarray | None = None
    p0: np.ndarray | None = None
    t1: np.ndarray | None = None
    p1: np.ndarray | None = None
    sign: np.ndarray | None = None
    seam_t: np.ndarray | None = None
    seam_sign: np.ndarray | None = None
    pole_inside: bool = False
    reason: SupportReason | None = None
    detail: str = ""

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for array in (self.t0, self.p0, self.t1, self.p1, self.sign, self.seam_t,
                                              self.seam_sign) if array is not None)


def _integrator(geometry: Mapping[str, Any]):
    """pyFAI's own geometry for a detached config, built without pyFAI's per-load ``gc.collect()`` (whose cost
    grows with the host process's heap)."""
    from pyFAI.geometry import Geometry
    loaded = Geometry()
    loaded.auto_gc = False
    loaded.set_config(dict(geometry))
    return loaded


def _qualification(ai) -> str:
    """Why this geometry is outside the kernel's qualified envelope, or ``""``."""
    from pyFAI import detectors
    detector = ai.detector
    if getattr(ai, "_parallax", None) is not None or bool(ai.get_config().get("parallax")):
        return "parallax correction"
    if isinstance(detector, detectors.HexDetector):
        return "hexagonal pixels"
    if not bool(getattr(detector, "IS_FLAT", False)):
        return "curved detector"
    if not bool(getattr(detector, "uniform_pixel", False)):
        return "non-uniform pixels"
    if getattr(detector, "spline", None) is not None:
        return "distortion spline"
    if detector.shape is None:
        return "unknown detector shape"
    return ""


def _lab(ai, rows: np.ndarray, cols: np.ndarray):
    """Lab (x, y, z) of detector points given in pixel-corner coordinates (row, column)."""
    z, y, x = ai.calc_pos_zyx(d1=np.asarray(rows, float) - .5, d2=np.asarray(cols, float) - .5)
    return np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)


def _two_theta(x, y, z):
    return np.arctan2(np.sqrt(x * x + y * y), z)


def _mapper(ai, convention: str, gi):
    """(rows, cols, side) → (2θ, φ) for one pose; ``side`` (+1/−1) selects the χ_GI half for cut points."""
    if convention == "detector_chi":
        def mapped(rows, cols, side=None):
            x, y, z = _lab(ai, rows, cols)
            return _two_theta(x, y, z), np.arctan2(y, x)
        return mapped
    from pyFAI.units import q_sample
    incident, tilt, orientation = math.radians(gi[0]), math.radians(gi[1]), int(gi[2])

    def mapped(rows, cols, side=None):
        x, y, z = _lab(ai, rows, cols)
        qb, qh, qv = q_sample(x, y, z, 1e-10, incident, tilt, orientation)
        sign = np.sign(qh) if side is None else side
        return _two_theta(x, y, z), np.arctan2(sign * np.sqrt(qb * qb + qh * qh), qv)
    return mapped


def _horizontal_q(ai, gi):
    from pyFAI.units import q_sample
    incident, tilt, orientation = math.radians(gi[0]), math.radians(gi[1]), int(gi[2])

    def horizontal(rows, cols):
        x, y, z = _lab(ai, rows, cols)
        return q_sample(x, y, z, 1e-10, incident, tilt, orientation)[1]
    return horizontal


def _boundary_edges(valid: np.ndarray):
    """Directed pixel edges with an unmasked pixel on their left, in (column a, row b) with b upward.

    Only boundary pixels are indexed: no detector-sized coordinate array is built.
    """
    height, width = valid.shape
    padded = np.zeros((height + 2, width + 2), bool)
    padded[1:-1, 1:-1] = valid
    starts, ends = [], []
    for neighbour, (r0, c0), (r1, c1) in (
            (padded[:-2, 1:-1], (0, 0), (0, 1)),      # low-b edge, toward +a
            (padded[1:-1, 2:], (0, 1), (1, 1)),       # high-a edge, toward +b
            (padded[2:, 1:-1], (1, 1), (1, 0)),       # high-b edge, toward −a
            (padded[1:-1, :-2], (1, 0), (0, 0))):     # low-a edge, toward −b
        rows, cols = np.nonzero(valid & ~neighbour)
        starts.append(np.stack((rows + r0, cols + c0), axis=1))
        ends.append(np.stack((rows + r1, cols + c1), axis=1))
    return np.concatenate(starts).astype(float), np.concatenate(ends).astype(float)


def _signed_area(mapper, row: int, col: int, side=None) -> float:
    """Signed area of one pixel (traversed with its inside on the left) in the polar plane (2θ cos φ, 2θ sin φ)."""
    rows = np.array([row, row, row + 1, row + 1], float)
    cols = np.array([col, col + 1, col + 1, col], float)
    t, p = mapper(rows, cols, side)
    x, y = t * np.cos(p), t * np.sin(p)
    return float(.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _orientation_pixels(masked: np.ndarray | None, shape: tuple[int, int], side=None, corner_sign=None,
                        wanted: int = 9):
    """Up to ``wanted`` unmasked pixels spread over the detector (wholly on χ_GI half ``side`` if given), found on a
    sparse grid first so a large detector never indexes, or inverts the mask of, all its pixels."""
    height, width = shape

    def admissible(rows, cols):
        keep = np.ones(rows.size, bool) if masked is None else ~masked[rows, cols]
        if corner_sign is not None:
            keep &= ((corner_sign[rows, cols] == side) & (corner_sign[rows + 1, cols] == side)
                     & (corner_sign[rows, cols + 1] == side) & (corner_sign[rows + 1, cols + 1] == side))
        return rows[keep], cols[keep]

    grid_rows, grid_cols = np.meshgrid(np.linspace(0, height - 1, 33).astype(int),
                                       np.linspace(0, width - 1, 33).astype(int), indexing="ij")
    rows, cols = admissible(grid_rows.ravel(), grid_cols.ravel())
    if not rows.size:
        rows, cols = np.nonzero(np.ones(shape, bool) if masked is None else ~masked)
        rows, cols = admissible(rows, cols)
    if not rows.size:
        return rows, cols
    picks = np.unique(np.linspace(0, rows.size - 1, min(rows.size, wanted)).astype(int))
    return rows[picks], cols[picks]


def _orientation(mapper, masked: np.ndarray | None, shape: tuple[int, int], side=None, corner_sign=None) -> int:
    """+1 when the pose keeps the left side on the left in the polar plane, else −1 (measured on spread pixels)."""
    rows, cols = _orientation_pixels(masked, shape, side, corner_sign)
    if not rows.size:
        return 0
    areas = [_signed_area(mapper, int(row), int(col), side) for row, col in zip(rows, cols)]
    signs = {int(np.sign(area)) for area in areas if abs(area) > 1e-18}
    if len(signs) != 1:
        raise ValueError("the detector's orientation in (2θ, φ) is not uniform")
    return signs.pop()


def _subdivide(mapper, start: np.ndarray, end: np.ndarray, side, cancel):
    """Map directed edges to (2θ, φ) sub-edges no wider than 0.25° in φ; φ is continuous along each."""
    t_a, p_a = mapper(start[:, 0], start[:, 1], side)
    t_b, p_b = mapper(end[:, 0], end[:, 1], side)
    step = np.abs((p_b - p_a + _PI) % _TWO_PI - _PI)
    pieces = np.maximum(1, np.ceil(step / _SUB_EDGE_PHI)).astype(np.int64)
    simple = pieces == 1
    out = [(t_a[simple], p_a[simple], t_b[simple], p_a[simple] + ((p_b[simple] - p_a[simple] + _PI) % _TWO_PI - _PI))]
    split = np.flatnonzero(~simple)
    for chunk in np.array_split(split, max(1, split.size // 4096)):
        _check(cancel)
        if not chunk.size:
            continue
        counts = pieces[chunk]
        owner = np.repeat(np.arange(chunk.size), counts + 1)
        offset = np.arange(owner.size) - np.repeat(np.cumsum(counts + 1) - (counts + 1), counts + 1)
        fraction = offset / counts[owner]
        a, b = start[chunk][owner], end[chunk][owner]
        point = a + (b - a) * fraction[:, None]
        t, p = mapper(point[:, 0], point[:, 1], side)
        steps = (np.diff(p) + _PI) % _TWO_PI - _PI
        boundaries = np.cumsum(counts + 1)
        steps = np.delete(steps, boundaries[:-1] - 1)
        if steps.size and np.max(np.abs(steps)) > _SINGULAR_STEP:
            raise ValueError("an unmasked pixel edge passes through the beam centre")
        keep = np.ones(t.size, bool)
        keep[boundaries - 1] = False
        t_start, p_start = t[keep], p[keep]
        tails = np.ones(t.size, bool)
        tails[np.concatenate(([0], boundaries[:-1]))] = False
        t_end = t[tails]
        unwrapped = p_start + steps
        out.append((t_start, p_start, t_end, unwrapped))
    return tuple(np.concatenate([part[i] for part in out]) for i in range(4))


def _pole(ai) -> tuple[float, float] | None:
    """Detector (row, column) corner coordinates where the direct beam lands, or None if it never does."""
    x, y, z = _lab(ai, np.array([0., 1., 0.]), np.array([0., 0., 1.]))
    jacobian = np.array([[x[1] - x[0], x[2] - x[0]], [y[1] - y[0], y[2] - y[0]]])
    if abs(np.linalg.det(jacobian)) < 1e-30:
        return None
    row, col = np.linalg.solve(jacobian, -np.array([x[0], y[0]]))
    rx, ry, rz = _lab(ai, np.array([row]), np.array([col]))
    if rz[0] <= 0 or math.hypot(rx[0], ry[0]) > 1e-9 * abs(rz[0]):
        return None
    return float(row), float(col)


def _seam_crossings(t0, p0, t1, p1, sign):
    """Crossings of the ray φ = ±π by each sub-edge (with p0 normalised into [−π, π))."""
    shift = np.floor((p0 + _PI) / _TWO_PI) * _TWO_PI
    a, b = p0 - shift, p1 - shift
    up = b >= _PI
    down = b < -_PI
    out_t, out_sign = [], []
    for mask, target, direction in ((up, _PI, 1.0), (down, -_PI, -1.0)):
        if not np.any(mask):
            continue
        fraction = (target - a[mask]) / (b[mask] - a[mask])
        out_t.append(t0[mask] + fraction * (t1[mask] - t0[mask]))
        out_sign.append(-sign[mask] * direction)
    if not out_t:
        return np.empty(0), np.empty(0)
    return np.concatenate(out_t), np.concatenate(out_sign)


def _cut_point(horizontal, a: np.ndarray, b: np.ndarray, ga: np.ndarray, gb: np.ndarray) -> np.ndarray:
    """Points where q_horz changes sign between corners a and b (bisection on pyFAI's own q_horz)."""
    lo, hi = np.zeros(len(a)), np.ones(len(a))
    for _ in range(40):
        middle = .5 * (lo + hi)
        point = a + (b - a) * middle[:, None]
        value = horizontal(point[:, 0], point[:, 1])
        same = np.sign(value) == np.sign(ga)
        lo, hi = np.where(same, middle, lo), np.where(same, hi, middle)
    fraction = .5 * (lo + hi)
    return a + (b - a) * fraction[:, None]


def _gi_pieces(ai, valid: np.ndarray, horizontal, cancel, pole=None):
    """Directed boundary edges of each χ_GI half: the unmasked boundary split at q_horz = 0, plus the cut.

    The cut contains the direct beam (q = 0), where χ_GI has no value: through the pixel holding ``pole`` the cut is
    split there, each half ending a millionth of a pixel short of it.
    """
    height, width = valid.shape
    corner_sign = np.empty((height + 1, width + 1), np.int8)
    rows = np.arange(height + 1, dtype=float)
    cols = np.arange(width + 1, dtype=float)
    for chunk in np.array_split(np.arange(height + 1), max(1, (height + 1) // 64)):
        _check(cancel)
        rr, cc = np.meshgrid(rows[chunk], cols, indexing="ij")
        value = horizontal(rr.ravel(), cc.ravel()).reshape(rr.shape)
        corner_sign[chunk] = np.where(value >= 0, 1, -1)
    start, end = _boundary_edges(valid)

    def sign_at(points):
        return corner_sign[points[:, 0].astype(int), points[:, 1].astype(int)]

    s_start, s_end = sign_at(start), sign_at(end)
    pieces = {1: ([], []), -1: ([], [])}
    same = s_start == s_end
    for side in (1, -1):
        keep = same & (s_start == side)
        pieces[side][0].append(start[keep])
        pieces[side][1].append(end[keep])
    # boundary edges that cross the cut: split them at the crossing
    cross = np.flatnonzero(~same)
    crossing_cache: dict[tuple, np.ndarray] = {}
    if cross.size:
        points = _cut_point(horizontal, start[cross], end[cross], s_start[cross].astype(float),
                            s_end[cross].astype(float))
        for index, point in zip(cross, points):
            key = tuple(sorted((tuple(start[index]), tuple(end[index]))))
            crossing_cache[key] = point
            pieces[int(s_start[index])][0].append(start[index][None])
            pieces[int(s_start[index])][1].append(point[None])
            pieces[int(s_end[index])][0].append(point[None])
            pieces[int(s_end[index])][1].append(end[index][None])
    # the cut through unmasked pixels whose corners straddle q_horz = 0
    corner = corner_sign
    straddle = valid & ~((corner[:-1, :-1] == corner[:-1, 1:]) & (corner[:-1, :-1] == corner[1:, :-1])
                         & (corner[:-1, :-1] == corner[1:, 1:]))
    for r, c in zip(*np.nonzero(straddle)):
        ring = [(r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c)]
        crossings = []
        for (ra, ca), (rb, cb) in zip(ring, ring[1:] + ring[:1]):
            if corner[ra, ca] != corner[rb, cb]:
                key = tuple(sorted(((float(ra), float(ca)), (float(rb), float(cb)))))
                if key not in crossing_cache:
                    crossing_cache[key] = _cut_point(
                        horizontal, np.array([[ra, ca]], float), np.array([[rb, cb]], float),
                        np.array([float(corner[ra, ca])]), np.array([float(corner[rb, cb])]))[0]
                crossings.append(crossing_cache[key])
        if len(crossings) != 2:
            raise ValueError("q_horz = 0 is not resolvable at this detector's pixel scale")
        first, second = crossings
        segments = [(first, second)]
        if pole is not None and (int(math.floor(pole[0])), int(math.floor(pole[1]))) == (r, c):
            centre = np.array(pole, float)
            near = [centre + 1e-6 * (end - centre) / max(np.linalg.norm(end - centre), 1e-300)
                    for end in (first, second)]
            segments = [(first, near[0]), (near[1], second)]
        direction = second - first
        plus = next(np.array(point, float) for point in ring if corner[point] == 1)
        # (row, col) vectors: in (col a, row b) the left normal of (Δcol, Δrow) is (−Δrow, Δcol)
        to_plus = plus - first
        plus_on_left = (-direction[0]) * to_plus[1] + direction[1] * to_plus[0] > 0
        for a, b in segments:
            if plus_on_left:
                pieces[1][0].append(a[None]); pieces[1][1].append(b[None])
                pieces[-1][0].append(b[None]); pieces[-1][1].append(a[None])
            else:
                pieces[1][0].append(b[None]); pieces[1][1].append(a[None])
                pieces[-1][0].append(a[None]); pieces[-1][1].append(b[None])
    out = {}
    for side, (starts, ends) in pieces.items():
        out[side] = (np.concatenate(starts) if starts else np.empty((0, 2)),
                     np.concatenate(ends) if ends else np.empty((0, 2)))
    return out, corner_sign


def _single_flight(lock, flights: dict, key, lookup: Callable[[], Any], compute: Callable[[], Any], cancel) -> Any:
    """``lookup()`` (called under ``lock``), else ``compute()`` by one caller while concurrent callers of the same
    ``key`` wait, cancellably, and then look again."""
    import threading
    while True:
        with lock:
            found = lookup()
            if found is not None:
                return found
            flight = flights.setdefault(key, threading.Lock())
        if flight.acquire(timeout=0.05):
            break
        _check(cancel)
    try:
        with lock:
            found = lookup()
        return found if found is not None else compute()
    finally:
        with lock:
            if flights.get(key) is flight:
                del flights[key]
        flight.release()


class SupportGeometryCache:
    """A small thread-safe cache one Run keeps across its poses (see :func:`support_footprint`); a missing entry is
    built by one caller while concurrent callers wait for it."""

    def __init__(self, limit: int = 4) -> None:
        import threading
        self._lock = threading.Lock()
        self._items: dict = {}
        self._flights: dict = {}
        self._limit = int(limit)

    def get(self, key):
        with self._lock:
            return self._items.get(key)

    def put(self, key, value) -> None:
        with self._lock:
            if key not in self._items and len(self._items) >= self._limit:
                self._items.pop(next(iter(self._items)))
            self._items[key] = value

    def once(self, key, build: Callable[[], Any], cancel=None) -> Any:
        def compute():
            value = build()
            self.put(key, value)
            return value
        return _single_flight(self._lock, self._flights, key, lambda: self._items.get(key), compute, cancel)


def support_footprint(pose: SupportPose, convention: str, *, cancel=None,
                      cache: SupportGeometryCache | None = None) -> SupportFootprint:
    """The wavelength-independent boundary of one pose in (2θ, φ), or the named reason it has none.

    ``cache`` (a caller-owned :class:`SupportGeometryCache`) keeps each geometry's loaded pyFAI geometry, and χ_GI's
    detector-plane pieces between poses differing only by incidence angle at zero tilt: q_horz, hence the cut, does
    not depend on it; only the mapping to (2θ, χ_GI) is recomputed.
    """
    geometry = json.dumps(pose.geometry, sort_keys=True)
    try:
        ai = (_integrator(pose.geometry) if cache is None
              else cache.once(("geometry", geometry), lambda: _integrator(pose.geometry), cancel))
    except SupportCancelled:
        raise
    except Exception as error:  # a geometry pyFAI cannot load is not qualified
        return SupportFootprint(pose.key, convention, reason=SupportReason.UNQUALIFIED_GEOMETRY,
                                detail=f"geometry not loadable: {type(error).__name__}")
    unqualified = _qualification(ai)
    if unqualified:
        return SupportFootprint(pose.key, convention, reason=SupportReason.UNQUALIFIED_GEOMETRY, detail=unqualified)
    shape = tuple(int(size) for size in ai.detector.shape)
    if pose.mask is not None and pose.mask.shape != shape:
        return SupportFootprint(pose.key, convention, reason=SupportReason.UNQUALIFIED_GEOMETRY,
                                detail=f"mask shape {pose.mask.shape} differs from the detector's {shape}")
    try:
        if convention == "detector_chi":
            valid = np.ones(shape, bool) if pose.mask is None else ~pose.mask
            # (row, col) corner coordinates; boundary edges are directed (col right, row up), region on the left
            start_ab, end_ab = _boundary_edges(valid)
            mapper = _mapper(ai, convention, None)
            sigma = _orientation(mapper, pose.mask, shape)
            if sigma == 0:
                return SupportFootprint(pose.key, convention, np.empty(0), np.empty(0), np.empty(0), np.empty(0),
                                        np.empty(0), np.empty(0), np.empty(0), False)
            t0, p0, t1, p1 = _subdivide(mapper, start_ab, end_ab, None, cancel)
            sign = np.full(t0.size, float(sigma))
            seam_t, seam_sign = _seam_crossings(t0, p0, t1, p1, sign)
            pole = _pole(ai)
            inside = bool(pole is not None and 0 <= pole[0] < shape[0] and 0 <= pole[1] < shape[1]
                          and valid[int(pole[0]), int(pole[1])])
            order = np.argsort(seam_t, kind="stable")
            return SupportFootprint(pose.key, convention, t0, p0, t1, p1, sign, seam_t[order], seam_sign[order],
                                    inside)
        def cut():
            valid = np.ones(shape, bool) if pose.mask is None else ~pose.mask
            horizontal = _horizontal_q(ai, pose.gi)
            pole = _pole(ai)
            inside = pole is not None and 0 <= pole[0] < shape[0] and 0 <= pole[1] < shape[1]
            return _gi_pieces(ai, valid, horizontal, cancel, pole if inside else None)
        if cache is not None and pose.gi[1] == 0.0:
            pieces, corner_sign = cache.once(("gi_cut", geometry, pose.mask_digest, pose.gi[2]), cut, cancel)
        else:
            pieces, corner_sign = cut()
        mapper = _mapper(ai, convention, pose.gi)
        parts = []
        for side in (1, -1):
            starts, ends = pieces[side]
            if not starts.size:
                continue
            sigma = _orientation(mapper, pose.mask, shape, side, corner_sign)
            if sigma == 0:
                continue
            t0, p0, t1, p1 = _subdivide(mapper, starts, ends, float(side), cancel)
            parts.append((t0, p0, t1, p1, np.full(t0.size, float(sigma))))
        if not parts:
            return SupportFootprint(pose.key, convention, np.empty(0), np.empty(0), np.empty(0), np.empty(0),
                                    np.empty(0), np.empty(0), np.empty(0), False)
        t0, p0, t1, p1, sign = (np.concatenate([part[i] for part in parts]) for i in range(5))
        return SupportFootprint(pose.key, convention, t0, p0, t1, p1, sign, np.empty(0), np.empty(0), False)
    except ValueError as error:
        return SupportFootprint(pose.key, convention, reason=SupportReason.UNQUALIFIED_GEOMETRY, detail=str(error))


# --------------------------------------------------------------------------------------------- sampling stage

def _native_two_theta(values: np.ndarray, unit: str, wavelength_m: float | None) -> np.ndarray:
    if unit == "2th_deg":
        return np.radians(values)
    if unit == "2th_rad":
        return np.asarray(values, float)
    if wavelength_m is None:
        raise ValueError("a q profile's support needs its wavelength")
    scale = 1e10 if unit == "q_A^-1" else 1e9
    s = np.clip(np.asarray(values, float) * wavelength_m * scale / (4 * _PI), 0.0, 1.0)
    return 2 * np.arcsin(s)


def _measure(t: np.ndarray, unit: str) -> np.ndarray:
    """The native radial measure as a function of 2θ: sin θ for q (∝ q), 2θ itself for 2θ."""
    return np.sin(.5 * t) if unit.startswith("q_") else t


def _window_radians(intervals) -> np.ndarray:
    out = []
    for lo, hi in intervals:
        for a, b in _circle(lo, hi) if hi - lo < 360.0 else _FULL:
            out.append((math.radians(a), math.radians(b)))
    return np.asarray(out, float).reshape(-1, 2)


def _line_support(footprints: Sequence[SupportFootprint], lines: np.ndarray, windows: Sequence[np.ndarray],
                  cancel) -> list[np.ndarray]:
    """Radians of each window reached on each radial line (2θ), by the union over footprints."""
    count = lines.size
    start_count = np.zeros(count)
    event_line, event_phi, event_delta = [], [], []
    for footprint in footprints:
        _check(cancel)
        if footprint.t0 is None or not footprint.t0.size:
            if footprint.pole_inside:
                start_count += 1.0
            continue
        t0, p0, t1, p1, sign = footprint.t0, footprint.p0, footprint.t1, footprint.p1, footprint.sign
        low, high = np.minimum(t0, t1), np.maximum(t0, t1)
        first = np.searchsorted(lines, low, side="right")
        last = np.searchsorted(lines, high, side="right")
        hits = last - first
        edges = np.flatnonzero(hits > 0)
        if edges.size:
            repeats = hits[edges]
            owner = np.repeat(edges, repeats)
            index = np.repeat(first[edges] - np.cumsum(repeats) + repeats, repeats) + np.arange(repeats.sum())
            tau = lines[index]
            phi = p0[owner] + (tau - t0[owner]) * (p1[owner] - p0[owner]) / (t1[owner] - t0[owner])
            event_line.append(index)
            event_phi.append((phi + _PI) % _TWO_PI - _PI)
            event_delta.append(sign[owner] * np.sign(t1[owner] - t0[owner]))
        if footprint.convention == "detector_chi":
            base = 1.0 if footprint.pole_inside else 0.0
            cumulative = np.concatenate(([0.0], np.cumsum(footprint.seam_sign)))
            start_count += base + cumulative[np.searchsorted(footprint.seam_t, lines, side="left")]
    covered = [np.zeros(count) for _ in windows]
    if event_line:
        line = np.concatenate(event_line)
        phi = np.concatenate(event_phi)
        delta = np.concatenate(event_delta)
        order = np.lexsort((phi, line))
        line, phi, delta = line[order], phi[order], delta[order]
        running = np.cumsum(delta)
        group_start = np.r_[0, np.flatnonzero(np.diff(line)) + 1]
        before = np.repeat(running[group_start] - delta[group_start], np.diff(np.r_[group_start, line.size]))
        after = start_count[line] + running - before
        next_phi = np.r_[phi[1:], np.nan]
        last_in_line = np.r_[line[1:] != line[:-1], True]
        next_phi[last_in_line] = _PI
        segments = after > 0.5
        seg_line, seg_lo, seg_hi = line[segments], phi[segments], next_phi[segments]
        heads = line[group_start]
        head_covered = start_count[heads] > 0.5
        seg_line = np.r_[seg_line, heads[head_covered]]
        seg_lo = np.r_[seg_lo, np.full(head_covered.sum(), -_PI)]
        seg_hi = np.r_[seg_hi, phi[group_start][head_covered]]
        touched = np.zeros(count, bool)
        touched[heads] = True
    else:
        seg_line, seg_lo, seg_hi = np.empty(0, int), np.empty(0), np.empty(0)
        touched = np.zeros(count, bool)
    whole = (~touched) & (start_count > 0.5)
    for result, window in zip(covered, windows):
        if not window.size:
            continue
        for w_lo, w_hi in window:
            if seg_line.size:
                length = np.clip(np.minimum(seg_hi, w_hi) - np.maximum(seg_lo, w_lo), 0.0, None)
                result += np.bincount(seg_line, weights=length, minlength=count)
            result[whole] += w_hi - w_lo
    return covered


def _bin_means(lines: np.ndarray, spacing: float, values: np.ndarray, edges_t: np.ndarray, unit: str) -> np.ndarray:
    """Mean of ``values`` (one per line cell) over each native bin, weighted by the native measure."""
    cell_edges = np.r_[lines - .5 * spacing, lines[-1] + .5 * spacing]
    measure = _measure(cell_edges, unit)
    cumulative = np.r_[0.0, np.cumsum(values * np.diff(measure))]

    def integral(t):
        position = np.clip((t - cell_edges[0]) / spacing, 0, lines.size)
        cell = np.minimum(position.astype(int), lines.size - 1)
        within = _measure(np.clip(t, cell_edges[0], cell_edges[-1]), unit) - measure[cell]
        return cumulative[cell] + values[cell] * within

    lo, hi = edges_t[:-1], edges_t[1:]
    width = _measure(hi, unit) - _measure(lo, unit)
    out = np.zeros(lo.size)
    positive = width > 0
    out[positive] = (integral(hi[positive]) - integral(lo[positive])) / width[positive]
    return out


def _pyfai_version() -> str:
    import pyFAI
    return str(pyFAI.version)


@dataclass(frozen=True, slots=True, eq=False)
class SupportLineTable:
    """Radians of the admitted and lost χ reached on uniform radial lines in 2θ (wavelength-independent).

    ``lines`` are cell centres ``spacing`` apart; any native bins inside ``[cell_start, cell_stop]`` integrate from it.
    """

    lines: np.ndarray
    spacing: float
    admitted: np.ndarray
    lost: np.ndarray

    @property
    def cell_start(self) -> float:
        return float(self.lines[0] - .5 * self.spacing)

    @property
    def cell_stop(self) -> float:
        return float(self.lines[-1] + .5 * self.spacing)

    def serves(self, edges_t: np.ndarray, spacing: float) -> bool:
        return (self.spacing <= spacing * (1 + 1e-12) and self.cell_start <= max(0.0, float(edges_t[0]))
                and self.cell_stop >= min(_PI, float(edges_t[-1])))


def support_line_table(footprints: Sequence[SupportFootprint], window: AdmittedWindow, *, spacing: float,
                       start: float, stop: float, cancel=None) -> SupportLineTable:
    """The union's reach on uniform 2θ lines over [start, stop] (radians), for ``window``'s admitted/lost χ."""
    start, stop = max(0.0, float(start)), min(_PI, float(stop))
    count = max(1, int(math.ceil((stop - start) / spacing)))
    lines = start + (np.arange(count) + .5) * spacing
    admitted, lost = _line_support(footprints, lines, (_window_radians(window.admitted_deg),
                                                       _window_radians(window.lost_deg)), cancel)
    for array in (lines, admitted, lost):
        array.setflags(write=False)
    return SupportLineTable(lines, float(spacing), admitted, lost)


def _needed_spacing(edges_t: np.ndarray, oversample: int) -> float:
    widths = np.diff(edges_t)
    return float(np.min(widths[widths > 0])) / oversample if np.any(widths > 0) else 0.0


def angular_support(request: AngularSupportRequest, *, oversample: int = 8, cancel=None,
                    footprints: Mapping[str, SupportFootprint] | None = None,
                    table: SupportLineTable | None = None) -> AngularSupportProfile:
    """The bin-mean angular support of one native profile, or its named reason.

    ``footprints`` may supply cached :func:`support_footprint` results keyed by pose key (the geometry stage is
    wavelength-independent); missing ones are computed. ``oversample`` lines per smallest native bin sample the radial
    direction (one further doubling changes a fraction by ≤0.0025 on the qualified fixtures). A ``table`` already
    computed for these footprints and window (at that spacing or finer) is reused; otherwise one is computed.
    """
    facts = dict(radial_unit=request.radial_unit, convention=request.convention, origin=request.origin,
                 requested_window_deg=request.window.requested_deg)
    window = request.window
    if request.radial_unit not in SUPPORTED_UNITS:
        return AngularSupportProfile.unavailable(SupportReason.UNSUPPORTED_UNIT, detail=request.radial_unit, **facts)
    if window.reason is not None:
        return AngularSupportProfile.unavailable(window.reason, detail=window.detail, **facts)
    if type(oversample) is not int or oversample < 1:
        raise ValueError("oversample must be a positive integer")
    unique: dict[str, SupportPose] = {}
    for pose in request.poses:
        unique.setdefault(pose.key, pose)
    computed = []
    for key, pose in unique.items():
        _check(cancel)
        footprint = (footprints or {}).get(key) or support_footprint(pose, request.convention, cancel=cancel)
        if footprint.reason is not None:
            return AngularSupportProfile.unavailable(footprint.reason, detail=footprint.detail, **facts)
        computed.append(footprint)
    edges_t = _native_two_theta(request.bin_edges, request.radial_unit, request.wavelength_m)
    spacing = _needed_spacing(edges_t, oversample)
    provenance = {
        "pyfai_version": _pyfai_version(),
        "wavelength_m": request.wavelength_m,
        "oversample": oversample,
        "unique_poses": len(unique),
        "poses_sha256": hashlib.sha256("".join(sorted(unique)).encode()).hexdigest(),
        **dict(request.provenance),
    }
    values = np.zeros(request.bin_edges.size - 1)
    lost = np.zeros_like(values)
    if spacing > 0:
        if table is None or not table.serves(edges_t, spacing):
            table = support_line_table(computed, window, spacing=spacing, start=float(edges_t[0]) - spacing,
                                       stop=float(edges_t[-1]) + spacing, cancel=cancel)
        values = np.degrees(_bin_means(table.lines, table.spacing, table.admitted, edges_t, request.radial_unit))
        lost = np.degrees(_bin_means(table.lines, table.spacing, table.lost, edges_t, request.radial_unit))
    if window.lost_deg and np.max(lost) > _TRUNCATION_SHARE * window.denominator_deg:
        return AngularSupportProfile.unavailable(
            SupportReason.WINDOW_TRUNCATED, detail=f"up to {np.max(lost):.2f}° of reachable χ not admitted",
            effective_window_deg=window.admitted_deg, **facts)
    values = np.clip(values, 0.0, window.denominator_deg)
    return AngularSupportProfile(
        origin=request.origin, supported_deg=values, window_deg=window.denominator_deg,
        requested_window_deg=window.requested_deg, effective_window_deg=window.admitted_deg,
        convention=request.convention, radial_unit=request.radial_unit,
        bin_edges=(float(request.bin_edges[0]), float(request.bin_edges[-1])), provenance=provenance)


# --------------------------------------------------------------------------------------------- run capture

#: What :func:`_referent` returns for an input whose weakly held object no longer exists.
_GONE = object()


def _weakly(value: Any) -> Any:
    """A weak reference to ``value`` (None stays None; an object refusing weak references is held strongly)."""
    import weakref
    if value is None:
        return None
    try:
        return weakref.ref(value)
    except TypeError:
        return lambda: value


def _referent(held: Any) -> Any:
    if held is None:
        return None
    value = held()
    return _GONE if value is None else value


def edges_from_centres(centres: Any, radial_range: tuple[float, float] | None = None) -> np.ndarray | None:
    """A reducer's uniform bin edges recovered from its float64 bin centres (never from rounded float32 ones)."""
    values = np.asarray(centres, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        return None
    if values.size == 1:
        if radial_range is None:
            return None
        lo, hi = (float(item) for item in radial_range)
        return np.array([lo, hi]) if hi > lo else None
    delta = (values[-1] - values[0]) / (values.size - 1)
    if not delta > 0:
        return None
    uniform = values[0] + delta * np.arange(values.size)
    if np.max(np.abs(values - uniform)) > 1e-9 * (values[-1] - values[0]):
        return None
    return values[0] - .5 * delta + delta * np.arange(values.size + 1)


class RunSupportCapture:
    """One Run's support capture: a footprint cache, the geometry-key budget (D3) and applied-mask digests.

    Shared by every reduction worker of one Run (thread-safe). It never touches intensity: a frame over the budget
    gets ``over_budget`` while its reduction is unchanged. Footprints are kept least-recently-used within
    ``cache_bytes`` and ``cache_keys``; distinct geometry keys are counted for the budget whether or not their
    footprints stay cached. A new footprint or line table is computed by one worker while the others wait for it.
    """

    def __init__(self, policy: AngularSupportPolicy | None = None, *, cache_bytes: int = 64 << 20,
                 cache_keys: int = 8) -> None:
        import threading
        self.policy = policy if policy is not None else AngularSupportPolicy()
        self._lock = threading.Lock()
        self._footprints: dict[tuple[str, str], SupportFootprint] = {}
        self._keys: set[tuple[str, str]] = set()
        self._masks: dict[tuple[int, int, bool], tuple[Any, Any, np.ndarray | None, str]] = {}
        self._applied: dict[str, np.ndarray] = {}
        self._tables: dict[tuple, SupportLineTable] = {}
        self._profiles: dict[tuple, AngularSupportProfile] = {}
        self._geometry = SupportGeometryCache()
        self._flights: dict[tuple, Any] = {}
        self._cache_bytes = int(cache_bytes)
        self._cache_keys = int(cache_keys)

    @property
    def geometry_keys(self) -> int:
        with self._lock:
            return len(self._keys)

    def applied_mask(self, integrator: Any, explicit: np.ndarray | None, *, bound: bool) -> tuple[Any, str]:
        """The geometric mask the reducer applied, with its digest, cached by the identities of its inputs.

        ``bound``: the run mask is bound into the integrator's detector (its mask is what pyFAI used). Otherwise
        pyFAI received :func:`~xrd_tools.integrate.detector_mask.mask_with_detector`'s union, exactly as the reducer
        built it (a detector mask of another shape is not applied, so it is not geometry either). The workers'
        deep-copied integrators hold equal masks, so one read-only copy is kept per mask content; the inputs are
        held weakly, so the cache never keeps a caller's per-frame mask alive.
        """
        detector = getattr(integrator, "detector", None)
        try:
            detector_mask = None if detector is None else detector.mask
        except Exception:
            detector_mask = None
        source = detector_mask if bound else explicit
        key = (id(source), id(detector_mask), bool(bound))
        with self._lock:
            cached = self._masks.get(key)
            if cached is not None and _referent(cached[0]) is source and _referent(cached[1]) is detector_mask:
                return cached[2], cached[3]
        if bound:
            applied = None if detector_mask is None else np.asarray(detector_mask, dtype=bool)
        else:
            from xrd_tools.integrate.detector_mask import mask_with_detector
            applied = mask_with_detector(integrator, explicit)
            applied = None if applied is None else np.asarray(applied, dtype=bool)
        if applied is not None:
            applied = np.array(applied, dtype=bool, order="C", copy=True)
            applied.setflags(write=False)
        digest = _mask_digest(applied)
        with self._lock:
            if applied is not None:
                applied = self._applied.pop(digest, applied)
                self._applied[digest] = applied
                while len(self._applied) > 4:
                    self._applied.pop(next(iter(self._applied)))
            for stale in [item for item, entry in self._masks.items()
                          if _referent(entry[0]) is _GONE or _referent(entry[1]) is _GONE]:
                del self._masks[stale]
            self._masks.pop(key, None)
            self._masks[key] = (_weakly(source), _weakly(detector_mask), applied, digest)
            while len(self._masks) > 16:
                self._masks.pop(next(iter(self._masks)))
        return applied, digest

    def _once(self, key: tuple, lookup: Callable[[], Any], compute: Callable[[], Any], cancel) -> Any:
        """``lookup()`` (under the lock), else ``compute()`` by one worker while the others wait for its result."""
        return _single_flight(self._lock, self._flights, key, lookup, compute, cancel)

    def _footprint(self, pose: SupportPose, convention: str, cancel) -> SupportFootprint | None:
        key = (pose.key, convention)
        with self._lock:
            cached = self._footprints.pop(key, None)
            if cached is not None:
                self._footprints[key] = cached
                return cached
            budget = self.policy.max_geometry_keys
            if key not in self._keys:
                if budget is not None and len(self._keys) >= budget:
                    return None
                self._keys.add(key)

        def compute() -> SupportFootprint:
            footprint = support_footprint(pose, convention, cancel=cancel, cache=self._geometry)
            with self._lock:
                self._footprints[key] = footprint
                total = sum(item.nbytes for item in self._footprints.values())
                while len(self._footprints) > 1 and (total > self._cache_bytes
                                                     or len(self._footprints) > self._cache_keys):
                    total -= self._footprints.pop(next(iter(self._footprints))).nbytes
            return footprint
        return self._once(("footprint", key), lambda: self._footprints.get(key), compute, cancel)

    def profile(self, *, integrator: Any, mask: np.ndarray | None, mask_digest: str, convention: str,
                radial_unit: str, centres: Any, wavelength_m: float | None, window: AdmittedWindow,
                gi: tuple[float, float, int] | None = None, radial_range=None,
                provenance: Mapping[str, Any] | None = None, cancel=None) -> AngularSupportProfile:
        """The Run's support for one frame's native profile (origin ``run``), or its named reason."""
        facts = dict(radial_unit=radial_unit, convention=convention, origin=SupportOrigin.RUN,
                     requested_window_deg=window.requested_deg)
        if not self.policy.enabled:
            return AngularSupportProfile.unavailable(SupportReason.RECORDING_DISABLED, **facts)
        if radial_unit not in SUPPORTED_UNITS:
            return AngularSupportProfile.unavailable(SupportReason.UNSUPPORTED_UNIT, detail=radial_unit, **facts)
        if window.reason is not None:
            return AngularSupportProfile.unavailable(window.reason, detail=window.detail, **facts)
        edges = edges_from_centres(centres, radial_range)
        if edges is None:
            return AngularSupportProfile.unavailable(
                SupportReason.UNQUALIFIED_GEOMETRY, detail="the radial bins are not a uniform grid", **facts)
        if not callable(getattr(integrator, "get_config", None)):
            return AngularSupportProfile.unavailable(
                SupportReason.UNQUALIFIED_GEOMETRY, detail="not a pyFAI integrator", **facts)
        pose = SupportPose.from_integrator(integrator, mask, gi=gi, mask_digest=mask_digest)
        reuse = (pose.key, convention, window.admitted_deg, window.lost_deg, window.denominator_deg,
                 window.requested_deg)
        exact = reuse + (radial_unit, edges.tobytes(), wavelength_m)
        with self._lock:
            done = self._profiles.get(exact)
            if done is not None:
                return done
        footprint = self._footprint(pose, convention, cancel)
        if footprint is None:
            return AngularSupportProfile.unavailable(
                SupportReason.OVER_BUDGET, detail=f"more than {self.policy.max_geometry_keys} detector geometries",
                **facts)
        request = AngularSupportRequest(
            poses=(pose,), convention=convention, radial_unit=radial_unit, bin_edges=edges,
            wavelength_m=wavelength_m, window=window, provenance=dict(provenance or {}, mask_sha256=mask_digest))
        table = None
        if footprint.reason is None:
            edges_t = _native_two_theta(edges, radial_unit, wavelength_m)
            needed = _needed_spacing(edges_t, 8)

            def serving() -> SupportLineTable | None:
                found = self._tables.get(reuse)
                return found if found is not None and found.serves(edges_t, needed) else None

            def build() -> SupportLineTable:
                with self._lock:
                    old = self._tables.get(reuse)
                # the first table is exactly the direct computation's; a later finer or wider need (an energy scan's
                # next wavelength, in either direction) rebuilds with headroom, never at every frame: the spacing
                # halves only when a finer one is needed, and the range grows by a quarter of the span on each side
                if old is None:
                    spacing, lo, hi = needed, float(edges_t[0]) - needed, float(edges_t[-1]) + needed
                else:
                    spacing = old.spacing if old.spacing <= needed * (1 + 1e-12) else needed / 2
                    margin = spacing + .25 * float(edges_t[-1] - edges_t[0])
                    lo = min(old.cell_start, float(edges_t[0]) - margin)
                    hi = max(old.cell_stop, float(edges_t[-1]) + margin)
                built = support_line_table((footprint,), window, spacing=spacing, start=lo, stop=hi, cancel=cancel)
                with self._lock:
                    self._tables[reuse] = built
                    while len(self._tables) > 16:
                        self._tables.pop(next(iter(self._tables)))
                return built
            if needed > 0:
                table = self._once(("table", reuse), serving, build, cancel)
        profile = angular_support(request, cancel=cancel, footprints={pose.key: footprint}, table=table)
        with self._lock:
            self._profiles[exact] = profile
            while len(self._profiles) > 256:
                self._profiles.pop(next(iter(self._profiles)))
        return profile


# --------------------------------------------------------------------------------------------- Stitch union

class StitchSupportAccumulator:
    """A Stitch's union of included contributors, streamed (MultiGeometry backend).

    Workers call :meth:`pose` with each frame's integrator and the operation's geometric mask (never the frame's
    value-invalid pixels); a new pose's footprint is computed once there. The merging thread calls :meth:`include`
    only for frames the screen keeps, so held-then-excluded frames never count. Repeated poses (same geometry and
    mask) are one footprint and one union member; every included frame still counts in the membership.
    """

    def __init__(self, policy: AngularSupportPolicy | None = None, *, backend: str = "multigeometry") -> None:
        import threading
        self.policy = policy if policy is not None else AngularSupportPolicy()
        self.backend = backend
        self._lock = threading.Lock()
        self._footprints: dict[str, SupportFootprint] = {}
        self._members: dict[str, SupportPose] = {}
        self._included = 0
        self._requested: tuple[float, float] | None = None
        self._used: tuple[float, float] | None = None
        self._wavelength: float | None = None
        self._refusal: tuple[SupportReason, str] | None = None

    def window(self, requested, used) -> None:
        """The profile's requested χ window and the window MultiGeometry actually received (scouted or given)."""
        self._requested = None if requested is None else tuple(float(value) for value in requested)
        self._used = None if used is None else tuple(float(value) for value in used)

    def pose(self, integrator: Any, mask: np.ndarray | None, *, cancel=None) -> SupportPose | None:
        """This frame's geometric pose (computed footprint cached), as a mask-free member key."""
        from xrd_tools.integrate.detector_mask import mask_with_detector
        applied = mask_with_detector(integrator, mask)
        applied = None if applied is None else np.array(applied, dtype=bool, order="C", copy=True)
        if applied is not None:
            applied.setflags(write=False)
        digest = _mask_digest(applied)
        pose = SupportPose(dict(integrator.get_config()), applied, None, digest)
        with self._lock:
            if self._wavelength is None:
                self._wavelength = getattr(integrator, "wavelength", None)
            known = pose.key in self._footprints
            budget = self.policy.max_geometry_keys
            if not known and budget is not None and len(self._footprints) >= budget:
                self._refusal = (SupportReason.OVER_BUDGET, f"more than {budget} contributor geometries")
                return None
        if not known:
            footprint = support_footprint(pose, "detector_chi", cancel=cancel)
            with self._lock:
                self._footprints.setdefault(pose.key, footprint)
        return SupportPose(pose.geometry, None, None, digest)

    def include(self, member: SupportPose | None) -> None:
        """One frame the screen kept (merging thread, frame order)."""
        if member is None:
            return
        with self._lock:
            self._included += 1
            self._members.setdefault(member.key, member)

    def profile(self, *, radial: Any, unit: str, radial_range=None, cancel=None) -> AngularSupportProfile:
        """The union's bin-mean support over the Stitch profile's own radial bins (origin ``run``)."""
        window = admitted_window(self._requested, reducer="stitch_multigeometry",
                                 scouted_deg=self._used if self._requested is None else None)
        facts = dict(radial_unit=unit, convention="detector_chi", origin=SupportOrigin.RUN,
                     requested_window_deg=window.requested_deg)
        if self._refusal is not None:
            return AngularSupportProfile.unavailable(self._refusal[0], detail=self._refusal[1], **facts)
        if unit not in SUPPORTED_UNITS:
            return AngularSupportProfile.unavailable(SupportReason.UNSUPPORTED_UNIT, detail=unit, **facts)
        if window.reason is not None:
            return AngularSupportProfile.unavailable(window.reason, detail=window.detail, **facts)
        for member in self._members.values():
            footprint = self._footprints.get(member.key)
            if footprint is not None and footprint.reason is not None:
                return AngularSupportProfile.unavailable(footprint.reason, detail=footprint.detail, **facts)
        edges = edges_from_centres(radial, radial_range)
        if edges is None or not self._members:
            return AngularSupportProfile.unavailable(
                SupportReason.UNQUALIFIED_GEOMETRY,
                detail="no included contributor" if not self._members else "the radial bins are not uniform",
                **facts)
        members = tuple(self._members.values())
        membership = hashlib.sha256("".join(sorted(member.key for member in members)).encode()).hexdigest()
        request = AngularSupportRequest(
            poses=members, convention="detector_chi", radial_unit=unit, bin_edges=edges,
            wavelength_m=self._wavelength, window=window,
            provenance={"backend": self.backend, "included_contributors": self._included,
                        "membership_sha256": membership})
        return angular_support(request, cancel=cancel,
                               footprints={key: self._footprints[key] for key in self._members})


# --------------------------------------------------------------------------------------------- XU geometry

def xu_corner_grid(geometry: Any, acquisition: Mapping[str, Any], *, nu: float, del_: float):
    """(2θ, χ) in radians at the (N+1)×(M+1) pixel corners of one XU pose, from xrayutilities' own geometry.

    χ follows XU's convention for the asset's camera (the pixel-centre lease's); 2θ comes from |q| at the mapping
    wavelength. pyFAI is not consulted: XU's accepted geometry is its own (D8).
    """
    from xrd_tools.integrate.xu_stitch import XuPowderQProvider
    provider = XuPowderQProvider(geometry, acquisition)
    with provider.frame(nu=nu, del_=del_, corners=True) as lease:
        q = np.array(lease.q_magnitude(), dtype=np.float64)
        chi = np.radians(np.array(lease.chi(), dtype=np.float64))
    wavelength = float(geometry.projection.runtime_xu_wavelength_A)
    with np.errstate(invalid="ignore"):
        two_theta = 2 * np.arcsin(np.clip(q * wavelength / (4 * _PI), 0.0, 1.0))
    return two_theta, chi


def _great_circle_subedges(t0: np.ndarray, p0: np.ndarray, t1: np.ndarray, p1: np.ndarray):
    """Sub-edges no wider than 0.25° in φ along the exact images of straight pixel edges of a flat detector.

    Seen from the sample, a straight edge is an arc of a great circle, so renormalised linear interpolation of its two
    corner directions traces it exactly (only the parametrisation differs). An edge through the beam is refused.
    """
    step = (p1 - p0 + _PI) % _TWO_PI - _PI
    pieces = np.maximum(1, np.ceil(np.abs(step) / _SUB_EDGE_PHI)).astype(np.int64)
    simple = pieces == 1
    out = [(t0[simple], p0[simple], t1[simple], p0[simple] + step[simple])]
    split = np.flatnonzero(~simple)
    if split.size:
        counts = pieces[split]
        owner = np.repeat(np.arange(split.size), counts + 1)
        offset = np.arange(owner.size) - np.repeat(np.cumsum(counts + 1) - (counts + 1), counts + 1)
        fraction = (offset / counts[owner])[:, None]

        def direction(t, p):
            return np.stack((np.cos(t), np.sin(t) * np.cos(p), np.sin(t) * np.sin(p)), axis=-1)
        vector = ((1 - fraction) * direction(t0[split], p0[split])[owner]
                  + fraction * direction(t1[split], p1[split])[owner])
        vector /= np.linalg.norm(vector, axis=1, keepdims=True)
        t = np.arccos(np.clip(vector[:, 0], -1.0, 1.0))
        p = np.arctan2(vector[:, 2], vector[:, 1])
        steps = (np.diff(p) + _PI) % _TWO_PI - _PI
        boundaries = np.cumsum(counts + 1)
        steps = np.delete(steps, boundaries[:-1] - 1)
        if steps.size and np.max(np.abs(steps)) > _SINGULAR_STEP:
            raise ValueError("an unmasked pixel edge passes through the beam")
        keep = np.ones(t.size, bool)
        keep[boundaries - 1] = False
        tails = np.ones(t.size, bool)
        tails[np.concatenate(([0], boundaries[:-1]))] = False
        out.append((t[keep], p[keep], t[tails], p[keep] + steps))
    return tuple(np.concatenate([part[i] for part in out]) for i in range(4))


def footprint_from_corners(key: str, two_theta: np.ndarray, chi: np.ndarray,
                           mask: np.ndarray | None) -> SupportFootprint:
    """A detector-χ footprint from a full (N+1)×(M+1) corner grid of (2θ, χ) (XU's corners).

    Pixel edges map corner to corner; an edge wider than 0.25° in χ is subdivided along its exact great-circle image
    (:func:`_great_circle_subedges`, no further XU evaluation), and an edge through the beam refuses the pose.
    """
    shape = (two_theta.shape[0] - 1, two_theta.shape[1] - 1)
    if chi.shape != two_theta.shape or min(shape) < 1 or (mask is not None and mask.shape != shape):
        return SupportFootprint(key, "detector_chi", reason=SupportReason.XU_GEOMETRY_UNQUALIFIED,
                                detail="corner grid and mask shapes disagree")
    if not (np.all(np.isfinite(two_theta)) and np.all(np.isfinite(chi))):
        return SupportFootprint(key, "detector_chi", reason=SupportReason.XU_GEOMETRY_UNQUALIFIED,
                                detail="corner directions are not finite")
    valid = np.ones(shape, bool) if mask is None else ~mask
    start, end = _boundary_edges(valid)

    def mapped(rows, cols, side=None):
        r, c = np.asarray(rows).astype(int), np.asarray(cols).astype(int)
        return two_theta[r, c], chi[r, c]

    sigma = _orientation(mapped, mask, shape)
    if sigma == 0:
        empty = np.empty(0)
        return SupportFootprint(key, "detector_chi", empty, empty, empty, empty, empty, empty, empty, False)
    t0, p0 = mapped(start[:, 0], start[:, 1])
    t1, p1 = mapped(end[:, 0], end[:, 1])
    try:
        t0, p0, t1, p1 = _great_circle_subedges(t0, p0, t1, p1)
    except ValueError as error:
        return SupportFootprint(key, "detector_chi", reason=SupportReason.XU_GEOMETRY_UNQUALIFIED, detail=str(error))
    sign = np.full(t0.size, float(sigma))
    seam_t, seam_sign = _seam_crossings(t0, p0, t1, p1, sign)
    # the beam lands in a valid pixel when that pixel's corners wind once around it
    winding = sum((np.roll(chi_ring, -1, axis=0) - chi_ring + _PI) % _TWO_PI - _PI
                  for chi_ring in [np.stack((chi[:-1, :-1], chi[:-1, 1:], chi[1:, 1:], chi[1:, :-1]))]).sum(axis=0)
    inside = bool(np.any(valid & (np.abs(winding) > _PI)))
    order = np.argsort(seam_t, kind="stable")
    return SupportFootprint(key, "detector_chi", t0, p0, t1, p1, sign, seam_t[order], seam_sign[order], inside)


class XuSupportAccumulator(StitchSupportAccumulator):
    """The union of an XU Stitch's included contributors, each footprint from XU's own corner grid."""

    def __init__(self, policy: AngularSupportPolicy | None = None) -> None:
        super().__init__(policy, backend="xu_hist")

    def xu_pose(self, geometry: Any, acquisition: Mapping[str, Any], *, nu: float, del_: float,
                mask: np.ndarray | None) -> SupportPose | None:
        """One XU frame's member (session thread): its corner footprint is computed once per distinct pose."""
        applied = None if mask is None else np.array(mask, dtype=bool, order="C", copy=True)
        digest = _mask_digest(applied)
        identity = {"xu_geometry": geometry.projection.fingerprint, "nu": float(nu), "del": float(del_)}
        member = SupportPose(identity, None, None, digest)
        with self._lock:
            if self._wavelength is None:
                self._wavelength = float(geometry.projection.runtime_xu_wavelength_A) * 1e-10
            known = member.key in self._footprints
            budget = self.policy.max_geometry_keys
            if not known and budget is not None and len(self._footprints) >= budget:
                self._refusal = (SupportReason.OVER_BUDGET, f"more than {budget} contributor geometries")
                return None
        if not known:
            two_theta, chi = xu_corner_grid(geometry, acquisition, nu=nu, del_=del_)
            footprint = footprint_from_corners(member.key, two_theta, chi, applied)
            with self._lock:
                self._footprints.setdefault(member.key, footprint)
        return member

    def window(self, requested, used=None) -> None:
        super().window(requested, None)

    def profile(self, *, radial: Any, unit: str, radial_range=None, cancel=None) -> AngularSupportProfile:
        window = admitted_window(self._requested, reducer="stitch_xu")
        facts = dict(radial_unit=unit, convention="detector_chi", origin=SupportOrigin.RUN,
                     requested_window_deg=window.requested_deg)
        if self._refusal is not None:
            return AngularSupportProfile.unavailable(self._refusal[0], detail=self._refusal[1], **facts)
        if unit not in SUPPORTED_UNITS:
            return AngularSupportProfile.unavailable(SupportReason.UNSUPPORTED_UNIT, detail=unit, **facts)
        if window.reason is not None:
            return AngularSupportProfile.unavailable(window.reason, detail=window.detail, **facts)
        for member in self._members.values():
            footprint = self._footprints.get(member.key)
            if footprint is not None and footprint.reason is not None:
                return AngularSupportProfile.unavailable(footprint.reason, detail=footprint.detail, **facts)
        edges = edges_from_centres(radial, radial_range)
        if edges is None or not self._members:
            return AngularSupportProfile.unavailable(
                SupportReason.UNQUALIFIED_GEOMETRY,
                detail="no included contributor" if not self._members else "the radial bins are not uniform",
                **facts)
        members = tuple(self._members.values())
        membership = hashlib.sha256("".join(sorted(member.key for member in members)).encode()).hexdigest()
        request = AngularSupportRequest(
            poses=members, convention="detector_chi", radial_unit=unit, bin_edges=edges,
            wavelength_m=self._wavelength, window=window,
            provenance={"backend": self.backend, "included_contributors": self._included,
                        "membership_sha256": membership})
        return angular_support(request, cancel=cancel,
                               footprints={key: self._footprints[key] for key in self._members})
