"""Image-stack → RSMVolume gridding.

Two paths:

* :func:`grid_img_data` — single-shot, in-memory.  Materialises the full
  ``(N, H, W)`` image stack and the per-pixel ``(qx, qy, qz)`` arrays
  simultaneously.  Fine for small scans; OOM territory for the
  9-scan × 126-frame × 514×1030 Eiger case.
* :class:`StreamingGridder` + :func:`grid_img_data_streaming` /
  :func:`grid_scans_streaming` — memory-bounded.  Uses
  ``xu.Gridder3D.KeepData(True)`` + a fixed ``dataRange`` so frame
  chunks accumulate into a single output grid.  Memory ceiling is
  ``chunk_size × H × W × (4 × 8 bytes)``, plus the final
  ``bins[0] × bins[1] × bins[2] × 8`` output buffer.

For multi-scan post-hoc concatenation of *already-gridded* volumes (e.g.
results saved across sessions) see :func:`combine_grids` /
:func:`get_common_grid`.
"""
from __future__ import annotations

from contextlib import contextmanager
import gc
import logging
import math
import threading
import weakref
from dataclasses import InitVar, dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from xrd_tools.core.geometry import DetectorHeader, PixelQMap
from xrd_tools.core.physical_memory import physical_root_fact
from xrd_tools.core.geometry.xu_runtime import (
    XuRuntimeSession,
    require_active_xu_runtime_session,
    xu_runtime_session,
)
from xrd_tools.rsm.volume import RSMVolume

# Private test seam. Production obtains Gridder3D only from an active shared
# XuRuntimeSession; importing this module never imports xrayutilities.
_GRIDDER3D_OVERRIDE = None
_RSM_GRID_CHUNK_FACTORY = object()
_RSM_GRID_RELEASE_FACTORY = object()


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The shared Σ(raw) / Σ(norm) accumulator (P6)
# ---------------------------------------------------------------------------
# RSM bins onto a 3D grid via two xu.Gridder3D running with Normalize(False)
# (so ``.data`` returns the bare SUM, not xu's count-mean): ``_grid_raw``
# accumulates Σraw, ``_grid_norm`` accumulates Σnorm, and the volume is
# ``Σraw/Σnorm`` — the SAME accumulator as the stitch histogram merge
# (``stitch_hist.stitch_q_grid``), and the convention the shared CorrectionStack
# documents (a multiplicative correction is the per-pixel ``norm``).  With
# ``norm = 1`` everywhere it reproduces the prior ``Σraw/Σcounts`` exactly, so
# the default (no corrections) is behaviour-preserving; a CorrectionStack weight
# only changes numbers when supplied.
#
# (Until Jun-2026 this accumulated ``Σ(raw·w)/Σw``, which only *weights* — it
# cannot apply a multiplicative correction: it returned ``Σ(true·C²)/ΣC ≠ true``.
# Unified onto ``Σraw/Σnorm`` per
# docs/design/design_stitch_rsm_accumulator_jun2026.md.)

@contextmanager
def _gridder_runtime(runtime_session: XuRuntimeSession | None):
    if runtime_session is not None:
        yield require_active_xu_runtime_session(runtime_session)
        return
    if _GRIDDER3D_OVERRIDE is not None:
        yield None
        return
    with xu_runtime_session() as owned:
        yield owned


def _gridder_type(runtime_session: XuRuntimeSession | None):
    if runtime_session is None:
        if _GRIDDER3D_OVERRIDE is None:
            raise RuntimeError("RSM gridder has no active XU runtime")
        return _GRIDDER3D_OVERRIDE
    xu_module, _numpy_module = runtime_session.active_modules()
    return xu_module.Gridder3D


def _new_gridder(
    bins: tuple[int, int, int],
    bounds: tuple[float, ...] | None = None,
    *,
    runtime_session: XuRuntimeSession | None,
):
    """A KeepData(True) + Normalize(False) Gridder3D (bare-SUM accumulator)."""
    g = _gridder_type(runtime_session)(*bins)
    g.KeepData(True)
    g.Normalize(False)
    if bounds is not None:
        g.dataRange(*bounds, fixed=True)
    return g


def _feed_pair(grid_raw, grid_norm, qx, qy, qz, img, weight) -> None:
    """Accumulate one chunk into the Σraw and Σnorm gridders.

    ``weight`` is the per-pixel ``norm`` of the shared correction convention: the
    raw channel accumulates Σraw (the bare image), the norm channel accumulates
    Σnorm, and the volume is ``Σraw/Σnorm`` — so a multiplicative correction
    ``raw = true·C`` is applied by passing ``norm = C`` (recovers ``true``), the
    SAME accumulator and convention as :func:`stitch_q_grid`.

    Bad pixels — non-finite q, non-finite image, or ``weight <= 0`` — are set
    to NaN in BOTH channels so xrayutilities drops them from each sum (the same
    good-mask :func:`stitch_q_grid` uses): a NaN q COORD would otherwise be
    clamped onto an edge bin, and a masked pixel's finite norm would pollute the
    denominator → a biased mean.
    """
    img = np.asarray(img, dtype=float)
    w = np.broadcast_to(np.asarray(weight, dtype=float), img.shape)
    bad = ~(np.isfinite(qx) & np.isfinite(qy) & np.isfinite(qz)
            & np.isfinite(img) & np.isfinite(w) & (w > 0))
    grid_raw(qx, qy, qz, np.where(bad, np.nan, img))
    grid_norm(qx, qy, qz, np.where(bad, np.nan, w))


def _pair_intensity(grid_raw, grid_norm) -> np.ndarray:
    """``Σraw / Σnorm``; NaN where the norm sum is non-positive (empty)."""
    out = np.asarray(grid_raw.data, dtype=float)
    den = np.asarray(grid_norm.data, dtype=float)
    positive = den > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(out, den, out=out, where=positive)
    np.logical_not(positive, out=positive)
    out[positive] = np.nan
    return out


def _qbounds(qx, qy, qz) -> tuple[float, float, float, float, float, float]:
    return (float(np.nanmin(qx)), float(np.nanmax(qx)),
            float(np.nanmin(qy)), float(np.nanmax(qy)),
            float(np.nanmin(qz)), float(np.nanmax(qz)))


class RSMGridChunkReleaseError(RuntimeError):
    """A consumed R2 chunk could not prove complete root release."""

    code = "RSM_CHUNK_RELEASE_FAILED"

    def __init__(self) -> None:
        super().__init__(self.code)


def _uncopyable_chunk_value(kind: str):
    def copy_value(self):
        raise TypeError(f"{kind} is not copyable")

    def deepcopy_value(self, _memo):
        raise TypeError(f"{kind} is not copyable")

    def reduce_value(self):
        raise TypeError(f"{kind} is not serializable")

    def reduce_ex_value(self, _protocol):
        raise TypeError(f"{kind} is not serializable")

    def replace_value(self, /, **_changes):
        raise TypeError(f"{kind} is not replaceable")

    return (
        copy_value,
        deepcopy_value,
        reduce_value,
        reduce_ex_value,
        replace_value,
    )


def _ndarray_root(value: np.ndarray, name: str) -> np.ndarray:
    try:
        root = physical_root_fact(value).root
    except ValueError as error:
        raise TypeError(f"{name} has no exact ndarray root") from error
    if type(root) is not np.ndarray:
        raise TypeError(f"{name} must have an exact ndarray root")
    return root


class RSMGridChunkLease:
    """One-shot owner of the two operation-owned R2 image roots."""

    __slots__ = (
        "_raw",
        "_conditioned",
        "_raw_root_ref",
        "_conditioned_root_ref",
        "_shape",
        "frame_count",
        "_consumed",
    )

    def __init__(
        self,
        raw_stack: np.ndarray,
        conditioned_stack: np.ndarray,
        frame_count: int,
        _claim: object = None,
    ) -> None:
        if (
            _claim is not _RSM_GRID_CHUNK_FACTORY
            or type(raw_stack) is not np.ndarray
            or raw_stack.dtype.kind not in "biuf"
            or raw_stack.ndim != 3
            or not raw_stack.flags.c_contiguous
            or type(conditioned_stack) is not np.ndarray
            or conditioned_stack.dtype != np.dtype(np.float64)
            or conditioned_stack.ndim != 3
            or not conditioned_stack.flags.c_contiguous
            or conditioned_stack.shape != raw_stack.shape
            or type(frame_count) is not int
            or frame_count < 1
            or frame_count != raw_stack.shape[0]
        ):
            raise TypeError("RSM grid chunk lease is not factory-owned")
        raw_root = _ndarray_root(raw_stack, "RSM raw chunk")
        conditioned_root = _ndarray_root(
            conditioned_stack,
            "RSM conditioned chunk",
        )
        if raw_root is conditioned_root:
            raise ValueError("RSM raw and conditioned chunks must not alias")
        object.__setattr__(self, "_raw", raw_stack)
        object.__setattr__(self, "_conditioned", conditioned_stack)
        object.__setattr__(self, "_raw_root_ref", weakref.ref(raw_root))
        object.__setattr__(
            self,
            "_conditioned_root_ref",
            weakref.ref(conditioned_root),
        )
        object.__setattr__(
            self,
            "_shape",
            tuple(int(value) for value in raw_stack.shape),
        )
        object.__setattr__(self, "frame_count", frame_count)
        object.__setattr__(self, "_consumed", False)

    @classmethod
    def from_arrays(
        cls,
        raw_stack: np.ndarray,
        conditioned_stack: np.ndarray,
        frame_count: int,
    ) -> "RSMGridChunkLease":
        if cls is not RSMGridChunkLease:
            raise TypeError("RSM grid chunk lease factory requires the exact class")
        return cls(
            raw_stack,
            conditioned_stack,
            frame_count,
            _RSM_GRID_CHUNK_FACTORY,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def consumed(self) -> bool:
        return self._consumed

    def _consume(self, claim: object):
        if claim is not _RSM_GRID_CHUNK_FACTORY or self._consumed:
            raise RuntimeError("RSM grid chunk lease is already consumed")
        raw = self._raw
        conditioned = self._conditioned
        raw_ref = self._raw_root_ref
        conditioned_ref = self._conditioned_root_ref
        if (
            type(raw) is not np.ndarray
            or raw.dtype.kind not in "biuf"
            or raw.ndim != 3
            or not raw.flags.c_contiguous
            or type(conditioned) is not np.ndarray
            or conditioned.dtype != np.dtype(np.float64)
            or conditioned.ndim != 3
            or not conditioned.flags.c_contiguous
            or type(self._shape) is not tuple
            or raw.shape != self._shape
            or conditioned.shape != self._shape
            or type(self.frame_count) is not int
            or self.frame_count != self._shape[0]
            or type(raw_ref) is not weakref.ReferenceType
            or type(conditioned_ref) is not weakref.ReferenceType
        ):
            raise TypeError("RSM grid chunk lease changed before consumption")
        raw_root = _ndarray_root(raw, "RSM raw chunk")
        conditioned_root = _ndarray_root(
            conditioned,
            "RSM conditioned chunk",
        )
        if (
            raw_root is conditioned_root
            or raw_ref() is not raw_root
            or conditioned_ref() is not conditioned_root
        ):
            raise TypeError("RSM grid chunk roots changed before consumption")
        object.__setattr__(self, "_consumed", True)
        object.__setattr__(self, "_raw", None)
        object.__setattr__(self, "_conditioned", None)
        object.__setattr__(self, "_raw_root_ref", None)
        object.__setattr__(self, "_conditioned_root_ref", None)
        return raw, conditioned, raw_ref, conditioned_ref

    def __setattr__(self, _name: str, _value: object) -> None:
        raise TypeError("RSM grid chunk lease is immutable")

    (
        __copy__,
        __deepcopy__,
        __reduce__,
        __reduce_ex__,
        __replace__,
    ) = _uncopyable_chunk_value("RSM grid chunk lease")


@dataclass(eq=False, frozen=True, slots=True, weakref_slot=True)
class RSMGridChunkReleaseReceipt:
    """Positive scalar-only proof that one consumed R2 chunk was released."""

    frame_count: int
    q_root_count: int
    release_passed: bool
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _RSM_GRID_RELEASE_FACTORY
            or type(self.frame_count) is not int
            or self.frame_count < 1
            or type(self.q_root_count) is not int
            or not 1 <= self.q_root_count <= 3
            or self.release_passed is not True
        ):
            raise TypeError("RSM grid chunk release receipt is not factory-owned")

    (
        __copy__,
        __deepcopy__,
        __reduce__,
        __reduce_ex__,
        __replace__,
    ) = _uncopyable_chunk_value("RSM grid chunk release receipt")


_RSM_GRID_RELEASE_ISSUANCE_LOCK = threading.RLock()
_RSM_GRID_RELEASE_ISSUANCE: weakref.WeakKeyDictionary[
    RSMGridChunkReleaseReceipt, tuple[int, int, bool]
] = weakref.WeakKeyDictionary()


def _rsm_grid_chunk_release_facts(
    receipt: object,
    *,
    expected_frame_count: int,
) -> tuple[int, int, bool]:
    """Return immutable factory-issued facts for one exact release receipt."""

    if (
        type(receipt) is not RSMGridChunkReleaseReceipt
        or type(expected_frame_count) is not int
        or expected_frame_count < 1
    ):
        raise RSMGridChunkReleaseError()
    with _RSM_GRID_RELEASE_ISSUANCE_LOCK:
        issued = _RSM_GRID_RELEASE_ISSUANCE.get(receipt)
    if issued is None:
        raise RSMGridChunkReleaseError()
    try:
        observed = (
            receipt.frame_count,
            receipt.q_root_count,
            receipt.release_passed,
        )
    except AttributeError:
        raise RSMGridChunkReleaseError() from None
    if (
        observed != issued
        or issued[0] != expected_frame_count
        or not 1 <= issued[1] <= 3
        or issued[2] is not True
    ):
        raise RSMGridChunkReleaseError()
    return issued


def _bounded_release_collect() -> None:
    """Perform the fixed CPython generation-zero release fence."""

    gc.collect(0)


# ---------------------------------------------------------------------------
# Single-shot path (in-memory)
# ---------------------------------------------------------------------------

def grid_img_data(
    mapper: PixelQMap,
    img: np.ndarray,
    angles: list[np.ndarray] | tuple[np.ndarray, ...],
    energy: float,
    *,
    UB: np.ndarray | None = None,
    bins: tuple[int, int, int] = (200, 200, 200),
    roi: tuple[int, int, int, int] | None = None,
    mask_static_pixels: bool = True,
    weight: np.ndarray | None = None,
    runtime_session: XuRuntimeSession | None = None,
) -> RSMVolume:
    """Map a 3D image stack to reciprocal space and bin onto a 3D grid.

    Single-shot, in-memory path: the full ``(N, H, W)`` image stack plus
    the per-pixel ``(qx, qy, qz)`` arrays are materialised simultaneously.
    For large multi-scan data sets prefer :func:`grid_img_data_streaming`.

    Parameters
    ----------
    mapper : PixelQMap
        Bundles the diffractometer convention + detector header.
    img : ndarray
        Image stack of shape ``(N_frame, H, W)``.
    angles : list of ndarray
        Per-frame angle arrays, in the order ``xu.QConversion`` expects.
    energy : float
        X-ray energy in eV.
    UB : (3, 3) ndarray, optional
        Sample orientation matrix.  ``None`` → raw lab-frame q.
    bins : tuple of int
        Grid bin counts along (qx, qy, qz).
    roi : (r0, r1, c0, c1), optional
        Crop ROI applied to ``img`` *and* the mapper's header.
    mask_static_pixels : bool
        Mask pixels whose per-frame variance is zero (typically hot
        masks or chip gaps).  Default ``True``.
    weight : ndarray, optional
        Per-pixel correction ``norm`` (``(H, W)`` or ``(N, H, W)``) — the
        denominator of the ``Σraw/Σnorm`` accumulator, e.g. a ``CorrectionStack``
        normalization (a multiplicative correction ``raw = true·C`` is applied by
        ``norm = C``).  ``None`` → unit norm (the count-mean).

    Returns
    -------
    RSMVolume
        Gridded H-K-L volume.
    """
    if runtime_session is not None:
        require_active_xu_runtime_session(runtime_session)
    img = np.array(img, dtype=float, copy=True)
    if img.ndim != 3:
        raise ValueError(
            f"img must be a 3D stack of shape (n, ny, nx), got {img.shape}"
        )

    if roi is not None:
        r0, r1, c0, c1 = roi
        img = img[:, r0:r1, c0:c1]

    # Mask temporally-static pixels (hot masks / chip gaps) by their zero
    # per-frame variance.  Two guards keep this from wiping the WHOLE image to
    # NaN (which yields a silently-empty RSMVolume with no error): skip a
    # single-frame stack (variance is meaningless and 0 everywhere), and skip
    # when EVERY pixel is static (a fully constant-in-time stack).  A real
    # multi-frame scene has photon noise, so the all-static case only trips on
    # degenerate/synthetic input; the caller can still pass an explicit mask.
    if mask_static_pixels and img.shape[0] >= 2:
        static = np.nanstd(img, axis=0) == 0
        if not static.all():
            img[:, static] = np.nan

    qx, qy, qz = mapper.pixel_q(
        angles,
        energy,
        UB=UB,
        roi=roi,
        image_shape=img.shape,
    )

    bounds = _qbounds(qx, qy, qz)
    with _gridder_runtime(runtime_session) as active_runtime:
        grid_raw = _new_gridder(
            bins, bounds, runtime_session=active_runtime
        )
        grid_norm = _new_gridder(
            bins, bounds, runtime_session=active_runtime
        )
        _feed_pair(
            grid_raw,
            grid_norm,
            qx,
            qy,
            qz,
            img,
            1.0 if weight is None else weight,
        )
        h = np.array(grid_raw.xaxis, dtype=float, copy=True)
        k = np.array(grid_raw.yaxis, dtype=float, copy=True)
        l = np.array(grid_raw.zaxis, dtype=float, copy=True)
        intensity = _pair_intensity(grid_raw, grid_norm)

    return RSMVolume(h=h, k=k, l=l, intensity=intensity)


# ---------------------------------------------------------------------------
# Streaming path (chunk-by-chunk, memory-bounded)
# ---------------------------------------------------------------------------

def _corner_pixel_q(
    mapper: PixelQMap,
    angles: list[np.ndarray] | tuple[np.ndarray, ...],
    energy: float,
    *,
    UB: np.ndarray | None = None,
    roi: tuple[int, int, int, int] | None = None,
    image_shape: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a tiny edge scout of ``(qx, qy, qz)`` for each frame.

    Cheap scout used to determine grid bounds without materialising the full
    per-pixel q arrays.  Uses a virtual ``3 × 3`` detector where possible
    (falling back to ``2`` along any two-pixel axis), with ``cch'`` and
    ``pwidth'`` adjusted so virtual pixels land on the original detector's
    first, middle, and last pixels.  The old corner-only scout could under-bound
    curved detector mappings whose extrema sit on an edge midpoint.

    Returns three arrays of shape ``(N_frame, n_sample_1, n_sample_2)``.
    """
    h = mapper.header
    if roi is not None:
        h = h.with_roi(roi)
    if image_shape is not None:
        h = h.with_image_shape(image_shape)
    if h.Nch1 < 2 or h.Nch2 < 2:
        raise ValueError(
            f"corner scout requires Nch1, Nch2 >= 2 (got {h.Nch1}, {h.Nch2})"
        )

    n1, n2 = h.Nch1, h.Nch2
    s1 = 3 if n1 > 2 else 2
    s2 = 3 if n2 > 2 else 2
    scale1 = (n1 - 1) / (s1 - 1)
    scale2 = (n2 - 1) / (s2 - 1)
    tiny = DetectorHeader(
        cch1=h.cch1 / scale1,
        cch2=h.cch2 / scale2,
        pwidth1=h.pwidth1 * scale1,
        pwidth2=h.pwidth2 * scale2,
        distance=h.distance,
        Nch1=s1,
        Nch2=s2,
    )
    tiny_mapper = PixelQMap(mapper.diff_config, tiny)
    return tiny_mapper.pixel_q(angles, energy, UB=UB)


@dataclass(frozen=True)
class StreamingScan:
    """One scan's data for :func:`grid_scans_streaming`.

    The ``img`` field is the eager in-memory stack.  For lazy / on-disk
    loading (e.g. v2 NeXus scans) prefer driving :class:`StreamingGridder`
    directly so frames can be materialised one chunk at a time.
    """

    img: np.ndarray
    angles: list[np.ndarray] = field(default_factory=list)
    energy: float = 0.0
    UB: np.ndarray | None = None
    roi: tuple[int, int, int, int] | None = None


class StreamingGridder:
    """Memory-bounded RSM gridder.

    Wraps ``xu.Gridder3D`` in ``KeepData(True)`` mode with a fixed
    ``dataRange``.  Frame chunks are fed in via :meth:`add`; the final
    volume is built by :meth:`to_volume`.

    Bounds must be set before the first :meth:`add` call, either via
    :meth:`set_bounds` (user supplies q ranges) or :meth:`scout` (compute
    q on a 3×3 virtual-detector grid — corners, edge midpoints, and centre —
    across a list of scans; C1).

    Memory ceiling (per chunk): ``chunk_size × H × W × (4 arrays × 8 bytes)``
    plus the gridder's bin buffer of ``bins[0] × bins[1] × bins[2] × 8``.
    """

    def __init__(
        self,
        mapper: PixelQMap,
        bins: tuple[int, int, int],
        *,
        runtime_session: XuRuntimeSession | None = None,
    ) -> None:
        if runtime_session is not None:
            require_active_xu_runtime_session(runtime_session)
        self.mapper = mapper
        self.bins = tuple(int(b) for b in bins)
        self._runtime_session = runtime_session
        # the Σraw and Σnorm accumulators (P6) — both KeepData+Normalize(False)
        self._grid_raw: object | None = None
        self._grid_norm: object | None = None
        self._bounds: tuple[float, float, float, float, float, float] | None = None
        self.n_frames_processed: int = 0
        self._poisoned = False

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    def set_bounds(
        self,
        qx_range: tuple[float, float],
        qy_range: tuple[float, float],
        qz_range: tuple[float, float],
    ) -> None:
        """Fix the q grid bounds explicitly.

        Must be called before :meth:`add`.  Bins outside the range are
        silently dropped by ``xu.Gridder3D``.
        """
        if self._poisoned:
            raise RSMGridChunkReleaseError()
        if self._grid_raw is not None:
            raise RuntimeError("bounds already set; create a fresh "
                               "StreamingGridder to re-bound")
        qxmin, qxmax = float(qx_range[0]), float(qx_range[1])
        qymin, qymax = float(qy_range[0]), float(qy_range[1])
        qzmin, qzmax = float(qz_range[0]), float(qz_range[1])
        for lo, hi, name in ((qxmin, qxmax, "qx"),
                             (qymin, qymax, "qy"),
                             (qzmin, qzmax, "qz")):
            if not (hi > lo):
                raise ValueError(f"{name} range must be (lo, hi) with hi > lo; "
                                 f"got ({lo}, {hi})")

        bounds = (qxmin, qxmax, qymin, qymax, qzmin, qzmax)
        with _gridder_runtime(self._runtime_session) as active_runtime:
            self._grid_raw = _new_gridder(
                self.bins, bounds, runtime_session=active_runtime
            )
            self._grid_norm = _new_gridder(
                self.bins, bounds, runtime_session=active_runtime
            )
        self._bounds = bounds

    def scout(
        self,
        scans: list[tuple[
            list[np.ndarray] | tuple[np.ndarray, ...],  # angles
            float,                                       # energy
            np.ndarray | None,                           # UB
            tuple[int, ...] | None,                      # image_shape
        ]],
        *,
        roi: tuple[int, int, int, int] | None = None,
        pad: float = 0.0,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Scout detector-edge q for each scan and set the union bounds.

        Parameters
        ----------
        scans : list of tuple
            Each entry is ``(angles, energy, UB, image_shape)`` — only
            the angle arrays, energy, UB, and detector size are needed
            (no actual image data).
        roi : (r0, r1, c0, c1), optional
            ROI applied to the header before scouting.  Same shape /
            semantics as in :meth:`add`.
        pad : float
            Multiplicative padding applied to each axis range to give
            edge pixels a little slack (e.g. 0.02 → +/- 2%).  Default 0.

        Returns
        -------
        ((qx_lo, qx_hi), (qy_lo, qy_hi), (qz_lo, qz_hi))
            The bounds that were set.
        """
        if self._poisoned:
            raise RSMGridChunkReleaseError()
        if not scans:
            raise ValueError("scout: scans list must not be empty")

        qx_lo = qy_lo = qz_lo = np.inf
        qx_hi = qy_hi = qz_hi = -np.inf
        for angles, energy, UB, image_shape in scans:
            qx_c, qy_c, qz_c = _corner_pixel_q(
                self.mapper, angles, energy,
                UB=UB, roi=roi, image_shape=image_shape,
            )
            qx_lo = float(min(qx_lo, np.nanmin(qx_c)))
            qx_hi = float(max(qx_hi, np.nanmax(qx_c)))
            qy_lo = float(min(qy_lo, np.nanmin(qy_c)))
            qy_hi = float(max(qy_hi, np.nanmax(qy_c)))
            qz_lo = float(min(qz_lo, np.nanmin(qz_c)))
            qz_hi = float(max(qz_hi, np.nanmax(qz_c)))

        if pad > 0:
            qx_lo, qx_hi = _pad_range(qx_lo, qx_hi, pad)
            qy_lo, qy_hi = _pad_range(qy_lo, qy_hi, pad)
            qz_lo, qz_hi = _pad_range(qz_lo, qz_hi, pad)

        self.set_bounds((qx_lo, qx_hi), (qy_lo, qy_hi), (qz_lo, qz_hi))
        return (qx_lo, qx_hi), (qy_lo, qy_hi), (qz_lo, qz_hi)

    # ------------------------------------------------------------------
    # Chunk feed
    # ------------------------------------------------------------------

    def add(
        self,
        img: np.ndarray,
        angles: list[np.ndarray] | tuple[np.ndarray, ...],
        energy: float,
        *,
        UB: np.ndarray | None = None,
        roi: tuple[int, int, int, int] | None = None,
        static_mask: np.ndarray | None = None,
        weight: np.ndarray | None = None,
    ) -> None:
        """Process one chunk of frames and accumulate it into the gridder.

        Parameters
        ----------
        img : ndarray
            Frame chunk of shape ``(n_chunk, H, W)`` or ``(H, W)`` for a
            single frame.
        angles : list of ndarray
            Per-frame angle arrays for the chunk, in ``xu.QConversion``
            order.  Each array's length must equal ``n_chunk``.
        energy : float
            X-ray energy in eV (constant across the chunk).
        UB : (3, 3) ndarray, optional
            Sample orientation matrix.
        roi : (r0, r1, c0, c1), optional
            ROI crop applied to the chunk and to the mapper header.
        static_mask : ndarray of bool, optional
            2D mask matching the per-frame image shape ``(H, W)``
            *after* ROI cropping.  ``True`` → pixel is set to NaN
            before gridding.  Use this to apply a detector / hot-pixel
            mask consistently across all chunks; computing it from a
            chunk-local std heuristic (as the old
            ``mask_static_pixels=True`` did) makes results depend on
            ``chunk_size``, which is a scientific-reproducibility bug.
            The single-shot :func:`grid_img_data` retains its own
            per-frame std heuristic only because there is only one
            "chunk" (the full stack) — see its docstring.
        weight : ndarray, optional
            Per-pixel correction ``norm`` (``(H, W)`` or ``(n_chunk, H, W)``) —
            the denominator of the ``Σraw/Σnorm`` accumulator (e.g. a
            ``CorrectionStack`` normalization; ``raw = true·C`` is applied by
            ``norm = C``).  ``None`` → unit norm (the count-mean).

        Notes
        -----
        The chunk-shape contract is ``(n_chunk, H, W)``; ``static_mask``
        must be ``(H, W)``.  Broadcast is applied along the frame axis.
        """
        if self._poisoned:
            raise RSMGridChunkReleaseError()
        if self._grid_raw is None:
            raise RuntimeError(
                "StreamingGridder bounds not set; call set_bounds() or "
                "scout() before add()."
            )

        img = np.array(img, dtype=float, copy=True)
        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        if img.ndim != 3:
            raise ValueError(
                f"img must be (n, H, W) or (H, W); got shape {img.shape}"
            )

        if roi is not None:
            r0, r1, c0, c1 = roi
            img = img[:, r0:r1, c0:c1]

        if static_mask is not None:
            sm = np.asarray(static_mask, dtype=bool)
            if sm.shape != img.shape[1:]:
                raise ValueError(
                    f"static_mask shape {sm.shape} must match per-frame "
                    f"image shape {img.shape[1:]} (after any ROI crop)"
                )
            img[:, sm] = np.nan

        with _gridder_runtime(self._runtime_session):
            qx, qy, qz = self.mapper.pixel_q(
                angles,
                energy,
                UB=UB,
                roi=roi,
                image_shape=img.shape,
            )
            if qx.shape != img.shape:
                raise ValueError(
                    f"per-pixel q shape {qx.shape} does not match chunk shape "
                    f"{img.shape}; check angle array lengths"
                )
            _feed_pair(
                self._grid_raw,
                self._grid_norm,
                qx,
                qy,
                qz,
                img,
                1.0 if weight is None else weight,
            )
        self.n_frames_processed += img.shape[0]

    def add_leased(
        self,
        lease: RSMGridChunkLease,
        angles: list[np.ndarray] | tuple[np.ndarray, ...],
        energy: float,
        *,
        UB: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
        weight: float | np.ndarray | None = None,
    ) -> RSMGridChunkReleaseReceipt:
        """Consume one R2-owned chunk and prove every downstream root dead.

        Unlike :meth:`add`, this path performs no hidden image copy and requires
        one explicit already-active XU runtime owner.  Any failure after lease
        consumption poisons the accumulator because one half of the raw/norm
        pair may already have changed.
        """

        if self._poisoned:
            raise RSMGridChunkReleaseError()
        if self._grid_raw is None or self._grid_norm is None:
            raise RuntimeError(
                "StreamingGridder bounds not set; call set_bounds() before "
                "add_leased()."
            )
        if type(lease) is not RSMGridChunkLease or lease.consumed:
            raise TypeError("add_leased requires one unused exact RSM chunk lease")
        if type(self.mapper) is not PixelQMap:
            raise TypeError("RSM leased gridding requires exact PixelQMap")
        if type(self._runtime_session) is not XuRuntimeSession:
            raise TypeError("RSM leased gridding requires an explicit XU runtime")
        session = require_active_xu_runtime_session(self._runtime_session)
        if type(angles) not in {tuple, list} or len(angles) != 6:
            raise ValueError("RSM leased gridding requires six angle arrays")
        angle_values = tuple(np.asarray(value, dtype=np.float64) for value in angles)
        if any(
            value.ndim != 1
            or len(value) != lease.frame_count
            or not np.all(np.isfinite(value))
            for value in angle_values
        ):
            raise ValueError("RSM leased angle arrays are not aligned and finite")
        if type(energy) not in {int, float} or not math.isfinite(float(energy)) or energy <= 0:
            raise ValueError("RSM leased energy must be finite and positive")
        if type(UB) is not np.ndarray:
            raise TypeError("RSM leased UB must be an exact ndarray")
        matrix = np.asarray(UB, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("RSM leased UB must be a finite 3 by 3 matrix")
        if roi is not None and (
            type(roi) is not tuple
            or len(roi) != 4
            or any(type(value) is not int for value in roi)
        ):
            raise TypeError("RSM leased ROI must be an exact integer tuple")
        header = self.mapper.header if roi is None else self.mapper.header.with_roi(roi)
        full_shape = (self.mapper.header.Nch1, self.mapper.header.Nch2)
        expected_shape = (lease.frame_count, header.Nch1, header.Nch2)
        if lease.shape[1:] != full_shape or any(value < 1 for value in expected_shape):
            raise ValueError("RSM leased chunk does not match detector geometry")
        if weight is not None:
            if type(weight) in {int, float}:
                if not math.isfinite(float(weight)) or float(weight) <= 0:
                    raise ValueError("RSM leased scalar weight must be positive")
            elif (
                type(weight) is not np.ndarray
                or weight.dtype.kind not in "biuf"
                or weight.shape != expected_shape[1:]
                or not weight.flags.c_contiguous
                or weight.flags.writeable
            ):
                raise TypeError(
                    "RSM leased weight must be a positive scalar or borrowed "
                    "read-only detector array"
                )

        raw = conditioned = conditioned_view = None
        qx = qy = qz = None
        q_values = None
        bad = norm_values = raw_payload = norm_payload = None
        raw_ref = conditioned_ref = None
        q_refs: tuple[weakref.ReferenceType, ...] = ()
        payload_refs: tuple[weakref.ReferenceType, ...] = ()
        q_root_count = 0
        pending: BaseException | None = None
        lease_consumed = False
        release_failed = False
        try:
            raw, conditioned, raw_ref, conditioned_ref = lease._consume(
                _RSM_GRID_CHUNK_FACTORY
            )
            lease_consumed = True
            if raw.shape != lease.shape or conditioned.shape != lease.shape:
                raise ValueError("RSM leased chunk changed shape before consumption")
            if roi is None:
                conditioned_view = conditioned
            else:
                r0, r1, c0, c1 = roi
                conditioned_view = conditioned[:, r0:r1, c0:c1]
            if conditioned_view.shape != expected_shape:
                raise ValueError("RSM leased ROI does not match q geometry")
            # The raw stack participates only in the ownership/release proof;
            # science consumes the already-conditioned numerator.
            raw = None
            q_values = self.mapper.pixel_q(
                angle_values,
                float(energy),
                UB=matrix,
                roi=roi,
                runtime_session=session,
            )
            qx, qy, qz = q_values
            unique_q: dict[int, weakref.ReferenceType] = {}
            for value in q_values:
                root = _ndarray_root(value, "RSM q chunk")
                unique_q.setdefault(id(root), weakref.ref(root))
                root = None
            value = None
            q_refs = tuple(unique_q.values())
            q_root_count = len(q_refs)
            unique_q.clear()
            if not 1 <= q_root_count <= 3:
                raise ValueError("RSM q chunk has an invalid root count")
            norm_values = np.broadcast_to(
                np.asarray(1.0 if weight is None else weight, dtype=np.float64),
                conditioned_view.shape,
            )
            bad = ~(
                np.isfinite(qx)
                & np.isfinite(qy)
                & np.isfinite(qz)
                & np.isfinite(conditioned_view)
                & np.isfinite(norm_values)
                & (norm_values > 0)
            )
            raw_payload = np.where(bad, np.nan, conditioned_view)
            norm_payload = np.where(bad, np.nan, norm_values)
            payload_refs = (
                weakref.ref(_ndarray_root(raw_payload, "RSM raw feed payload")),
                weakref.ref(_ndarray_root(norm_payload, "RSM norm feed payload")),
            )
            self._grid_raw(qx, qy, qz, raw_payload)
            self._grid_norm(qx, qy, qz, norm_payload)
        except BaseException as error:
            try:
                error.__traceback__ = None
                error.__context__ = None
                error.__cause__ = None
            except BaseException:
                pass
            pending = error
        finally:
            raw = conditioned = conditioned_view = None
            qx = qy = qz = None
            q_values = None
            bad = norm_values = raw_payload = norm_payload = None
            angle_values = ()
            matrix = None
            session = None
            try:
                _bounded_release_collect()
            except BaseException:
                release_failed = True
            references = tuple(
                reference
                for reference in (
                    raw_ref,
                    conditioned_ref,
                    *q_refs,
                    *payload_refs,
                )
                if reference is not None
            )
            if any(reference() is not None for reference in references):
                release_failed = True
            references = ()
            q_refs = ()
            payload_refs = ()
        if release_failed or (pending is not None and lease_consumed):
            self._poisoned = True
        if release_failed:
            raise RSMGridChunkReleaseError() from None
        if pending is not None:
            raise pending from None
        self.n_frames_processed += lease.frame_count
        receipt = RSMGridChunkReleaseReceipt(
            lease.frame_count,
            q_root_count,
            True,
            _RSM_GRID_RELEASE_FACTORY,
        )
        with _RSM_GRID_RELEASE_ISSUANCE_LOCK:
            _RSM_GRID_RELEASE_ISSUANCE[receipt] = (
                receipt.frame_count,
                receipt.q_root_count,
                receipt.release_passed,
            )
        return receipt

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_volume(self) -> RSMVolume:
        """Build the final :class:`RSMVolume` from the accumulated grid."""
        if self._poisoned:
            raise RSMGridChunkReleaseError()
        if self._grid_raw is None:
            raise RuntimeError(
                "bounds not set; call set_bounds() or scout() before to_volume()."
            )
        if self.n_frames_processed == 0:
            raise RuntimeError(
                "no chunks processed; call add() at least once before to_volume()."
            )
        with _gridder_runtime(self._runtime_session):
            h = np.array(self._grid_raw.xaxis, dtype=float, copy=True)
            k = np.array(self._grid_raw.yaxis, dtype=float, copy=True)
            l = np.array(self._grid_raw.zaxis, dtype=float, copy=True)
            intensity = _pair_intensity(self._grid_raw, self._grid_norm)
        return RSMVolume(h=h, k=k, l=l, intensity=intensity)


def _pad_range(lo: float, hi: float, pad: float) -> tuple[float, float]:
    span = hi - lo
    return lo - pad * span, hi + pad * span


# ---------------------------------------------------------------------------
# Streaming convenience wrappers
# ---------------------------------------------------------------------------

def grid_img_data_streaming(
    mapper: PixelQMap,
    img: np.ndarray,
    angles: list[np.ndarray] | tuple[np.ndarray, ...],
    energy: float,
    *,
    UB: np.ndarray | None = None,
    bins: tuple[int, int, int] = (200, 200, 200),
    chunk_size: int = 8,
    q_bounds: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ] | None = None,
    roi: tuple[int, int, int, int] | None = None,
    static_mask: np.ndarray | None = None,
    scout_pad: float = 0.0,
    weight: np.ndarray | None = None,
    runtime_session: XuRuntimeSession | None = None,
) -> RSMVolume:
    """Stream a single in-memory image stack through :class:`StreamingGridder`.

    Equivalent to :func:`grid_img_data` but memory-bounded: only a
    ``chunk_size``-frame slice of the stack and its q-arrays live in
    memory at a time.  Importantly the result is **independent of
    ``chunk_size``** — see :meth:`StreamingGridder.add` for why we use
    an explicit ``static_mask`` rather than a chunk-local std heuristic.

    Parameters
    ----------
    chunk_size : int
        Number of frames per chunk.  Memory ceiling per chunk is
        ``chunk_size × H × W × 32`` bytes (1 image + 3 q-arrays in
        float64).  Default 8 (~130 MB on a 514×1030 Eiger frame).
    q_bounds : ((qx_lo, qx_hi), (qy_lo, qy_hi), (qz_lo, qz_hi)), optional
        Explicit grid bounds.  If omitted, a scout pass over the frame
        corners is run to determine bounds (cheap — no real image data
        is read).
    static_mask : ndarray of bool, optional
        2D mask applied to every chunk before gridding.  Apply your
        detector mask (from pyFAI / a calibration step) here.
    scout_pad : float
        Multiplicative padding applied to scouted bounds.  Ignored if
        ``q_bounds`` is given.
    """
    if img.ndim != 3:
        raise ValueError(f"img must be (N, H, W); got shape {img.shape}")
    n_frames = img.shape[0]
    if any(len(np.atleast_1d(a)) != n_frames for a in angles):
        raise ValueError(
            "angles arrays must each have length N matching img.shape[0]"
        )

    # A 3D (N, H, W) *per-frame* weight must be sliced to each chunk's frames,
    # exactly like the image + angle arrays — otherwise every chunk gets the full
    # N-frame weight and _feed_pair fails / mis-broadcasts (the chunk-safety bug).
    # A 2D (H, W) per-pixel weight (what rsm_correction_weight returns) broadcasts
    # over frames and is passed through unchanged.  The weight's spatial dims must
    # already match the POST-ROI image (the caller crops a per-pixel weight to the
    # ROI, as rsm_correction_weight does — add()/grid_img_data crop only the image,
    # never the weight, so cropping it here too would double-crop).
    weight_arr = None if weight is None else np.asarray(weight, dtype=float)
    per_frame_weight = weight_arr is not None and weight_arr.ndim == 3
    if per_frame_weight and weight_arr.shape[0] != n_frames:
        raise ValueError(
            f"per-frame weight has {weight_arr.shape[0]} frames but the stack has "
            f"{n_frames}; a 3D weight must be (N, H, W) with N == img.shape[0].")

    sg = StreamingGridder(
        mapper, bins, runtime_session=runtime_session
    )
    if q_bounds is None:
        sg.scout(
            [(list(angles), energy, UB, img.shape[-2:])],
            roi=roi,
            pad=scout_pad,
        )
    else:
        sg.set_bounds(*q_bounds)

    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        img_chunk = img[start:end]
        angles_chunk = [np.asarray(a)[start:end] for a in angles]
        weight_chunk = weight_arr[start:end] if per_frame_weight else weight_arr
        sg.add(
            img_chunk,
            angles_chunk,
            energy,
            UB=UB,
            roi=roi,
            static_mask=static_mask,
            weight=weight_chunk,
        )
    return sg.to_volume()


def grid_scans_streaming(
    mapper: PixelQMap,
    scans: list[StreamingScan],
    *,
    bins: tuple[int, int, int] = (200, 200, 200),
    chunk_size: int = 8,
    q_bounds: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ] | None = None,
    static_mask: np.ndarray | None = None,
    scout_pad: float = 0.0,
) -> RSMVolume:
    """Stream multiple in-memory scans into a single :class:`RSMVolume`.

    Memory-bounded equivalent of looping :func:`grid_img_data` over each
    scan and then calling :func:`combine_grids` — but a single
    accumulating gridder is used, so no per-scan volume is ever
    materialised, and the post-hoc :class:`RegularGridInterpolator`
    re-binning is avoided.

    Parameters
    ----------
    scans : list of StreamingScan
        Each entry carries its own ``img`` stack, ``angles``, ``energy``,
        ``UB``, and optional ``roi``.  Different scans may have
        different UB / energy / ROI.
    static_mask : ndarray of bool, optional
        2D mask applied to every chunk of every scan before gridding.
        Use this for a detector / hot-pixel mask that's constant across
        the run.
    """
    if not scans:
        raise ValueError("scans must not be empty")

    sg = StreamingGridder(mapper, bins)
    if q_bounds is None:
        sg.scout(
            [(s.angles, s.energy, s.UB, s.img.shape[-2:]) for s in scans],
            roi=None,  # roi is per-scan; scout takes the union without it
            pad=scout_pad,
        )
    else:
        sg.set_bounds(*q_bounds)

    for s in scans:
        n_frames = s.img.shape[0]
        for start in range(0, n_frames, chunk_size):
            end = min(start + chunk_size, n_frames)
            img_chunk = s.img[start:end]
            angles_chunk = [np.asarray(a)[start:end] for a in s.angles]
            sg.add(
                img_chunk,
                angles_chunk,
                s.energy,
                UB=s.UB,
                roi=s.roi,
                static_mask=static_mask,
            )
    return sg.to_volume()


# ---------------------------------------------------------------------------
# Post-hoc volume union (safety valve for cross-session work)
# ---------------------------------------------------------------------------

def get_common_grid(
    volumes: list[RSMVolume],
    bins: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not volumes:
        raise ValueError("volumes must not be empty")

    hmin = min(np.nanmin(v.h) for v in volumes)
    hmax = max(np.nanmax(v.h) for v in volumes)
    kmin = min(np.nanmin(v.k) for v in volumes)
    kmax = max(np.nanmax(v.k) for v in volumes)
    lmin = min(np.nanmin(v.l) for v in volumes)
    lmax = max(np.nanmax(v.l) for v in volumes)

    h = np.linspace(hmin, hmax, bins[0])
    k = np.linspace(kmin, kmax, bins[1])
    l = np.linspace(lmin, lmax, bins[2])
    return h, k, l


def combine_grids(
    volumes: list[RSMVolume],
    bins: tuple[int, int, int],
) -> RSMVolume:
    """Re-grid a list of volumes onto a common grid and sum the intensities.

    Post-hoc safety valve for the case where individual scans were
    gridded separately (different sessions, machines, partial results).
    For in-session multi-scan processing prefer
    :func:`grid_scans_streaming`.
    """
    if not volumes:
        raise ValueError("volumes must not be empty")

    h, k, l = get_common_grid(volumes, bins)
    combined = np.zeros((len(h), len(k), len(l)), dtype=float)

    # Avoid materialising three dense ``(H, K, L)`` coordinate volumes.  A
    # realistic grid can make those temporaries hundreds of MB before the
    # interpolated output even exists.  Interpolate one H-slab at a time from a
    # reusable 2-D K/L point template.
    kl_size = len(k) * len(l)
    k_col = np.repeat(k, len(l))
    l_col = np.tile(l, len(k))
    pts = np.empty((kl_size, 3), dtype=float)
    pts[:, 1] = k_col
    pts[:, 2] = l_col

    for vol in volumes:
        vals = np.nan_to_num(vol.intensity, nan=0.0)
        rgi = RegularGridInterpolator(
            (vol.h, vol.k, vol.l),
            vals,
            bounds_error=False,
            fill_value=0.0,
        )
        for i, h_value in enumerate(h):
            pts[:, 0] = h_value
            combined[i] += rgi(pts).reshape(len(k), len(l))

    return RSMVolume(h=h, k=k, l=l, intensity=combined)
