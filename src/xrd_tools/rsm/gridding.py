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

import logging
import types
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from xrd_tools.core.geometry import DetectorHeader, PixelQMap
from xrd_tools.rsm.volume import RSMVolume

# xrayutilities is required for any real gridding call; the try/except
# lets the module import in environments where xu isn't installed (e.g.
# CI sandboxes), so tests can monkeypatch ``xu.Gridder3D`` and exercise
# the chunk-handoff machinery without the full RSM stack.
try:
    import xrayutilities as xu
except ModuleNotFoundError:  # pragma: no cover — exercised in sandbox only
    xu = types.SimpleNamespace(Gridder3D=None)


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

def _new_gridder(bins: tuple[int, int, int],
                 bounds: tuple[float, ...] | None = None):
    """A KeepData(True) + Normalize(False) Gridder3D (bare-SUM accumulator)."""
    g = xu.Gridder3D(*bins)
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
    num = np.asarray(grid_raw.data, dtype=float)
    den = np.asarray(grid_norm.data, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~(den > 0)] = np.nan
    return out


def _qbounds(qx, qy, qz) -> tuple[float, float, float, float, float, float]:
    return (float(np.nanmin(qx)), float(np.nanmax(qx)),
            float(np.nanmin(qy)), float(np.nanmax(qy)),
            float(np.nanmin(qz)), float(np.nanmax(qz)))


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
    img = np.array(img, dtype=float, copy=True)
    if img.ndim != 3:
        raise ValueError(
            f"img must be a 3D stack of shape (n, ny, nx), got {img.shape}"
        )

    if roi is not None:
        r0, r1, c0, c1 = roi
        img = img[:, r0:r1, c0:c1]

    if mask_static_pixels:
        std_dev = np.nanstd(img, axis=0)
        img[:, std_dev == 0] = np.nan

    qx, qy, qz = mapper.pixel_q(
        angles,
        energy,
        UB=UB,
        roi=roi,
        image_shape=img.shape,
    )

    bounds = _qbounds(qx, qy, qz)
    grid_raw = _new_gridder(bins, bounds)
    grid_norm = _new_gridder(bins, bounds)
    _feed_pair(grid_raw, grid_norm, qx, qy, qz, img,
               1.0 if weight is None else weight)

    return RSMVolume(
        h=np.asarray(grid_raw.xaxis, dtype=float),
        k=np.asarray(grid_raw.yaxis, dtype=float),
        l=np.asarray(grid_raw.zaxis, dtype=float),
        intensity=_pair_intensity(grid_raw, grid_norm),
    )


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

    def __init__(self, mapper: PixelQMap, bins: tuple[int, int, int]) -> None:
        self.mapper = mapper
        self.bins = tuple(int(b) for b in bins)
        # the Σraw and Σnorm accumulators (P6) — both KeepData+Normalize(False)
        self._grid_raw: xu.Gridder3D | None = None
        self._grid_norm: xu.Gridder3D | None = None
        self._bounds: tuple[float, float, float, float, float, float] | None = None
        self.n_frames_processed: int = 0

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
        self._grid_raw = _new_gridder(self.bins, bounds)
        self._grid_norm = _new_gridder(self.bins, bounds)
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
        _feed_pair(self._grid_raw, self._grid_norm, qx, qy, qz, img,
                   1.0 if weight is None else weight)
        self.n_frames_processed += img.shape[0]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_volume(self) -> RSMVolume:
        """Build the final :class:`RSMVolume` from the accumulated grid."""
        if self._grid_raw is None:
            raise RuntimeError(
                "bounds not set; call set_bounds() or scout() before to_volume()."
            )
        if self.n_frames_processed == 0:
            raise RuntimeError(
                "no chunks processed; call add() at least once before to_volume()."
            )
        return RSMVolume(
            h=np.asarray(self._grid_raw.xaxis, dtype=float),
            k=np.asarray(self._grid_raw.yaxis, dtype=float),
            l=np.asarray(self._grid_raw.zaxis, dtype=float),
            intensity=_pair_intensity(self._grid_raw, self._grid_norm),
        )


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

    sg = StreamingGridder(mapper, bins)
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
        sg.add(
            img_chunk,
            angles_chunk,
            energy,
            UB=UB,
            roi=roi,
            static_mask=static_mask,
            weight=weight,
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
