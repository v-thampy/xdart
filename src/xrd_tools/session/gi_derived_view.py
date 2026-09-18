"""A q–χ VIEW derived from a grazing-incidence q_ip–q_oop cake.  Display only.

A GI run that integrated only ``qip_qoop`` can still be looked at in (q, χ)
without touching the detector frame again, because the two are one coordinate
change (pyFAI's fiber convention, which xdart's direct ``q_chi`` mode uses)::

    q = hypot(q_ip, q_oop)        chi = degrees(atan2(q_ip, q_oop))

χ is 0 along +q_oop (the surface normal) and ±90 along ±q_ip.

This is NOT the ``q_chi`` integration and must never be stored as one.  It
re-bins an already binned map, so it is approximate:

* every source bin is spread over the target bins it overlaps (area weights from
  a 4x4 sub-sampling), which smooths sharp features by about one source bin;
* next to detector gaps and the edge of coverage a target bin mixes measured and
  missing source bins.  A target bin is kept only when at least
  ``min_valid_fraction`` of its weight is measured, so coverage is not invented,
  but intensity within about two bins of a gap is unreliable;
* at low q a polar bin is smaller than a Cartesian one.  Those bins are filled
  from the single source bin under their centre, i.e. interpolated rather than
  re-binned;
* it carries no uncertainty.  Use the direct ``q_chi`` mode (or the combined
  ``Q-χ + Qip-Qoop`` choice) for quantitative χ profiles and texture.

On one real Rayonix GI scan, 500x500 bins, the median deviation from the direct
integration was about 0.5 %, the 95th percentile about 3 %, and about 10 % (95th
percentile) below q = 0.5 Å⁻¹.  Those are observations, not tolerances.

Arrays follow the saved-cake / ``FrameView`` convention: ``intensity[y, x]``
with x = q_ip (in) or q (out) and y = q_oop (in) or χ (out).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading

import numpy as np

__all__ = ["DerivedQChi", "derive_q_chi", "derived_q_chi_cache_bytes"]

_SUBSAMPLES = 4
#: Sub-samples handled per construction block (bounds peak build memory).
_BLOCK_SUBSAMPLES = 1 << 21
#: The mapping depends only on the two grids, so it is reused across frames.
#: Bounded both ways: a scan has one grid, a second entry covers a comparison.
_CACHE_ENTRIES = 2
_CACHE_BYTES = 128 * 1024 * 1024

_cache: "OrderedDict[tuple[bytes, ...], tuple[object, np.ndarray, int]]" = OrderedDict()
_cache_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class DerivedQChi:
    """A derived (q, χ) map: ``intensity[chi, q]``, NaN where nothing measured."""

    intensity: np.ndarray
    q: np.ndarray
    chi: np.ndarray


def _centres(values, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all():
        raise ValueError(f"{name} must be a finite 1-D axis of at least two bins")
    if not (np.diff(axis) > 0).all():
        raise ValueError(f"{name} must be strictly increasing")
    return axis


def _edges(centres: np.ndarray) -> np.ndarray:
    middle = (centres[:-1] + centres[1:]) / 2
    return np.concatenate((
        [centres[0] - (centres[1] - centres[0]) / 2],
        middle,
        [centres[-1] + (centres[-1] - centres[-2]) / 2],
    ))


def _linear_centres(low: float, high: float, count: int) -> np.ndarray:
    step = (high - low) / count
    return low + step * (np.arange(count) + 0.5)


def _target_axes(qip_e: np.ndarray, qoop_e: np.ndarray, n_q: int, n_chi: int):
    """The (q, χ) extent the source rectangle actually covers — and no more."""
    x_lo, x_hi, y_lo, y_hi = qip_e[0], qip_e[-1], qoop_e[0], qoop_e[-1]
    nearest_x = min(max(0.0, x_lo), x_hi)
    nearest_y = min(max(0.0, y_lo), y_hi)
    q_lo = float(np.hypot(nearest_x, nearest_y))
    q_hi = float(max(np.hypot(x, y) for x in (x_lo, x_hi) for y in (y_lo, y_hi)))
    straddles_cut = x_lo < 0.0 < x_hi and y_lo < 0.0
    if straddles_cut:
        # Covers the origin, or wraps across χ = ±180: only the full circle is
        # a single increasing axis.
        chi_lo, chi_hi = -180.0, 180.0
    else:
        corners = [np.degrees(np.arctan2(x, y))
                   for x in (x_lo, x_hi) for y in (y_lo, y_hi)]
        chi_lo, chi_hi = float(min(corners)), float(max(corners))
    if not (q_hi > q_lo and chi_hi > chi_lo):
        raise ValueError("the source map covers no (q, chi) area")
    return _linear_centres(q_lo, q_hi, n_q), _linear_centres(chi_lo, chi_hi, n_chi)


def _build(qip: np.ndarray, qoop: np.ndarray, q: np.ndarray, chi: np.ndarray):
    from scipy import sparse

    qip_e, qoop_e, q_e, chi_e = (_edges(axis) for axis in (qip, qoop, q, chi))
    n_ip, n_oop, n_q, n_chi = qip.size, qoop.size, q.size, chi.size
    shape = (n_chi * n_q, n_oop * n_ip)
    sub = (np.arange(_SUBSAMPLES) + 0.5) / _SUBSAMPLES
    y = (qoop_e[:-1, None] + np.diff(qoop_e)[:, None] * sub).ravel()
    y_bin = np.repeat(np.arange(n_oop), _SUBSAMPLES)
    weight = 1.0 / (_SUBSAMPLES * _SUBSAMPLES)
    columns_per_block = max(1, _BLOCK_SUBSAMPLES // (y.size * _SUBSAMPLES))
    matrix = sparse.csr_matrix(shape, dtype=np.float64)
    for start in range(0, n_ip, columns_per_block):
        stop = min(n_ip, start + columns_per_block)
        x = (qip_e[start:stop, None]
             + np.diff(qip_e)[start:stop, None] * sub).ravel()
        x_bin = np.repeat(np.arange(start, stop), _SUBSAMPLES)
        xx, yy = np.meshgrid(x, y, indexing="xy")             # (sub_y, sub_x)
        qi = np.searchsorted(q_e, np.hypot(xx, yy), side="right") - 1
        ci = np.searchsorted(chi_e, np.degrees(np.arctan2(xx, yy)), side="right") - 1
        inside = (qi >= 0) & (qi < n_q) & (ci >= 0) & (ci < n_chi)
        rows = (ci * n_q + qi)[inside]
        cols = (y_bin[:, None] * n_ip + x_bin[None, :])[inside]
        matrix = matrix + sparse.coo_matrix(
            (np.full(rows.size, weight), (rows, cols)), shape=shape,
        ).tocsr()
    # A target bin smaller than the source bins can receive no sub-sample.  Give
    # it the one source bin under its centre (interpolation, documented above),
    # but only when that centre lies inside the source map.
    empty = np.flatnonzero(np.diff(matrix.indptr) == 0)
    if empty.size:
        ci, qi = np.divmod(empty, n_q)
        angle = np.radians(chi[ci])
        cx, cy = q[qi] * np.sin(angle), q[qi] * np.cos(angle)
        ix = np.searchsorted(qip_e, cx, side="right") - 1
        iy = np.searchsorted(qoop_e, cy, side="right") - 1
        inside = (ix >= 0) & (ix < n_ip) & (iy >= 0) & (iy < n_oop)
        matrix = matrix + sparse.coo_matrix(
            (np.ones(int(inside.sum())),
             (empty[inside], (iy * n_ip + ix)[inside])), shape=shape,
        ).tocsr()
    total = np.asarray(matrix.sum(axis=1)).ravel()
    size = int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
               + total.nbytes)
    return matrix, total, size


def _mapping(qip, qoop, q, chi):
    key = tuple(axis.tobytes() for axis in (qip, qoop, q, chi))
    with _cache_lock:
        entry = _cache.get(key)
        if entry is not None:
            _cache.move_to_end(key)
            return entry
    entry = _build(qip, qoop, q, chi)
    if entry[2] <= _CACHE_BYTES:
        with _cache_lock:
            _cache[key] = entry
            _cache.move_to_end(key)
            while len(_cache) > _CACHE_ENTRIES or (
                sum(item[2] for item in _cache.values()) > _CACHE_BYTES
            ):
                _cache.popitem(last=False)
    return entry


def derived_q_chi_cache_bytes() -> int:
    """Bytes currently retained by the grid-mapping cache (for diagnostics)."""
    with _cache_lock:
        return sum(item[2] for item in _cache.values())


def derive_q_chi(
    intensity,
    qip,
    qoop,
    *,
    valid=None,
    q=None,
    chi=None,
    min_valid_fraction: float = 0.5,
) -> DerivedQChi:
    """Re-bin a q_ip–q_oop cake ``intensity[q_oop, q_ip]`` onto (q, χ).

    *qip* / *qoop* are the bin centres of the source map.  A source bin is used
    when it is finite and, if given, true in *valid*.  *q* / *chi* (bin centres;
    χ in degrees) select the target grid; by default it spans exactly the (q, χ)
    area the source rectangle covers, with the source's bin counts, so a cropped
    source yields a cropped view.  The result is NaN wherever less than
    *min_valid_fraction* of a target bin's weight comes from used source bins.
    """
    qip = _centres(qip, "qip")
    qoop = _centres(qoop, "qoop")
    image = np.asarray(intensity, dtype=float)
    if image.shape != (qoop.size, qip.size):
        raise ValueError(
            f"intensity shape {image.shape} is not (q_oop, q_ip) = "
            f"{(qoop.size, qip.size)}"
        )
    if not 0.0 < float(min_valid_fraction) <= 1.0:
        raise ValueError("min_valid_fraction must be in (0, 1]")
    used = np.isfinite(image)
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != image.shape:
            raise ValueError("valid must have the shape of intensity")
        used &= mask
    if q is None or chi is None:
        auto_q, auto_chi = _target_axes(_edges(qip), _edges(qoop), qip.size, qoop.size)
        q = auto_q if q is None else q
        chi = auto_chi if chi is None else chi
    q = _centres(q, "q")
    chi = _centres(chi, "chi")
    matrix, total, _size = _mapping(qip, qoop, q, chi)
    weight = matrix @ used.ravel().astype(np.float64)
    signal = matrix @ np.where(used, image, 0.0).ravel()
    keep = (total > 0) & (weight >= float(min_valid_fraction) * total)
    out = np.full(weight.shape, np.nan)
    np.divide(signal, weight, out=out, where=keep)
    return DerivedQChi(out.reshape(chi.size, q.size), q, chi)
