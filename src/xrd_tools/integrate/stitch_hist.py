"""Histogram stitch backend — the shared per-pixel merge for ``pyfai_hist`` and
``xu_hist`` (design_stitching_jun2026.md §2.6).

A stitched pattern is ``I(q[,χ]) = Σ raw / Σ normalization`` over the pixels of
ALL frames falling in a (q[,χ]) bin — the SAME accumulator scheme as the per-frame
:class:`~xrd_tools.corrections.CorrectionStack` (P2a) and as RSM's gridder, only
the bin space differs.  Unlike pyFAI ``MultiGeometry`` this **streams** (one frame's
q-map + image resident at a time) and is engine-agnostic: it consumes a per-frame
**q-provider** yielding ``(|q|, χ, signal, normalization)``, so ``pyfai_hist`` (pyFAI
q/χ maps) and ``xu_hist`` (``Diffractometer.to_qconversion`` → ``Ang2Q.area``) share
one merge and differ only in the provider.

Intensity convention: the normalization is the per-pixel correction weight
(solid-angle · polarization · … from :class:`CorrectionStack`), i.e. the same
normalized solid-angle pyFAI ``integrate1d`` uses — so a single frame reproduces
``ai.integrate1d`` exactly.  (pyFAI ``MultiGeometry`` normalizes by the *absolute*
solid angle, so it agrees in shape up to a global scale.)
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Iterator

import numpy as np

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D

logger = logging.getLogger(__name__)

#: a per-frame q-provider yields ``(q_A, chi_deg, signal, normalization)`` arrays.
QFrame = "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]"


def _finite_range(provider_frames: list, idx: int) -> tuple[float, float]:
    lo, hi = np.inf, -np.inf
    for f in provider_frames:
        a = np.asarray(f[idx], dtype=float)
        m = np.isfinite(a)
        if m.any():
            lo = min(lo, float(a[m].min()))
            hi = max(hi, float(a[m].max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("could not determine a finite q/chi range from the frames")
    return lo, hi


def stitch_q_grid(
    frames: Iterable[Any],
    *,
    mode: str = "1d",
    npt: int = 2000,
    npt_azim: int = 720,
    unit: str = "q_A^-1",
    radial_range: tuple[float, float] | None = None,
    azimuth_range: tuple[float, float] | None = None,
) -> IntegrationResult1D | IntegrationResult2D:
    """Merge a per-frame q-provider into a stitched 1D or 2D pattern.

    Parameters
    ----------
    frames : iterable of ``(q_A, chi_deg, signal, normalization)``
        One tuple per frame (e.g. from :func:`pyfai_q_frames`).  ``signal`` is the
        raw image; ``normalization`` the per-pixel correction weight (pixels with
        ``normalization <= 0`` or non-finite q are dropped — that is the mask).
    mode : {"1d", "2d"}
        1D → ``I(q)``; 2D → ``I(q, χ)``.
    npt, npt_azim : int
        Radial / azimuthal bin counts.
    radial_range, azimuth_range : (lo, hi), optional
        Explicit ranges; ``None`` scouts them from the frames.

    Returns
    -------
    IntegrationResult1D | IntegrationResult2D
    """
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("stitch_q_grid: no frames")

    qlo, qhi = radial_range if radial_range is not None else _finite_range(frame_list, 0)
    q_edges = np.linspace(qlo, qhi, npt + 1)
    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])

    if mode == "1d":
        sig = np.zeros(npt, dtype=float)
        nrm = np.zeros(npt, dtype=float)
        for q, _chi, signal, norm in frame_list:
            q = np.asarray(q, dtype=float).ravel()
            s = np.asarray(signal, dtype=float).ravel()
            w = np.asarray(norm, dtype=float).ravel()
            good = np.isfinite(q) & np.isfinite(s) & np.isfinite(w) & (w > 0)
            sig += np.histogram(q[good], q_edges, weights=s[good])[0]   # Σ raw
            nrm += np.histogram(q[good], q_edges, weights=w[good])[0]   # Σ norm
        with np.errstate(divide="ignore", invalid="ignore"):
            intensity = sig / nrm                                       # Σraw / Σnorm
        intensity[nrm <= 0] = np.nan
        return IntegrationResult1D(radial=q_centers, intensity=intensity, unit=unit)

    if mode != "2d":
        raise ValueError(f"mode must be '1d' or '2d', got {mode!r}")

    clo, chi_hi = (azimuth_range if azimuth_range is not None
                   else _finite_range(frame_list, 1))
    chi_edges = np.linspace(clo, chi_hi, npt_azim + 1)
    chi_centers = 0.5 * (chi_edges[:-1] + chi_edges[1:])
    sig2 = np.zeros((npt, npt_azim), dtype=float)
    nrm2 = np.zeros((npt, npt_azim), dtype=float)
    for q, chi, signal, norm in frame_list:
        q = np.asarray(q, dtype=float).ravel()
        chi = np.asarray(chi, dtype=float).ravel()
        s = np.asarray(signal, dtype=float).ravel()
        w = np.asarray(norm, dtype=float).ravel()
        good = (np.isfinite(q) & np.isfinite(chi) & np.isfinite(s)
                & np.isfinite(w) & (w > 0))
        sig2 += np.histogram2d(q[good], chi[good], [q_edges, chi_edges],
                               weights=s[good])[0]
        nrm2 += np.histogram2d(q[good], chi[good], [q_edges, chi_edges],
                               weights=w[good])[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        intensity2 = sig2 / nrm2
    intensity2[nrm2 <= 0] = np.nan
    return IntegrationResult2D(
        radial=q_centers, azimuthal=chi_centers, intensity=intensity2,
        unit=unit, azimuthal_unit="chi_deg")


def pyfai_q_frames(
    images: Iterable[np.ndarray],
    integrators: Iterable[Any],
    *,
    corrections: Any = None,
    mask: np.ndarray | None = None,
    normalization: Iterable[float] | None = None,
) -> Iterator[Any]:
    """``pyfai_hist`` q-provider: per-frame ``(|q|_Å, χ_deg, signal, weight)`` from
    pyFAI ``AzimuthalIntegrator``s.

    ``|q|`` from ``ai.qArray()`` (nm⁻¹ → Å⁻¹), ``χ`` from ``ai.chiArray()``
    (deg); the per-pixel weight is ``corrections.normalization(ai)`` (an all-ones
    array when ``corrections`` is None).  A detector ``mask`` zeroes excluded
    pixels' weight; a per-frame monitor ``normalization`` scalar divides the
    signal (matching the MultiGeometry path).
    """
    images = list(images)
    integrators = list(integrators)
    mon = (np.asarray(list(normalization), dtype=float)
           if normalization is not None else None)
    for i, (ai, img) in enumerate(zip(integrators, images)):
        shape = np.asarray(img).shape
        q = np.asarray(ai.qArray(shape=shape), dtype=float) / 10.0
        chi = np.degrees(np.asarray(ai.chiArray(shape=shape), dtype=float))
        if corrections is not None:
            w = np.asarray(corrections.normalization(ai, shape), dtype=float)
        else:
            w = np.ones(shape, dtype=float)
        if mask is not None:
            w = np.where(np.asarray(mask, dtype=bool), 0.0, w)
        signal = np.asarray(img, dtype=float)
        if mon is not None:
            signal = signal / mon[i]
        yield q, chi, signal, w


__all__ = ["pyfai_q_frames", "stitch_q_grid"]
