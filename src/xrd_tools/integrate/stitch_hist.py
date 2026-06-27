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


def _materialize_and_validate(images, integrators, normalization, *, who):
    """Materialize the image/integrator iterables and fail loud on a corrupting
    input — shared by the plain and GI q-providers.

    Catches the silent-corruption class: a frame/integrator length desync (``zip``
    would truncate, dropping whole frames) and a bad per-frame monitor (zero/NaN
    drops a frame; **negative** flips its sign and cancels healthy frames).
    """
    images = list(images)
    integrators = list(integrators)
    if len(images) != len(integrators):
        raise ValueError(
            f"{who}: {len(images)} images but {len(integrators)} integrators — "
            "every frame needs exactly one integrator")
    mon = None
    if normalization is not None:
        mon = np.asarray(list(normalization), dtype=float)
        if len(mon) != len(images):
            raise ValueError(
                f"{who}: {len(mon)} monitor/normalization values for "
                f"{len(images)} images")
        bad = ~np.isfinite(mon) | (mon <= 0)
        if bad.any():
            raise ValueError(
                "stitch monitor/normalization has invalid value(s) (non-finite "
                f"or <= 0) at frame index/indices {np.flatnonzero(bad).tolist()}: "
                f"{mon[bad].tolist()}")
    return images, integrators, mon


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
    images, integrators, mon = _materialize_and_validate(
        images, integrators, normalization, who="pyfai_q_frames")
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


def _as_fiber(ai: Any) -> Any:
    """Promote a pyFAI integrator to a ``FiberIntegrator`` with the same geometry.

    ``FiberIntegrator`` *is* an ``AzimuthalIntegrator`` (same dist/poni/rot/detector/
    wavelength) — we only need its GI unit machinery for the per-pixel αf and q_oop
    maps, so a plain integrator is rebuilt as a fiber one without changing geometry.
    """
    from pyFAI.integrator.fiber import FiberIntegrator  # noqa: PLC0415
    if isinstance(ai, FiberIntegrator):
        return ai
    return FiberIntegrator(
        dist=ai.dist, poni1=ai.poni1, poni2=ai.poni2,
        rot1=ai.rot1, rot2=ai.rot2, rot3=ai.rot3,
        detector=ai.detector, wavelength=ai.wavelength)


def pyfai_gi_q_frames(
    images: Iterable[np.ndarray],
    integrators: Iterable[Any],
    *,
    gi: Any,
    incident_angles_deg: Iterable[float],
    sample_orientation: int = 1,
    tilt_deg: float = 0.0,
    corrections: Any = None,
    mask: np.ndarray | None = None,
    normalization: Iterable[float] | None = None,
) -> Iterator[Any]:
    """``pyfai_hist`` **grazing-incidence** q-provider — like :func:`pyfai_q_frames`
    but applies a :class:`~xrd_tools.corrections.grazing.GICorrectionStack`.

    Per pixel: the GI intensity factors (footprint·Fresnel·absorption) multiply into
    the ``Σnorm`` weight via ``gi.gi_normalization``; if ``gi.refraction`` the q-map
    is rewritten by ``gi.refract_q`` (a position correction).  The per-pixel exit
    angle αf and out-of-plane q_z come from **pyFAI's own fiber geometry**
    (``FiberIntegrator`` + the ``exit_angle_vert``/``qoop`` units, after
    ``reset_integrator(incident_angle=…)``) — the SAME convention as the reduction
    GI path, so q_oop ≡ k0·(sin αf + sin αi).  Per-frame αi (degrees) is
    ``incident_angles_deg`` (one per frame, from
    ``Diffractometer.to_pyfai_per_frame(...)['incident_angle']``).

    ⚠ The GI sample geometry (``sample_orientation``/``tilt_deg``) + the P2b
    composition signs are **pending real-data (GIXSGUI) validation** — the per-pixel
    αf/q_z maps are pyFAI's (gate-checked), but the absolute correction direction is
    not yet confirmed against a worked GI example.
    """
    import pyFAI.units as U  # noqa: PLC0415

    if gi is None:
        raise ValueError("pyfai_gi_q_frames requires a GICorrectionStack (gi=)")
    images, integrators, mon = _materialize_and_validate(
        images, integrators, normalization, who="pyfai_gi_q_frames")
    inc = np.asarray(list(incident_angles_deg), dtype=float)
    if inc.shape != (len(images),):
        raise ValueError(
            f"pyfai_gi_q_frames: {inc.shape} incident angle(s) for "
            f"{len(images)} images — need exactly one αi per frame")
    if not np.all(np.isfinite(inc)):
        raise ValueError(
            "pyfai_gi_q_frames: non-finite incident angle(s) at "
            f"{np.flatnonzero(~np.isfinite(inc)).tolist()}")

    tilt_rad = float(np.radians(tilt_deg))
    so = int(sample_orientation)
    for i, (ai, img) in enumerate(zip(integrators, images)):
        shape = np.asarray(img).shape
        air = float(np.radians(inc[i]))
        fi = _as_fiber(ai)
        # populate the fiber geometry cache so the unit maps recompute for this αi
        fi.reset_integrator(incident_angle=air, tilt_angle=tilt_rad,
                            sample_orientation=so)
        af_u = U.get_unit_fiber("exit_angle_vert_rad", incident_angle=air,
                                tilt_angle=tilt_rad, sample_orientation=so)
        qoop_u = U.get_unit_fiber("qoop_A^-1", incident_angle=air,
                                  tilt_angle=tilt_rad, sample_orientation=so)
        af = np.asarray(fi.array_from_unit(shape, "center", af_u), dtype=float)
        q_oop = np.asarray(fi.array_from_unit(shape, "center", qoop_u), dtype=float)
        q = np.asarray(fi.qArray(shape=shape), dtype=float) / 10.0
        chi = np.degrees(np.asarray(fi.chiArray(shape=shape), dtype=float))
        # base normalization (solid angle/polarization) × the GI intensity weight
        if corrections is not None:
            w = np.asarray(corrections.normalization(fi, shape), dtype=float)
        else:
            w = np.ones(shape, dtype=float)
        w = w * gi.gi_normalization(incident_angle_deg=float(inc[i]), alpha_f_rad=af)
        if mask is not None:
            w = np.where(np.asarray(mask, dtype=bool), 0.0, w)
        if getattr(gi, "refraction", False):
            q = gi.refract_q(incident_angle_deg=float(inc[i]), alpha_f_rad=af,
                             q_total=q, q_z=q_oop)
        signal = np.asarray(img, dtype=float)
        if mon is not None:
            signal = signal / mon[i]
        yield q, chi, signal, w


__all__ = ["pyfai_gi_q_frames", "pyfai_q_frames", "stitch_q_grid"]
