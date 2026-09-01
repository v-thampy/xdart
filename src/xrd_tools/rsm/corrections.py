"""Per-pixel intensity corrections for the RSM grid (P6).

RSM bins onto a 3D ``(qx, qy, qz)`` grid via the shared ``Σraw/Σnorm``
accumulator (:mod:`xrd_tools.rsm.gridding`).  The per-pixel ``norm`` is the
SAME :class:`~xrd_tools.corrections.CorrectionStack` normalization the stitch
histogram merge uses — so RSM, stitching, and pyFAI all share one intensity
convention (a multiplicative correction ``raw = true·C`` is applied by ``norm = C``).

``CorrectionStack.normalization`` is expressed against a pyFAI
``AzimuthalIntegrator``; RSM's geometry is xrayutilities (a
:class:`~xrd_tools.core.geometry.DetectorHeader`).  This module bridges the two:
:func:`detector_header_to_ai` builds the matching fixed-lab pyFAI integrator, and
:func:`rsm_correction_weight` returns the per-pixel weight.

The angular corrections (solid angle, polarization, 2θ) are **wavelength-
independent** — pure detector geometry — so the bridge uses a placeholder
wavelength and the weight is computed ONCE per detector geometry (it does not
vary with energy).  It is **geometry-static**: the RSM detector is treated as a
fixed lab detector (``rot = 0``); per-frame detector-arm rotation and the GI
weights (footprint/Fresnel/refraction) are deferred to a later step.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["detector_header_to_ai", "gi_grid_weight", "rsm_correction_weight"]


def detector_header_to_ai(header: Any, *, wavelength_m: float = 1.0e-10):
    """A fixed lab-frame pyFAI ``AzimuthalIntegrator`` matching a DetectorHeader.

    The xrayutilities header is in **mm** (``pwidth*``/``distance``) with the
    beam centre in **pixels** (``cch*``); pyFAI wants metres and a PONI in
    metres: ``poni_i = cch_i · pwidth_i`` (m), ``dist = distance`` (m), pixel
    size ``pwidth_i`` (m), ``rot = 0``.  ``wavelength`` only affects q (not the
    angular corrections), so its value here is a harmless placeholder.
    """
    from pyFAI.detectors import Detector  # noqa: PLC0415
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # noqa: PLC0415

    p1 = float(header.pwidth1) * 1.0e-3
    p2 = float(header.pwidth2) * 1.0e-3
    det = Detector(pixel1=p1, pixel2=p2,
                   max_shape=(int(header.Nch1), int(header.Nch2)))
    return AzimuthalIntegrator(
        dist=float(header.distance) * 1.0e-3,
        poni1=float(header.cch1) * p1,
        poni2=float(header.cch2) * p2,
        rot1=0.0, rot2=0.0, rot3=0.0,
        detector=det, wavelength=wavelength_m)


def gi_grid_weight(
    header: Any,
    gi: Any,
    *,
    incident_angle_deg: float,
    wavelength_m: float,
    sample_orientation: int = 1,
    tilt_deg: float = 0.0,
    roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """The per-pixel grazing-incidence INTENSITY weight (footprint·Fresnel·
    absorption) for the RSM grid, from a
    :class:`~xrd_tools.corrections.grazing.GICorrectionStack`.

    The per-pixel exit angle is reconstructed from pyFAI's q-oop map and the
    supplied mapping wavelength.  The pyFAI exit-angle map is never an input to
    xdart's Fresnel/absorption corrections.

    ⚠ **Refraction is NOT applied here** (it is a position correction that would
    rewrite qz in 3-D — deferred, real-data-gated), and the absolute composition
    signs are the same P2b flag (verify vs GIXSGUI).  Only the intensity weight
    is built; αi is the (fixed) ``incident_angle_deg``.
    """
    import pyFAI.units as U  # noqa: PLC0415
    from pyFAI.detectors import Detector  # noqa: PLC0415
    from pyFAI.integrator.fiber import FiberIntegrator  # noqa: PLC0415
    from xrd_tools.corrections.grazing import (  # noqa: PLC0415
        physical_exit_angle_from_qoop,
    )

    wavelength = float(wavelength_m)
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("GI RSM mapping wavelength must be positive and finite")
    h = header.with_roi(roi) if roi is not None else header
    shape = (int(h.Nch1), int(h.Nch2))
    p1 = float(h.pwidth1) * 1.0e-3
    p2 = float(h.pwidth2) * 1.0e-3
    fi = FiberIntegrator(
        dist=float(h.distance) * 1.0e-3, poni1=float(h.cch1) * p1,
        poni2=float(h.cch2) * p2, rot1=0.0, rot2=0.0, rot3=0.0,
        detector=Detector(pixel1=p1, pixel2=p2, max_shape=shape),
        wavelength=wavelength)
    air = float(np.radians(incident_angle_deg))
    tilt = float(np.radians(tilt_deg))
    fi.reset_integrator(incident_angle=air, tilt_angle=tilt,
                        sample_orientation=int(sample_orientation))
    qoop_u = U.get_unit_fiber(
        "qoop_A^-1",
        incident_angle=air,
        tilt_angle=tilt,
        sample_orientation=int(sample_orientation),
    )
    qoop = np.asarray(fi.array_from_unit(shape, "center", qoop_u), dtype=float)
    af = physical_exit_angle_from_qoop(
        qoop,
        incident_angle_rad=air,
        wavelength_m=fi.wavelength,
    )
    return np.asarray(
        gi.gi_normalization(incident_angle_deg=float(incident_angle_deg),
                            alpha_f_rad=af), dtype=float)


def rsm_correction_weight(
    header: Any,
    corrections: Any,
    *,
    gi: Any = None,
    wavelength_m: float | None = None,
    roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray | None:
    """Per-pixel ``Σnorm`` weight for the RSM grid, or ``None``.

    ``corrections`` is a :class:`~xrd_tools.corrections.CorrectionStack` (solid
    angle / polarization / air absorption).  The weight is computed on the
    ROI-cropped detector so it lines up with the chunk images
    :meth:`StreamingGridder.add` feeds (which are ROI-cropped the same way).

    When ``gi`` (a :class:`~xrd_tools.corrections.GISettings`) carrying a
    ``GICorrectionStack`` is given, the GI intensity weight
    (:func:`gi_grid_weight`) is multiplied in — ``gi.incident_angle_deg`` is the
    fixed incidence αi (required; per-frame-varying αi + refraction are the
    real-data-gated tail).  ``corrections is None`` and ``gi`` empty → ``None``
    (unit weight, the count-mean).
    """
    base = None
    if corrections is not None:
        h = header.with_roi(roi) if roi is not None else header
        ai = detector_header_to_ai(h)
        base = np.asarray(
            corrections.normalization(ai, (int(h.Nch1), int(h.Nch2))), dtype=float)
    if gi is not None and gi.corrections is not None:
        from xrd_tools.corrections.grazing import (  # noqa: PLC0415
            GI_EXIT_ANGLE_CONVENTION,
            validate_gi_exit_angle_convention,
        )
        convention = validate_gi_exit_angle_convention(
            gi.gi_exit_angle_convention
        )
        if convention != GI_EXIT_ANGLE_CONVENTION:
            raise ValueError(
                "legacy GI exit-angle science cannot be rerun under the current "
                "pyFAI runtime; rerun raw science with xdart_reflection_qoop_v1"
            )
        if gi.incident_angle_deg is None:
            raise ValueError(
                "GI RSM (RSMPlan.gi) requires GISettings.incident_angle_deg (a "
                "fixed incidence angle); per-frame-varying αi is the real-data-"
                "gated tail.")
        # Fail loud rather than silently diverge from GI stitch: the RSM grid
        # applies only the GI *intensity* weight (footprint/Fresnel/absorption);
        # it does NOT yet rewrite the q-coordinates for refraction the way the
        # stitch GI provider (pyfai_gi_q_frames → refract_q) does.  Accepting a
        # refraction=True stack here would produce a q-space that disagrees with
        # the GI stitch while looking like the same correction object.  RSM
        # q-refraction is gated on real-data GI validation (don't guess geometry).
        if getattr(gi.corrections, "refraction", False):
            raise NotImplementedError(
                "GI refraction (q-coordinate rewriting) is not implemented in the "
                "RSM path — it would silently diverge from GI stitch.  Pass a "
                "GICorrectionStack(refraction=False) for RSM until RSM q-refraction "
                "lands (real-data-gated); the intensity weights "
                "(footprint/Fresnel/absorption) still apply.")
        recorded_wavelength = gi.integrator_wavelength_m
        if wavelength_m is None:
            wavelength_m = recorded_wavelength
        if wavelength_m is None:
            raise ValueError(
                "GI RSM corrections require the exact mapping wavelength"
            )
        if recorded_wavelength is not None and not np.isclose(
            float(wavelength_m), float(recorded_wavelength),
            rtol=1.0e-12, atol=0.0,
        ):
            raise ValueError(
                "GI RSM mapping wavelength differs from persisted GI intent"
            )
        giw = gi_grid_weight(
            header, gi.corrections, incident_angle_deg=gi.incident_angle_deg,
            wavelength_m=float(wavelength_m),
            sample_orientation=gi.sample_orientation, tilt_deg=gi.tilt_deg, roi=roi)
        base = giw if base is None else base * giw
    return base
