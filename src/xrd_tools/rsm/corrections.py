"""Per-pixel intensity corrections for the RSM grid (P6).

RSM bins onto a 3D ``(qx, qy, qz)`` grid via the shared ``Σ(raw·w)/Σ(w)``
accumulator (:mod:`xrd_tools.rsm.gridding`).  The per-pixel weight ``w`` is the
SAME :class:`~xrd_tools.corrections.CorrectionStack` normalization the stitch
histogram merge uses — so RSM, stitching, and pyFAI all share one intensity
convention.

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

__all__ = ["detector_header_to_ai", "rsm_correction_weight"]


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


def rsm_correction_weight(
    header: Any,
    corrections: Any,
    *,
    roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray | None:
    """Per-pixel ``Σnorm`` weight for the RSM grid, or ``None``.

    ``corrections`` is a :class:`~xrd_tools.corrections.CorrectionStack` (solid
    angle / polarization / air absorption).  The weight is computed on the
    ROI-cropped detector so it lines up with the chunk images
    :meth:`StreamingGridder.add` feeds (which are ROI-cropped the same way).
    ``corrections is None`` → ``None`` (unit weight, the count-mean).
    """
    if corrections is None:
        return None
    h = header.with_roi(roi) if roi is not None else header
    ai = detector_header_to_ai(h)
    return np.asarray(
        corrections.normalization(ai, (int(h.Nch1), int(h.Nch2))), dtype=float)
