# -*- coding: utf-8 -*-
"""Always-on geometric detector mask (maintainer decision 2026-07-12).

pyFAI applies the detector's own geometric mask (``detector.calc_mask()`` —
module gaps, i.e. NON-pixels with no sensor behind them) only when an
``integrate*`` call receives ``mask=None``; an explicit mask **replaces** it.
xrd-tools/xdart always pass an explicit mask (an empty index array means
"mask nothing"), which silently disabled the geometric mask wherever the
wrangler-built ``global_mask`` was absent — reintegrating an old ``.nxs`` with
no persisted mask, a PONI whose detector name doesn't resolve, or data whose
gap pixels are zero-filled rather than sentinel-valued (smooth 1-D dips at the
gap radii; bl17-2, 2026-07-12).

Every pyFAI call site therefore unions the integrator's OWN detector mask into
the explicit mask via :func:`mask_with_detector`.  This is detector GEOMETRY,
deliberately **not** gated by the GUI "Auto Mask Saturated" toggle — that
toggle governs VALUE-based masking (saturation / the uint32 sentinel) only.
For detectors without a geometric mask (Rayonix, Perkin — ``calc_mask() is
None``) this is a strict no-op, so the live≡batch≡reload equivalence spine is
numerically unchanged.
"""

from __future__ import annotations

import numpy as np

__all__ = ["mask_with_detector"]


def mask_with_detector(ai, mask):
    """Union *mask* with ``ai.detector``'s geometric (calc) mask.

    Parameters
    ----------
    ai : pyFAI AzimuthalIntegrator / FiberIntegrator (anything with
        ``.detector``); ``detector.mask`` is pyFAI's lazily-cached
        ``calc_mask()``, so the per-frame cost is one boolean OR.
    mask : ndarray or None
        The explicit mask about to be handed to pyFAI (full-frame bool/int
        array), or ``None``.

    Returns
    -------
    ndarray or None
        The union when both exist; the geometric mask alone when *mask* is
        ``None`` (equivalent to pyFAI's own fallback, made explicit); *mask*
        untouched when the detector has no geometric mask or the shapes
        disagree (never guess about a non-frame-shaped mask).
    """
    det = getattr(ai, "detector", None)
    if det is None:
        return mask
    try:
        det_mask = det.mask
    except Exception:
        det_mask = None
    if det_mask is None:
        return mask
    if mask is None:
        return det_mask
    m = np.asarray(mask)
    if m.shape != np.asarray(det_mask).shape:
        return mask
    return np.logical_or(m, det_mask)
