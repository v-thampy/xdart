"""Pins the stitch ↔ RSM merge-accumulator equivalence (Jun 2026 reconciliation).

Both reduction pipelines that share ``xrd_tools.corrections`` weights now merge
with the SAME accumulator, ``I = Σraw/Σnorm`` (stitch: ``stitch_hist.stitch_q_grid``;
RSM: ``rsm.gridding``).  This is the only accumulator that correctly *applies* a
multiplicative correction: with ``raw = true·C`` and ``norm = C`` it recovers
``true``.  (Until Jun-2026 RSM accumulated ``Σ(raw·w)/Σw``, which only *weights* —
it returned ``Σ(true·C²)/ΣC ≠ true`` and could not apply a multiplicative
correction.  See ``docs/design/design_stitch_rsm_accumulator_jun2026.md``.)

Both tests below assert the recovery of ``true``; a regression in either pipeline's
accumulator (e.g. a slide back to ``Σ(raw·w)/Σw``) re-fails the RSM gate.
"""
from __future__ import annotations

import numpy as np
import pytest


# raw = true · C, a per-pixel multiplicative boost (e.g. solid angle / footprint).
# The merge must recover `true`; with norm = C only Σraw/Σnorm does.
_TRUE = 100.0
_C = np.array([1.0, 2.0, 4.0, 8.0, 1.0, 2.0, 4.0, 8.0])
_RAW = _TRUE * _C


def test_stitch_accumulator_recovers_true_from_multiplicative_correction():
    """Σraw/Σnorm (stitch) recovers `true` from raw=true·C with norm=C."""
    pytest.importorskip("pyFAI")
    from xrd_tools.integrate.stitch_hist import stitch_q_grid
    q = np.linspace(1.1, 1.9, _C.size)
    chi = np.zeros_like(q)
    frame = (q, chi, _RAW, _C)            # (q, chi, signal=raw, norm=C)
    res = stitch_q_grid([frame], mode="1d", npt=1, radial_range=(1.0, 2.0))
    # Σraw/Σnorm = Σ(true·C)/ΣC = true
    assert float(res.intensity[0]) == pytest.approx(_TRUE, rel=1e-6)


def test_rsm_accumulator_recovers_true_from_multiplicative_correction():
    """Σraw/Σnorm (RSM gridder) recovers `true` from raw=true·C, norm=C — the SAME
    way stitch does (unified Jun 2026; was Σ(raw·w)/Σw, which could not)."""
    pytest.importorskip("xrayutilities")
    from xrd_tools.rsm.gridding import _feed_pair, _new_gridder, _pair_intensity
    n = _C.size
    qx = np.full(n, 0.5); qy = np.full(n, 0.5); qz = np.full(n, 0.5)
    bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    gr = _new_gridder((1, 1, 1), bounds)
    gn = _new_gridder((1, 1, 1), bounds)
    _feed_pair(gr, gn, qx, qy, qz, _RAW, _C)
    vol = _pair_intensity(gr, gn)
    got = float(np.asarray(vol).ravel()[0])
    assert got == pytest.approx(_TRUE, rel=1e-6)
