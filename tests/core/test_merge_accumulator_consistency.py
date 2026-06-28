"""Pins the stitch ↔ RSM merge-accumulator inconsistency (Jun 2026 finding).

Stitch merges with ``I = Σraw/Σnorm`` (``stitch_hist.stitch_q_grid``), RSM with
``I = Σ(raw·w)/Σw`` (``rsm.gridding``) — two different accumulators that share one
correction weight (``GICorrectionStack.gi_normalization`` / ``CorrectionStack``).
Only ``Σraw/Σnorm`` correctly applies a multiplicative correction; the
``Σ(raw·w)/Σw`` form can only *weight*, not *correct*.  See
``docs/design/design_stitch_rsm_accumulator_jun2026.md``.

The RSM test below is ``xfail(strict)`` — it documents the defect and will start
PASSING (and fail the strict-xfail) once RSM is unified onto ``Σraw/Σnorm``, which
is the cue to drop the marker.
"""
from __future__ import annotations

import numpy as np
import pytest


# raw = true · C, a per-pixel multiplicative boost (e.g. solid angle / footprint).
# The merge must recover `true`; with norm/weight = C only Σraw/Σnorm does.
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


@pytest.mark.xfail(strict=True, reason=(
    "RSM uses Σ(raw·w)/Σw (rsm/gridding.py:86), NOT the stitch Σraw/Σnorm — "
    "rsm/gridding.py:51 wrongly claims they are the same accumulator. A "
    "Σ(raw·w)/Σw merge cannot apply a multiplicative correction via w: it "
    "returns Σ(true·C²)/ΣC ≠ true. Unify RSM onto Σraw/Σnorm (see "
    "docs/design/design_stitch_rsm_accumulator_jun2026.md), then drop this "
    "marker."))
def test_rsm_accumulator_recovers_true_from_multiplicative_correction():
    """The RSM gridder should recover `true` from raw=true·C, weight=C the same
    way stitch does — it currently does not (different accumulator)."""
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
