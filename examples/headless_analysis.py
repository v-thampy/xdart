"""Headless example for the analysis-agnostic runner contract.

The parallel to ``headless_sanity.py`` (reduction spine) and
``headless_scan_session.py`` (live-scan boundary), for the **analysis** layer.
It runs the ``Analyzer`` contract end-to-end with **no Qt / GUI imports**, so
the same calls work from a notebook, a batch job, or an autonomous loop — and
are exactly what xdart's live + batch peak-fit runners drive internally.

What it exercises (``xrd_tools.analysis.runner``):
  * wrap a ``PeakFitPlan`` in a ``PeakFitAnalyzer`` (a frame-unit ``Analyzer``);
  * single frame: ``analyzer.analyze(AnalysisInput(label, x, y))`` -> a uniform
    ``AnalysisOutcome`` with a flat ``params`` dict (``center_i`` / ``fwhm_i`` /
    ``amplitude_i``) and an ``overlay`` (fit / background / residual traces +
    peak-center markers);
  * a whole scan: ``run_batch(analyzer, inputs)`` -> one outcome per frame, then
    ``batch_params_table(outcomes)`` -> the aligned vs-frame parameter series
    (the table the GUI plots and a notebook would drop into a DataFrame).

The ``unit`` attribute ('frame' vs 'scan') is the single switch the runner
branches on, so a future ``PhaseFitAnalyzer`` / ``Sin2PsiAnalyzer`` plugs into
the SAME ``run_batch`` with no runner change.

Needs the ``[fitting]`` extra (lmfit).  Run it where importing ``xdart`` / Qt
would fail — it must still pass::

    python examples/headless_analysis.py
"""

from __future__ import annotations

import sys

import numpy as np


def _synthetic_frame(centers, amps, *, sigma=0.06, slope=400.0, x=None):
    """A 1-D pattern: Gaussian peaks on a sloped background (+ a NaN gap, to
    prove the analyzer drops non-finite samples like a real detector mask)."""
    if x is None:
        x = np.linspace(1.0, 5.0, 600)
    y = 2000.0 + slope * x
    for c, a in zip(centers, amps):
        y = y + a * np.exp(-0.5 * ((x - c) / sigma) ** 2)
    y[100:105] = np.nan  # masked detector gap
    return x, y


def main() -> None:
    from xrd_tools.analysis.plans import PeakFitPlan
    from xrd_tools.analysis.runner import (
        AnalysisInput,
        PeakFitAnalyzer,
        batch_params_table,
        run_batch,
    )

    # --- one analyzer, reused for single-frame and batch --------------------
    analyzer = PeakFitAnalyzer(PeakFitPlan(model="gaussian", n_peaks=2))
    assert analyzer.kind == "peak_fit" and analyzer.unit == "frame"

    # --- single frame -------------------------------------------------------
    x, y = _synthetic_frame(centers=(2.0, 3.5), amps=(1.0e5, 6.0e4))
    out = analyzer.analyze(AnalysisInput(label="0", x=x, y=y, x_unit="q (Å⁻¹)"))
    assert out.ok, out.message
    centers = sorted(out.params[f"center_{i}"] for i in range(2))
    assert abs(centers[0] - 2.0) < 0.05 and abs(centers[1] - 3.5) < 0.05
    assert {"fit", "residual"} <= set(out.overlay.traces)
    assert len(out.overlay.markers) == 2
    print(f"single frame: centers={centers[0]:.3f}, {centers[1]:.3f}; "
          f"overlay traces={sorted(out.overlay.traces)}")

    # --- a whole scan: the first peak drifts frame-to-frame -----------------
    inputs = []
    for i in range(8):
        fx, fy = _synthetic_frame(centers=(2.0 + 0.01 * i, 3.5),
                                  amps=(1.0e5, 6.0e4))
        inputs.append(AnalysisInput(label=str(i), x=fx, y=fy, x_unit="q (Å⁻¹)"))

    outcomes = run_batch(analyzer, inputs)
    assert all(o.ok for o in outcomes)
    labels, columns = batch_params_table(outcomes)
    drift = columns["center_0"]
    assert labels == [str(i) for i in range(8)]
    assert drift[-1] - drift[0] > 0.05            # the drift is recovered
    print(f"batch: {len(labels)} frames; center_0 {drift[0]:.3f} -> "
          f"{drift[-1]:.3f} (vs-frame series ready for a DataFrame/plot)")

    # --- a SCAN-UNIT analyzer: the SAME contract, one outcome for a set -----
    # sin²ψ aggregates χ-sectors within one (q, χ) polar map -> unit='scan'.
    _scan_unit_demo()

    # --- the whole point: no xdart GUI on the import graph ------------------
    gui_roots = {"xdart", "pyqtgraph"}
    leaked = sorted({m.split(".")[0] for m in sys.modules} & gui_roots)
    assert not leaked, f"headless example pulled in the xdart GUI stack: {leaked}"
    print("OK: analysis runner exercised single-frame + batch + scan-unit with "
          "no xdart/pyqtgraph GUI stack.")


def _scan_unit_demo() -> None:
    """sin²ψ strain — a scan-unit analyzer driven through the SAME contract.

    The unit is the whole (q, χ) polar map (not per-frame); it rides on
    ``AnalysisInput.result_1d`` and the runner branches on ``unit='scan'`` to
    call ``analyze_scan`` once.  Here a synthetic map whose Bragg peak walks with
    sin²ψ gives a clean d-vs-sin²ψ regression."""
    from xrd_tools.analysis.plans import Sin2PsiPlan
    from xrd_tools.analysis.runner import AnalysisInput, Sin2PsiAnalyzer
    from xrd_tools.core.containers import IntegrationResult2D

    q = np.linspace(2.0, 4.0, 400)
    chi = np.linspace(-40.0, 40.0, 33)
    intensity = np.zeros((q.size, chi.size))
    for j, c in enumerate(chi):
        s2 = np.sin(np.deg2rad(abs(c))) ** 2
        intensity[:, j] = 1.0e4 * np.exp(
            -0.5 * ((q - (3.0 + 0.02 * s2)) / 0.03) ** 2) + 100.0
    polar = IntegrationResult2D(radial=q, azimuthal=chi, intensity=intensity,
                                unit="q_A^-1", azimuthal_unit="chi_deg")

    analyzer = Sin2PsiAnalyzer(Sin2PsiPlan(q_range=(2.7, 3.3), chi_width=5.0))
    assert analyzer.unit == "scan"
    out = analyzer.analyze_scan(
        [AnalysisInput(label="sin2psi", x=q, y=intensity[:, 0], result_1d=polar)])
    assert out.ok, out.message
    print(f"scan-unit (sin2psi): d0={out.params['d0']:.4f} Å, "
          f"slope={out.params['slope']:+.4f}, R²={out.params['r_squared']:.3f} "
          f"over {int(out.params['n_sectors'])} χ-sectors")


if __name__ == "__main__":
    main()
