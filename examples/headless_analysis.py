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

    # --- the whole point: no xdart GUI on the import graph ------------------
    gui_roots = {"xdart", "pyqtgraph"}
    leaked = sorted({m.split(".")[0] for m in sys.modules} & gui_roots)
    assert not leaked, f"headless example pulled in the xdart GUI stack: {leaked}"
    print("OK: analysis runner exercised single-frame + batch with no "
          "xdart/pyqtgraph GUI stack.")


if __name__ == "__main__":
    main()
