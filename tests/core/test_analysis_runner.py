"""Step 1: the analysis-agnostic runner contract + the PeakFitAnalyzer.

Headless / Qt-free.  Locks the Analyzer protocol (so the live/batch runners can
drive any analysis) and proves it end-to-end on peak fitting.
"""
import numpy as np
import pytest


def test_peak_fit_analyzer_satisfies_protocol_and_is_frame_unit():
    from xrd_tools.analysis.runner import Analyzer, PeakFitAnalyzer

    a = PeakFitAnalyzer()
    # runtime_checkable Protocol: has kind / unit / analyze / analyze_scan
    assert isinstance(a, Analyzer)
    assert a.kind == "peak_fit"
    assert a.unit == "frame"


def test_peak_fit_analyzer_projects_overlay_and_params():
    pytest.importorskip("lmfit")  # the xrd-tools[fitting] extra
    from xrd_tools.analysis.plans import PeakFitPlan
    from xrd_tools.analysis.runner import AnalysisInput, PeakFitAnalyzer

    x = np.linspace(1.0, 5.0, 600)

    def g(c, s, a):
        return a * np.exp(-0.5 * ((x - c) / s) ** 2)

    y = g(2.0, 0.05, 1.0e5) + g(3.5, 0.07, 6.0e4) + 2000.0 + 500.0 * x

    analyzer = PeakFitAnalyzer(PeakFitPlan(model="gaussian", n_peaks=2))
    out = analyzer.analyze(AnalysisInput(label="7", x=x, y=y, x_unit="q"))

    assert out.ok and out.label == "7"
    # Flat params drive the vs-frame batch plot: two centers near truth + width/amp.
    centers = sorted(out.params[f"center_{i}"] for i in range(2))
    assert abs(centers[0] - 2.0) < 0.05 and abs(centers[1] - 3.5) < 0.05
    assert "fwhm_0" in out.params and "amplitude_0" in out.params
    # Overlay drives the live display: the fit trace IS the backend best_fit.
    assert "fit" in out.overlay.traces and "residual" in out.overlay.traces
    np.testing.assert_allclose(
        out.overlay.traces["fit"], out.result.payload.best_fit
    )
    assert len(out.overlay.markers) == 2  # peak centers


def test_peak_fit_analyzer_is_inert_on_failure():
    """A bad pattern yields ok=False with empty params and no exception, so a
    live/batch runner driving it stays alive."""
    pytest.importorskip("lmfit")
    from xrd_tools.analysis.runner import AnalysisInput, PeakFitAnalyzer

    out = PeakFitAnalyzer().analyze(
        AnalysisInput(label="0", x=np.array([1.0, 2.0, 3.0]),
                      y=np.array([np.nan, np.nan, np.nan]))
    )
    assert out.ok is False
    assert out.params == {}
    assert out.overlay is None
