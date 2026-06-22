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


# ── Batch driver (analysis-agnostic) ──────────────────────────────────────


def test_run_batch_frame_unit_accumulates_per_input():
    """Frame-unit: one outcome per input, in order, with progress ticks; the
    table is the aligned vs-index series."""
    from xrd_tools.analysis.runner import (
        AnalysisInput, AnalysisOutcome, batch_params_table, run_batch)

    class CountAnalyzer:
        kind = "count"
        unit = "frame"

        def analyze(self, inp):
            return AnalysisOutcome(label=inp.label, ok=True,
                                   params={"center_0": float(inp.label)})

        def analyze_scan(self, inputs):
            raise AssertionError("frame-unit must not call analyze_scan")

    inputs = [AnalysisInput(label=str(i), x=np.array([1.0]), y=np.array([1.0]))
              for i in range(3)]
    ticks = []
    outs = run_batch(CountAnalyzer(), inputs,
                     on_progress=lambda d, t: ticks.append((d, t)))
    assert [o.label for o in outs] == ["0", "1", "2"]
    assert ticks == [(1, 3), (2, 3), (3, 3)]
    labels, cols = batch_params_table(outs)
    assert labels == ["0", "1", "2"]
    assert cols["center_0"] == [0.0, 1.0, 2.0]


def test_run_batch_is_inert_when_a_frame_raises():
    """A raising analyze() yields ok=False; the batch still completes and the
    failed frame is nan in the table (kept aligned with labels)."""
    from xrd_tools.analysis.runner import (
        AnalysisInput, AnalysisOutcome, batch_params_table, run_batch)

    class FlakyAnalyzer:
        kind = "flaky"
        unit = "frame"

        def analyze(self, inp):
            if inp.label == "1":
                raise RuntimeError("bad frame")
            return AnalysisOutcome(label=inp.label, ok=True,
                                   params={"center_0": 5.0})

        def analyze_scan(self, inputs):
            raise AssertionError

    inputs = [AnalysisInput(label=str(i), x=np.array([1.0]), y=np.array([1.0]))
              for i in range(3)]
    outs = run_batch(FlakyAnalyzer(), inputs)
    assert [o.ok for o in outs] == [True, False, True]
    labels, cols = batch_params_table(outs)
    assert labels == ["0", "1", "2"]
    assert cols["center_0"][0] == 5.0
    assert np.isnan(cols["center_0"][1])       # raising frame -> nan, still aligned
    assert cols["center_0"][2] == 5.0


def test_run_batch_scan_unit_is_one_outcome():
    """Scan-unit: the whole set is analyzed once -> a single-element list."""
    from xrd_tools.analysis.runner import (
        AnalysisInput, AnalysisOutcome, run_batch)

    class ScanAnalyzer:
        kind = "scan"
        unit = "scan"

        def analyze(self, inp):
            raise AssertionError("scan-unit must not call analyze")

        def analyze_scan(self, inputs):
            return AnalysisOutcome(label="scan", ok=True,
                                   params={"n": len(inputs)})

    inputs = [AnalysisInput(label=str(i), x=np.array([1.0]), y=np.array([1.0]))
              for i in range(4)]
    ticks = []
    outs = run_batch(ScanAnalyzer(), inputs,
                     on_progress=lambda d, t: ticks.append((d, t)))
    assert len(outs) == 1 and outs[0].params["n"] == 4
    assert ticks == [(4, 4)]


# ── Sin2PsiAnalyzer (scan-unit, the first scan-unit concrete analyzer) ─────


def _synthetic_polar_map():
    """A (q, χ) polar map with one Bragg peak per χ sector whose center drifts
    linearly in sin²ψ (ψ=|χ|), so the sin²ψ regression has a clean slope."""
    from xrd_tools.core.containers import IntegrationResult2D

    q = np.linspace(2.0, 4.0, 400)
    chi = np.linspace(-40.0, 40.0, 33)
    intensity = np.zeros((q.size, chi.size), dtype=float)
    for j, c in enumerate(chi):
        s2 = np.sin(np.deg2rad(abs(c))) ** 2
        center = 3.0 + 0.02 * s2                      # peak walks with sin²ψ
        intensity[:, j] = 1.0e4 * np.exp(-0.5 * ((q - center) / 0.03) ** 2) + 100.0
    return IntegrationResult2D(radial=q, azimuthal=chi, intensity=intensity,
                               unit="q_A^-1", azimuthal_unit="chi_deg")


def test_sin2psi_analyzer_is_scan_unit_and_rejects_frame_call():
    from xrd_tools.analysis.plans import Sin2PsiPlan
    from xrd_tools.analysis.runner import Analyzer, AnalysisInput, Sin2PsiAnalyzer

    a = Sin2PsiAnalyzer(Sin2PsiPlan(q_range=(2.7, 3.3)))
    assert isinstance(a, Analyzer)
    assert a.kind == "sin2psi" and a.unit == "scan"
    with pytest.raises(NotImplementedError):
        a.analyze(AnalysisInput(label="0", x=np.array([1.0]), y=np.array([1.0])))


def test_sin2psi_analyzer_projects_regression_and_overlay():
    pytest.importorskip("lmfit")
    from xrd_tools.analysis.plans import Sin2PsiPlan
    from xrd_tools.analysis.runner import AnalysisInput, Sin2PsiAnalyzer

    result2d = _synthetic_polar_map()
    analyzer = Sin2PsiAnalyzer(Sin2PsiPlan(q_range=(2.7, 3.3), chi_width=5.0))
    # The 2-D map rides on result_1d; x/y are unused for a scan-unit analyzer.
    inp = AnalysisInput(label="scan", x=result2d.radial,
                        y=result2d.intensity[:, 0], result_1d=result2d)
    out = analyzer.analyze_scan([inp])

    assert out.ok and out.label == "scan"
    assert {"d0", "slope", "r_squared", "n_sectors"} <= set(out.params)
    assert out.params["n_sectors"] >= 2
    assert out.result.payload.r_squared > 0.9        # clean synthetic regression
    # Overlay is the sin²ψ regression: measured d points + the fitted line.
    assert {"data", "fit"} <= set(out.overlay.traces)
    assert out.overlay.x.size == out.overlay.traces["data"].size


def test_sin2psi_analyzer_is_inert_without_a_2d_map():
    """No IntegrationResult2D on any input -> ok=False, no exception, so a runner
    driving it stays alive."""
    from xrd_tools.analysis.plans import Sin2PsiPlan
    from xrd_tools.analysis.runner import AnalysisInput, Sin2PsiAnalyzer

    out = Sin2PsiAnalyzer(Sin2PsiPlan(q_range=(2.7, 3.3))).analyze_scan(
        [AnalysisInput(label="0", x=np.array([1.0]), y=np.array([1.0]))]
    )
    assert out.ok is False and "IntegrationResult2D" in out.message
