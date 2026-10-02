"""Shared notebook host actions: one public callback, committed science, presentation-only rerender, disposal."""
from __future__ import annotations

import numpy as np

from xrd_tools.analysis.fit_snapshot import FitAxis, FitInputKey, FitInputSnapshot
from xrd_tools.analysis.runner import AnalysisOutcome, Overlay


def test_fit_viewer_callbacks_boundaries_and_disposal():
    from xrd_tools.gui.widgets import PeakFitViewer
    from xrd_tools.gui.widgets.pattern_viewer import PatternViewer
    from xrd_tools.gui.widgets.fit_controls import PeakFitControls
    x = np.linspace(1., 4., 101)
    snapshot = FitInputSnapshot(FitInputKey("current", "derived", "receipt", 1),
                                (FitAxis("q", "A^-1", x),), np.ones(x.size))
    loads, fits = [], []
    def load(index):
        loads.append(index)
        return snapshot
    def fit(inp, settings):
        fits.append((inp, settings))
        return AnalysisOutcome(inp.key.result_identity, True, {"center_0": 2.},
                               Overlay(x, {"fit": np.ones(x.size), "residual": np.zeros(x.size)}))
    viewer = PeakFitViewer(["current", "next"], load, fit,
                           settings={"model": "gaussian", "selection_mode": "count", "n_peaks": 1})
    assert isinstance(viewer.pattern_viewer, PatternViewer)
    assert isinstance(viewer.controls, PeakFitControls)
    assert not fits and loads == [0]
    viewer.ranges.fit_bounds.value = (1.5, 3.)
    viewer.controls._fit_button.click()
    assert len(fits) == 1 and fits[0][1]["fit_bounds"] == (1.5, 3.)
    viewer._log.value = True
    viewer.ranges.display_min.value = 1.1
    viewer.ranges.reset.click()
    assert len(fits) == 1
    assert viewer.figure_widget.layout.yaxis.type == "log"
    assert viewer.figure_widget.layout.yaxis2.type == "linear"
    viewer.selector.selection.value = 1
    assert loads == [0, 1] and viewer.outcome is None and len(fits) == 1
    viewer.dispose()
    viewer.dispose()
    viewer.controls._fit_button.click()
    assert len(fits) == 1


def test_phase_controls_use_actual_current_settings_and_explicit_phase_selection():
    from xrd_tools.gui.widgets import PhaseFitControls
    calls = []
    controls = PhaseFitControls(["first", "second"], on_fit=calls.append)
    settings = {"cif_paths": ["a.cif", "b.cif"], "phase_names": ["first", "second"],
                "expected_hashes": [None, None], "phase_profile": "gaussian", "prefit_background": "none"}
    controls.configure_displayed(settings)
    controls._phase_toggles["first"].value = False
    controls._fit_button.click()
    assert calls == [{**settings, "cif_paths": ("b.cif",), "phase_names": ("second",), "expected_hashes": (None,)}]
    controls.dispose()
    controls._fit_button.click()
    assert len(calls) == 1


def test_region_controls_use_each_native_axis_and_slice_coordinate():
    from xrd_tools.gui.widgets import PeakFitViewer, RegionRangeControls
    axes = (FitAxis("H", "r.l.u.", np.linspace(-1., 1., 11)),
            FitAxis("K", "r.l.u.", np.linspace(-2., 2., 13)),
            FitAxis("L", "r.l.u.", np.linspace(1., 3., 15)))
    snapshot = FitInputSnapshot(FitInputKey("native box", "derived", "receipt", 1), axes,
                                np.ones((11,13,15)))
    fits = []
    viewer = PeakFitViewer(["native box"], lambda index: snapshot,
                           lambda inp, settings: fits.append(settings) or AnalysisOutcome("native box", False),
                           tool="region3d", settings={"seeds": [["peak", [0., 0., 2.], None, None]]})
    assert isinstance(viewer.region_ranges, RegionRangeControls)
    viewer.region_ranges.ranges["K"].fit_bounds.value = (-1., 1.)
    viewer.controls._fit_button.click()
    assert len(fits) == 1
    assert dict(fits[0]["region_bounds"])["K"] == (-1., 1.)
    viewer.region_ranges.slice_axis.value = "H"
    viewer.region_ranges.slice_coordinate.value = .2
    assert viewer.region_ranges.plane == ("H", 6)
    assert len(fits) == 1
    viewer.dispose()


def test_plotly_semantic_axis_conversion_and_peak_components():
    from xrd_tools.viz.plotly import plot_analysis_outcome
    from xrd_tools.core.containers import convert_radial_axis
    from xrd_tools.analysis.runner import DisplayedPeakFitAnalyzer
    x = np.linspace(1., 4., 151)
    y = 3. + 10.*np.exp(-.5*((x-2.5)/.08)**2)
    snapshot = FitInputSnapshot(FitInputKey("q input", "derived", "receipt", 1),
                                (FitAxis("q", "A^-1", x),), y, wavelength_angstrom=1.)
    outcome = DisplayedPeakFitAnalyzer({"model": "gaussian", "selection_mode": "count", "n_peaks": 1}).analyze(snapshot)
    figure = plot_analysis_outcome(snapshot, outcome, axis_unit="2th_deg", peak_model="gaussian",
                                   overlays=("fit", "residual", "components"), log_y=True)
    np.testing.assert_allclose(figure.data[0].x, convert_radial_axis(x, "q_A^-1", "2th_deg", 1.))
    assert figure.layout.xaxis2.title.text == "2θ (2th_deg)"
    assert figure.layout.yaxis2.type == "linear"
    assert any(trace.name == "p0" for trace in figure.data)
