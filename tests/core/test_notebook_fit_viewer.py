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
