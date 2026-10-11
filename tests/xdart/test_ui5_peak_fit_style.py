"""UI-5 — Peak Fitting fit-plot curve styling.

(1) DATA = markers only, marker size 4 (match the 1D display), NO connecting
    line so the fit reads on top of the points.
(2) BACKGROUND (and any non-total-fit component) = DASHED, line width +30%.
(3) the TOTAL fit stays SOLID.
Drives the real PeakFitDialog._draw_outcome and reads each drawn item's style. FIT-TOOLS-2026-10: the dialog draws
on the shared FitView, whose series roles carry these looks; the rows read the real plot's items (the fake host
these rows once bound ``_draw_outcome`` to is gone with the dialog's own plots).
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pg = pytest.importorskip("pyqtgraph")
from pyqtgraph.Qt import QtCore

from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog


def _draw():
    from pyqtgraph.Qt import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    x = np.linspace(0.0, 5.0, 20)
    overlay = SimpleNamespace(
        x=x, traces={"fit": np.ones_like(x), "background": np.zeros_like(x),
                     "residual": np.zeros_like(x)})
    payload = SimpleNamespace(
        peak_centers=[], peak_centers_err=[], peak_sigmas=[], peak_amplitudes=[],
        params=None, success=True, model_name="Gaussian", background_name="linear")
    outcome = SimpleNamespace(
        overlay=overlay, result=SimpleNamespace(payload=payload),
        label="current", params={})
    dialog = PeakFitDialog(analysis_context=None)
    try:
        dialog.set_live_pattern(x, np.ones_like(x), "q")
        dialog._draw_outcome(outcome)
        return {item.name(): item.opts for item in dialog.plot.listDataItems() if item.name()}
    finally:
        dialog.close()
        dialog.deleteLater()


def test_data_is_markers_only():
    data = _draw()["data"]
    assert data.get("pen") is None            # NO connecting line
    assert data.get("symbol") == "o"
    assert data.get("symbolSize") == 4        # matches the 1D display markers


def test_total_fit_is_solid():
    fit = _draw()["fit"]
    assert fit["pen"].style() == QtCore.Qt.PenStyle.SolidLine


def test_background_is_dashed_and_thicker():
    bg = _draw()["background"]
    assert bg["pen"].style() == QtCore.Qt.PenStyle.DashLine
    assert bg["pen"].widthF() == pytest.approx(1.3)   # +30% over the old width 1
