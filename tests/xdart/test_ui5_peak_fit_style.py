"""UI-5 — Peak Fitting fit-plot curve styling.

(1) DATA = markers only, marker size 4 (match the 1D display), NO connecting
    line so the fit reads on top of the points.
(2) BACKGROUND (and any non-total-fit component) = DASHED, line width +30%.
(3) the TOTAL fit stays SOLID.
Drives the real PeakFitDialog._draw_outcome; captures each plot() call's style.
"""

import os
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pg = pytest.importorskip("pyqtgraph")
from pyqtgraph.Qt import QtCore

from xdart.gui.tabs.static_scan.peak_fit_dialog import PeakFitDialog


def _draw():
    calls: list = []

    class _Plot:
        def clear(self):
            pass

        def addItem(self, *_):
            pass

        def setLabel(self, *_a, **_k):
            pass

        def plot(self, *a, **k):
            calls.append(k)
            return SimpleNamespace()

        def addLine(self, *_a, **_k):
            pass

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

    host = SimpleNamespace(
        plot=_Plot(), resid_plot=_Plot(), region=object(),
        _x=x, _y=np.ones_like(x), _x_label="q",
        _clear_fit=lambda: None,
        table=SimpleNamespace(setRowCount=lambda *_: None,
                              setItem=lambda *_: None),
        status=SimpleNamespace(setText=lambda *_: None))
    host._draw_outcome = MethodType(PeakFitDialog._draw_outcome, host)
    host._draw_outcome(outcome)
    return {k.get("name"): k for k in calls if k.get("name")}


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
