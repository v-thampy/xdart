"""Stitch's calibration fit windows on the shared fit display (Codex's Stitch/Int reuse audit R2: they drew
on two page-built plots with their own legends, colours and clearing). What each fit shows is pinned with
real fits in ``test_stitch_calibration_fit_dialog.py`` and ``test_stitch_guided_calibration.py``; these rows
pin the windows on ``FitView``, from small stand-in results."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from pyqtgraph.Qt import QtWidgets

from xdart.gui.gui_utils import RectViewBox
from xdart.gui.tabs.scattering.fit_view import FitView
from xdart.gui.tabs.calibration.fit_windows import CalibrationFitWindow, DirectBeamFitWindow
from xdart.gui.widgets.scientific_plot import trace_colors


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _names(plot):
    return [item.name() for item in plot.listDataItems()]


def test_the_direct_beam_fit_window_shows_its_frames_model_and_residuals_on_the_shared_fit_display():
    _app()
    window = DirectBeamFitWindow()
    result = SimpleNamespace(
        distance_mm=1014.7, distance_std_m=1e-4, zero_beam_pixel=(243.0, 97.0), model='linear',
        angles_deg=[0., 1., 2., 3.], centres_px=[10., 11., 12., np.nan], used=[True, True, False, False],
        predictions_px=[10., 11., 12., 13.], residuals_px=[0., 0.1, 0., 0.])
    window.show_result(result, energy_eV=17000.)
    assert isinstance(window.view, FitView) and window.view.parent() is window
    assert isinstance(window.plot.getPlotItem().vb, RectViewBox)
    assert window.summary.text().startswith('Direct-beam distance: 1014.7 mm')
    assert window.summary.text().endswith('linear model, 2 of 4 frames used')
    assert _names(window.plot) == ['used frames', 'left out', 'linear fit']
    left_out = window.plot.listDataItems()[1]
    np.testing.assert_array_equal(left_out.xData, [2.])  # the frame left out with a finite centre only
    assert len(window.residuals.listDataItems()) == 1
    assert window.residuals.getPlotItem().getAxis('bottom').labelText == 'Motor angle (°)'
    assert window.plot.getPlotItem().getAxis('left').labelText == 'Beam centre along the motion (px)'


def test_the_calibration_fit_window_groups_rows_in_the_1d_plots_colours_and_marks_rejected_points():
    _app()
    window = CalibrationFitWindow()
    frames = [(0, {'del': 10., 'nu': 0.}), (1, {'del': 20., 'nu': 0.}), (2, {'del': 10., 'nu': 1.5})]
    fit = SimpleNamespace(
        asset={'provenance': {'source_motors': {'del_angle': 'del', 'nu_angle': 'nu'}}},
        point_residuals={'frames': frames, 'frame': np.array([0, 1, 2, 2]), 'kept': np.array([1, 1, 1, 0], bool),
                         'dq': np.array([1e-3, -1e-3, 2e-3, 5e-3]), 'q': np.array([1., 2., 1., 2.])},
        rms_q=1e-3, n_points=3, message='')
    window.show_result(fit, 'refine')
    assert isinstance(window.view, FitView) and window.windowTitle() == 'Automatic refinement'
    assert _names(window.plot) == ['nu 0°', 'nu 1.5°', 'rejected']
    colours = [tuple(item.opts['symbolBrush']) for item in window.plot.listDataItems()[:2]]
    assert colours == [tuple(colour) for colour in trace_colors('Default', 2)]
    assert window.plot.getPlotItem().getAxis('bottom').labelText == 'del motor angle (°)'
    assert len(window.residuals.listDataItems()) == 3  # the two rows and the rejected crosses, unnamed
    window.show_result(SimpleNamespace(asset={}, point_residuals=None, rms_q=0., n_points=0, message=''))
    assert not window.plot.listDataItems()
    assert window.plot.getPlotItem().titleLabel.text == 'This fit reports no control points to plot'
