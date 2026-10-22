"""Stitch's calibration fit windows on the shared fit display (Codex's Stitch/Int reuse audit R2: they drew
on two page-built plots with their own legends, colours and clearing). What each fit shows is pinned with
real fits in ``test_stitch_calibration_fit_dialog.py`` and ``test_stitch_guided_calibration.py``; these rows
pin the windows on ``FitView``, from small stand-in results."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from pyqtgraph.Qt import QtWidgets

from xdart.gui.gui_utils import RectViewBox
from xdart.gui.tabs.scattering.fit_view import FitView
from xdart.gui.tabs.calibration.fit_windows import CalibrationFitWindow, DirectBeamFitWindow
from xdart.gui.widgets.scientific_plot import trace_colors
from tests.core.test_calibrant_preview import refined_scan


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


def test_calibrant_preview_uses_native_stitch_chi_grid_and_shared_image_rect(refined_scan, tmp_path):
    from pathlib import Path
    from xrd_tools.analysis.plans import run_stitch
    from xrd_tools.analysis.stitch_operation import (
        StitchGeometryInput, StitchGeometryKind, StitchOperationPlan, _OrientedSource,
        _runtime_geometry, _runtime_plan, capture_stitch_geometry,
    )
    from xrd_tools.integrate.stitch_calibration import save_calibration_fit, stitch_calibrant_preview
    from xrd_tools.session.display_logic import PanelKey, PanelRole
    from xrd_tools.session.rsm_viewer_model import rsm_axis_bounds
    from xrd_tools.sources import open_source
    from xdart.gui.tabs.scattering.scientific_view import ScientificView

    app = _app()
    spec, fit = refined_scan
    # Keep the fitted asset; a later calibrant scan's forward angles have narrow χ coverage.
    from tests.core.test_stitch_autocalibration import ring_scan
    spec = ring_scan(tmp_path / 'forward', [(20., 0.), (25., 0.), (30., 0.)])
    path = Path(save_calibration_fit(fit, tmp_path / 'fit.json'))
    provenance = fit.asset['provenance']
    geometry = capture_stitch_geometry(StitchGeometryInput(path, StitchGeometryKind.PYFAI_GONIOMETER_JSON,
        source_motors=tuple(sorted(provenance['source_motors'].items())),
        image_rotation=provenance['image_rotation'], image_flip=provenance.get('image_flip', False)))
    diff, orientation = _runtime_geometry(geometry, path.read_bytes())
    native = StitchOperationPlan(geometry, mode='2d', native_records=True,
        npt_rad_2d=80, npt_azim_2d=36, npt_1d=120, radial_range=(.1, 6.),
        map_radial_range=(.1, 6.), polarization_factor=None)
    runtime = _runtime_plan(native, diff)
    assert runtime.azimuth_range is None
    source = open_source(spec)
    try:
        oriented = _OrientedSource(source, orientation, 512 << 20)
        main_stitch = run_stitch(runtime, oriented,
            frame_indices=source.frame_indices, workers=1, collect_frame_records=False)
        # Preview retains its existing BBox integration; main Stitch uses full splitting.
        # Their Auto χ domain comes from the same geometry scout, independently of splitting.
        expected = run_stitch(replace(runtime, method='BBox'), oriented,
            frame_indices=source.frame_indices, workers=1, collect_frame_records=False)
        np.testing.assert_array_equal(expected.payload.azimuthal, main_stitch.payload.azimuthal)
    finally:
        close = getattr(source, 'close', None)
        if close is not None:
            close()
    expected_chi = rsm_axis_bounds(expected.payload.azimuthal)
    assert expected_chi[1] - expected_chi[0] < 180.
    window = CalibrationFitWindow()
    key = PanelKey(PanelRole.SLICE_2D, 'calibrant')
    try:
        for zero_counts in (False, True):
            if zero_counts:
                for image in spec.uri.parent.joinpath('images').glob('*.raw'):
                    np.zeros((195, 1475), dtype='<i4').tofile(image)
            result = stitch_calibrant_preview(fit, spec, radial_range=(.1, 6.),
                npt_rad=80, npt_azim=36, npt_1d=120)
            window.show_preview(result)
            window.show()
            app.processEvents()
            assert isinstance(window.pattern_view, ScientificView)
            canvas = window.pattern_view.keyed_panel(key).pane.canvas
            rect = canvas.imageItem.mapRectToParent(canvas.imageItem.boundingRect())
            np.testing.assert_allclose([rect.top(), rect.bottom()], expected_chi, rtol=0, atol=1e-10)
            np.testing.assert_array_equal(result.payload.azimuthal, expected.payload.azimuthal)
            np.testing.assert_array_equal(result.auxiliary.coverage, expected.auxiliary.coverage)
            np.testing.assert_array_equal(canvas.raw_image, result.payload.intensity)
            if zero_counts:
                assert np.any(result.auxiliary.coverage > 0)
                assert not np.any(result.payload.intensity)
                covered_profile = result.profile[1].coverage > 0
                assert np.any(covered_profile)
                assert np.all(result.profile[0].intensity[covered_profile] == 0.)
            else:
                np.testing.assert_array_equal(result.payload.intensity, expected.payload.intensity)
                np.testing.assert_array_equal(result.profile[0].radial, expected.profile[0].radial)
                np.testing.assert_array_equal(result.profile[0].intensity, expected.profile[0].intensity)
            canvas.imageViewBox.setYRange(-180., 180., padding=0)
            window.pattern_view.log_scale.click()
            app.processEvents()
            np.testing.assert_allclose(canvas.imageViewBox.viewRange()[1], [-180., 180.], rtol=0, atol=1e-10)
            rect = canvas.imageItem.mapRectToParent(canvas.imageItem.boundingRect())
            np.testing.assert_allclose([rect.top(), rect.bottom()], expected_chi, rtol=0, atol=1e-10)
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
