"""The refinement popup mounts Int's full view and uses the calibration worker for a real scan."""
import time
from dataclasses import replace
import copy
import json

import numpy as np
from pyqtgraph.Qt import QtWidgets

from tests.core.test_calibrant_preview import refined_scan
from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind


def test_confirmed_save_replaces_the_current_geometry_and_companion_through_the_worker(refined_scan, tmp_path, monkeypatch):
    from xrd_tools.integrate.stitch_calibration import paired_xu_calibration
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _spec, fit = refined_scan
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    target = tmp_path / 'confirmed.json'
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *args: (str(target), '')))
    try:
        for index in range(2):
            if index:
                asset = copy.deepcopy(fit.asset)
                record = asset['calibration']
                record['param'][record['param_names'].index('dist')] += .001
                fit = replace(fit, asset=asset)
            dialog.fit = fit
            dialog._save()
            until = time.monotonic() + 15
            while dialog.active() and time.monotonic() < until:
                app.processEvents()
                time.sleep(.01)
            assert not dialog.active()
            assert json.loads(target.read_text()) == fit.asset['calibration'], dialog.status.text()
            assert paired_xu_calibration(target).exists()
    finally:
        dialog.shutdown()
        dialog.close()


def test_preview_worker_draws_the_stitched_map_and_pattern_and_retires_on_edit(refined_scan, tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    spec, fit = refined_scan
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        for key, value in (('source', str(spec.uri)), ('scan', '1'),
                           ('images', spec.options['image_dir']), ('stem', 'ring_scan1_')):
            dialog._set_field_text(key, value)
        dialog._fields['shape'].setText('195, 1475')
        dialog._fields['dtype'].setText('int32')
        dialog.fit, dialog._fit_source = fit, dialog._spec()
        dialog._fit_kind = 'refine'
        dialog.show()
        dialog._show_calibration_fit()
        window = dialog._calibration_fit_window
        assert window.pattern_view.vertical_splitter.isHidden(), 'never show the initial Int layout while stitching'
        assert window.pattern_view.viewer_intensity_row.isHidden()
        assert 'calculated |q| − expected ring |q|' in window.summary.text()
        assert window.plot.getPlotItem().getAxis('bottom').labelText == 'del motor angle (°)'
        assert isinstance(window.pattern_view, ScientificView)
        assert window.panels.tabText(0) == 'Stitched calibrant'
        assert window.panels.currentWidget() is window.pattern_view
        until = time.monotonic() + 15
        while window.pattern_view.keyed_projection is None and time.monotonic() < until:
            app.processEvents()
            time.sleep(.02)
        state = window.pattern_view.keyed_projection
        assert state is not None, dialog.status.text()
        assert state.share_axis, 'the stacked preview shares Q by default'
        assert '3 frames' in state.status
        image, trace = state.panels
        assert image.image.shape == (image.y_axis.values.size, image.x_axis.values.size)
        assert np.nanmax(trace.intensity) > 0
        pane = window.pattern_view.keyed_panel(image.key).pane
        (reference,) = pane.overlay_families['calibrant-reference']
        from xrd_tools.integrate.calibration import pyfai_calibrant
        q = np.sort(2 * np.pi / np.asarray(pyfai_calibrant('LaB6').dspacing))
        wavelength = fit.asset['calibration']['wavelength'] * 1e10
        expected = q[(q >= image.x_axis.values.min()) & (q <= image.x_axis.values.max())
                     & (q <= 4 * np.pi / wavelength)]
        assert expected.size
        np.testing.assert_allclose(reference.xData[::3], expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(reference.xData[1::3], expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(reference.yData[::3], image.y_axis.edges[0])
        np.testing.assert_allclose(reference.yData[1::3], image.y_axis.edges[1])
        assert 'LaB6' in reference.toolTip() and 'Display only' in reference.toolTip()
        marker_x = reference.xData.copy()
        app.processEvents()
        rows = window.pattern_view._keyed_area
        assert isinstance(rows, QtWidgets.QSplitter) and rows.count() == 2
        before = rows.sizes()
        assert abs(before[0] - before[1]) <= 3
        rows.moveSplitter(sum(before) // 3, 1)
        resized = rows.sizes()
        assert resized[0] < resized[1]
        window.pattern_view.commandRequested.emit(ShellCommand(ShellCommandKind.SET_LOG_SCALE, True))
        assert window.pattern_view.keyed_projection.log_scale
        assert rows.sizes() == resized
        (reference,) = pane.overlay_families['calibrant-reference']
        np.testing.assert_array_equal(reference.xData, marker_x)
        window.panels.setCurrentWidget(window.view)
        fit = replace(fit)
        dialog.fit = fit
        dialog._show_calibration_fit()
        assert window.panels.currentWidget() is window.pattern_view
        until = time.monotonic() + 15
        while window.pattern_view.keyed_projection is None and time.monotonic() < until:
            app.processEvents()
            time.sleep(.02)
        assert window.pattern_view.keyed_projection is not None
        window.clear_preview()
        assert not pane.overlay_families
        window.previewRequested.emit()
        window.close()  # actual worker cancellation; completion must not reopen/adopt a hidden view
        until = time.monotonic() + 15
        while dialog.active() and time.monotonic() < until:
            app.processEvents()
            time.sleep(.02)
        assert not dialog.active() and dialog.fit is fit
        assert not window.isVisible() and window.pattern_view.keyed_projection is None
        dialog._invalidate_fit()
        assert not window.isVisible() and window.pattern_view.keyed_projection is None
    finally:
        dialog.shutdown()
        dialog.close()
