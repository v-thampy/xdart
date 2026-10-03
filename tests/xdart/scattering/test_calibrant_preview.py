"""The refinement popup mounts Int's full view and uses the calibration worker for a real scan."""
import time

import numpy as np
from pyqtgraph.Qt import QtWidgets

from tests.core.test_calibrant_preview import refined_scan
from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind


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
        assert 'calculated |q| − expected ring |q|' in window.summary.text()
        assert window.plot.getPlotItem().getAxis('bottom').labelText == 'del motor angle (°)'
        assert isinstance(window.pattern_view, ScientificView)
        window.previewRequested.emit()
        until = time.monotonic() + 15
        while window.pattern_view.keyed_projection is None and time.monotonic() < until:
            app.processEvents()
            time.sleep(.02)
        state = window.pattern_view.keyed_projection
        assert state is not None, dialog.status.text()
        assert '3 frames' in state.status
        image, trace = state.panels
        assert image.image.shape == (image.y_axis.values.size, image.x_axis.values.size)
        assert np.nanmax(trace.intensity) > 0
        window.pattern_view.commandRequested.emit(ShellCommand(ShellCommandKind.SET_LOG_SCALE, True))
        assert window.pattern_view.keyed_projection.log_scale
        window.clear_preview()
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
