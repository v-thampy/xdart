"""Frozen P1 E3: real non-ROI refusal in the mounted context-help consumer."""
import os
from pathlib import Path

import pytest
import numpy as np
from pyqtgraph.Qt import QtWidgets


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_real_scan_refusal_opens_context_with_literal_guidance(tmp_path, qapp, monkeypatch):
    from tests.xdart.scattering.test_roi_combined_window import _open, _settle
    from tests.xdart.scattering.test_p37_analysis_mount import _page, _close_page, _roi_navigation_table
    from xdart.gui.widgets.context_help import owner_for
    from xrd_tools.analysis.scan_operations import run_scan_plot, ScanPlotPlan
    _roi_navigation_table(tmp_path, missing_label=7); page = _page()
    try:
        dialog, picker = _open(page, tmp_path/'navigation.nexus', qapp)
        table = dialog._vnext_table_result
        result = run_scan_plot(ScanPlotPlan(None, ('absent_numeric_column',)), table)
        dialog.set_vnext_scan_result(result)
        dialog.help_button.click(); window = owner_for(dialog).window
        assert 'Next action: select a current available column' in window.pane.toPlainText()
        assert result.code == 'SCAN_PLOT_COLUMN_NOT_FOUND'
        assert window.topic_id == 'analysis.plot-roi' and '!' in dialog.help_button.text()
        assert result.code in window.pane.toPlainText()
        # Opening help must use only retained value text; no source construction.
        import xrd_tools.analysis.scan_operations as operations
        def forbidden(*args, **kwargs):
            raise AssertionError('help must not reopen a source')
        monkeypatch.setattr(operations, 'open_source', forbidden)
        paths = [Path(os.environ[key]) for key in ('XDART_SESSION_FILE', 'XDART_SETTINGS_FILE')]
        before = [(p.exists(), p.read_bytes() if p.exists() else None) for p in paths]
        dialog.help_button.click()
        assert before == [(p.exists(), p.read_bytes() if p.exists() else None) for p in paths]
        monkeypatch.undo()
        from xrd_tools.analysis.scan_operations import RoiScanPlan, run_roi_scan
        warning = run_roi_scan(RoiScanPlan.from_table(table,
            signals=tuple(picker.roi_signals()), mask=np.zeros((2, 2), dtype=bool)))
        dialog.set_vnext_roi_result(warning)
        assert warning.code == 'OK'
        assert warning.diagnostics  # the real missing-frame warning, not a fabricated result
        dialog.help_button.click()
        for literal in warning.diagnostics:
            assert literal in window.pane.toPlainText()
        dialog.set_vnext_metadata(table); dialog.help_button.click()
        assert 'SCAN_PLOT_COLUMN_NOT_FOUND' not in window.pane.toPlainText()
    finally:
        _close_page(page, qapp)
