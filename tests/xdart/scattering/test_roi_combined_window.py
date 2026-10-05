"""Frozen ROI-COMBINED-WINDOW C1–C10/T1–T9 through real owners and source execution."""
from threading import Event

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets, QtTest

from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.trend_view import PlotSeries, SeriesProjection, TrendView
from xdart.gui.tabs.scattering.events import CleanupStatus
from tests.xdart.scattering.test_p37_analysis_mount import (
    _page, _close_page, _finish, _roi_navigation_table,
)


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _settle(page, qapp):
    qapp.processEvents(); page._run_timer.stop()
    while page._analysis_slot.current_identity is not None:
        page._consume_analysis_update(_finish(page._analysis_slot, page._analysis_slot.current_identity))
        qapp.processEvents(); page._run_timer.stop()
        page._drain_executor(); page._run_timer.stop()


def _open(page, path, qapp):
    page._open_analysis_mount("scan_roi")
    dialog = page._scan_roi_dialog
    dialog.load_uri(str(path)); _settle(page, qapp)
    assert dialog._vnext_table_result is not None
    dialog.roi_btn.click(); _settle(page, qapp)
    assert dialog._roi_dialog is not None
    return dialog, dialog._roi_dialog


def _markers(view):
    items = (*view.plot.getPlotItem().listDataItems(), *view.right_view_box.childGroup.childItems())
    return [item for item in items if item.objectName() == "currentFrameMarker"]


def _click(view, curve, row, qapp):
    spot = next(spot for spot in curve.scatter.points() if spot.data() == row)
    qapp.processEvents()
    pos = view.plot.mapFromScene(curve.scatter.mapToScene(spot.pos()))
    QtTest.QTest.mouseClick(view.plot.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=pos)
    qapp.processEvents()


def test_combined_roi_window_cursor_compute_tooltips_and_click(tmp_path, qapp):
    """C1/C2/C3/C6/C10, T1/T9: admitted scan, actual buttons, real preview/reduction/gesture."""
    _roi_navigation_table(tmp_path)
    page = _page()
    try:
        dialog, picker = _open(page, tmp_path / "navigation.nexus", qapp)
        assert not picker.isWindow(), "Plot ROI must embed the picker in its owning tool"
        assert picker.window() is dialog
        dialog.roi_btn.click(); _settle(page, qapp)
        assert dialog._roi_dialog is picker and len(dialog.findChildren(ScientificView)) == 1
        picker.bg_check.setChecked(True)
        definitions = tuple(picker.roi_signals())
        picker.scale_log_btn.click()
        assert tuple(picker.roi_signals()) == definitions
        for widget in (picker.add_btn, picker.remove_btn, picker.compute_btn, picker.close_btn,
                       picker.cancel_btn, picker.status, dialog.roi_caption, picker.scale_default_btn, picker.scale_log_btn,
                       dialog.log_btn, *picker._numeric_rows.values()):
            assert widget.toolTip(), widget.objectName()
        assert "image" in picker.scale_log_btn.toolTip().lower()
        assert "left" in dialog.log_btn.toolTip().lower()
        picker.bg_check.setChecked(False)
        picker.compute_btn.click(); _settle(page, qapp)
        result = dialog._vnext_roi_result
        assert result.completed_labels == (3, 7, 11)
        np.testing.assert_array_equal(result.signal_values[0], [3, 7, 11])
        assert "roi1" in dialog._checked_y() and "roi1" not in dialog.r_browser.checked_results()
        assert dialog._roi_dialog is picker and picker._frame_label == 3
        assert "Frame 3" in dialog.roi_caption.text()
        assert _markers(dialog.trend)
        assert all(marker.toolTip() for marker in _markers(dialog.trend))
        dialog.resize(1200, 800); dialog.show(); qapp.processEvents()
        curve = next(item for item in dialog.plot.getPlotItem().listDataItems() if item.name() == "roi1")
        _click(dialog.trend, curve, 1, qapp); _settle(page, qapp)
        assert picker._frame_label == 7 and page._roi_preview_binding[4] == 7
        assert "Frame 7" in dialog.roi_caption.text()
        assert all(marker.scatter.points()[0].data() == 1 for marker in _markers(dialog.trend))
        dialog.norm_combo.setCurrentText("frame_index"); _settle(page, qapp)
        assert all(float(marker.getOriginalDataset()[1][0]) == 1 for marker in _markers(dialog.trend)
                   if marker.scatter.points()[0].data() == 1)
        dialog.norm_combo.setCurrentIndex(0); _settle(page, qapp)
        dialog.trend.reconcile_series(SeriesProjection((PlotSeries("a", [1, 2, 3], [10, 20, 30]),
            PlotSeries("b", [2, 1, 3], [20, 10, 30])), current_index=1))
        qapp.processEvents()
        curve = next(item for item in dialog.plot.getPlotItem().listDataItems() if item.name() == "a")
        _click(dialog.trend, curve, 0, qapp)
        assert picker._frame_label == 7 and "3, 7" in dialog.status.text()
        dialog.refresh_roi_cursor()
        picker.reducer_combo.setCurrentText("sum")
        assert "Out of date" in dialog.roi_caption.text()
        picker.reducer_combo.setCurrentText("mean")
        assert "Out of date" in dialog.roi_caption.text(), "staleness is latched until current adoption"
        picker.compute_btn.click(); _settle(page, qapp)
        assert "Out of date" not in dialog.roi_caption.text()
        picker.add_btn.click(); picker.compute_btn.click(); _settle(page, qapp)
        assert "roi2" in dialog._selector_columns
        picker.remove_btn.click(); picker.compute_btn.click(); _settle(page, qapp)
        assert "roi2" not in dialog._selector_columns
    finally:
        _close_page(page, qapp)


@pytest.mark.parametrize("shape", [(4, 4), (5, 5)])
def test_combined_roi_source_shape_policy_and_image_release(tmp_path, qapp, shape):
    """D2/T7: independently admitted source, retained view, cleared results, exact shape policy."""
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    _roi_navigation_table(first); _roi_navigation_table(second, shape=shape)
    page = _page()
    try:
        dialog, picker = _open(page, first / "navigation.nexus", qapp)
        picker.bg_check.setChecked(True); picker.reducer_combo.setCurrentText("sum")
        definitions = tuple(picker.roi_signals())
        picker.compute_btn.click(); _settle(page, qapp)
        dialog.load_uri(str(second / "navigation.nexus")); _settle(page, qapp)
        assert dialog._roi_dialog is picker, "source replacement must retain the embedded owner"
        assert picker._image is None and dialog._vnext_roi_result is None
        assert not _markers(dialog.trend) and not dialog._roi_stale
        assert page._roi_preview_binding is None and "roi1" not in dialog._selector_columns
        dialog.roi_btn.click(); _settle(page, qapp)
        assert dialog._roi_dialog is picker and picker._image.shape == shape
        if shape == (4, 4):
            assert tuple(picker.roi_signals()) == definitions
        else:
            signal, = picker.roi_signals()
            assert signal.reducer == "mean" and signal.background is None
            assert (signal.roi.width_x, signal.roi.width_y) == (5, 5)
            assert "reset" in picker.status.text().lower()
        dialog.close(); qapp.processEvents()
        assert picker._image is None and page._roi_preview_binding is None
    finally:
        _close_page(page, qapp)


def test_combined_roi_latest_navigation_waits_for_real_plot(tmp_path, qapp, monkeypatch):
    """T3: hold an actual scan_plot terminal, use real frame selector, dispatch only latest label."""
    import xrd_tools.analysis.scan_operations as operations
    _roi_navigation_table(tmp_path)
    page = _page(); ready, release = Event(), Event()
    real = operations.run_scan_plot

    def held(*args, **kwargs):
        result = real(*args, **kwargs); ready.set(); assert release.wait(5)
        return result

    try:
        dialog, picker = _open(page, tmp_path / "navigation.nexus", qapp)
        monkeypatch.setattr(operations, "run_scan_plot", held)
        dialog.log_btn.setChecked(True); page._run_timer.stop()
        assert ready.wait(5)
        panel = picker.scientific.keyed_panel(picker._image_key)
        panel.selector.setCurrentIndex(1); panel.selector.setCurrentIndex(2)
        assert dialog._roi_preview_label == 11 and dialog._roi_preview_waiting
        assert picker._frame_label == 3 and not picker.compute_btn.isEnabled()
        release.set(); _settle(page, qapp)
        assert dialog._roi_dialog is picker and picker._frame_label == 11
        assert page._roi_preview_binding[4] == 11 and picker.compute_btn.isEnabled()
    finally:
        release.set(); _close_page(page, qapp)


@pytest.mark.parametrize("action", ["edit", "source", "outer", "page", "cancel"])
def test_combined_roi_real_scan_cancellation_cleanup(tmp_path, qapp, monkeypatch, action):
    """C6/C7/T4/T5/T7/T9: pause a real reduction with its source open, then cancel through UI/owner."""
    import h5py
    import xrd_tools.analysis.scan_operations as operations
    _roi_navigation_table(tmp_path)
    page = _page(); ready, release = Event(), Event()
    real = operations.run_roi_scan
    handles = h5py.h5f.get_obj_count()

    def held(plan, **kwargs):
        progress = kwargs.get("progress_callback")
        def paused(done, total):
            if done == 1:
                ready.set(); assert release.wait(5)
            if progress is not None: progress(done, total)
        return real(plan, **{**kwargs, "progress_callback": paused})

    try:
        dialog, picker = _open(page, tmp_path / "navigation.nexus", qapp)
        picker.compute_btn.click(); _settle(page, qapp)
        accepted = dialog._vnext_roi_result
        monkeypatch.setattr(operations, "run_roi_scan", held)
        picker.compute_btn.click(); page._run_timer.stop()
        identity = page._analysis_identity
        assert ready.wait(5)
        assert not picker.compute_btn.isEnabled()
        assert picker.cancel_btn.isEnabled()
        dialog.resize(1200, 800); dialog.show(); qapp.processEvents()
        curve = next(item for item in dialog.plot.getPlotItem().listDataItems() if item.name() == "roi1")
        _click(dialog.trend, curve, 1, qapp)
        assert picker._frame_label == 3 and "running" in dialog.status.text().lower()
        if action == "edit":
            picker.reducer_combo.setCurrentText("sum")
            assert "Out of date" in dialog.roi_caption.text()
            assert not picker.compute_btn.isEnabled()
        elif action == "source":
            dialog.source_widget.dir_check.setChecked(True)
            assert page._roi_preview_binding is None and picker._image is None
        elif action == "outer":
            dialog.close()
        elif action == "cancel":
            picker.cancel_btn.click()
        else:
            before = page._scan_roi_generation
            receipt = page.close_workspace()
            assert page._scan_roi_generation > before
            assert receipt.cleanup_status is not CleanupStatus.CLEANED
        release.set()
        if action == "page":
            page._analysis_slot._worker.join(2)
            receipt = page.close_workspace()
            assert receipt.cleanup_status is CleanupStatus.CLEANED
        else:
            _settle(page, qapp)
            if action == "edit":
                assert dialog._vnext_roi_result is accepted
                assert picker.compute_btn.isEnabled()
                assert "Out of date" in dialog.roi_caption.text()
            elif action == "cancel":
                assert dialog._vnext_roi_result is accepted and picker.compute_btn.isEnabled()
                assert "cancel" in dialog.status.text().lower()
        assert not page._analysis_slot.owned
        assert h5py.h5f.get_obj_count() == handles
    finally:
        release.set(); _close_page(page, qapp)


def test_combined_roi_missing_raw_has_no_marker(tmp_path, qapp):
    """T8: real missing raw provenance, no thumbnail/zero/neighbor substitution."""
    _roi_navigation_table(tmp_path, missing_label=7)
    page = _page()
    try:
        dialog, picker = _open(page, tmp_path / "navigation.nexus", qapp)
        picker.compute_btn.click(); _settle(page, qapp)
        assert dialog._vnext_roi_result.no_raw_labels == (7,)
        # Preview7 refuses: the accepted image/cursor must remain on3. Its ROI data is still NaN.
        picker.scientific.keyed_panel(picker._image_key).selector.setCurrentIndex(1)
        _settle(page, qapp)
        assert picker._frame_label == 3
        assert "raw" in dialog.roi_caption.text().lower()
        assert np.isnan(dialog._vnext_roi_result.signal_values[0][1])
        # A prepared projection of that exact admitted missing row has no marker.
        dialog.trend.reconcile_series(SeriesProjection((PlotSeries("roi1", [3, 7, 11],
            dialog._vnext_roi_result.signal_values[0]),), current_index=1))
        assert not _markers(dialog.trend)
    finally:
        _close_page(page, qapp)


@pytest.mark.parametrize("log", [False, True])
def test_combined_roi_marker_axis_identity_and_ambiguous_real_click(qapp, log):
    """C4/C5/T2/T9: repeated X, normalized missing row, real scene clicks and cross-curve ambiguity."""
    view = TrendView(series_only=True); clicked = []
    try:
        view.reconcile_series(SeriesProjection((PlotSeries("left", [1, 1, 2], [10, 100, np.nan]),
            PlotSeries("right", [1, 1, 2], [-2, 0, 3], True)), log_left=log, current_index=1))
        markers = _markers(view)
        assert len(markers) == 2
        assert sorted(float(item.getData()[1][0]) for item in markers) == [0, 2 if log else 100]
        with pytest.raises(ValueError, match="index"):
            view.reconcile_series(SeriesProjection((PlotSeries("short", [1], [1]),), current_index=1))
        assert _markers(view) == markers, "refuse invalid index before touching accepted display"
        view.pointRowsClicked.connect(clicked.append)
        view.resize(800, 450); view.show(); qapp.processEvents()
        left = next(item for item in view.plot.getPlotItem().listDataItems() if item.name() == "left")
        _click(view, left, 1, qapp)
        assert clicked[-1] == (1,), "duplicate X must use the point's own original row"
        _click(view, view.right_view_box.addedItems[0], 0, qapp)
        assert clicked[-1] == (0,), "right-axis negative data uses its exact source row"
        view.reconcile_series(SeriesProjection((PlotSeries("a", [1, 2], [10, 100]),
            PlotSeries("b", [2, 1], [100, 10])), log_left=log))
        qapp.processEvents()
        curve = next(item for item in view.plot.getPlotItem().listDataItems() if item.name() == "a")
        _click(view, curve, 0, qapp)
        assert clicked[-1] == (0, 1), "cross-curve hit must report ambiguity rather than jump"
        view.reconcile_series(SeriesProjection())
        assert not _markers(view), "right markers excluded from autorange still retire on redraw"
    finally:
        view.close(); view.deleteLater()


@pytest.mark.parametrize("x,y,right,log", [(np.nan, 2, False, False),
    (1, np.nan, False, False), (1, 0, False, True), (1, -2, False, True),
    (1, 0, True, True), (1, -2, True, True)])
def test_combined_roi_finite_marker_policy(qapp, x, y, right, log):
    view = TrendView(series_only=True)
    try:
        view.reconcile_series(SeriesProjection((PlotSeries("value", [x], [y], right),),
            log_left=log, current_index=0))
        assert bool(_markers(view)) == bool(right and np.isfinite(x) and np.isfinite(y))
    finally:
        view.close(); view.deleteLater()


def test_combined_roi_legacy_wrapper_and_detached_owners(tmp_path, qapp):
    """C8/C9/T6: legacy wrapper shares controls; detached metadata never mounts image owners."""
    from xdart.gui.analysis.roi_select_dialog import RoiSelectDialog
    from xdart.gui.analysis.scan_plot_dialog import ScanPlotDialog
    from xdart.gui.tabs.scattering.roi_selection_view import RoiSelectionView
    from xdart.gui.tabs.scattering.analysis_mount import MetadataSource
    picker = RoiSelectDialog(np.ones((4, 4)))
    detached = ScanPlotDialog()
    try:
        assert len(picker.findChildren(RoiSelectionView)) == 1
        assert len(picker.findChildren(ScientificView)) == 1
        table = _roi_navigation_table(tmp_path)
        detached.set_metadata_rows(MetadataSource("selected", table, (7, 3)))
        assert not detached.findChildren(ScientificView)
        assert not detached.findChildren(RoiSelectionView)
        assert detached.roi_btn.isHidden()
        assert not _markers(detached.trend)
    finally:
        picker.close(); detached.close()
