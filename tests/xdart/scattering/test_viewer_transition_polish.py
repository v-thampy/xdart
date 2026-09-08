"""Real viewer replacement keeps scalar presentation, never old payloads."""

from threading import Event

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest

from test_viewer_1d_selection import viewer as viewer_1d, _ready, _mode
from test_viewer_2d_navigation import viewer as viewer_2d, _wait


class _HideEvents(QtCore.QObject):
    def __init__(self, widget):
        super().__init__(widget)
        self.events = []
        widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.Type.Hide:
            self.events.append(watched.objectName())
        return False


def test_new_hdf_file_has_no_teardown_or_zero_one_range(viewer_2d, monkeypatch):
    import xdart.gui.tabs.scattering.hydration_transport as transport

    page, app, paths, values = viewer_2d
    view = page._shell.scientific
    entered, release = Event(), Event()
    read = transport.read_viewer_2d_frame

    def gated_read(*args, **kwargs):
        entered.set()
        assert release.wait(6)
        return read(*args, **kwargs)

    monkeypatch.setattr(transport, "read_viewer_2d_frame", gated_read)
    hidden = _HideEvents(view.image_splitter)
    plot = view.raw.canvas.imageViewBox
    plot.setRange(xRange=(3, 10), yRange=(2, 8), padding=0)
    target = plot.targetRect()
    levels = tuple(view.raw.image.levels)
    ranges = []
    plot.sigRangeChanged.connect(lambda *_: ranges.append(tuple(plot.viewRange()[0])))
    image_id = id(view.raw.image)
    try:
        page._open_viewer_2d_path(str(paths[1]))
        _wait(page, app, entered.is_set)
        print("VIEWER_TRANSITION", dict(mode="2D", hidden=hidden.events,
              ranges=ranges, target=tuple(plot.targetRect().getRect()),
              levels=view.raw.image.levels))
        assert not hidden.events
        assert plot.targetRect() == target
        assert not any(np.allclose(bounds, (0, 1)) for bounds in ranges)
        assert tuple(view.raw.canvas.histogram.levels()) == levels
        assert view._viewer_2d_payload is None
        assert view.raw.image.image is None and view.raw.image.qimage is None
        assert view.raw.canvas.raw_image.size == 0
        assert view.viewer_loading_snapshot_visible
        assert view._viewer_loading_notice.text() == "Loading — previous view"
        assert 0 < view.viewer_loading_snapshot_pixels <= 2_000_000
        pixmap = view._viewer_loading_pixmap.pixmap()
        assert view.viewer_loading_snapshot_pixels == pixmap.width() * pixmap.height()
        assert view._viewer_loading_overlay.geometry() == QtCore.QRect(
            view.raw.canvas.mapTo(view, QtCore.QPoint()),
            view.raw.canvas.size(),
        )
        snapshot = view._viewer_loading_pixmap.pixmap().toImage()
        sample_x = (0, snapshot.width() // 2, snapshot.width() - 1)
        sample_y = (0, snapshot.height() // 2, snapshot.height() - 1)
        assert len({snapshot.pixelColor(x, y).rgba()
                    for x in sample_x for y in sample_y}) > 1
        release.set()
        _wait(page, app, lambda: page._context_controller.viewer_2d_frame is not None
              and view._viewer_2d_payload is page._context_controller.viewer_2d_frame.array)
        assert not view.viewer_loading_snapshot_visible
        assert view.viewer_loading_snapshot_pixels == 0
        assert id(view.raw.image) == image_id
        assert plot.targetRect() == target
        np.testing.assert_array_equal(page._context_controller.viewer_2d_frame.array, values[0])
    finally:
        release.set()
        _wait(page, app, lambda: not page._context_controller.viewer_2d_loading)


def test_new_xye_batch_retires_curves_without_hiding_panel(viewer_1d, monkeypatch):
    import xdart.gui.tabs.scattering.hydration_transport as transport

    app, page, paths = viewer_1d
    view = page._shell.scientific
    entered, release = Event(), Event()
    read = transport.begin_viewer_1d_read

    def gated_read(*args, **kwargs):
        entered.set()
        assert release.wait(6)
        return read(*args, **kwargs)

    monkeypatch.setattr(transport, "begin_viewer_1d_read", gated_read)
    bottom = view.vertical_splitter.widget(1)
    hidden = _HideEvents(bottom)
    curve_id = id(view.curve)
    plot = view.curve.getViewBox()
    plot.setRange(xRange=(0.2, 1.8), yRange=(10, 25), padding=0)
    target = plot.targetRect()
    try:
        page._open_viewer_1d_paths((paths[-1],))
        _wait(page, app, entered.is_set)
        print("VIEWER_TRANSITION", dict(mode="1D", hidden=hidden.events,
              visible=bottom.isVisible(), target=tuple(plot.targetRect().getRect())))
        assert not hidden.events and bottom.isVisible()
        assert not view.curve.listDataItems() and not view.trace_history_keys
        assert plot.targetRect() == target
        assert view.viewer_loading_snapshot_visible
        assert view._viewer_loading_notice.text() == "Loading — previous view"
        assert 0 < view.viewer_loading_snapshot_pixels <= 2_000_000
        release.set()
        _ready(app, page)
        assert id(view.curve) == curve_id and bottom.isVisible()
        assert len(view.curve.listDataItems()) == 1
        assert not view.viewer_loading_snapshot_visible
    finally:
        release.set()
        _ready(app, page)


def test_failed_xye_replacement_drops_pending_snapshot(viewer_1d, monkeypatch):
    import xdart.gui.tabs.scattering.hydration_transport as transport

    app, page, paths = viewer_1d
    view = page._shell.scientific
    entered, release = Event(), Event()
    read = transport.begin_viewer_1d_read

    def failed_read(*args, **kwargs):
        entered.set()
        assert release.wait(6)
        raise OSError("test viewer read failure")

    monkeypatch.setattr(transport, "begin_viewer_1d_read", failed_read)
    try:
        page._open_viewer_1d_paths((paths[-1],))
        _wait(page, app, entered.is_set)
        assert view.viewer_loading_snapshot_visible
        assert not view.curve.listDataItems() and not view.trace_history_keys
        release.set()
        _wait(page, app, lambda: not page._context_controller.viewer_1d_loading)
        assert not view.viewer_loading_snapshot_visible
        assert view.viewer_loading_snapshot_pixels == 0
    finally:
        release.set()
        page._ensure_timer()
        _wait(page, app, lambda: not page._context_controller.viewer_1d_loading)
        monkeypatch.setattr(transport, "begin_viewer_1d_read", read)


def test_one_d_actions_align_with_top_background_button(viewer_1d):
    app, page, _paths = viewer_1d
    view = page._shell.scientific
    app.processEvents()
    top = view.background.mapTo(view, QtCore.QPoint()).x()
    bottom = view.plot_mode.mapTo(view, QtCore.QPoint()).x()
    assert abs(top - bottom) <= 1


@pytest.mark.parametrize("mode", ["Single", "Overlay", "Waterfall"])
def test_one_d_manual_intensity_survives_frame_selection(viewer_1d, mode):
    app, page, _paths = viewer_1d
    _mode(app, page, mode)
    view = page._shell.scientific
    controls = view.viewer_intensity
    assert controls.isVisible() and controls.autoscale.isChecked()
    # Exercise the same commit signal as the slider/numeric editor; the widget
    # itself has separate real drag/editor tests.
    controls.sync((0, 100), (12, 18))
    controls.autoscale.setChecked(False)
    controls.rangeChanged.emit(*controls.values())
    current = page._context_controller.navigation.frames[-1]
    page._context_controller.select_viewer_1d(current, (current,))
    page._refresh_shell()
    app.processEvents()
    target = (view.waterfall.canvas.histogram.levels() if mode == "Waterfall"
              else view.curve.getViewBox().viewRange()[1])
    np.testing.assert_allclose(target, (12, 18))
    assert controls.values() == (12, 18)
    controls.autoscale.setChecked(True)
    app.processEvents()
    target = (view.waterfall.canvas.histogram.levels() if mode == "Waterfall"
              else view.curve.getViewBox().viewRange()[1])
    assert target[1] > 18


def test_hdf_manual_color_range_survives_next_frame(viewer_2d):
    page, app, _paths, _values = viewer_2d
    view = page._shell.scientific
    controls = view.viewer_intensity
    assert controls.isVisible() and controls.autoscale.isChecked()
    controls.sync((0, 1600), (100, 150))
    controls.autoscale.setChecked(False)
    controls.rangeChanged.emit(*controls.values())
    QtTest.QTest.mouseClick(view.next_frame, QtCore.Qt.LeftButton)
    _wait(page, app, lambda: page._context_controller.viewer_2d_frame is not None
          and page._context_controller.viewer_2d_frame.label == 1
          and view._viewer_2d_payload is page._context_controller.viewer_2d_frame.array)
    np.testing.assert_array_equal(view.raw.image.levels, (100, 150))
    np.testing.assert_array_equal(view.raw.canvas.histogram.levels(), (100, 150))
    controls.autoscale.setChecked(True)
    app.processEvents()
    assert view.raw.image.levels[1] > 150
