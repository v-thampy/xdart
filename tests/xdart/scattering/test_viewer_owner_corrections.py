"""Owner-reported viewer behavior through the real page and renderer."""

import numpy as np
import h5py
import pytest
from threading import Event
from matplotlib import colormaps
from pyqtgraph.Qt import QtCore, QtTest

from test_viewer_1d_selection import viewer as viewer_1d, _mode, _ready
from test_viewer_2d_navigation import viewer as viewer_2d, _wait
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind


@pytest.fixture
def processed_page(tmp_path, monkeypatch):
    from tests.xdart.scattering.test_e4_preview_transport import _write_processed
    from tests.xdart.scattering.test_p3_experiment_operation_composition import _page
    from test_browse_selected_slices import _wait as wait_browse, _ready as ready_browse, _close
    from pyqtgraph.Qt import QtWidgets
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    path, raw = _write_processed(tmp_path / "source", schema_version=3, labels=(1, 2, 3))
    with h5py.File(path, "a") as handle:
        handle.require_group("entry/instrument/monochromator").create_dataset("wavelength", data=1.)
    page, _ = _page(tmp_path, monkeypatch)
    page.resize(1400, 1000)
    page.show()
    controller = page._context_controller
    controller.begin_browse(str(path.resolve()))
    assert wait_browse(app, controller.poll_browse, page).status is BrowseLoadStatus.READY
    wait_browse(app, lambda: ready_browse(page, 1), page)
    try:
        yield app, page, path, raw
    finally:
        _close(page, app)


def test_cached_browse_keeps_wavelength_for_two_theta(processed_page):
    app, page, _, _ = processed_page
    from test_browse_selected_slices import _wait as wait_browse, _ready as ready_browse
    view = page._shell.scientific
    view.plot_axis.setCurrentIndex(view.plot_axis.findData("2theta"))
    wait_browse(app, lambda: ready_browse(page, 1), page)
    assert view.plot_axis.currentData() == "2theta"
    trace = page._last_scientific_projection.traces[0]
    assert trace.axis.unit == "2th_deg"
    np.testing.assert_allclose(trace.axis.values,
        2 * np.rad2deg(np.arcsin(np.array([.1, .2, .3]) / (4 * np.pi))))


def test_switch_processed_browse_to_2d_viewer_loads_current_source(processed_page):
    app, page, path, _ = processed_page
    page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PROCESSING_MODE, "2D Viewer"))
    _wait(page, app, lambda: page._context_controller.viewer_2d_frame is not None
          and page._shell.scientific._viewer_2d_payload is page._context_controller.viewer_2d_frame.array)
    assert page._context_controller.viewer_2d_context.original_path == str(path)
    assert "Loading" not in page._shell.scientific.status.text()
    np.testing.assert_array_equal(page._context_controller.viewer_2d_frame.array,
                                  np.arange(16).reshape(4, 4))


def test_xye_mode_with_selected_browse_enters_2d_through_mode_widget(processed_page):
    app, page, path, _ = processed_page
    page._shell.run_controls.modeCombo.setCurrentText("Int 1D (XYE)")
    page._shell.run_controls.modeCombo.setCurrentText("2D Viewer")
    _wait(page, app, lambda: page._context_controller.viewer_2d_frame is not None
          and page._shell.scientific._viewer_2d_payload is not None)
    assert page._context_controller.viewer_2d_context.original_path == str(path)


@pytest.mark.parametrize("select_in_viewer", (False, True))
def test_mode_widget_from_1d_viewer_keeps_selected_nexus(
        processed_page, select_in_viewer):
    app, page, path, _ = processed_page
    controller = page._context_controller
    page._shell.run_controls.modeCombo.setCurrentText("1D Viewer")
    app.processEvents()
    if select_in_viewer:
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SELECT_SCAN, str(path), path=("artifact",),
            artifacts=(str(path),)))
        _wait(page, app, lambda: bool(controller.viewer_1d_diagnostic))
    page._shell.run_controls.modeCombo.setCurrentText("2D Viewer")
    _wait(page, app, lambda: controller.viewer_2d_frame is not None
          and page._shell.scientific._viewer_2d_payload is not None)
    assert controller.viewer_2d_context.original_path == str(path)
    assert "Loading" not in page._shell.scientific.status.text()
    np.testing.assert_array_equal(controller.viewer_2d_frame.array,
                                  np.arange(16).reshape(4, 4))


def test_ready_processed_2d_viewer_qualifies_its_exact_nexus_revision(
        processed_page):
    app, page, path, _ = processed_page
    page._handle_shell_command(ShellCommand(
        ShellCommandKind.SET_PROCESSING_MODE, "2D Viewer",
    ))
    _wait(page, app, lambda: (
        page._context_controller.viewer_2d_frame is not None
        and page._shell.scientific._viewer_2d_payload
        is page._context_controller.viewer_2d_frame.array
    ))
    catalog = page._context_controller._runtime._viewer_2d_catalog
    assert catalog is not None
    assert catalog.primary_revision.canonical_path == str(path)
    assert page._qualify_external_nexus(validate_disk=False).target == str(path)
    assert page._qualify_external_nexus(validate_disk=True).target == str(path)

    path.write_bytes(b"replacement")
    qualification = page._qualify_external_nexus(validate_disk=True)
    assert qualification.target is None
    assert "changed after it was opened in 2D Viewer" in qualification.reason


def test_switch_refused_nexus_1d_viewer_to_2d_viewer_loads_current_source(
        processed_page):
    app, page, path, _ = processed_page
    page._open_viewer_1d_paths((str(path),), current_path=str(path))
    _wait(page, app, lambda: (
        page._context_controller.viewer_1d_context is not None
        and page._context_controller.viewer_1d_context.state.value == "empty"
        and page._context_controller.viewer_1d_diagnostic
    ))
    assert "viewer suffix" in page._context_controller.viewer_1d_diagnostic
    page._handle_shell_command(ShellCommand(
        ShellCommandKind.SET_PROCESSING_MODE, "2D Viewer",
    ))
    _wait(page, app, lambda: (
        page._context_controller.viewer_2d_frame is not None
        and page._shell.scientific._viewer_2d_payload
        is page._context_controller.viewer_2d_frame.array
    ))
    assert page._context_controller.viewer_2d_context.original_path == str(path)


@pytest.mark.parametrize("viewer_mode", ("1D Viewer", "2D Viewer"))
@pytest.mark.parametrize("integration_mode", ("Int 1D", "Int 2D"))
def test_return_from_viewer_reopens_same_processed_file(
        processed_page, viewer_mode, integration_mode):
    from test_browse_selected_slices import _wait as wait_browse, _ready as ready_browse
    app, page, path, _ = processed_page
    controller = page._context_controller
    def ready():
        if integration_mode == "Int 2D":
            return bool(ready_browse(page, 1))
        page._drain_executor()
        page._refresh_shell()
        projection = page._last_scientific_projection
        return (projection is not None and len(projection.traces) == 1
                and projection.traces[0].frame is controller.navigation.current
                and len(page._shell.scientific.curve.listDataItems()) == 1
                and "Loading" not in page._shell.scientific.status.text())
    page._handle_shell_command(ShellCommand(
        ShellCommandKind.SET_PROCESSING_MODE, viewer_mode))
    if viewer_mode == "1D Viewer":
        # Selecting the already shown NeXus in 1D Viewer reaches its real
        # unsupported-suffix refusal and releases the previous Browse owner.
        page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_SCAN,
            str(path), path=("artifact",), artifacts=(str(path),)))
        _wait(page, app, lambda: controller.viewer_1d_context is not None
              and controller.viewer_1d_context.state.value == "empty"
              and controller.viewer_1d_diagnostic)
    else:
        _wait(page, app, lambda: controller.viewer_2d_frame is not None
              and page._shell.scientific._viewer_2d_payload is not None)
    assert controller.browse_context is None
    page._handle_shell_command(ShellCommand(
        ShellCommandKind.SET_PROCESSING_MODE, integration_mode))
    wait_browse(app, ready, page)
    assert controller.browse_context.requested_path == str(path)
    assert not controller.viewer_1d_owned and not controller.viewer_2d_owned
    browser = page._shell.browser
    second = browser.frame_model.index(1, 0)
    QtTest.QTest.mouseClick(browser.frames.viewport(), QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier, browser.frames.visualRect(second).center())
    wait_browse(app, lambda: (controller.navigation.current.local_frame_label == 2
                            and ready()), page)
    assert "Loading" not in page._shell.scientific.status.text()


def test_uncached_browse_retains_only_capped_raster_until_current_is_ready(
        processed_page, monkeypatch):
    from xdart.gui.tabs.scattering.browse_1d_hydration import FrameViewReader
    from test_browse_selected_slices import _wait as wait_browse, _ready as ready_browse

    app, page, _, _ = processed_page
    view = page._shell.scientific
    controller = page._context_controller
    target = next(frame for frame in controller.navigation.frames
                  if frame is not controller.navigation.current)
    entered, release = Event(), Event()
    read = FrameViewReader.read_1d_rows

    def gated_read(*args, **kwargs):
        entered.set()
        assert release.wait(8)
        return read(*args, **kwargs)

    monkeypatch.setattr(FrameViewReader, "read_1d_rows", gated_read)
    try:
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SELECT_FRAME, frame=target, frames=(target,)))
        wait_browse(app, entered.is_set, page)
        assert view.viewer_loading_snapshot_visible
        assert 0 < view.viewer_loading_snapshot_pixels <= 2_000_000
        assert not view.curve.listDataItems()
        assert view.raw.image.image is None
        release.set()
        state = wait_browse(app, lambda: ready_browse(page, 1), page)
        assert state.heavy.frame is target
        assert not view.viewer_loading_snapshot_visible
        assert view.viewer_loading_snapshot_pixels == 0
    finally:
        release.set()


def test_rapid_raw_frame_selection_keeps_one_preview_and_adopts_latest(
        viewer_2d, monkeypatch):
    import xdart.gui.tabs.scattering.hydration_transport as transport

    page, app, _, values = viewer_2d
    view = page._shell.scientific
    browser = page._shell.browser
    entered, release = Event(), Event()
    read = transport.read_viewer_2d_frame

    class BlankRawPaints(QtCore.QObject):
        paints = 0
        total = 0

        def eventFilter(self, watched, event):
            if event.type() == QtCore.QEvent.Type.Paint:
                self.total += 1
            if (
                event.type() == QtCore.QEvent.Type.Paint
                and view.raw.isVisible()
                and view.raw.image.image is None
                and not view.viewer_loading_snapshot_visible
                ):
                self.paints += 1
            return False

    def gated_read(*args, **kwargs):
        entered.set()
        assert release.wait(8)
        return read(*args, **kwargs)

    monkeypatch.setattr(transport, "read_viewer_2d_frame", gated_read)
    frames = page._context_controller.navigation.frames
    blank_paints = BlankRawPaints(view.raw.canvas)
    view.raw.canvas.image_win.viewport().installEventFilter(blank_paints)
    try:
        browser.frames.setFocus()
        QtTest.QTest.keyClick(browser.frames, QtCore.Qt.Key_Down)
        _wait(page, app, entered.is_set)
        assert view.viewer_loading_snapshot_visible
        snapshot = view._viewer_loading_pixmap.pixmap().cacheKey()
        for key in (QtCore.Qt.Key_Down, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
            QtTest.QTest.keyClick(browser.frames, key)
            page._refresh_shell()
            assert view.viewer_loading_snapshot_visible
            assert view._viewer_loading_pixmap.pixmap().cacheKey() == snapshot
            assert view.raw.image.image is None
        release.set()
        _wait(page, app, lambda: page._context_controller.viewer_2d_frame is not None
              and page._context_controller.viewer_2d_frame.label == 2
              and view._viewer_2d_payload is page._context_controller.viewer_2d_frame.array)
        np.testing.assert_array_equal(view._viewer_2d_payload, values[2])
        assert not view.viewer_loading_snapshot_visible
        app.processEvents()
        assert blank_paints.total > 0
        assert blank_paints.paints == 0
    finally:
        release.set()


def test_top_log_changes_actual_1d_values_and_restores_linear(viewer_1d):
    app, page, _ = viewer_1d
    view = page._shell.scientific
    current = page._context_controller.navigation.current
    page._context_controller.select_viewer_1d(current, (current,))
    page._refresh_shell()
    before = [item.getData()[1].copy() for item in view.curve.listDataItems()]
    QtTest.QTest.mouseClick(view.log_scale, QtCore.Qt.LeftButton)
    app.processEvents()
    for prior, item in zip(before, view.curve.listDataItems(), strict=True):
        np.testing.assert_allclose(item.getData()[1], np.log10(prior))
    QtTest.QTest.mouseClick(view.log_scale, QtCore.Qt.LeftButton)
    app.processEvents()
    for prior, item in zip(before, view.curve.listDataItems(), strict=True):
        np.testing.assert_array_equal(item.getData()[1], prior)


def test_selected_colormap_colors_actual_1d_curves(viewer_1d):
    app, page, _ = viewer_1d
    view = page._shell.scientific
    items = tuple(view.curve.listDataItems())
    before = tuple(item.opts["pen"].color().getRgb() for item in items)
    view.color_map.setCurrentText("magma")
    app.processEvents()
    after = tuple(item.opts["pen"].color().getRgb() for item in items)
    assert after != before
    assert len(set(after)) == len(items)
    for color, point in zip(after, np.linspace(.15, .85, len(items)), strict=True):
        np.testing.assert_allclose(color, colormaps["magma"](point, bytes=True), atol=1)
    assert tuple(view.curve.listDataItems()) == items


def test_rounded_single_xye_displays_with_inherited_waterfall(viewer_1d, tmp_path):
    app, page, _ = viewer_1d
    _mode(app, page, "Waterfall")
    path = tmp_path / "iq_new_grid.xye"
    axis = np.linspace(.987654321, 8.3456789, 1000)
    values = np.column_stack((axis, np.arange(1000) + 1, np.ones(1000)))
    np.savetxt(path, values, fmt="%.9g")
    page._open_viewer_1d_paths((str(path),))
    _ready(app, page)
    view = page._shell.scientific
    assert "refused" not in view.status.text().lower()
    assert len(view.curve.listDataItems()) == 1
    np.testing.assert_array_equal(view.curve.listDataItems()[0].getData()[0], np.loadtxt(path)[:, 0])
    assert not view.bottom_waterfall_active


def test_viewer_frame_captions_are_positions_not_stored_ids(viewer_2d):
    page, app, _, _ = viewer_2d
    view = page._shell.scientific
    assert page._context_controller.navigation.current.local_frame_label == 0
    assert view.frame_selector.currentText() == "1"
    assert "frame 1" in view.title.text()
    QtTest.QTest.mouseClick(view.next_frame, QtCore.Qt.LeftButton)
    _wait(page, app, lambda: page._context_controller.viewer_2d_frame is not None
          and page._context_controller.viewer_2d_frame.label == 1
          and view._viewer_2d_payload is page._context_controller.viewer_2d_frame.array)
    assert view.frame_selector.currentText() == "2"
    assert "frame 2" in view.title.text()


def test_intensity_controls_above_status_with_actual_hover_and_entry(viewer_1d):
    app, page, _ = viewer_1d
    view = page._shell.scientific
    controls = view.viewer_intensity
    controls.sync((0, 100), (12, 18))
    assert "12" in controls.slider.toolTip() and "18" in controls.slider.toolTip()
    assert controls.mapTo(view, QtCore.QPoint()).y() < view.status.mapTo(view, QtCore.QPoint()).y()
    assert abs(controls.mapTo(view, controls.rect().center()).y()
               - view.plot_mode.mapTo(view, view.plot_mode.rect().center()).y()) <= 2
    QtTest.QTest.mouseDClick(controls.slider, QtCore.Qt.LeftButton)
    app.processEvents()
    assert controls._entry_popup.isVisible()
    low, high = controls._entry_editors()
    low.setText("13")
    high.setText("17")
    QtTest.QTest.keyClick(high, QtCore.Qt.Key_Return)
    app.processEvents()
    np.testing.assert_allclose(view.curve.getViewBox().viewRange()[1], (13, 17))
