"""Saved Stitch contributors are inspected through the real scientific and worker owners."""
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest

from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tabs.stitch.page import StitchPage
from xdart.gui.tools.stitch_values import StitchFrameSelector, prepare_stitch_tool
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.stitch_operation import run_stitch_operation
from xrd_tools.integrate.stitch_conditioning import StitchConditioning
from xrd_tools.io.analysis_artifact import read_analysis_artifact
from xrd_tools.io.frame_view import read_contributing_frame_view
from tests.xdart.pages.test_stitch_page import _Status, _wait, qapp  # noqa: F401
from tests.xdart.tools.test_stitch_native_values import _native_form


def _command(page, kind, value=None):
    page._view_command(ShellCommand(kind, value))


def _click(page, row, qapp, *, settle=True):
    item = page.frames.item(row)
    QtTest.QTest.mouseClick(page.frames.viewport(), QtCore.Qt.MouseButton.LeftButton,
                           QtCore.Qt.KeyboardModifier.NoModifier, page.frames.visualItemRect(item).center())
    if settle:
        _wait(page, qapp)


def _saved_page(tmp_path, qapp, backend="multigeometry", scans="1", mode="2d", *, conditioning=None,
                edge_pixels=None):
    form = replace(_native_form(tmp_path, backend, scans=scans), mode=mode, npt_1d=39,
                   npt_rad_2d=23, npt_azim_2d=19, q_range=(1., 5.2),
                   map_radial_range=(1., 5.2), chi_range=(-180., 180.),
                   conditioning=conditioning or StitchConditioning("manual", threshold_max=50000))
    if edge_pixels is not None:
        original = next(Path(form.image_dir).glob("*scan1_0000.raw"))
        pixels = np.fromfile(original, dtype=form.raw_dtype)
        pixels[:len(edge_pixels)] = edge_pixels
        pixels.tofile(original)
    page = StitchPage(empty_host_services(_Status()))
    page.resize(1400, 900)
    page.show()
    page.load_form(form)
    _wait(page, qapp)
    page.run_bar.startButton.click()
    _wait(page, qapp)
    target = page.browser.current_result()
    assert target and Path(target).is_file()
    page.open_result(target)  # the persisted file, independent of the live request
    _wait(page, qapp)
    return page, form, target


def _close(page, qapp):
    until = time.monotonic() + 5
    receipt = page.shutdown()
    while receipt.status is not PageCleanup.CLEAN and time.monotonic() < until:
        qapp.processEvents()
        time.sleep(.005)
        receipt = page.shutdown()
    assert receipt.status is PageCleanup.CLEAN
    page.deleteLater()
    qapp.processEvents()


@pytest.mark.parametrize("backend", ["multigeometry", "xu_hist"])
def test_reopened_stitch_frame_inspector_matches_single_frame_reduction(qapp, tmp_path, backend):
    page, form, target = _saved_page(tmp_path, qapp, backend)
    try:
        saved = read_analysis_artifact(target)
        assert page.frames.item(0).text() == "Full"
        np.testing.assert_array_equal(page._last_map_state.heavy.cake, saved.intensity.T)
        np.testing.assert_array_equal(page._last_map_state.traces[0].intensity, saved.profile.intensity)
        # An independent, actual one-frame operation at the captured pose and settings.
        reference = prepare_stitch_tool(replace(form, frame_selector=StitchFrameSelector(0, 0),
            output_path=tmp_path / "output" / "reference_stitch2d.nexus", output_family="reference"))
        result = run_stitch_operation(reference.request)
        assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
        raw = read_contributing_frame_view(target, 0, scan=saved.contributing_frames[0]["key"].split("/")[0][5:],
                                          artifact_payload=saved, full_raw=True).raw
        expected_raw = np.where(raw <= 50000, raw, np.nan)
        page.set_field("npt_1d", "7")  # editable intent cannot change inspection science
        page.set_field("npt_rad_2d", "9")
        _click(page, 1, qapp)
        state = page._last_map_state
        assert not page.scientific.raw.isHidden() and not page.scientific.cake.isHidden()
        assert len(state.traces) == len(page.curve.listDataItems()) == 1
        np.testing.assert_array_equal(state.heavy.raw, expected_raw)
        # The public artifact quantizes scientific intensities to float32.
        np.testing.assert_array_equal(state.heavy.cake.astype(np.float32), result.payload.intensity.T)
        np.testing.assert_array_equal(state.traces[0].intensity.astype(np.float32), result.payload.profile.intensity)
        assert page.browser.current_result() == target
        _click(page, 0, qapp)
        assert page.scientific.raw.isHidden()
        np.testing.assert_array_equal(page._last_map_state.heavy.cake, saved.intensity.T)
    finally:
        _close(page, qapp)


@pytest.mark.parametrize("backend", ["multigeometry", "xu_hist"])
@pytest.mark.parametrize("mode", ["manual", "auto", "off"])
def test_full_raw_preview_uses_saved_conditioning_without_changing_native_reader(qapp, tmp_path, backend, mode):
    edges = [-1, 0, 10, 50000, 50001, 1048574, 1048575, np.iinfo(np.int32).max]
    policy = (StitchConditioning("manual", 10, 50000) if mode == "manual" else StitchConditioning(mode))
    page, form, target = _saved_page(tmp_path, qapp, backend, mode="1d", conditioning=policy,
                                    edge_pixels=edges)
    try:
        saved = read_analysis_artifact(target)
        row = saved.contributing_frames[0]
        scan = row["key"].split("/")[0][5:]
        native = read_contributing_frame_view(target, 0, scan=scan, full_raw=True, artifact_payload=saved)
        original = next(Path(form.image_dir).glob("*scan1_0000.raw"))
        np.testing.assert_array_equal(native.raw.ravel()[:len(edges)], edges)
        thumbnail = native.thumbnail.copy()
        curve = page.curve.listDataItems()[0].yData.copy()
        page.set_field("threshold_min", "2")
        page.set_field("threshold", "3")  # current intent must not change a saved preview
        page._show_raw()
        _wait(page, qapp)
        page.scientific.raw_popup_full.click()
        _wait(page, qapp)
        actual = page._raw_frame_view.raw.ravel()[:len(edges)]
        finite = ([False, False, True, True, False, False, False, False] if mode == "manual" else
                  [False, True, True, True, True, True, False, False] if mode == "auto" else [True] * len(edges))
        np.testing.assert_array_equal(np.isfinite(actual), finite)
        np.testing.assert_array_equal(actual[finite], np.asarray(edges)[finite])
        np.testing.assert_array_equal(page._raw_frame_view.thumbnail, thumbnail)
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, curve)
        np.testing.assert_array_equal(np.fromfile(original, dtype=form.raw_dtype)[:len(edges)], edges)
        np.testing.assert_array_equal(read_analysis_artifact(target).intensity, saved.intensity)
        assert "saved" in page.scientific.raw_popup_full.toolTip().lower()
    finally:
        _close(page, qapp)


def test_full_restores_file_overlay_options_and_pins(qapp, tmp_path):
    page, _form, _target = _saved_page(tmp_path, qapp, scans="1,2,[1,2]")
    try:
        _command(page, ShellCommandKind.SET_PLOT_MODE, "Overlay")
        first = page.browser.listed_results()[0]
        page.results.setCurrentRow(next(row for row in range(page.results.count())
            if page.results.item(row).data(QtCore.Qt.ItemDataRole.UserRole) == first),
            QtCore.QItemSelectionModel.SelectionFlag.Select)
        _wait(page, qapp)
        _command(page, ShellCommandKind.SET_SLICE_CENTER, -120.)
        _command(page, ShellCommandKind.SET_SLICE_WIDTH, 20.)
        _command(page, ShellCommandKind.SET_SLICE_ENABLED, True)
        _command(page, ShellCommandKind.PIN_SLICE)
        selected, preferences = page.browser.selected_results(), page._view
        before = page._last_map_state
        assert page.frames.item(0).text() == "Full" and before.pinned_traces
        _click(page, 1, qapp)
        assert page._last_map_state.plot_mode == "Single" and not page._last_map_state.pinned_traces
        assert len(page._last_map_state.traces) == 1
        _command(page, ShellCommandKind.SET_SLICE_ENABLED, False)
        _click(page, 0, qapp)
        assert page._view is preferences and page.browser.selected_results() == selected
        after = page._last_map_state
        assert after.plot_options is before.plot_options and after.slice_center == before.slice_center
        assert after.pinned_traces[0].pin is before.pinned_traces[0].pin
        for old, new in zip(before.traces, after.traces, strict=True):
            assert old.frame is new.frame
            np.testing.assert_array_equal(old.intensity, new.intensity)
    finally:
        _close(page, qapp)


def test_directory_cursor_keeps_file_selection_without_active_scientific_navigation(qapp, tmp_path):
    page, _form, target = _saved_page(tmp_path, qapp, scans="1,2,[1,2]")
    try:
        _command(page, ShellCommandKind.SET_PLOT_MODE, "Overlay")
        selected = page.browser.selected_results()
        assert selected and page.results.item(0).text().startswith("..")
        page.results.setCurrentRow(0, QtCore.QItemSelectionModel.SelectionFlag.Select)
        _wait(page, qapp)
        assert page.browser.current_result() is None
        assert page.browser.selected_results() == selected
        _keys, navigation = page._navigation()
        assert navigation.current is None and navigation.selected == ()
        assert page._last_map_state.traces == ()
        page.browser.set_current_result(target, keep_selection=True)
        _wait(page, qapp)
        assert page.browser.selected_results() == selected
        np.testing.assert_array_equal(page._last_map_state.traces[0].intensity,
                                      read_analysis_artifact(target).profile.intensity)
    finally:
        _close(page, qapp)


@pytest.mark.parametrize("change", ["missing", "changed", "calibration"])
def test_frame_inspector_refuses_missing_or_changed_originals(qapp, tmp_path, change):
    page, form, target = _saved_page(tmp_path, qapp)
    try:
        raw = next(Path(form.image_dir).glob("*scan1_0000.raw"))
        original = Path(form.geometry_path) if change == "calibration" else raw
        if change == "missing":
            original.unlink()
        else:
            original.write_bytes(original.read_bytes() + b" ")
        assert page.frames.item(0).text() == "Full"
        _click(page, 1, qapp)
        status = page.run_bar.readinessLabel.full_text().lower()
        assert any(reason in status for reason in ("unavailable", "changed", "does not match")), status
        assert not page.run_bar.readinessDot.property("ready")
        assert page._last_map_state.heavy is None and page._last_map_state.traces == ()
        assert page.curve.listDataItems() == [] and page.scientific.cake.canvas.raw_image.size == 0
        _click(page, 0, qapp)
        np.testing.assert_array_equal(page._last_map_state.heavy.cake, read_analysis_artifact(target).intensity.T)
    finally:
        _close(page, qapp)


def test_frame_inspector_cancels_rapid_frame_file_and_close(qapp, tmp_path):
    page, _form, _target = _saved_page(tmp_path, qapp, scans="1,2,[1,2]")
    try:
        assert page.frames.item(0).text() == "Full"
        _click(page, 1, qapp, settle=False)
        qapp.processEvents()
        _click(page, 2, qapp, settle=False)
        other = next(path for path in page.browser.listed_results() if path != page.browser.current_result())
        page.results.setCurrentRow(next(row for row in range(page.results.count())
                                        if page.results.item(row).data(QtCore.Qt.ItemDataRole.UserRole) == other))
        _wait(page, qapp)
        assert page.frames.currentRow() == 0 and page.frames.item(0).text() == "Full"
        assert page._last_map_state.heavy.raw is None
        assert all(trace.frame is page._result_keys[page.browser.current_result()]
                   for trace in page._last_map_state.traces)
        _click(page, 1, qapp, settle=False)
        qapp.processEvents()
    finally:
        _close(page, qapp)


def test_xu_stitch_1d_keeps_full_and_original_raw_without_inventing_a_cake(qapp, tmp_path):
    page, _form, target = _saved_page(tmp_path, qapp, "xu_hist", mode="1d")
    try:
        saved = read_analysis_artifact(target)
        _click(page, 1, qapp)
        assert "no frozen 2D settings" in page.run_bar.readinessLabel.full_text()
        assert page._member_preferences is None and page._last_map_state is None
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, saved.intensity)
        page._show_raw()
        _wait(page, qapp)
        page.scientific.raw_popup_full.click()
        _wait(page, qapp)
        assert page._raw_frame_view.raw is not None
        _click(page, 0, qapp)
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, saved.intensity)
    finally:
        _close(page, qapp)
