"""Stitch's Data Browser is Int's own (AUD-9 stage 2b G3b; hard rule 14; Codex's Stitch/Int reuse audit R1):
the page mounts ``BrowserView`` with a results profile (several results for an overlay, each result's
contributing frames, Open… · Metadata · Raw image) instead of its copied menus, heading and two lists. Each row
keeps one item of Codex's preservation list, now through the owner."""
import json
from pathlib import Path

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup, SCATTERING_PAGE_KEY
from xdart.gui.tabs.scattering.browser_view import BrowserView
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tabs.stitch.page import StitchPage
from tests.xdart.pages.test_stitch_page import _Status, _wait, qapp  # noqa: F401
from tests.xdart.tools.test_stitch_native_values import _native_form

_ROLE = QtCore.Qt.ItemDataRole.UserRole


def _close(page, qapp):
    assert page.shutdown().status is PageCleanup.CLEAN
    page.deleteLater()
    qapp.processEvents()


def test_stitch_mounts_ints_data_browser_with_its_menus_lists_and_actions(qapp, monkeypatch):
    page = StitchPage(empty_host_services(_Status()))
    tools, opened = [], []
    page.toolRequested.connect(tools.append)
    monkeypatch.setattr(page, "open_result", lambda path=None: opened.append(path))
    try:
        assert type(page.browser) is BrowserView
        profile = page.browser.profile
        assert profile.results and profile.result_selection == "extended" and profile.result_frames
        assert page.results is page.browser.scans and page.frames is page.browser.result_frames
        for name in ("stitchResults", "stitchFrames", "browserHeading"):  # the copies are gone
            assert page.findChild(QtWidgets.QWidget, name) is None
        assert [action.text() for action in page.browser.menus["File"].actions()] == [
            "Open result…", "Integration workspace"]
        hosts = page.mount_points()
        assert hosts.config_menu is page.browser.menus["Config"] and hosts.help_menu is page.browser.menus["Help"]
        page.browser.menus["File"].actions()[1].trigger()
        page.browser.menus["File"].actions()[0].trigger()
        page.browser.open_result.click()
        assert tools == [str(SCATTERING_PAGE_KEY)] and opened == [None, None]
        assert (page.open_button, page.metadata_button, page.raw_button) == (
            page.browser.open_result, page.browser.metadata, page.browser.raw_image)
        assert page.browser.show_all.isHidden() and page.browser.auto_last.isHidden()
        assert not page.metadata_button.isEnabled() and not page.raw_button.isEnabled()  # no result yet
    finally:
        _close(page, qapp)


def test_stitch_frames_keep_identities_grouped_labels_and_excluded_marks_in_ints_browser(qapp, tmp_path):
    from tests.core.test_analysis_stitch_auxiliary import _publish
    from tests.core.test_frame_screen import test_grouped_scans_are_screened_only_against_themselves
    path, _raw, _project, _receipt = _publish(tmp_path)
    screened = tmp_path / "screened"
    screened.mkdir()
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.open_result(path)
        _wait(page, qapp)
        frames = page.browser.result_frames
        assert [frames.item(i).text() for i in range(frames.count())] == ["2-0", "2-1", "3-0"]
        identities = [tuple(json.loads(frames.item(i).data(_ROLE))) for i in range(frames.count())]
        assert [frame for _scan, frame in identities] == [0, 1, 0]
        assert identities[0][0] == identities[1][0] != identities[2][0]  # two scans' own storage groups
        assert page.raw_button.isEnabled() and page.browser.current_result_frame_label() == "2-0"
        page._show_raw()
        _wait(page, qapp)
        page.frames.setCurrentRow(2)  # a user choosing a frame: the raw read follows it
        assert page.browser.current_result_frame() == frames.item(2).data(_ROLE)
        assert page.scientific.raw_popup_status.text() == "Loading selected frame…"
        _wait(page, qapp)
        assert "3-0" in page.scientific.raw_popup_status.text() and page.browser.current_result_frame_label() == "3-0"
        page.scientific.raw_popup_dialog.close()
        page.open_result(test_grouped_scans_are_screened_only_against_themselves(screened))
        _wait(page, qapp)
        items = [frames.item(i) for i in range(frames.count())]
        assert [item.text() for item in items][-2:] == ["dim-3", "dim-4 · excluded"]
        assert items[-1].foreground().color() != items[-2].foreground().color()  # muted
        assert "Frame check: empty, excluded from the result" in items[-1].toolTip()
    finally:
        _close(page, qapp)


@pytest.fixture
def no_recent(monkeypatch):
    from xdart.gui.tabs.stitch import page as module
    monkeypatch.setattr(module, "_RECENT_BYTES", 0)


def test_stitch_overlay_selection_stepping_and_read_back_live_in_ints_browser(qapp, tmp_path, no_recent):
    page = StitchPage(empty_host_services(_Status()))
    select = QtCore.QItemSelectionModel.SelectionFlag
    try:
        page.load_form(_native_form(tmp_path, "multigeometry"))  # three outputs
        _wait(page, qapp)
        page.run_bar.startButton.click()
        _wait(page, qapp)
        paths = [page.results.item(row).data(_ROLE) for row in range(page.results.count())]
        assert len(paths) == 3 and page.browser.current_result() == paths[-1]
        page._view_command(ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Overlay"))
        page.results.setCurrentRow(2, select.ClearAndSelect)
        page.results.setCurrentRow(0, select.Select)  # a user adding a result to the overlay
        _wait(page, qapp)  # the one Single let go is read back, the selection kept
        assert page.browser.current_result() == paths[0]
        assert set(page.browser.selected_results()) == {paths[0], paths[2]}
        assert set(page._payloads) == {paths[0], paths[2]} and len(page.curve.listDataItems()) == 2
        page.scientific.next_frame.click()  # Overlay visits add the current curve, as Int's do
        _wait(page, qapp)
        assert page.browser.current_result() == paths[1]
        assert set(page.browser.selected_results()) == set(paths)
        assert set(page._payloads) == set(paths) and len(page.curve.listDataItems()) == 3
        page._view_command(ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Single"))
        _wait(page, qapp)  # Single: the current result alone, read back if it was let go
        assert page.browser.selected_results() == (paths[1],) and page.browser.current_result() == paths[1]
        assert set(page._payloads) == {paths[1]} and len(page.curve.listDataItems()) == 1
        page.clear_plot()
        assert page.browser.current_result() is None and page.browser.selected_results() == ()
        assert page.frames.count() == 0 and not page.raw_button.isEnabled()
    finally:
        _close(page, qapp)


def test_a_contributing_frame_opens_in_the_views_own_raw_popup(qapp, tmp_path):
    """Codex's Stitch/Int reuse audit R2a: the browser's Raw image opens the scientific view's own raw pop-up,
    not a page-built dialog of its own. The frame stays through the page's own redraws, Full Raw reads it
    again, and closing it releases the page's frame."""
    from tests.core.test_analysis_stitch_auxiliary import _publish
    path, raw, _project, _receipt = _publish(tmp_path)
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.open_result(path)
        _wait(page, qapp)
        page.raw_button.click()
        _wait(page, qapp)
        view = page.scientific
        assert view.raw_popup_shown() and view.raw_popup_dialog.windowTitle() == "Stitch contributing frame"
        assert view.raw_popup_image.image.image is not None
        assert view.raw_popup_status.text() == "2-0 · Thumbnail"
        page._draw()  # the page's own redraw reconciles the view (Int 1D, no heavy frame)
        assert view.raw_popup_image.image.image is not None
        view.raw_popup_full.click()
        _wait(page, qapp)
        assert view.raw_popup_status.text() == "2-0 · Full Raw"
        np.testing.assert_array_equal(page._raw_frame_view.raw, raw)
        view.raw_popup_dialog.close()
        qapp.processEvents()
        assert page._raw_frame_view is None and view.raw_popup_image.canvas.raw_image.size == 0
    finally:
        _close(page, qapp)
