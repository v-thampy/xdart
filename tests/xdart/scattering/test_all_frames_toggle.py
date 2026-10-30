"""ALL-FRAMES-128: Int's Show All becomes the persistent "All frames" toggle (frozen oracle, Claude, 2026-10-29).

The maintainer's scenario: several samples cycled for hours, one output each, seen one result at a time in
Overlay/Waterfall without clicking Show All after every switch. His words: "making the show all button a persistent
toggle"; then "overlay/waterfall mode should alway overlay. even when all frames is toggled". All frames sets which of
the shown result's frames are drawn (every one) and resets nothing: earlier results stay overlaid until Clear.

Real pages and real processed outputs. The toggle is the real BrowserView button, frames are real list clicks
and the next result is a real Down-arrow key on the results list. The session row goes through the application's own
close and restart.
"""
from __future__ import annotations

import os
import shutil

import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from tests.xdart.scattering.test_e4_accumulation_selection_parity import _click_frame
from tests.xdart.scattering.test_e4_preview_transport import _write_processed
from tests.xdart.scattering.test_p3_experiment_operation_composition import _page
from tests.xdart.scattering.test_p34_reintegrate_operation import _wait
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent

LEFT = QtCore.Qt.MouseButton.LeftButton
CTRL = QtCore.Qt.KeyboardModifier.ControlModifier
SHIFT = QtCore.Qt.KeyboardModifier.ShiftModifier


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def two_results(tmp_path):
    """Two three-frame outputs in one folder, as two samples' files sit side by side."""
    first, _ = _write_processed(tmp_path / "a", labels=(1, 2, 3))
    other, _ = _write_processed(tmp_path / "b", labels=(1, 2, 3))
    second = first.parent / "scan_b.nexus"
    shutil.copyfile(other, second)
    return first, second


@pytest.fixture
def workspace(qapp, tmp_path, monkeypatch):
    page, _ = _page(tmp_path, monkeypatch, store=RunIntentStore(RunIntent(
        project_root=str(tmp_path), processing_mode="Int 1D")))
    page.resize(1200, 800)
    page.show()
    try:
        yield page
    finally:
        _wait(lambda: True if page.close_workspace().cleanup_status is CleanupStatus.CLEANED else None)
        page.deleteLater()
        qapp.processEvents()


def _idle(page):
    return (not page._context_controller.browse_pending and not page._scientific_repaint_pending
            and not page._shell.browser.frame_selection_pending)


def _settle(page):
    _wait(lambda: True if _idle(page) else None)


def _open(page, path):
    """Open ``path`` as a click on its row does, and wait until it is painted."""
    page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_SCAN, str(path), path=("artifact",)))
    _shown(page, path)


def _shown(page, path):
    controller, view = page._context_controller, page._shell.scientific

    def painted():
        frame = controller.navigation.current
        state = page._last_scientific_projection
        return True if (frame is not None and frame.artifact == str(path) and _idle(page)
                        and state is not None and state.traces and frame in view.trace_history_keys) else None
    _wait(painted)


def _toggle(page):
    QtTest.QTest.mouseClick(page._shell.browser.show_all, LEFT)
    QtWidgets.QApplication.processEvents()
    _settle(page)


def _mode(page, mode):
    page._shell.scientific.plot_mode.setCurrentText(mode)
    QtWidgets.QApplication.processEvents()
    _settle(page)


def _on(page):
    """The toggle's state, from the button and its owner alike."""
    button = page._shell.browser.show_all.isChecked()
    owner = getattr(page._processed_browser, "all_frames", None)
    assert button is owner, (button, owner)
    return button


def _all_selected(page):
    navigation = page._context_controller.navigation
    return bool(navigation.frames) and {id(f) for f in navigation.selected} == {id(f) for f in navigation.frames}


def _rows(page):
    return [(os.path.basename(str(frame.artifact)), frame.local_frame_label)
            for frame in page._shell.scientific.trace_history_keys]


A = [("scan.nexus", label) for label in (1, 2, 3)]
B = [("scan_b.nexus", label) for label in (1, 2, 3)]


# AF1, AF2: every frame of each result opened, earlier results kept; the next result is one Down-arrow key ----------

@pytest.mark.parametrize("plot_mode", ("Overlay", "Waterfall"))
def test_af1_af2_each_result_opened_shows_all_its_frames_and_keeps_the_earlier(workspace, two_results, plot_mode):
    page, (first, second) = workspace, two_results
    _mode(page, plot_mode)
    _toggle(page)
    assert _on(page) is True
    page._set_browser_directory(str(first.parent), explicit=True)
    _open(page, first)
    assert _all_selected(page) and _rows(page) == A
    scans = page._shell.browser.scans
    _wait(lambda: True if any(scans.item(row).data(QtCore.Qt.ItemDataRole.UserRole) == str(second)
                              for row in range(scans.count())) else None)
    scans.setFocus()
    QtTest.QTest.keyClick(scans, QtCore.Qt.Key.Key_Down)
    _shown(page, second)
    assert _all_selected(page), "the next result shows every one of its frames"
    assert _rows(page) == A + B, "Overlay and Waterfall keep the earlier result's curves"
    assert _on(page) is True


# AF3: off again, the next result is today's: its first frame, the earlier result kept (preservation on the parent) --

def test_af3_turning_it_off_gives_back_the_one_frame_open(workspace, two_results):
    page, (first, second) = workspace, two_results
    _mode(page, "Overlay")
    _toggle(page)
    _open(page, first)
    _toggle(page)
    assert not page._shell.browser.show_all.isChecked()
    _open(page, second)
    assert [frame.local_frame_label for frame in page._context_controller.navigation.selected] == [1]
    assert _rows(page) == A + B[:1]


# AF4: turning it on keeps the other results' curves and adds every frame of the shown one ---------------------------

def test_af4_turning_it_on_adds_the_shown_results_frames_to_the_overlay(workspace, two_results):
    page, (first, second) = workspace, two_results
    _mode(page, "Overlay")
    _open(page, first)
    _open(page, second)
    assert _rows(page) == A[:1] + B[:1]
    _toggle(page)
    assert _on(page) is True
    assert _all_selected(page) and _rows(page) == A[:1] + B


# AF5: a plain click keeps it on; leaving a frame out (Ctrl-click, Shift-click) turns it off --------------------------

def test_af5_leaving_a_frame_out_turns_it_off_and_a_plain_click_does_not(qapp, workspace, two_results):
    page, (first, _second) = workspace, two_results
    shell = page._shell
    _mode(page, "Overlay")
    _toggle(page)
    _open(page, first)
    _click_frame(qapp, shell, 1)
    _settle(page)
    assert _on(page) is True and _all_selected(page), "a plain click visits a frame and keeps every frame drawn"
    _click_frame(qapp, shell, 2, modifiers=CTRL)
    _settle(page)
    assert _on(page) is False and not _all_selected(page), "a Ctrl-click leaves a frame out"
    _toggle(page)
    assert _on(page) is True and _all_selected(page)
    _click_frame(qapp, shell, 1)
    _click_frame(qapp, shell, 2, modifiers=SHIFT)
    _settle(page)
    assert _on(page) is False and not _all_selected(page), "a Shift-click leaves a range out"


# AF6: Clear starts a fresh overlay of the shown result's frames, all of them, and the toggle stays on ----------------

def test_af6_clear_keeps_every_frame_of_the_shown_result_and_the_toggle(workspace, two_results):
    page, (first, second) = workspace, two_results
    _mode(page, "Overlay")
    _toggle(page)
    _open(page, first)
    _open(page, second)
    assert _rows(page) == A + B
    QtTest.QTest.mouseClick(page._shell.scientific.clear, LEFT)
    QtWidgets.QApplication.processEvents()
    _settle(page)
    assert _on(page) is True
    assert _all_selected(page) and _rows(page) == B


# AF7, AF8: Single (one frame), Average and Sum (Browse draws nothing there) grey it out and keep its check; back in
# Overlay every frame shows again -----------------------------------------------------------------------------------

@pytest.mark.parametrize("plot_mode", ("Single", "Average", "Sum"))
def test_af7_af8_other_plot_modes_grey_it_out_and_keep_the_check(workspace, two_results, plot_mode):
    page, (first, _second) = workspace, two_results
    button = page._shell.browser.show_all
    _mode(page, "Overlay")
    _toggle(page)
    _open(page, first)
    _mode(page, plot_mode)
    assert not button.isEnabled() and button.isChecked() and _on(page) is True
    if plot_mode == "Single":
        assert len(page._context_controller.navigation.selected) == 1
    _mode(page, "Overlay")
    assert button.isEnabled() and _all_selected(page)


# AF9: remembered between sessions through the application's own close and restart; a fresh session neither ---------

def test_af9_remembered_between_sessions_and_never_in_a_fresh_one(qapp, tmp_path, monkeypatch):
    from xdart._gui_main import Main
    from xdart.utils.session import load_session

    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    monkeypatch.setenv("XDART_PHASE_LIBRARY", str(tmp_path / "phases"))
    monkeypatch.delenv("XDART_SESSION_FRESH", raising=False)

    def started():
        window = Main()
        return window, window.page_handle.widget

    def closed(window):
        window.close()
        window.deleteLater()
        qapp.processEvents()

    window, page = started()
    try:
        assert not page._shell.browser.show_all.isChecked(), "off until it is turned on"
        _toggle(page)
        assert _on(page) is True
    finally:
        closed(window)
    assert load_session()["scattering_browser"] == {"all_frames": True}
    window, page = started()
    try:
        assert _on(page) is True, "the next session starts with it on"
    finally:
        closed(window)
    saved = (tmp_path / "session.json").read_bytes()
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    window, page = started()
    try:
        assert _on(page) is False, "a fresh session reads nothing"
        _toggle(page)
    finally:
        closed(window)
    assert (tmp_path / "session.json").read_bytes() == saved, "and writes nothing"


# AF10: the button says what it does --------------------------------------------------------------------------------

def test_af10_the_button_is_a_checkable_all_frames_toggle_with_its_tooltip(workspace):
    button = workspace._shell.browser.show_all
    assert button.text() == "All frames" and button.isCheckable()
    tip = button.toolTip()
    for phrase in ("Overlay or Waterfall", "every one of its frames", "until Clear", "Up and Down arrow keys",
                   "Ctrl-click", "Greyed out in Single, Average and Sum", "Remembered between sessions"):
        assert phrase in tip, (phrase, tip)
