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
from pathlib import Path
import shutil
from threading import Event
import time

import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from tests.xdart.scattering.test_e4_accumulation_selection_parity import _click_frame
from tests.xdart.scattering.test_browse_latest_selection import burst
from tests.xdart.scattering.test_e4_preview_transport import _write_processed
from tests.xdart.scattering.test_p3_experiment_operation_composition import _page
from tests.xdart.scattering.test_p34_reintegrate_operation import _wait
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.browse_1d_hydration import Browse1DHydrationLane
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
    """Two three-frame outputs in one folder, as two samples' files sit side by side. The second carries Int 1D's
    output slot (``_int1d``), as a real output does, so the results list's Auto filter lists it (declared oracle repair,
    ALL-FRAMES-128 implementation: Auto hid ``scan_b.nexus`` and AF1/AF2 waited for its row)."""
    first, _ = _write_processed(tmp_path / "a", labels=(1, 2, 3))
    other, _ = _write_processed(tmp_path / "b", labels=(1, 2, 3))
    second = first.parent / "scan_b_int1d.nexus"
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
B = [("scan_b_int1d.nexus", label) for label in (1, 2, 3)]


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
    assert page._processed_browser.auto_last is False, "earlier-frame visits still disable Auto Last"
    shell.scientific.frame_selector.setCurrentIndex(2)
    _settle(page)
    assert _on(page) is True and _all_selected(page), "the footer only moves the current frame"
    shell.browser.frames.setFocus()
    QtTest.QTest.keyClick(shell.browser.frames, QtCore.Qt.Key.Key_A, CTRL)
    _settle(page)
    assert _on(page) is True and _all_selected(page)
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


# AF7 (his live check, 2026-10-30: "previously even in single mode, when show all was clicked, it selected all the
# frames and showed the frames together"; his choice "Same as Overlay (Recommended)"): in Single too, every frame of
# each result opened is drawn; a plain click shows one frame and turns it off. The first assertions hold on the
# parent before ALL-FRAMES-128, where Show All drew every frame in Single --------------------------------------------

def test_af7_single_draws_every_frame_and_a_plain_click_turns_it_off(qapp, workspace, two_results):
    page, (first, second) = workspace, two_results
    shell = page._shell
    _mode(page, "Single")
    page._set_browser_directory(str(first.parent), explicit=True)
    _open(page, first)
    assert len(page._context_controller.navigation.selected) == 1
    _toggle(page)
    assert _all_selected(page) and _rows(page) == A, "as Show All did in Single, every frame is drawn"
    assert _on(page) is True
    scans = shell.browser.scans
    _wait(lambda: True if any(scans.item(row).data(QtCore.Qt.ItemDataRole.UserRole) == str(second)
                              for row in range(scans.count())) else None)
    scans.setFocus()
    QtTest.QTest.keyClick(scans, QtCore.Qt.Key.Key_Down)
    _shown(page, second)
    assert _all_selected(page) and _rows(page) == B, "the next result draws every one of its frames"
    _click_frame(qapp, shell, 1)
    _settle(page)
    assert _on(page) is False, "a plain click in Single leaves the other frames out"
    assert [frame.local_frame_label for frame in page._context_controller.navigation.selected] == [2]
    assert _rows(page) == B[1:2]
    _toggle(page)
    _mode(page, "Overlay")
    _mode(page, "Single")
    assert _on(page) is True and _all_selected(page), "entering Single with it on keeps every frame"


# AF8: Average and Sum (Browse draws nothing there) grey it out and keep its check; back in Overlay every frame shows
# again -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("plot_mode", ("Average", "Sum"))
def test_af8_average_and_sum_grey_it_out_and_keep_the_check(workspace, two_results, plot_mode):
    page, (first, _second) = workspace, two_results
    button = page._shell.browser.show_all
    _mode(page, "Overlay")
    _toggle(page)
    _open(page, first)
    _mode(page, plot_mode)
    assert not button.isEnabled() and button.isChecked() and _on(page) is True
    QtTest.QTest.mouseClick(page._shell.scientific.clear, LEFT)
    QtWidgets.QApplication.processEvents()
    _settle(page)
    assert _on(page) is True and len(page._context_controller.navigation.selected) == 1
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
        _mode(page, "Overlay")
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
        _mode(page, "Overlay")
        _toggle(page)
    finally:
        closed(window)
    assert (tmp_path / "session.json").read_bytes() == saved, "and writes nothing"


# AF10: the button says what it does --------------------------------------------------------------------------------

def test_af10_the_button_is_a_checkable_all_frames_toggle_with_its_tooltip(workspace):
    button = workspace._shell.browser.show_all
    assert button.text() == "All frames" and button.isCheckable()
    tip = button.toolTip()
    for phrase in ("Single, Overlay or Waterfall", "every one of its frames", "until Clear", "Up and Down arrow keys",
                   "Ctrl-click", "in Single a plain click", "Greyed out in Average and Sum",
                   "Remembered between sessions"):
        assert phrase in tip, (phrase, tip)


def test_af11_same_result_reselect_uses_the_existing_context_and_all_its_frames(workspace, two_results):
    page, (first, _second) = workspace, two_results
    _mode(page, "Overlay")
    _toggle(page)
    _open(page, first)
    context = page._context_controller.browse_context
    _open(page, first)
    assert page._context_controller.browse_context is context
    assert _all_selected(page) and _rows(page) == A
    assert _on(page) is True


@pytest.mark.parametrize("plot_mode", ("Overlay", "Waterfall"))
def test_af16_same_result_reselect_supersedes_a_pending_frame_removal(
    qapp, workspace, two_results, plot_mode,
):
    page, (first, _second) = workspace, two_results
    _mode(page, plot_mode)
    _toggle(page)
    _open(page, first)
    context = page._context_controller.browse_context
    _click_frame(qapp, page._shell, 2, modifiers=CTRL, settle=False)
    assert page._shell.browser.frame_selection_pending
    page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_SCAN, str(first), path=("artifact",)))
    assert not page._shell.browser.frame_selection_pending, "the later accepted result choice supersedes the gesture"
    _shown(page, first)
    assert page._context_controller.browse_context is context
    assert _on(page) is True and _all_selected(page)
    assert _rows(page) == A


@pytest.mark.parametrize("toggle_state", ("enabled", "disabled", "enabled_while_pending"))
def test_af12_latest_request_uses_the_current_toggle_at_adoption(burst, toggle_state):
    (page, loader, original, paths, select, entered, release_read,
     closing, release_close, opened, closed) = burst
    _mode(page, "Overlay")
    if toggle_state != "enabled_while_pending":
        _toggle(page)
    active = select(paths[0])
    assert entered.wait(5)
    skipped = select(paths[1])
    latest = select(paths[2])
    assert not loader.owns_request(skipped) and loader.owns_request(latest)
    page._handle_shell_command(ShellCommand(
        ShellCommandKind.SELECT_SCAN, original.requested_path, path=("artifact",)))
    assert page._context_controller._browse_request is latest, "the held old target cannot erase newer intent"
    if toggle_state != "enabled":
        QtTest.QTest.mouseClick(page._shell.browser.show_all, LEFT)
        QtWidgets.QApplication.processEvents()
    release_read.set()
    assert closing.wait(5)
    newest = select(paths[3])
    assert newest is not latest and newest is not active
    release_close.set()
    _shown(page, paths[3])
    assert opened == closed == [paths[0], paths[3]], "replaced files are never adopted"
    navigation = page._context_controller.navigation
    expected = [2] if toggle_state == "disabled" else [2, 5, 9]
    assert [frame.local_frame_label for frame in navigation.selected] == expected
    assert _on(page) is (toggle_state != "disabled")


@pytest.mark.parametrize("action", ("replay", "replace", "refresh"))
def test_af13_pending_read_cleanup_replay_replacement_and_refresh(
    workspace, two_results, tmp_path, monkeypatch, action,
):
    page, (first, second) = workspace, two_results
    third, _ = _write_processed(tmp_path / "third", labels=(4, 5, 6))
    _open(page, first)
    entered, release = Event(), Event()
    real_read = Browse1DHydrationLane._open_and_read

    def held_read(self, task, labels):
        entered.set()
        assert release.wait(10)
        return real_read(self, task, labels)

    monkeypatch.setattr(Browse1DHydrationLane, "_open_and_read", held_read)
    try:
        _mode(page, "Overlay")
        QtTest.QTest.mouseClick(page._shell.browser.show_all, LEFT)
        QtWidgets.QApplication.processEvents()
        assert entered.wait(5), "the real all-frame read is held at its I/O boundary"
        page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_SCAN, str(second), path=("artifact",)))
        assert page._deferred_browse_reopen.value == str(second)
        if action == "replace":
            page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_SCAN, str(third), path=("artifact",)))
            assert page._deferred_browse_reopen.value == str(third)
        elif action == "refresh":
            page._handle_shell_command(ShellCommand(ShellCommandKind.REFRESH_BROWSER))
            assert page._deferred_browse_reopen is None
        release.set()
        target = first if action == "refresh" else third if action == "replace" else second
        if action == "refresh":
            # Catalog refresh cancels the kept click; it does not admit/repaint a fresh Browse context.
            controller = page._context_controller
            _wait(lambda: True if not controller.browse_pending
                  and not controller.browse_preview_polling_needed else None)
            assert controller.navigation.current.artifact == str(first)
        else:
            _shown(page, target)
            _wait(lambda: True if not page._context_controller.browse_preview_polling_needed else None)
        assert page._deferred_browse_reopen is None
        assert _all_selected(page) and _on(page) is True
        if action != "replay":
            assert all(frame.artifact != str(second) for frame in page._shell.scientific.trace_history_keys)
    finally:
        release.set()


def test_af14_fresh_open_reads_a_grown_result_and_selects_its_new_frame(workspace, two_results, tmp_path):
    page, (first, second) = workspace, two_results
    _mode(page, "Waterfall")
    _toggle(page)
    _open(page, first)
    prior = page._context_controller.browse_context
    _open(page, second)
    grown, _ = _write_processed(tmp_path / "grown", labels=(1, 2, 3, 4))
    shutil.copyfile(grown, first)
    _open(page, first)
    assert page._context_controller.browse_context is not prior
    navigation = page._context_controller.navigation
    assert [frame.local_frame_label for frame in navigation.frames] == [1, 2, 3, 4]
    assert _all_selected(page) and _on(page) is True
    assert ("scan.nexus", 4) in _rows(page)


# AF15 (added at the implementation terminal, Codex REVIEW-1: "Measure actual long-file toggle-switch time"): the
# maintainer's case at its measured size, read-only on test_data. Design probe at the parent: open 1.80-2.77 s, Show All
# +0.33-0.34 s on these 2500-frame bursts.

_BURST = ("bo_0_506V_0p1ms_sfpx_45p19_halpha_0p30_burst_00001_int2d.nexus",
          "bo_2_716V_5p9ms_sfpx_23p70_halpha_0p30_burst_00001_int2d.nexus")


@pytest.mark.skipif(not os.environ.get("XDART_TEST_DATA"), reason="needs test_data's 2500-frame burst outputs")
@pytest.mark.parametrize("all_frames", (True, False), ids=("on", "off"))
def test_af15_two_2500_frame_results_switch_with_all_frames_on(qapp, tmp_path, monkeypatch, all_frames):
    """Off is the same switch's one-frame open, run beside it so the difference is All frames' own cost."""
    folder = Path(os.environ["XDART_TEST_DATA"]) / "xdart_processed_data" / "bo_burst"
    first, second = (folder / name for name in _BURST)
    if not (first.is_file() and second.is_file()):
        pytest.skip("the burst outputs are not in this test_data")
    page, _ = _page(tmp_path, monkeypatch, store=RunIntentStore(RunIntent(
        project_root=str(folder), processing_mode="Int 2D")))
    page.resize(1200, 800)
    page.show()
    try:
        _mode(page, "Waterfall")
        if all_frames:
            _toggle(page)
        times = []
        for path in (first, second):
            start = time.perf_counter()
            _open(page, path)
            times.append(time.perf_counter() - start)
            navigation = page._context_controller.navigation
            assert len(navigation.frames) == 2500
            assert _all_selected(page) if all_frames else len(navigation.selected) == 1
        rows = _rows(page)
        assert sum(name == second.name for name, _label in rows) == (2500 if all_frames else 1)
        assert any(name == first.name for name, _label in rows), "the earlier result stays overlaid"
        state = "on" if all_frames else "off"
        print(f"AF15 result switch with All frames {state}: {', '.join(f'{seconds:.2f} s' for seconds in times)}")
    finally:
        _wait(lambda: True if page.close_workspace().cleanup_status is CleanupStatus.CLEANED else None)
        page.deleteLater()
        qapp.processEvents()
