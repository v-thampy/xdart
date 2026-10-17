"""The Data Browser's profile (AUD-9 stage 2b G3a): another page mounts Int's ``BrowserView`` with its own
menus, captions and actions instead of copying them (hard rule 14; Codex's RSM reuse audit R1)."""
from __future__ import annotations

import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from xdart.gui.tabs.scattering.browser_view import (
    INT_BROWSER_MENUS,
    BrowserProfile,
    BrowserResult,
    BrowserView,
)
from xdart.gui.tabs.scattering.shell_values import ShellCommandKind


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


_RESULTS = BrowserProfile(
    menus=(("File", ("Open result…", "Integration workspace")), ("Config", ()), ("Help", ())),
    raw_resolution_menus=False, list_caption="Results", directory=False, frames=False,
    actions=("Open…",), date_sort=False, refresh=False, results=True,
)


def _close(qapp, browser):
    browser.close()
    browser.deleteLater()
    qapp.processEvents()


def test_ints_browser_is_the_default_profile_unchanged(qapp):
    browser = BrowserView()
    try:
        assert browser.profile == BrowserProfile() and browser.profile.menus == INT_BROWSER_MENUS
        assert {title: [action.text() for action in menu.actions() if not action.menu()]
                for title, menu in browser.menus.items()} == {
            title: list(actions) for title, actions in INT_BROWSER_MENUS}
        assert [action.text() for action in browser.menus["Config"].actions() if action.menu()] == [
            "Raw image resolution", "Heavy residency"]
        for name in ("fileMenuButton", "configMenuButton", "helpMenuButton"):
            assert browser.findChild(QtWidgets.QToolButton, name) is not None
        assert browser.scans_label.text() == "Scans" and browser.frames_label.text() == "Frames"
        assert browser.open_result is None
        for widget in (browser.show_all, browser.metadata, browser.auto_last, browser.date_sort,
                       browser.refresh, browser.frames, browser.directory_label):
            assert not widget.isHidden()
    finally:
        _close(qapp, browser)


def test_a_results_profile_shows_the_pages_menus_one_list_and_open(qapp):
    browser = BrowserView(profile=_RESULTS)
    commands = []
    browser.commandRequested.connect(commands.append)
    try:
        assert [action.text() for action in browser.menus["File"].actions()] == [
            "Open result…", "Integration workspace"]
        assert browser.menus["Config"].actions() == [] and browser.menus["Help"].actions() == []
        assert browser.scans_label.text() == "Results"
        for widget in (browser.show_all, browser.metadata, browser.auto_last, browser.date_sort,
                       browser.refresh, browser.directory_label):
            assert widget.isHidden()
        assert browser.frames.parentWidget().isHidden() and not browser.open_result.isHidden()
        browser.menus["File"].actions()[0].trigger()
        browser.open_result.click()
        assert [(command.kind, command.value) for command in commands] == [
            (ShellCommandKind.MENU, "File:Open result…"), (ShellCommandKind.OPEN_RESULT, None)]
        commands.clear()
        entries = (BrowserResult("/a/one.nexus", "one.nexus", "/a/one.nexus"),
                   BrowserResult("/a/two.nexus", "two.nexus", "/a/two.nexus"))
        browser.reconcile_results(entries, "/a/one.nexus")
        browser.reconcile_results(entries, "/a/two.nexus")  # the page moves the current one: no emit
        assert commands == [] and browser.scans.count() == 2
        assert browser.scans.currentItem().text() == "two.nexus"
        assert browser.scans.item(0).toolTip() == "/a/one.nexus"
        browser.scans.setCurrentRow(0)  # the user's choice
        assert [(command.kind, command.value) for command in commands] == [
            (ShellCommandKind.SELECT_RESULT, "/a/one.nexus")]
        commands.clear()
        browser.scans.itemActivated.emit(browser.scans.item(1))  # no Int directory or artifact semantics
        assert commands == []
        browser.set_action_enabled("Open…", False)
        assert not browser.open_result.isEnabled()
        browser.reconcile_results((), None)
        assert browser.scans.count() == 0 and commands == []
    finally:
        _close(qapp, browser)


def test_reconcile_results_is_only_for_a_results_profile(qapp):
    browser = BrowserView()
    try:
        with pytest.raises(RuntimeError):
            browser.reconcile_results((), None)
        with pytest.raises(TypeError):
            BrowserView(profile=object())
    finally:
        _close(qapp, browser)


_STITCH_LIKE = BrowserProfile(
    menus=(("File", ("Open result…",)), ("Config", ()), ("Help", ())), raw_resolution_menus=False,
    list_caption="Results", directory=False, actions=("Open…", "Metadata", "Raw image"), date_sort=False,
    refresh=False, results=True, result_selection="extended", result_frames=True,
)


@pytest.mark.parametrize("profile", (_RESULTS, _STITCH_LIKE))
def test_empty_results_click_preserves_current_and_explicit_selection(qapp, profile):
    browser = BrowserView(profile=profile)
    commands = []
    browser.commandRequested.connect(commands.append)
    try:
        browser.resize(400, 500)
        browser.show()
        browser.reconcile_results(tuple(BrowserResult(key, key) for key in "ab"), "a")
        if profile.result_selection == "extended":
            browser.scans.item(1).setSelected(True)
        qapp.processEvents()
        before = browser.current_result(), browser.selected_results()
        commands.clear()
        blank = QtCore.QPoint(15, browser.scans.viewport().height() - 10)
        assert not browser.scans.indexAt(blank).isValid()
        QtTest.QTest.mouseClick(browser.scans.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=blank)
        qapp.processEvents()
        assert (browser.current_result(), browser.selected_results()) == before
        assert commands == []
        browser.clear_result_selection()  # Explicit Clear still has its existing meaning.
        assert browser.current_result() is None and browser.selected_results() == ()
    finally:
        _close(qapp, browser)


def test_an_extended_results_list_reports_the_current_result_and_the_selection(qapp):
    """G3b (Codex's Stitch/Int reuse audit R1): Stitch's results list selects several results for an overlay.
    The owner reports the current result and the selection as the list's own signals do, and lets the page move
    the current result keeping or replacing the selection, as a user would."""
    from pyqtgraph.Qt import QtCore
    browser = BrowserView(profile=_STITCH_LIKE)
    commands = []
    browser.commandRequested.connect(commands.append)
    try:
        browser.reconcile_results(tuple(BrowserResult(key, key.upper()) for key in "abc"), "a")
        assert commands == [] and browser.current_result() == "a" and browser.selected_results() == ("a",)
        select = QtCore.QItemSelectionModel.SelectionFlag
        browser.scans.setCurrentRow(2, select.ClearAndSelect)
        browser.scans.setCurrentRow(0, select.Select)  # a user extending the selection
        kinds = {command.kind for command in commands}
        assert kinds == {ShellCommandKind.SELECT_RESULT, ShellCommandKind.SELECT_RESULTS}
        assert [command.value for command in commands if command.kind is ShellCommandKind.SELECT_RESULT] == [
            "c", "a"]
        latest = [command for command in commands if command.kind is ShellCommandKind.SELECT_RESULTS][-1]
        assert set(latest.artifacts) == {"a", "c"} == set(browser.selected_results())
        commands.clear()
        browser.set_current_result("b", keep_selection=True)  # an overlay stepping: the selection stays
        assert browser.current_result() == "b" and set(browser.selected_results()) == {"a", "c"}
        assert [command.value for command in commands] == ["b"]
        browser.set_current_result("b", keep_selection=False)  # Single: the current result alone
        assert browser.selected_results() == ("b",) and browser.current_result_label() == "B"
        assert browser.current_result_row() == 1
        commands.clear()
        browser.clear_result_selection()
        assert browser.current_result() is None and browser.selected_results() == ()
        assert ShellCommandKind.SELECT_RESULTS in {command.kind for command in commands}
        assert [command.value for command in commands if command.kind is ShellCommandKind.SELECT_RESULT] == [
            None]
    finally:
        _close(qapp, browser)


def test_extended_selection_never_notifies_nonempty_results_without_a_current(qapp):
    from pyqtgraph.Qt import QtCore

    browser = BrowserView(profile=_STITCH_LIKE)
    observed = []
    browser.commandRequested.connect(lambda command: observed.append(
        (command, browser.current_result(), browser.selected_results())))
    try:
        browser.reconcile_results(tuple(BrowserResult(key, key.upper()) for key in "ab"), None)
        browser.scans.setCurrentRow(0, QtCore.QItemSelectionModel.SelectionFlag.Select)
        assert browser.current_result() == "a" and browser.selected_results() == ("a",)
        assert any(command.kind is ShellCommandKind.SELECT_RESULT for command, _, _ in observed)
        assert all(current is not None or not selected for command, current, selected in observed
                   if command.kind is ShellCommandKind.SELECT_RESULTS)
        observed.clear()
        browser.clear_result_selection()
        assert browser.current_result() is None and browser.selected_results() == ()
        assert any(command.kind is ShellCommandKind.SELECT_RESULTS and not selected
                   for command, _, selected in observed)
    finally:
        _close(qapp, browser)


def test_a_results_list_cleared_from_outside_is_filled_again(qapp):
    browser = BrowserView(profile=_RESULTS)
    try:
        entries = (BrowserResult("one", "one.nexus"),)
        browser.reconcile_results(entries, "one")
        browser.scans.clear()
        browser.reconcile_results(entries, "one")
        assert browser.scans.count() == 1 and browser.current_result() == "one"
    finally:
        _close(qapp, browser)


def test_the_result_frames_pane_lists_a_results_entries_muted_and_rebuilt_on_every_call(qapp):
    """Stitch's contributing frames: their texts, tooltips and excluded marks, the pane rebuilt each time the
    page shows a result (one ``clear``), its current entry set silently and a user's choice reported."""
    browser = BrowserView(profile=_STITCH_LIKE)
    commands = []
    browser.commandRequested.connect(commands.append)
    try:
        assert browser.frames.isHidden() and not browser.result_frames.isHidden()
        assert browser.frames_label.text() == "Frames"
        clears = []
        clear = browser.result_frames.clear
        browser.result_frames.clear = lambda: (clears.append(1), clear())[1]
        entries = (BrowserResult("f0", "2-0", "scan 2 · frame 0"),
                   BrowserResult("f1", "2-1 · excluded", "scan 2 · frame 1", muted=True))
        browser.reconcile_result_frames(entries, "f0")
        assert commands == [] and clears == [1] and browser.current_result_frame() == "f0"
        assert browser.result_frame_count() == 2 and browser.current_result_frame_label() == "2-0"
        items = [browser.result_frames.item(index) for index in range(2)]
        assert [item.text() for item in items] == ["2-0", "2-1 · excluded"]
        assert items[1].toolTip() == "scan 2 · frame 1"
        assert items[1].foreground().color() != items[0].foreground().color()
        browser.reconcile_result_frames(entries, "f0")
        assert clears == [1, 1]
        browser.result_frames.setCurrentRow(1)
        assert [(command.kind, command.value) for command in commands] == [
            (ShellCommandKind.SELECT_RESULT_FRAME, "f1")]
        commands.clear()
        browser.raw_image.click()
        browser.set_action_enabled("Raw image", False)
        assert not browser.raw_image.isEnabled()
        assert [command.kind for command in commands] == [ShellCommandKind.SHOW_RAW_IMAGE]
    finally:
        _close(qapp, browser)


def test_a_profile_refuses_an_unknown_selection_and_frames_without_results(qapp):
    with pytest.raises(ValueError):
        BrowserView(profile=BrowserProfile(results=True, result_selection="multiple"))
    with pytest.raises(ValueError):
        BrowserView(profile=BrowserProfile(result_frames=True))
