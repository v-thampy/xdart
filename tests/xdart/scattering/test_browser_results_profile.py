"""The Data Browser's profile (AUD-9 stage 2b G3a): another page mounts Int's ``BrowserView`` with its own
menus, captions and actions instead of copying them (hard rule 14; Codex's RSM reuse audit R1)."""
from __future__ import annotations

import pytest
from pyqtgraph.Qt import QtWidgets

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
