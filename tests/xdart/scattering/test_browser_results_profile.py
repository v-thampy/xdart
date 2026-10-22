"""The Data Browser's profile (AUD-9 stage 2b G3a): another page mounts Int's ``BrowserView`` with its own
menus, captions and actions instead of copying them (hard rule 14; Codex's RSM reuse audit R1)."""
from __future__ import annotations

import time

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


@pytest.mark.parametrize("mode,tokens", [
    ("Int 1D", ("int-1d", "int-2d", "reintegrate-1d", "average")),
    ("Int 2D", ("int-2d", "reintegrate-2d", "average")),
    ("Stitch 1D", ("stitch-1d", "stitch-2d")), ("Stitch 2D", ("stitch-2d",)), ("RSM", ("rsm",)),
])
def test_filename_filter_projects_real_catalog_and_keeps_navigation(qapp, tmp_path, monkeypatch, mode, tokens):
    """The filename hint changes resident rows, never catalog admission or scientific reads."""
    import h5py
    from xrd_tools.io.output_path import FINITE_OPERATION_SLOTS
    from xdart.gui.tabs.scattering.browser_catalog import enumerate_processed_artifacts
    from xdart.gui.tabs.scattering.shell_values import (
        BrowserProjection, FrameNavigationProjection, build_browser_scan_index,
    )
    for slot in FINITE_OPERATION_SLOTS.values():
        (tmp_path / f"sample{slot}.nexus").write_bytes(b"not an admitted scientific record")
    for name in ("sample_int1d_average.nexus", "sample_int2d_average.nexus"):
        (tmp_path / name).write_bytes(b"catalog only")
    (tmp_path / "CUSTOM[ABC].nexus").write_bytes(b"not HDF5")
    (tmp_path / "sample_rsm_middle_int2d.nexus").write_bytes(b"catalog only")
    (tmp_path / "folder").mkdir()
    catalog = enumerate_processed_artifacts(str(tmp_path))
    state = BrowserProjection(directory=str(tmp_path), scans=build_browser_scan_index(catalog, False).scans)
    browser = BrowserView()
    try:
        browser.set_filename_mode(mode)
        assert browser.filename_filter.text() == ", ".join(FINITE_OPERATION_SLOTS[token] for token in tokens)
        assert browser.filename_filter.isClearButtonEnabled()
        assert browser.filename_filter.toolTip() and browser.filename_filter.accessibleName()
        assert browser._root_layout.itemAt(1).layout().indexOf(browser.filename_filter) + 1 == (
            browser._root_layout.itemAt(1).layout().indexOf(browser.date_sort))
        browser.reconcile(state, FrameNavigationProjection(), plot_mode="Single")
        names = lambda: [browser.scans.item(i).text() for i in range(browser.scans.count())]
        expected = {"..", "folder/", *(f"sample{FINITE_OPERATION_SLOTS[token]}.nexus" for token in tokens)}
        if "int-2d" in tokens:
            expected.add("sample_rsm_middle_int2d.nexus")
        if "average" in tokens:
            expected.update(("sample_int1d_average.nexus", "sample_int2d_average.nexus"))
        assert set(names()) == expected
        def forbidden(*args, **kwargs):
            raise AssertionError("filtering must not read files or HDF5")
        monkeypatch.setattr(h5py, "File", forbidden)
        monkeypatch.setattr("os.scandir", forbidden)
        browser.filename_filter.setText("custom[abc], _RSM")
        assert set(names()) == {"..", "folder/", "CUSTOM[ABC].nexus", "sample_rsm.nexus",
                                "sample_rsm_middle_int2d.nexus"}
        browser.set_filename_mode("Int 2D" if mode != "Int 2D" else "RSM")
        browser.reconcile(state, FrameNavigationProjection(), plot_mode="Single")
        assert browser.filename_filter.text() == "custom[abc], _RSM"
        browser.filename_filter.clear()
        assert len(names()) == len(catalog)
        browser.set_filename_mode(mode)
        browser.reconcile(state, FrameNavigationProjection(), plot_mode="Single")
        assert browser.filename_filter.text() == "" and len(names()) == len(catalog)
    finally:
        _close(qapp, browser)


def test_untouched_filename_default_follows_mode_and_open_pins_custom_names(qapp):
    """The untouched default follows the mode; the result being viewed stays listed through it (the filter
    narrows the other rows, never what is shown); opening a renamed file keeps that filter intact."""
    browser = BrowserView(profile=_STITCH_LIKE)
    commands = []
    browser.commandRequested.connect(commands.append)
    entries = tuple(BrowserResult(name, name) for name in (
        "one_stitch1d.nexus", "two_stitch2d.nexus", "renamed.nexus"))
    try:
        browser.set_filename_mode("Stitch 1D")  # a Stitch 2D record's I(q) shows there too
        browser.reconcile_results(entries, entries[0].identifier)
        assert browser.listed_results() == (entries[0].identifier, entries[1].identifier)
        browser.set_filename_mode("Stitch 2D")  # the Stitch 1D result being viewed stays, current
        assert browser.listed_results() == (entries[0].identifier, entries[1].identifier)
        assert browser.current_result() == entries[0].identifier
        browser.clear_result_selection()
        browser.reconcile_results(entries, None)  # no longer viewed: the mode's rows only
        assert browser.listed_results() == (entries[1].identifier,)
        assert browser.current_result() is None and browser.selected_results() == ()
        browser.reconcile_results(entries, "renamed.nexus")
        assert browser.filename_filter.text() == "_stitch2d"
        assert browser.listed_results() == (entries[1].identifier, "renamed.nexus")
        assert browser.current_result() == "renamed.nexus"
        browser.set_filename_mode("Stitch 1D")
        assert browser.filename_filter.text() == "_stitch1d, _stitch2d"
        assert browser.current_result() == "renamed.nexus"
        browser.clear_result_selection()
        browser.filename_filter.setText("stitch")
        browser.filename_filter.clear()
        assert browser.current_result() is None and browser.selected_results() == ()
    finally:
        _close(qapp, browser)


@pytest.mark.parametrize("profile", [_STITCH_LIKE, _RESULTS], ids=["extended", "single"])
def test_a_row_kept_while_viewed_goes_when_the_view_moves_on(qapp, profile):
    """A result the filter kept only because it was viewed goes once a click moves the view to another result,
    as Int's re-projection drops it (it used to stay until the folder next changed, and a no-selection batch
    would have included it)."""
    browser = BrowserView(profile=profile)
    entries = tuple(BrowserResult(name, name) for name in (
        "one_stitch1d.nexus", "two_stitch1d.nexus", "renamed.nexus"))
    try:
        browser.set_filename_mode("Stitch 1D")
        browser.reconcile_results(entries, "renamed.nexus")  # opened explicitly: listed while viewed
        assert browser.listed_results() == tuple(entry.identifier for entry in entries)
        row = next(i for i in range(browser.scans.count())
                   if browser.scans.item(i).data(QtCore.Qt.ItemDataRole.UserRole) == entries[1].identifier)
        browser.scans.setCurrentRow(row)  # a click on a result already listed (no page re-publish)
        deadline = time.monotonic() + 5
        while "renamed.nexus" in browser.listed_results() and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(.005)
        assert browser.listed_results() == (entries[0].identifier, entries[1].identifier)
        assert browser.current_result() == entries[1].identifier
        browser.filename_filter.clear()  # All results
        assert browser.listed_results() == tuple(entry.identifier for entry in entries)
    finally:
        _close(qapp, browser)


def test_the_viewed_scan_and_its_frames_stay_listed_through_the_filter(qapp):
    """In Int the filter narrows the other rows, never the scan being viewed: its row and its frames stay, as do
    an overlay's selected scans, so the list, its frames and the plot cannot disagree (a mode switch or typing
    used to hide the shown scan and empty its frames while the plot kept it)."""
    from dataclasses import replace
    from xdart.gui.tabs.scattering.shell_values import BrowserScan, FrameNavigationProjection
    from tests.xdart.scattering.e3_shell_support import make_shell_projection
    projection = make_shell_projection(frame_count=4)
    state = replace(projection.browser, scans=(*projection.browser.scans,
                                               BrowserScan("x_int2d.nexus", "x_int2d.nexus", "x_int2d.nexus")))
    names = lambda: [browser.scans.item(i).text() for i in range(browser.scans.count())]
    browser = BrowserView()
    try:
        browser.set_filename_mode("Int 2D")
        browser.reconcile(state, FrameNavigationProjection(), plot_mode="Single")
        assert names() == ["scan-a", "x_int2d.nexus"] and browser.scans.currentItem().text() == "scan-a"
        assert browser.frame_model.rowCount() == len(state.frames) == 4
        overlay = replace(state, selected_artifacts=("scan-a", "scan-b"), multi_artifact_selection=True)
        browser.reconcile(overlay, FrameNavigationProjection(), plot_mode="Overlay")
        assert names() == ["scan-a", "scan-b", "x_int2d.nexus"]
        moved = replace(state, selected_scan="x_int2d.nexus", frames=())
        browser.reconcile(moved, FrameNavigationProjection(), plot_mode="Single")
        assert names() == ["x_int2d.nexus"]
    finally:
        _close(qapp, browser)
