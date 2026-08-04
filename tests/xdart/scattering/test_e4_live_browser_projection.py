from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from types import SimpleNamespace

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.adapters.source import (
    FilesystemSourceAdapter,
)
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.browser_catalog import (
    BrowserCatalogEntry,
    enumerate_processed_artifacts,
)
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.events import RunIdentity
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_projection import (
    build_browser_projection,
)
from xdart.gui.tabs.scattering.shell_values import (
    FrameNavigationProjection,
    ShellCommand,
    ShellCommandKind,
)
from xdart.gui.tabs.scattering.workspace_shell import (
    ScatteringWorkspaceShell,
)
from xdart.gui.themes import render_qss
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent

from tests.xdart.scattering.e3_shell_support import make_shell_projection


def test_processed_catalog_includes_parent_and_child_directory_navigation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    (root / "nested").mkdir()
    (root / "scan_10.nxs").touch()
    (root / "scan_2.nxs").touch()
    (root / "ignored.txt").touch()

    catalog = enumerate_processed_artifacts(str(root))

    assert tuple(
        (entry.label, entry.is_directory)
        for entry in catalog
    ) == (
        ("..", True),
        ("nested/", True),
        ("scan_2.nxs", False),
        ("scan_10.nxs", False),
    )
    assert catalog[0].artifact == str(tmp_path)
    assert catalog[1].artifact == str(root / "nested")


def test_deleted_processed_directory_retains_parent_navigation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    root.rmdir()

    catalog = enumerate_processed_artifacts(str(root))

    assert tuple(
        (entry.label, entry.artifact, entry.is_directory)
        for entry in catalog
    ) == (("..", str(tmp_path), True),)


def test_processed_catalog_naturally_interleaves_directories_and_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    (root / "scan_2").mkdir()
    (root / "scan_20").mkdir()
    (root / "scan_1.nxs").touch()
    (root / "scan_3.nexus").touch()
    (root / "scan_10.nxs").touch()

    catalog = enumerate_processed_artifacts(str(root))

    assert tuple(entry.label for entry in catalog) == (
        "..",
        "scan_1.nxs",
        "scan_2/",
        "scan_3.nexus",
        "scan_10.nxs",
        "scan_20/",
    )


def test_processed_directory_maps_explicit_nexus_artifact_to_parent(tmp_path):
    from xdart.gui.tabs.scattering.browser_catalog import processed_directory

    target = tmp_path / "out" / "scan.nexus"
    assert processed_directory(str(target)) == str(tmp_path / "out")


def test_browser_time_sort_interleaves_directories_and_artifacts() -> None:
    catalog = (
        BrowserCatalogEntry("/out/old-dir", "old-dir/", 10, True),
        BrowserCatalogEntry("/out/new-file.nxs", "new-file.nxs", 40),
        BrowserCatalogEntry("/", "..", 50, True),
        BrowserCatalogEntry("/out/new-dir", "new-dir/", 30, True),
        BrowserCatalogEntry("/out/old-file.nxs", "old-file.nxs", 20),
    )

    projected = build_browser_projection(
        contexts=(),
        selection=None,
        navigation=FrameNavigationProjection(),
        browser_directory="/out",
        date_sorted=True,
        auto_last=True,
        catalog=catalog,
    )

    assert tuple(scan.label for scan in projected.scans) == (
        "..",
        "new-file.nxs",
        "new-dir/",
        "old-file.nxs",
        "old-dir/",
    )


def _keys() -> tuple[DisplayFrameKey, ...]:
    identity = RunIdentity(9, "e4-live-browser")
    return (
        DisplayFrameKey(identity, "scan-a", "/out/a.nxs", 0, 1),
        DisplayFrameKey(identity, "scan-a", "/out/a.nxs", 1, 2),
        DisplayFrameKey(identity, "scan-b", "/out/b.nxs", 0, 3),
        DisplayFrameKey(identity, "scan-b", "/out/b.nxs", 1, 4),
    )


def test_browser_projection_borrows_only_inflight_artifact_keys() -> None:
    keys = _keys()
    navigation = FrameNavigationProjection(keys, keys[3], (keys[3],))

    projected = build_browser_projection(
        contexts=(),
        selection=None,
        navigation=navigation,
        browser_directory="/out",
        date_sorted=False,
        auto_last=True,
        transient_frame=keys[3],
    )

    assert tuple(scan.identifier for scan in projected.scans) == (
        "/out/b.nxs",
    )
    assert tuple(scan.label for scan in projected.scans) == ("b.nxs",)
    assert projected.selected_scan == "/out/b.nxs"
    assert getattr(projected, "frames", ()) == keys[2:]
    assert all(
        actual is expected
        for actual, expected in zip(
            projected.frames,
            keys[2:],
            strict=True,
        )
    )
    assert navigation.frames is keys


def test_authoritative_catalog_does_not_resurrect_historical_navigation(
) -> None:
    keys = _keys()
    navigation = FrameNavigationProjection(keys, keys[3], (keys[3],))
    retained_context = SimpleNamespace(
        source="/out/missing.nxs",
        context_token="retained-browse",
        scan_key="missing",
    )

    projected = build_browser_projection(
        contexts=(retained_context,),
        selection=None,
        navigation=navigation,
        browser_directory="/out",
        date_sorted=False,
        auto_last=True,
        catalog=(BrowserCatalogEntry("/", "..", 1, True),),
    )

    assert tuple(scan.identifier for scan in projected.scans) == ("/",)
    assert projected.selected_scan == ""
    assert projected.frames == ()
    assert navigation.frames is keys


def test_catalog_artifact_remains_selectable_after_transient_identity_ends(
) -> None:
    keys = _keys()
    navigation = FrameNavigationProjection(keys, keys[3], (keys[3],))

    projected = build_browser_projection(
        contexts=(),
        selection=None,
        navigation=navigation,
        browser_directory="/out",
        date_sorted=False,
        auto_last=True,
        catalog=(BrowserCatalogEntry("/out/b.nxs", "b.nxs", 2),),
    )

    assert tuple(scan.identifier for scan in projected.scans) == (
        "/out/b.nxs",
    )
    assert projected.selected_scan == "/out/b.nxs"
    assert all(
        actual is expected
        for actual, expected in zip(
            projected.frames,
            keys[2:],
            strict=True,
        )
    )


def test_equal_distinct_frame_cannot_lend_a_transient_browser_artifact(
) -> None:
    keys = _keys()
    navigation = FrameNavigationProjection(keys, keys[3], (keys[3],))
    latest = keys[3]
    equal_distinct = DisplayFrameKey(
        latest.run_identity,
        latest.source_scan,
        latest.artifact,
        latest.local_frame_label,
        latest.work_ordinal,
    )

    projected = build_browser_projection(
        contexts=(),
        selection=None,
        navigation=navigation,
        browser_directory="/out",
        date_sorted=False,
        auto_last=True,
        transient_frame=equal_distinct,
    )

    assert equal_distinct == latest
    assert equal_distinct is not latest
    assert projected.scans == ()
    assert projected.selected_scan == ""


def test_browser_menu_theme_uses_canonical_indicator_and_font_contract() -> None:
    font_selector = (
        "QToolButton#fileMenuButton,\n"
        "QToolButton#configMenuButton,\n"
        "QToolButton#helpMenuButton {"
    )
    indicator_selector = (
        "QToolButton#fileMenuButton::menu-indicator,\n"
        "QToolButton#configMenuButton::menu-indicator,\n"
        "QToolButton#helpMenuButton::menu-indicator {"
    )
    default = render_qss("dark", font_scale="default")
    extra_large = render_qss("dark", font_scale="extra_large")

    assert indicator_selector in default
    default_block = default[
        default.index(font_selector):default.index(
            "}", default.index(font_selector)
        )
    ]
    extra_large_block = extra_large[
        extra_large.index(font_selector):extra_large.index(
            "}", extra_large.index(font_selector)
        )
    ]
    assert "font-size: 13px;" in default_block
    assert "font-size: 15px;" in extra_large_block


def test_footer_and_left_frames_reconcile_to_same_exact_key() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    keys = _keys()
    navigation = FrameNavigationProjection(keys, keys[1], (keys[1],))
    browser = build_browser_projection(
        contexts=(),
        selection=None,
        navigation=navigation,
        browser_directory="/out",
        date_sorted=False,
        auto_last=True,
        transient_frame=keys[1],
    )
    base = make_shell_projection(
        frame_count=1,
        selected_index=0,
        plot_mode="Single",
    )
    state = replace(
        base,
        browser=browser,
        navigation=navigation,
        scientific=replace(
            base.scientific,
            traces=(),
            heavy=None,
            heavy_available=frozenset(),
            retain_display=True,
        ),
    )
    shell = ScatteringWorkspaceShell()
    try:
        shell.apply_state(state)
        selected = shell.browser.frames.selectionModel().selectedRows()
        assert len(selected) == 1
        assert selected[0].data(QtCore.Qt.ItemDataRole.UserRole) is keys[1]
        assert shell.browser.frames.currentIndex().data(
            QtCore.Qt.ItemDataRole.UserRole
        ) is keys[1]
        assert shell.scientific.frame_selector.currentData() is keys[1]

        # Simulate the live failure class: Qt's painted selection has drifted
        # while the passive view's identity cache still describes navigation.
        selection = shell.browser.frames.selectionModel()
        blocker = QtCore.QSignalBlocker(selection)
        selection.clearSelection()
        selection.setCurrentIndex(
            shell.browser.frame_model.index(0, 0),
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
        )
        del blocker
        shell.apply_state(replace(state, revision=state.revision + 1))
        selected = selection.selectedRows()
        assert len(selected) == 1
        assert selected[0].data(QtCore.Qt.ItemDataRole.UserRole) is keys[1]
        assert selection.currentIndex().data(
            QtCore.Qt.ItemDataRole.UserRole
        ) is keys[1]

        next_navigation = FrameNavigationProjection(
            keys,
            keys[0],
            (keys[0],),
        )
        selection_changes: list[tuple[object, object]] = []
        selection.selectionChanged.connect(
            lambda selected, deselected: selection_changes.append(
                (selected, deselected)
            )
        )
        shell.apply_state(
            replace(
                state,
                revision=state.revision + 2,
                browser=build_browser_projection(
                    contexts=(),
                    selection=None,
                    navigation=next_navigation,
                    browser_directory="/out",
                    date_sorted=False,
                    auto_last=True,
                    transient_frame=keys[1],
                ),
                navigation=next_navigation,
            )
        )
        selected = shell.browser.frames.selectionModel().selectedRows()
        assert len(selected) == 1
        assert selected[0].data(QtCore.Qt.ItemDataRole.UserRole) is keys[0]
        assert shell.scientific.frame_selector.currentData() is keys[0]
        assert selection_changes, (
            "the passive reconcile must notify Qt's selection view so the "
            "highlight repaints in the same event-loop turn"
        )
    finally:
        shell.close()
        shell.deleteLater()
        app.processEvents()


def test_open_folder_and_refresh_publish_processed_catalog(
    tmp_path: Path,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configured = tmp_path / "configured"
    configured.mkdir()
    (configured / "initial.nxs").touch()
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "scan_10.nxs").touch()
    (selected / "scan_2.nxs").touch()
    nested = selected / "nested"
    nested.mkdir()
    (nested / "inside.nxs").touch()
    chooser_calls: list[str] = []

    def choose(current: str) -> str:
        chooser_calls.append(current)
        return str(selected)

    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(
                project_root=str(tmp_path / "raw"),
                save_path=str(configured),
            )
        ),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
        browser_directory_chooser=choose,
    )
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None

    def wait_for(predicate) -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError("processed browser catalog did not settle")

    try:
        wait_for(
            lambda: shell.browser.directory_label._path
            == str(configured)
        )
        menu_buttons = {
            button.objectName(): button.text()
            for button in shell.browser.findChildren(QtWidgets.QToolButton)
            if button.objectName() in {
                "fileMenuButton",
                "configMenuButton",
                "helpMenuButton",
            }
        }
        assert menu_buttons == {
            "fileMenuButton": "File",
            "configMenuButton": "Config",
            "helpMenuButton": "Help",
        }
        file_menu = shell.browser.findChild(
            QtWidgets.QToolButton,
            "fileMenuButton",
        )
        assert file_menu is not None
        open_folder = next(
            action
            for action in file_menu.menu().actions()
            if action.text() == "Open Folder"
        )
        open_folder.trigger()
        wait_for(
            lambda: (
                shell.browser.directory_label._path == str(selected)
                and shell.browser.scans.count() == 4
            )
        )
        assert chooser_calls == [str(configured)]
        artifacts = [
            shell.browser.scans.item(index).data(
                QtCore.Qt.ItemDataRole.UserRole
            )
            for index in range(shell.browser.scans.count())
        ]
        assert artifacts == [
            str(tmp_path),
            str(nested),
            str(selected / "scan_2.nxs"),
            str(selected / "scan_10.nxs"),
        ]

        (selected / "scan_1.nxs").touch()
        shell.browser.refresh.click()
        wait_for(lambda: shell.browser.scans.count() == 5)
        assert [
            shell.browser.scans.item(index).data(
                QtCore.Qt.ItemDataRole.UserRole
            )
            for index in range(shell.browser.scans.count())
        ][2] == str(selected / "scan_1.nxs")

        # An idle external deletion/recreation has no shell command or run
        # event to drive a refresh.  The background catalog poll must evict
        # and later restore the exact filesystem member on its own.
        (selected / "scan_10.nxs").unlink()
        wait_for(
            lambda: (
                shell.browser.scans.count() == 4
                and all(
                    shell.browser.scans.item(index).data(
                        QtCore.Qt.ItemDataRole.UserRole
                    ) != str(selected / "scan_10.nxs")
                    for index in range(shell.browser.scans.count())
                )
            )
        )
        (selected / "scan_10.nxs").touch()
        wait_for(
            lambda: any(
                shell.browser.scans.item(index).data(
                    QtCore.Qt.ItemDataRole.UserRole
                ) == str(selected / "scan_10.nxs")
                for index in range(shell.browser.scans.count())
            )
        )

        nested_item = next(
            shell.browser.scans.item(index)
            for index in range(shell.browser.scans.count())
            if shell.browser.scans.item(index).data(
                QtCore.Qt.ItemDataRole.UserRole
            ) == str(nested)
        )
        shell.browser.scans.setCurrentItem(nested_item)
        wait_for(
            lambda: (
                shell.browser.directory_label._path == str(nested)
                and shell.browser.scans.count() == 2
            )
        )

        (nested / "inside.nxs").unlink()
        nested.rmdir()
        wait_for(
            lambda: (
                shell.browser.directory_label._path == str(nested)
                and shell.browser.scans.count() == 1
                and shell.browser.scans.item(0).text() == ".."
                and shell.browser.scans.item(0).data(
                    QtCore.Qt.ItemDataRole.UserRole
                )
                == str(selected)
            )
        )
        shell.browser.scans.setCurrentItem(shell.browser.scans.item(0))
        wait_for(
            lambda: shell.browser.directory_label._path == str(selected)
        )
    finally:
        page.close_workspace()
        assert not page._browser_catalog_timer.isActive()
        page.deleteLater()
        app.processEvents()


def test_published_artifact_auto_follows_until_user_opens_folder(
    tmp_path: Path,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configured = tmp_path / "configured"
    configured.mkdir()
    produced = tmp_path / "processed"
    produced.mkdir()
    first_artifact = produced / "scan_2.nxs"
    first_artifact.touch()
    selected = tmp_path / "selected"
    selected.mkdir()
    selected_artifact = selected / "browse.nxs"
    selected_artifact.touch()
    other = tmp_path / "other"
    other.mkdir()
    later_artifact = other / "later.nxs"
    later_artifact.touch()

    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(save_path=str(configured))
        ),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
        browser_directory_chooser=lambda _current: str(selected),
    )
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None

    def wait_for(predicate) -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError("processed browser auto-follow did not settle")

    identity = RunIdentity(10, "browser-auto-follow")
    first = DisplayFrameKey(
        identity,
        "scan_2",
        str(first_artifact),
        0,
        1,
    )
    later = DisplayFrameKey(
        identity,
        "later",
        str(later_artifact),
        0,
        2,
    )
    try:
        page._follow_processed_artifact(first)
        wait_for(
            lambda: (
                shell.browser.directory_label._path == str(produced)
                and shell.browser.scans.count() == 2
                and shell.browser.scans.item(1).data(
                    QtCore.Qt.ItemDataRole.UserRole
                )
                == str(first_artifact)
            )
        )

        page._handle_shell_command(
            ShellCommand(
                ShellCommandKind.MENU,
                "File:Open Folder",
            )
        )
        wait_for(
            lambda: shell.browser.directory_label._path == str(selected)
        )
        page._follow_processed_artifact(later)
        wait_for(lambda: shell.browser.scans.count() == 2)
        assert shell.browser.directory_label._path == str(selected)
        assert shell.browser.scans.item(1).data(
            QtCore.Qt.ItemDataRole.UserRole
        ) == str(selected_artifact)

        next_identity = RunIdentity(11, "next-browser-auto-follow")
        next_frame = DisplayFrameKey(
            next_identity,
            "later",
            str(later_artifact),
            0,
            1,
        )
        page._begin_browser_follow(next_identity)
        page._follow_processed_artifact(next_frame)
        wait_for(
            lambda: (
                shell.browser.directory_label._path == str(other)
                and shell.browser.scans.count() == 2
            )
        )
        assert shell.browser.scans.item(1).data(
            QtCore.Qt.ItemDataRole.UserRole
        ) == str(later_artifact)
    finally:
        page.close_workspace()
        page.deleteLater()
        app.processEvents()


def test_terminal_transient_is_retained_until_catalog_barrier(
    tmp_path: Path,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configured = tmp_path / "configured"
    configured.mkdir()
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(save_path=str(configured))),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )

    def wait_for(predicate) -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError("browser terminal catalog barrier did not settle")

    identity = RunIdentity(12, "browser-terminal-barrier")
    frame = DisplayFrameKey(
        identity,
        "atomic-output",
        str(configured / "atomic-output.nxs"),
        0,
        1,
    )
    try:
        wait_for(lambda: page._browser_catalog_operation is None)
        page._browser_transient_frame = frame
        page._request_browser_catalog()
        operation = page._browser_catalog_operation
        assert operation is not None
        page._browser_transient_clear_token = operation.token

        # The terminal refresh is asynchronous.  Do not create a one-turn
        # blank browser/display seam by dropping the only exact transient
        # owner before the authoritative filesystem result arrives.
        assert page._browser_transient_frame is frame
        wait_for(lambda: page._browser_catalog_operation is None)
        assert page._browser_transient_frame is None
        assert page._browser_transient_clear_token is None
    finally:
        page.close_workspace()
        page.deleteLater()
        app.processEvents()
