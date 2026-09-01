"""Joined appearance/UI7 authority at the mounted-shell boundary."""

from __future__ import annotations

import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from tests.xdart.scattering.e3_shell_support import make_shell_projection
from xdart.gui.pages.descriptors import PageDescriptor
from xdart.gui.pages.handle import AppMenuHosts, PageHandle
from xdart.gui.pages.values import (
    CloseReceipt,
    PageCapability,
    PageCleanup,
    PageKey,
    PageLifecycle,
)
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xdart.gui.themes import apply_theme
from xdart.gui.themes.spacing import (
    SPACING_SETTINGS_KEY,
    current_spacing,
)
from xdart.gui.themes.typography import application_settings


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _MenuHost(QtWidgets.QWidget):
    """Small host for the production ``Main`` Config-menu owner."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.config_menu = QtWidgets.QMenu(self)
        self.help_menu = QtWidgets.QMenu(self)


class _MenuPort:
    def __init__(self, host: _MenuHost) -> None:
        self._host = host

    def mount_points(self) -> AppMenuHosts:
        return AppMenuHosts(self._host.config_menu, self._host.help_menu)


def _main_window(monkeypatch):
    from xdart import _gui_main

    del monkeypatch

    key = PageKey("appearance-test")

    def build(_services, parent):
        widget = _MenuHost(parent)
        return PageHandle(
            key=key,
            widget=widget,
            close=lambda: CloseReceipt(PageCleanup.CLEAN, "closed"),
            app_menus=_MenuPort(widget),
        )

    descriptor = PageDescriptor(
        key=key,
        label="Appearance test",
        order=0,
        build=build,
        lifecycle=PageLifecycle.EXIT_ONLY,
        capabilities=frozenset({PageCapability.APP_MENU_HOSTS}),
    )
    return _gui_main.Main(
        page_descriptors=(descriptor,),
        selected_page_key=descriptor.key,
    )


def _trigger(window, submenu_title: str, label: str) -> None:
    menu = window._attached_config_menu
    assert menu is not None
    for owner in menu.actions():
        submenu = owner.menu()
        if submenu is None or submenu.title() != submenu_title:
            continue
        for action in submenu.actions():
            if action.text() == label:
                action.trigger()
                return
    raise AssertionError(f"missing Config > {submenu_title} > {label}")


def _shell_spacing(shell: ScatteringWorkspaceShell) -> tuple[int, int, int]:
    margins = shell.tools.tool_content.layout().contentsMargins()
    return (
        shell.browser.list_splitter.handleWidth(),
        shell.browser.layout().spacing(),
        margins.top(),
    )


def _current_frame(shell: ScatteringWorkspaceShell):
    index = shell.browser.frames.currentIndex()
    return index.data(int(QtCore.Qt.ItemDataRole.UserRole))


def test_config_spacing_drives_the_real_shell_without_changing_selection(
    qapp: QtWidgets.QApplication,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "appearance.ini"))
    apply_theme(qapp, "dark", spacing="normal")
    window = _main_window(monkeypatch)
    shell = ScatteringWorkspaceShell()
    commands: list[object] = []
    shell.commandRequested.connect(commands.append)
    projection = make_shell_projection(selected_index=3)
    shell.apply_state(projection)
    shell.show()
    qapp.processEvents()
    selected = projection.navigation.current
    assert _current_frame(shell) is selected
    columns = [
        shell.splitter.widget(index).geometry()
        for index in range(shell.splitter.count())
    ]
    assert [column.y() for column in columns] == [0, 0, 0]
    assert [column.height() for column in columns] == [shell.height()] * 3
    normal = _shell_spacing(shell)
    try:
        _trigger(window, "Spacing", "Extra Spacious")
        qapp.processEvents()
        qapp.processEvents()
        spacious = _shell_spacing(shell)
        assert current_spacing() == "extra_spacious"
        assert application_settings().value(SPACING_SETTINGS_KEY) == (
            "extra_spacious"
        )
        assert all(after > before for before, after in zip(normal, spacious))
        assert _current_frame(shell) is selected
        assert commands == []

        _trigger(window, "Spacing", "Extra Tight")
        qapp.processEvents()
        qapp.processEvents()
        tight = _shell_spacing(shell)
        assert current_spacing() == "extra_tight"
        assert all(after < before for before, after in zip(spacious, tight))
        assert _current_frame(shell) is selected
        assert commands == []
    finally:
        apply_theme(qapp, "dark", spacing="normal")
        shell.close()
        window.close()
        shell.deleteLater()
        window.deleteLater()
        qapp.processEvents()
