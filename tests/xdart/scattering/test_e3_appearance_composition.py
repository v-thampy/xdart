"""Joined appearance/UI7 authority at the mounted-shell boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from tests.xdart.scattering.e3_shell_support import make_shell_projection
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

    def __init__(self) -> None:
        super().__init__()
        self.h5viewer = SimpleNamespace(
            paramMenu=QtWidgets.QMenu(self),
            helpMenu=QtWidgets.QMenu(self),
        )
        self.ui = SimpleNamespace(
            leftFrame=QtWidgets.QFrame(self),
            middleFrame=QtWidgets.QFrame(self),
            rightFrame=QtWidgets.QFrame(self),
        )

    def enable_async_hydration(self) -> None:
        pass


def _main_window(monkeypatch):
    from xdart import _gui_main

    monkeypatch.setattr(
        _gui_main.tabs.static_scan,
        "staticWidget",
        _MenuHost,
    )
    return _gui_main.Main()


def _trigger(window, submenu_title: str, label: str) -> None:
    for owner in window.main_widget.h5viewer.paramMenu.actions():
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
