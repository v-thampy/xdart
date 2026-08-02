"""Frozen visual geometry for the post-E3 browser polish packet."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph.Qt import QtCore, QtWidgets
from shiboken6 import delete, isValid

from xdart.gui.themes.spacing import set_current_spacing
from xdart.gui.tabs.scattering.browser_view import BrowserView
from xdart.gui.tabs.scattering.tools_view import ToolsView


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _normal_spacing():
    set_current_spacing("normal")
    yield
    set_current_spacing("normal")


@pytest.mark.parametrize("width", [255, 289, 360])
def test_scan_and_frame_lists_have_an_aligned_eight_pixel_gutter(qapp, width):
    browser = BrowserView()
    browser.resize(width, 700)
    browser.show()
    qapp.processEvents()
    try:
        splitter = browser.findChild(QtWidgets.QSplitter, "e3BrowserLists")
        assert splitter is not None
        assert splitter.handleWidth() >= 8

        scan_right = browser.scans.mapTo(
            browser, browser.scans.rect().topRight()
        ).x()
        frame_left = browser.frames.mapTo(
            browser, browser.frames.rect().topLeft()
        ).x()
        assert frame_left - scan_right - 1 >= 8

        label_left = browser.frames_label.mapTo(
            browser, QtCore.QPoint(0, 0)
        ).x()
        assert abs(label_left - frame_left) <= 2
    finally:
        browser.close()
        browser.deleteLater()
        qapp.processEvents()


def test_tools_have_symmetric_breathing_room_and_scroll_reachability(
    qapp, monkeypatch
):
    monkeypatch.setattr(
        ToolsView,
        "_TOOLS",
        tuple((f"Tool {index}", f"tool_{index}") for index in range(12)),
    )
    tools = ToolsView()
    tools.resize(255, 140)
    tools.show()
    qapp.processEvents()
    try:
        layout = tools.tool_content.layout()
        margins = layout.contentsMargins()
        assert margins.top() == margins.bottom()
        assert margins.top() >= 12
        assert layout.spacing() >= 8

        buttons = tools.tool_content.findChildren(QtWidgets.QPushButton)
        assert len(buttons) == 12
        for before, after in zip(buttons, buttons[1:]):
            gap = after.geometry().top() - before.geometry().bottom() - 1
            assert gap >= 8

        scroll = tools.tool_scroll.verticalScrollBar()
        scroll.setValue(scroll.maximum())
        qapp.processEvents()
        last_center = buttons[-1].mapTo(
            tools.tool_scroll.viewport(),
            buttons[-1].rect().center(),
        )
        assert tools.tool_scroll.viewport().rect().contains(last_center)
    finally:
        tools.close()
        tools.deleteLater()
        qapp.processEvents()


def test_browser_actions_expose_one_compact_theme_seam(qapp):
    browser = BrowserView()
    try:
        assert {
            browser.date_sort.objectName(),
            browser.show_all.objectName(),
            browser.metadata.objectName(),
            browser.auto_last.objectName(),
        } == {"e3BrowserCompactButton"}
        assert browser.refresh.objectName() == "e3RefreshBrowser"
    finally:
        browser.close()
        browser.deleteLater()
        qapp.processEvents()


def test_browser_and_tools_follow_one_live_spacing_tier(qapp):
    browser = BrowserView()
    tools = ToolsView()
    browser.show()
    tools.show()
    qapp.processEvents()
    try:
        observed = []
        for tier in (
            "extra_tight",
            "tight",
            "normal",
            "spacious",
            "extra_spacious",
        ):
            set_current_spacing(tier)
            event = QtCore.QEvent(QtCore.QEvent.Type.StyleChange)
            QtWidgets.QApplication.sendEvent(browser, event)
            event = QtCore.QEvent(QtCore.QEvent.Type.StyleChange)
            QtWidgets.QApplication.sendEvent(tools, event)
            qapp.processEvents()
            margins = tools.tool_content.layout().contentsMargins()
            observed.append(
                (
                    browser.list_splitter.handleWidth(),
                    browser.layout().spacing(),
                    tools.tool_content.layout().spacing(),
                    margins.top(),
                )
            )
        for column in zip(*observed):
            assert tuple(column) == tuple(sorted(column))
            assert len(set(column)) == 5
    finally:
        browser.close()
        tools.close()
        browser.deleteLater()
        tools.deleteLater()
        qapp.processEvents()


def test_pending_tools_refit_is_cancelled_with_its_destroyed_owner(qapp):
    tools = ToolsView()
    set_current_spacing("extra_spacious")
    QtWidgets.QApplication.sendEvent(
        tools, QtCore.QEvent(QtCore.QEvent.Type.StyleChange)
    )

    delete(tools)
    qapp.processEvents()

    assert not isValid(tools)
