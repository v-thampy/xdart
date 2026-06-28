from __future__ import annotations

import pytest

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_xye_overlay_filter_is_inert_during_qt_teardown(qapp):
    from PySide6 import QtCore, QtWidgets

    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _XyeOverlayInputFilter,
    )

    items = QtWidgets.QListWidget()
    event_filter = _XyeOverlayInputFilter(
        items, lambda: True, lambda: "Overlay")

    delattr(event_filter, "_is_active")
    event = QtCore.QEvent(QtCore.QEvent.Type.MouseButtonPress)

    assert event_filter.eventFilter(items, event) is False


def test_xye_overlay_filter_accumulating_is_inert_after_method_teardown(qapp):
    from PySide6 import QtWidgets

    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _XyeOverlayInputFilter,
    )

    items = QtWidgets.QListWidget()
    event_filter = _XyeOverlayInputFilter(
        items, lambda: True, lambda: "Overlay")

    delattr(event_filter, "_get_method")

    assert event_filter._accumulating() is False
