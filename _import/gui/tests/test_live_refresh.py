"""Focused tests for throttled live-scan GUI refresh helpers."""

from __future__ import annotations

import os
from types import MethodType, SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from xdart.gui.tabs.static_scan.h5viewer import H5Viewer
from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget


class _FakeItem:
    def __init__(self, text):
        self._text = str(text)
        self._selected = False

    def text(self):
        return self._text

    def setSelected(self, selected):
        self._selected = bool(selected)


class _FakeListWidget:
    def __init__(self, items=()):
        self._items = []
        self._current_row = -1
        self._signals_blocked = False
        self.addItems([str(item) for item in items])

    def addItems(self, items):
        self._items.extend(_FakeItem(item) for item in items)

    def insertItems(self, row, items):
        for offset, item in enumerate(items):
            self._items.insert(row + offset, _FakeItem(item))

    def clear(self):
        self._items.clear()
        self._current_row = -1

    def clearSelection(self):
        for item in self._items:
            item.setSelected(False)

    def selectAll(self):
        for item in self._items:
            item.setSelected(True)

    def count(self):
        return len(self._items)

    def item(self, row):
        return self._items[row]

    def selectedItems(self):
        return [item for item in self._items if item._selected]

    def currentRow(self):
        return self._current_row

    def blockSignals(self, blocked):
        self._signals_blocked = bool(blocked)

    def findItems(self, text, _flags):
        return [item for item in self._items if item.text() == text]

    def setCurrentRow(self, row, _flags=None):
        self.clearSelection()
        self._current_row = row
        if 0 <= row < len(self._items):
            self._items[row].setSelected(True)

    def setCurrentItem(self, item, _flags=None):
        self.clearSelection()
        try:
            self._current_row = self._items.index(item)
        except ValueError:
            self._current_row = -1
            return
        item.setSelected(True)


def _viewer(frame_ids, displayed):
    list_data = _FakeListWidget(displayed)

    calls = []
    viewer = SimpleNamespace(
        sphere=SimpleNamespace(
            name="scan",
            arches=SimpleNamespace(index=list(frame_ids)),
        ),
        ui=SimpleNamespace(listData=list_data),
        new_scan_loaded=False,
        arch_ids=[],
        auto_last=True,
        latest_idx=None,
        data_changed=lambda *args, **kwargs: calls.append((args, kwargs)),
        _displayed_list_count=0,
        _displayed_last_label=None,
    )
    viewer.set_current_frame = MethodType(H5Viewer.set_current_frame, viewer)
    viewer._remember_displayed_frames = MethodType(
        H5Viewer._remember_displayed_frames, viewer,
    )
    viewer.update_data = MethodType(H5Viewer.update_data, viewer)
    viewer._remember_displayed_frames()
    return viewer, list_data, calls


def _labels(list_widget):
    return [
        list_widget.item(row).text()
        for row in range(list_widget.count())
    ]


def test_update_data_append_autolast_selects_only_latest_without_emit():
    viewer, list_data, calls = _viewer([1, 2, 3], [1, 2])
    list_data.selectAll()
    first_item = list_data.item(0)
    viewer.latest_idx = 3

    viewer.update_data(emit_update=False)

    assert _labels(list_data) == ["1", "2", "3"]
    assert list_data.item(0) is first_item
    assert [item.text() for item in list_data.selectedItems()] == ["3"]
    assert calls == []


def test_update_data_same_length_changed_labels_rebuilds_list():
    viewer, list_data, _calls = _viewer([3, 4], [1, 2])
    viewer.auto_last = False
    first_item = list_data.item(0)

    viewer.update_data(emit_update=False)

    assert _labels(list_data) == ["3", "4"]
    assert list_data.item(0) is not first_item


def test_update_data_append_emits_once_when_requested():
    viewer, _list_data, calls = _viewer([1, 2, 3], [1, 2])
    viewer.latest_idx = 3

    viewer.update_data()

    assert len(calls) == 1


def test_flush_pending_update_owns_single_repaint():
    calls = []
    widget = SimpleNamespace(
        _pending_update_idx=5,
        h5viewer=SimpleNamespace(
            auto_last=True,
            update_data=lambda **kwargs: calls.append(("update_data", kwargs)),
        ),
        latest_arch=lambda **kwargs: calls.append(("latest_arch", kwargs)),
        displayframe=SimpleNamespace(
            update=lambda: calls.append(("display", {})),
        ),
        metawidget=SimpleNamespace(
            update=lambda: calls.append(("meta", {})),
        ),
    )

    staticWidget._flush_pending_update(widget)

    assert widget._pending_update_idx is None
    assert calls == [
        ("update_data", {"emit_update": False}),
        ("latest_arch", {"emit_update": False}),
        ("display", {}),
        ("meta", {}),
    ]
