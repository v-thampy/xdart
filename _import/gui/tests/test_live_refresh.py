"""Focused tests for throttled live-scan GUI refresh helpers."""

from __future__ import annotations

import os
from types import MethodType, SimpleNamespace
from threading import RLock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from xdart.gui.tabs.static_scan.h5viewer import H5Viewer
from xdart.gui.tabs.static_scan.display_data import DisplayDataMixin
from xdart.gui.tabs.static_scan.display_frame_widget import displayFrameWidget
from xdart.gui.tabs.static_scan.display_plot import DisplayPlotMixin
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

    def addItem(self, item):
        self._items.append(_FakeItem(item))

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
        prev = self._signals_blocked
        self._signals_blocked = bool(blocked)
        return prev

    def setSelectionMode(self, mode):
        self.selection_mode = mode

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


class _FakeImageWidget:
    def __init__(self):
        self.images = []
        self.rects = []

    def setImage(self, data, *args, **kwargs):
        self.images.append(np.asarray(data))

    def setRect(self, rect):
        self.rects.append(rect)


class _FakeCurve:
    def __init__(self):
        self.cleared = False

    def clear(self):
        self.cleared = True


class _FakeLegend:
    def __init__(self):
        self.cleared = False

    def clear(self):
        self.cleared = True


class _FakeLabel:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = text


class _FakeTimer:
    def __init__(self, active=True):
        self.active = active

    def isActive(self):
        return self.active

    def stop(self):
        self.active = False


class _FakeWorker:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class _FakeControl:
    def __init__(self, checked=False):
        self._checked = bool(checked)
        self._enabled = True
        self._visible = True
        self.blocked = False

    def blockSignals(self, blocked):
        self.blocked = bool(blocked)

    def setChecked(self, checked):
        self._checked = bool(checked)

    def isChecked(self):
        return self._checked

    def setEnabled(self, enabled):
        self._enabled = bool(enabled)

    def isEnabled(self):
        return self._enabled

    def setVisible(self, visible):
        self._visible = bool(visible)

    def isVisible(self):
        return self._visible


class _FakeCombo:
    def __init__(self, text):
        self._text = text

    def currentText(self):
        return self._text


class _FakeIndexedCombo(_FakeCombo):
    def __init__(self, text, index=0):
        super().__init__(text)
        self._index = int(index)

    def currentIndex(self):
        return self._index

    def setCurrentIndex(self, index):
        self._index = int(index)


class _FakeMutableCombo(_FakeIndexedCombo):
    def __init__(self):
        super().__init__("", index=0)
        self._items = []
        self._enabled = True
        self._visible = True
        self.blocked = False

    def blockSignals(self, blocked):
        self.blocked = bool(blocked)

    def clear(self):
        self._items.clear()
        self._index = 0

    def addItem(self, text):
        self._items.append(text)
        if len(self._items) == 1:
            self._text = text

    def count(self):
        return len(self._items)

    def currentText(self):
        if 0 <= self._index < len(self._items):
            return self._items[self._index]
        return ""

    def setCurrentIndex(self, index):
        super().setCurrentIndex(index)
        if 0 <= self._index < len(self._items):
            self._text = self._items[self._index]

    def setEnabled(self, enabled):
        self._enabled = bool(enabled)

    def setVisible(self, visible):
        self._visible = bool(visible)


class _FakeSignal:
    def __init__(self):
        self.emitted = []

    def emit(self, *args):
        self.emitted.append(args)


class _FakeAction:
    def __init__(self):
        self.enabled = True

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)


def _viewer(frame_ids, displayed):
    list_data = _FakeListWidget(displayed)

    calls = []
    viewer = SimpleNamespace(
        scan=SimpleNamespace(
            name="scan",
            frames=SimpleNamespace(index=list(frame_ids)),
        ),
        ui=SimpleNamespace(listData=list_data),
        new_scan_loaded=False,
        frame_ids=[],
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
            data_changed=lambda: calls.append(("data_changed", {})),
        ),
        latest_frame=lambda **kwargs: calls.append(("latest_frame", kwargs)),
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
        ("latest_frame", {"emit_update": False}),
        ("data_changed", {}),
    ]


def _display_host():
    image_widget = _FakeImageWidget()
    binned_widget = _FakeImageWidget()
    wf_widget = _FakeImageWidget()
    label = _FakeLabel()
    curve = _FakeCurve()
    legend = _FakeLegend()
    host = SimpleNamespace(
        image_data=(np.ones((3, 3)), None),
        binned_data=(np.ones((3, 3)), None),
        plot_data=[np.arange(3), np.ones((1, 3))],
        plot_data_range=[[0, 3], [0, 1]],
        frame_names=["old"],
        image_widget=image_widget,
        binned_widget=binned_widget,
        wf_widget=wf_widget,
        curves=[curve],
        legend=legend,
        ui=SimpleNamespace(labelCurrent=label),
    )
    host.clear_overlay = MethodType(displayFrameWidget.clear_overlay, host)
    host.clear_image_view = MethodType(displayFrameWidget.clear_image_view, host)
    host.clear_binned_view = MethodType(displayFrameWidget.clear_binned_view, host)
    host.clear_plot_view = MethodType(displayFrameWidget.clear_plot_view, host)
    host.clear_display_state = MethodType(displayFrameWidget.clear_display_state, host)
    return host, image_widget, binned_widget, wf_widget, curve, legend, label


def test_clear_display_state_resets_visible_and_cached_state():
    host, image_widget, binned_widget, wf_widget, curve, legend, label = _display_host()

    host.clear_display_state("XYE Viewer")

    np.testing.assert_array_equal(host.image_data[0], np.zeros((2, 2)))
    np.testing.assert_array_equal(host.binned_data[0], np.zeros((2, 2)))
    assert host.plot_data[0].size == 0
    assert host.plot_data[1].size == 0
    assert host.plot_data_range == [[0, 0], [0, 0]]
    assert host.frame_names == []
    assert host.overlaid_idxs == []
    assert host.curves == []
    assert curve.cleared is True
    assert legend.cleared is True
    assert label.text == "XYE Viewer"
    np.testing.assert_array_equal(image_widget.images[-1], np.zeros((2, 2)))
    np.testing.assert_array_equal(binned_widget.images[-1], np.zeros((2, 2)))
    np.testing.assert_array_equal(wf_widget.images[-1], np.zeros((2, 2)))


def test_display_generation_bumps_on_mode_switch_and_selection():
    # Stage 2: the monotonic display generation must advance on a mode
    # switch (the exact case that caused stale renders) and on a selection
    # change, but not when nothing changed.
    from unittest.mock import MagicMock

    host = SimpleNamespace(
        viewer_mode=None,
        display_generation=0,
        _last_selection_sig=None,
        idxs=[],
        overall=False,
        _viewer_x_axis_label=None,
        ui=MagicMock(),
        _showImageBtn=MagicMock(),
    )
    host.clear_display_state = MagicMock()
    host._set_2d_controls_visible = MagicMock()
    host._bump_display_generation = MethodType(
        displayFrameWidget._bump_display_generation, host)
    host._note_selection_generation = MethodType(
        displayFrameWidget._note_selection_generation, host)
    host.set_viewer_display_mode = MethodType(
        displayFrameWidget.set_viewer_display_mode, host)

    # Mode switch bumps; re-selecting the same mode does not.
    host.set_viewer_display_mode('image')
    assert host.display_generation == 1
    host.set_viewer_display_mode('image')
    assert host.display_generation == 1
    host.set_viewer_display_mode('xye')
    assert host.display_generation == 2

    # Selection: first call records the baseline (no bump), then changes bump.
    host._note_selection_generation()
    assert host.display_generation == 2
    host.idxs = [0, 1]
    host._note_selection_generation()
    assert host.display_generation == 3
    host._note_selection_generation()          # unchanged
    assert host.display_generation == 3


def _render_host():
    """A host that records which draw/clear delegates render_display calls."""
    from unittest.mock import MagicMock
    from xdart.gui.tabs.static_scan import display_logic as dl

    calls = []
    def rec(name):
        return lambda *a, **k: calls.append(name)

    host = SimpleNamespace(
        ui=MagicMock(),
        plot=MagicMock(),
        binned_widget=MagicMock(),
        update_image=rec("draw_image"),
        update_binned=rec("draw_binned"),
        update_plot=rec("draw_plot"),
        _update_image_viewer=rec("draw_viewer_image"),
        _update_xye_viewer=rec("draw_viewer_xye"),
        clear_image_view=rec("clear_image"),
        clear_binned_view=rec("clear_binned"),
        clear_plot_view=rec("clear_plot"),
        _apply_1d_only_visibility=rec("apply_1d_only"),
        update_2d_label=rec("label_2d"),
        _update_image_preview=rec("preview"),
    )
    host.ui.shareAxis.isChecked.return_value = False
    host.ui.imageUnit.currentIndex.return_value = 0
    for name in ("_draw_delegate", "_clear_delegate", "render_display"):
        setattr(host, name, MethodType(getattr(displayFrameWidget, name), host))
    return host, calls, dl


def test_render_display_int2d_draws_all_panels():
    host, calls, dl = _render_host()
    state = dl.compute_display_state(
        mode=dl.Mode.INT_2D, selected_ids=(0,), all_frame_index=[0],
        loaded_1d_keys={0}, loaded_2d_keys={0}, gi=False, plot_unit='q_A^-1',
        method='Single', unit_changed=False, prev_overlaid_ids=(),
        raw_availability={0: dict(has_raw=True)}, titles={}, generation=1)
    host.render_display(state, dl.build_payload(state))
    assert "draw_plot" in calls and "draw_image" in calls and "draw_binned" in calls
    assert "label_2d" in calls and "preview" in calls
    assert not any(c.startswith("clear_") for c in calls)


def test_render_display_image_viewer_draws_raw_clears_others():
    host, calls, dl = _render_host()
    state = dl.compute_display_state(
        mode=dl.Mode.IMAGE_VIEWER, selected_ids=(0,), all_frame_index=[],
        loaded_1d_keys=set(), loaded_2d_keys={0}, gi=False, plot_unit='q_A^-1',
        method='Single', unit_changed=False, prev_overlaid_ids=(),
        raw_availability={0: dict(has_raw=True)}, titles={}, generation=1)
    host.render_display(state, dl.build_payload(state))
    assert "draw_viewer_image" in calls         # RAW_2D via the viewer method
    assert "clear_binned" in calls and "clear_plot" in calls  # absent panels blanked
    assert "label_2d" not in calls              # viewer sets its own title
    assert "draw_image" not in calls


def test_render_display_drops_stale_generation():
    host, calls, dl = _render_host()
    state = dl.compute_display_state(
        mode=dl.Mode.INT_2D, selected_ids=(0,), all_frame_index=[0],
        loaded_1d_keys={0}, loaded_2d_keys={0}, gi=False, plot_unit='q_A^-1',
        method='Single', unit_changed=False, prev_overlaid_ids=(),
        raw_availability={0: dict(has_raw=True)}, titles={}, generation=7)
    stale = dl.DisplayPayload(generation=6, raw_image=None, cake_image=None, plot=None)
    host.render_display(state, stale)
    assert calls == []                           # nothing drawn or cleared


def test_shadow_display_state_check_agrees_and_never_raises(caplog):
    # Stage 2 shadow mode: building a DisplayState from live inputs must
    # agree with the existing render path's idxs and never raise into the
    # session.  (Evidence for "no shadow assert fires" during use.)
    import logging

    host = SimpleNamespace(
        viewer_mode=None,
        display_generation=3,
        frame_ids=['0', '1'],
        idxs=[0, 1], idxs_1d=[0, 1], idxs_2d=[0, 1],
        overall=True,
        overlaid_idxs=[],
        data_1d={0: object(), 1: object()},
        data_2d={
            0: {'map_raw': np.ones((2, 2)), 'thumbnail': None},
            1: {'map_raw': np.ones((2, 2)), 'thumbnail': None},
        },
        data_lock=RLock(),
        scan=SimpleNamespace(scan_lock=RLock(),
                             frames=SimpleNamespace(index=[0, 1]), gi=False),
        ui=SimpleNamespace(plotMethod=SimpleNamespace(currentText=lambda: 'Single')),
    )
    for name in ('_live_mode', '_live_display_state', '_shadow_check_display_state'):
        setattr(host, name, MethodType(getattr(displayFrameWidget, name), host))

    logger_name = 'xdart.gui.tabs.static_scan.display_frame_widget'
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        host._shadow_check_display_state()       # must not raise

    assert not any('shadow: render_ids' in r.getMessage() for r in caplog.records)


def test_enter_viewer_mode_cleanup_clears_lists_and_cancels_loader():
    worker = _FakeWorker()
    timer = _FakeTimer(active=True)
    list_data = _FakeListWidget([1, 2])
    list_scans = _FakeListWidget(["old.nxs"])
    list_data.selectAll()
    list_scans.selectAll()
    viewer = SimpleNamespace(
        data_lock=RLock(),
        data_1d={1: object()},
        data_2d={1: {"map_raw": np.ones((2, 2))}},
        frame_ids=["1"],
        latest_idx=1,
        new_scan_loaded=True,
        _raw_cache_order=[1],
        _load_generation=4,
        _load_worker=worker,
        _update_coalesce_timer=timer,
        _viewer_image_path="/tmp/old.tif",
        _viewer_image_nframes=10,
        _viewer_is_xdart=True,
        ui=SimpleNamespace(listData=list_data, listScans=list_scans),
        _displayed_list_count=2,
        _displayed_last_label="2",
    )
    viewer._clear_raw_cache = MethodType(H5Viewer._clear_raw_cache, viewer)
    viewer._remember_displayed_frames = MethodType(
        H5Viewer._remember_displayed_frames, viewer,
    )
    viewer.cancel_pending_loads = MethodType(H5Viewer.cancel_pending_loads, viewer)
    viewer.enter_viewer_mode_cleanup = MethodType(
        H5Viewer.enter_viewer_mode_cleanup, viewer,
    )

    viewer.enter_viewer_mode_cleanup()

    assert worker.cancelled is True
    assert timer.isActive() is False
    assert viewer._load_generation == 5
    assert viewer.data_1d == {}
    assert viewer.data_2d == {}
    assert viewer.frame_ids == []
    assert viewer._raw_cache_order == []
    assert viewer.latest_idx is None
    assert viewer.new_scan_loaded is False
    assert list_data.count() == 0
    assert list_scans.selectedItems() == []
    assert viewer._displayed_list_count == 0
    assert not hasattr(viewer, "_viewer_image_path")


def test_viewer_mode_change_blocks_scan_list_autoload():
    calls = []
    list_scans = _FakeListWidget(["old.xye"])

    def update_scans():
        calls.append(("update_scans_blocked", list_scans._signals_blocked))
        if not list_scans._signals_blocked:
            calls.append("autoload")

    widget = SimpleNamespace(
        wrangler=object(),
        h5viewer=SimpleNamespace(
            ui=SimpleNamespace(listScans=list_scans),
            actionNewFile=_FakeAction(),
            actionSaveDataAs=_FakeAction(),
            viewer_mode="xye",
            _suspend_scan_selection_loads=False,
            enter_viewer_mode_cleanup=lambda: calls.append(
                ("cleanup_suspend", widget.h5viewer._suspend_scan_selection_loads),
            ),
            cancel_pending_loads=lambda: calls.append("cancel"),
            update_scans=update_scans,
        ),
        displayframe=SimpleNamespace(
            _wrangler=None,
            set_viewer_display_mode=lambda mode: calls.append(("display", mode)),
            clear_display_state=lambda: calls.append("clear_display"),
        ),
    )

    staticWidget._on_viewer_mode_changed(widget, "image")

    assert ("cleanup_suspend", True) in calls
    assert ("update_scans_blocked", True) in calls
    assert "autoload" not in calls
    assert widget.h5viewer._suspend_scan_selection_loads is False
    assert list_scans._signals_blocked is False


def test_scan_selection_handlers_ignore_suspended_loads():
    calls = []
    item = _FakeItem("scan.nxs")
    list_scans = _FakeListWidget(["scan.nxs"])
    list_scans.item(0).setSelected(True)
    viewer = SimpleNamespace(
        _suspend_scan_selection_loads=True,
        viewer_mode="image",
        ui=SimpleNamespace(listScans=list_scans),
        scans_clicked=lambda q: calls.append(("scan", q.text())),
        _load_xye_files=lambda: calls.append("xye"),
    )

    H5Viewer._scans_single_clicked(viewer, item)
    H5Viewer._scans_current_changed(viewer, item, None)
    viewer.viewer_mode = "xye"
    H5Viewer._scans_selection_changed(viewer)

    assert calls == []


def test_old_processed_xdart_nxs_not_loaded_as_generic_image(tmp_path):
    import h5py

    path = tmp_path / "old_processed.nxs"
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.create_group("reduction")

    calls = []
    viewer = SimpleNamespace(
        data_lock=RLock(),
        data_1d={1: object()},
        data_2d={1: {"map_raw": np.ones((2, 2))}},
        frame_ids=["1"],
        ui=SimpleNamespace(listData=_FakeListWidget([1])),
        _raw_cache_order=[1],
        _viewer_is_xdart=False,
        _remember_displayed_frames=lambda: calls.append("remember"),
        sigUpdate=_FakeSignal(),
        _is_xdart_processed=H5Viewer._is_xdart_processed,
        _load_single_frame=lambda *args, **kwargs: calls.append("load"),
    )
    # Bind the cache clear after the namespace exists.
    viewer._clear_raw_cache = MethodType(H5Viewer._clear_raw_cache, viewer)
    viewer._load_image_file = MethodType(H5Viewer._load_image_file, viewer)

    viewer._load_image_file(str(path))

    assert viewer._viewer_is_xdart is True
    assert calls == ["remember"]
    assert viewer.ui.listData.count() == 0
    assert viewer.data_1d == {}
    assert viewer.data_2d == {}
    assert viewer.frame_ids == []
    assert viewer.sigUpdate.emitted == [()]


def test_reduction_group_raw_nexus_loads_as_generic_image(tmp_path):
    import h5py

    path = tmp_path / "raw_with_reduction.nxs"
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.create_group("reduction")
        data = entry.create_group("data")
        data.create_dataset("data", data=np.ones((2, 3, 4)))

    calls = []
    viewer = SimpleNamespace(
        data_lock=RLock(),
        data_1d={1: object()},
        data_2d={1: {"map_raw": np.ones((2, 2))}},
        frame_ids=["1"],
        ui=SimpleNamespace(listData=_FakeListWidget([1])),
        _raw_cache_order=[1],
        _viewer_is_xdart=True,
        _remember_displayed_frames=lambda: calls.append("remember"),
        sigUpdate=_FakeSignal(),
        _is_xdart_processed=H5Viewer._is_xdart_processed,
        _load_single_frame=lambda _path, frame_idx=0, frame_id=1: calls.append(
            ("load", frame_idx, frame_id),
        ),
    )
    viewer._clear_raw_cache = MethodType(H5Viewer._clear_raw_cache, viewer)
    viewer._populate_image_viewer_rows = MethodType(
        H5Viewer._populate_image_viewer_rows, viewer,
    )
    viewer._load_image_file = MethodType(H5Viewer._load_image_file, viewer)

    viewer._load_image_file(str(path))

    assert viewer._viewer_is_xdart is False
    assert viewer.ui.listData.count() == 2
    assert [viewer.ui.listData.item(i).text() for i in range(2)] == ["1", "2"]
    assert viewer.frame_ids == ["1"]
    assert calls == [("load", 0, 1), "remember"]
    assert viewer.sigUpdate.emitted == [()]


def test_processed_xdart_markers_still_short_circuit_image_loader(tmp_path):
    import h5py

    for marker in ("integrated_1d", "integrated_2d", "frames"):
        path = tmp_path / f"processed_{marker}.nxs"
        with h5py.File(path, "w") as f:
            entry = f.create_group("entry")
            entry.create_group(marker)

        assert H5Viewer._is_xdart_processed(str(path)) is True


def test_image_viewer_missing_raw_and_thumbnail_clears_panel():
    calls = []
    host = SimpleNamespace(
        idxs_2d=[1],
        data_lock=RLock(),
        data_2d={1: {"map_raw": None, "thumbnail": None}},
        _wrangler=None,
        clear_image_view=lambda: calls.append("clear"),
    )

    displayFrameWidget._update_image_viewer(host)

    assert calls == ["clear"]


def test_image_viewer_all_sentinel_image_clears_safely():
    calls = []
    host = SimpleNamespace(
        idxs_2d=[1],
        data_lock=RLock(),
        data_2d={1: {"map_raw": np.full((2, 2), 4294967295.0),
                     "thumbnail": None}},
        _wrangler=None,
        clear_image_view=lambda: calls.append("clear"),
        _set_viewer_title=lambda idxs: calls.append(("title", tuple(idxs))),
        _sanitize_display_image=DisplayDataMixin._sanitize_display_image,
    )

    displayFrameWidget._update_image_viewer(host)

    assert calls == [("title", (1,)), "clear"]


def test_image_viewer_masks_uint16_ceiling_sentinel_before_autoscale():
    raw = np.array([[10.0, 65535.0], [20.0, 30.0]])
    host = SimpleNamespace(
        idxs_2d=[1],
        data_lock=RLock(),
        data_2d={1: {"map_raw": raw, "thumbnail": None}},
        _wrangler=None,
        clear_image_view=lambda: None,
        _set_viewer_title=lambda idxs: None,
        _sanitize_display_image=DisplayDataMixin._sanitize_display_image,
        update_image_view=lambda: None,
    )

    displayFrameWidget._update_image_viewer(host)

    assert np.isnan(host.image_data[0]).sum() == 1
    assert np.nanmax(host.image_data[0]) == 30.0


def _plot_host(method="Overlay"):
    unit = _FakeIndexedCombo("Q", 0)
    method_combo = _FakeCombo(method)
    slice_control = _FakeControl(False)
    slice_control.setEnabled(True)
    host = SimpleNamespace(
        scan=SimpleNamespace(name="scan", series_average=False),
        frame_ids=["1"],
        idxs=[1],
        idxs_1d=[1],
        idxs_2d=[],
        data_1d={1: object(), 2: object(), 3: object()},
        data_2d={},
        plot_data=[np.zeros(0), np.zeros(0)],
        plot_data_range=[[0, 0], [0, 0]],
        frame_names=[],
        overlaid_idxs=[],
        bkg_1d=None,
        _last_plot_unit=-1,
        _plot_axis_info=[],
        ui=SimpleNamespace(
            plotUnit=unit,
            plotMethod=method_combo,
            slice=slice_control,
            slice_center=SimpleNamespace(value=lambda: 0.0),
            slice_width=SimpleNamespace(value=lambda: 1.0),
        ),
        update_plot_view=lambda: None,
    )

    def get_frames_int_1d(idxs=None, rv="all"):
        ids = list(host.idxs_1d if idxs is None else idxs)
        x = np.array([0.0, 1.0]) + host.ui.plotUnit.currentIndex() * 10.0
        rows = np.vstack([
            np.array([float(idx), float(idx) + 0.5])
            for idx in ids if int(idx) in host.data_1d
        ])
        if rows.shape[0] == 1:
            rows = rows[0]
        return rows, x

    def get_int_1d(_frame, _frame_2d, idx):
        x = np.array([0.0, 1.0]) + host.ui.plotUnit.currentIndex() * 10.0
        return x, np.array([float(idx), float(idx) + 0.5])

    host.get_frames_int_1d = get_frames_int_1d
    host.get_int_1d = get_int_1d
    host.update_plot = MethodType(DisplayPlotMixin.update_plot, host)
    host._loaded_1d_overlay_labels = MethodType(
        DisplayPlotMixin._loaded_1d_overlay_labels, host,
    )
    return host


def test_overlay_unit_switch_rebuilds_all_accumulated_curves():
    host = _plot_host("Overlay")
    for idx in (1, 2, 3):
        host.idxs = [idx]
        host.idxs_1d = [idx]
        host.update_plot()

    assert host.plot_data[1].shape == (3, 2)
    assert host.frame_names == ["scan_1", "scan_2", "scan_3"]
    assert host.overlaid_idxs == [1, 2, 3]

    host.ui.plotUnit.setCurrentIndex(1)
    host.idxs = [3]
    host.idxs_1d = [3]
    host.update_plot()

    assert host.plot_data[1].shape == (3, 2)
    assert host.frame_names == ["scan_1", "scan_2", "scan_3"]
    assert host.overlaid_idxs == [1, 2, 3]
    np.testing.assert_array_equal(host.plot_data[0], np.array([10.0, 11.0]))


def test_waterfall_unit_switch_rebuilds_all_accumulated_curves():
    host = _plot_host("Waterfall")
    for idx in (1, 2, 3):
        host.idxs = [idx]
        host.idxs_1d = [idx]
        host.update_plot()

    host.ui.plotUnit.setCurrentIndex(1)
    host.update_plot()

    assert host.plot_data[1].shape == (3, 2)
    assert host.frame_names == ["scan_1", "scan_2", "scan_3"]


def test_single_multiselect_unit_switch_still_uses_current_selection():
    host = _plot_host("Single")
    host.idxs = [1, 2, 3]
    host.idxs_1d = [1, 2, 3]
    host.update_plot()

    host.ui.plotUnit.setCurrentIndex(1)
    host.update_plot()

    assert host.plot_data[1].shape == (3, 2)
    assert host.frame_names == ["scan_1", "scan_2", "scan_3"]
    assert host.overlaid_idxs == [1, 2, 3]


def test_xye_single_method_change_clears_accumulated_traces_immediately():
    calls = []
    host = SimpleNamespace(
        viewer_mode="xye",
        ui=SimpleNamespace(plotMethod=_FakeCombo("Single")),
        sigPlotMethodChanged=_FakeSignal(),
        plot_data=[np.arange(2), np.ones((2, 2))],
        plot_data_range=[[0, 1], [0, 1]],
        frame_names=["old_a", "old_b"],
        overlaid_idxs=[1, 2],
        _update_xye_viewer=lambda: calls.append("xye"),
    )
    host.clear_overlay = MethodType(displayFrameWidget.clear_overlay, host)

    DisplayPlotMixin._on_plotMethod_changed(host)

    assert calls == ["xye"]
    assert host.plot_data[0].size == 0
    assert host.plot_data[1].size == 0
    assert host.frame_names == []
    assert host.overlaid_idxs == []


def test_xye_axis_label_uses_file_unit_not_hidden_transform_combo():
    host = SimpleNamespace(
        viewer_mode="xye",
        _viewer_x_axis_label=(u"2\u03b8", u"\u00b0"),
        ui=SimpleNamespace(plotUnit=_FakeCombo("Q (\u212b\u207b\u00b9)")),
    )

    assert DisplayPlotMixin._current_plot_axis_label(host) == (
        u"2\u03b8", u"\u00b0",
    )


def test_xye_loader_marks_unknown_units_without_guessing(monkeypatch, tmp_path):
    import xdart.gui.tabs.static_scan.h5viewer as h5viewer_mod

    monkeypatch.setattr(
        h5viewer_mod, "read_xye",
        lambda _path: (
            np.array([0.0, 1.0]),
            np.array([2.0, 3.0]),
            np.array([0.1, 0.1]),
        ),
    )
    for name in ("iq_scan.xye", "itth_scan.xye", "plain_scan.xye"):
        (tmp_path / name).write_text("0 1 0.1\n", encoding="utf-8")
    viewer = SimpleNamespace(
        dirname=str(tmp_path),
        data_lock=RLock(),
        data_1d={},
        data_2d={},
        frame_ids=[],
        ui=SimpleNamespace(
            listScans=SimpleNamespace(
                selectedItems=lambda: [
                    _FakeItem("iq_scan.xye"),
                    _FakeItem("itth_scan.xye"),
                    _FakeItem("plain_scan.xye"),
                ],
            ),
            listData=_FakeListWidget(),
        ),
        _remember_displayed_frames=lambda: None,
        sigUpdate=_FakeSignal(),
    )

    H5Viewer._load_xye_files(viewer)

    assert viewer.data_1d[1].int_1d.unit == "q_A^-1"
    assert viewer.data_1d[2].int_1d.unit == "2th_deg"
    assert viewer.data_1d[3].int_1d.unit == "unknown"


def test_standard_plot_axis_defaults_to_integrated_2theta_unit():
    plot_unit = _FakeMutableCombo()
    image_unit = _FakeMutableCombo()
    host = SimpleNamespace(
        scan=SimpleNamespace(
            gi=False,
            bai_1d_args={"unit": "2th_deg"},
            bai_2d_args={},
        ),
        ui=SimpleNamespace(plotUnit=plot_unit, imageUnit=image_unit),
        _plot_axis_info=[],
        _on_plotUnit_changed=lambda: None,
    )

    displayFrameWidget.set_axes(host)

    assert plot_unit.currentIndex() == 1
    assert "2" in plot_unit.currentText()


def test_processed_image_viewer_falls_back_to_thumbnail(monkeypatch):
    import ssrl_xrd_tools.io as ssrl_io

    thumb = np.array([[5.0, 6.0], [7.0, 8.0]])

    def fake_get_raw_frame(_path, *, frame, allow_thumbnail=True):
        assert frame == 5
        if not allow_thumbnail:
            raise OSError("missing source")
        return thumb

    monkeypatch.setattr(ssrl_io, "get_raw_frame", fake_get_raw_frame)
    viewer = SimpleNamespace(
        _viewer_is_xdart=True,
        data_lock=RLock(),
        data_1d={},
        data_2d={},
    )

    H5Viewer._load_single_frame(
        viewer, "processed.nxs", frame_idx=0, frame_id=5,
    )

    np.testing.assert_array_equal(viewer.data_2d[5]["map_raw"], thumb)
    np.testing.assert_array_equal(viewer.data_2d[5]["thumbnail"], thumb)


def test_processed_image_viewer_reads_thumbnail_directly_if_helper_fails(
        monkeypatch, tmp_path):
    import h5py
    import ssrl_xrd_tools.io as ssrl_io

    def failing_get_raw_frame(*_args, **_kwargs):
        raise KeyError("missing source and helper fallback")

    monkeypatch.setattr(ssrl_io, "get_raw_frame", failing_get_raw_frame)
    path = tmp_path / "processed_thumbnail_only.nxs"
    with h5py.File(path, "w") as f:
        ds = f.create_dataset(
            "entry/frames/frame_0005/thumbnail",
            data=(np.ones((3, 4)) * 128).astype(np.uint8),
        )
        ds.attrs["vmin"] = 10.0
        ds.attrs["vmax"] = 20.0
        ds.attrs["dtype"] = "uint8"

    viewer = SimpleNamespace(
        _viewer_is_xdart=True,
        data_lock=RLock(),
        data_1d={},
        data_2d={},
    )
    viewer._read_processed_thumbnail = H5Viewer._read_processed_thumbnail

    loaded = H5Viewer._load_single_frame(
        viewer, str(path), frame_idx=5, frame_id=5,
    )

    assert loaded is True
    assert 5 in viewer.data_2d
    assert viewer.data_2d[5]["map_raw"].shape == (3, 4)
    np.testing.assert_allclose(
        viewer.data_2d[5]["map_raw"],
        10.0 + (128 / 255) * 10.0,
    )


def test_image_viewer_single_raw_file_gets_selectable_frame(tmp_path):
    path = tmp_path / "single.raw"
    raw = np.arange(195 * 487, dtype=np.int32).reshape(195, 487)
    path.write_bytes(raw.tobytes())

    viewer = SimpleNamespace(
        data_lock=RLock(),
        data_1d={},
        data_2d={},
        frame_ids=[],
        ui=SimpleNamespace(listData=_FakeListWidget()),
        _raw_cache_order=[],
        _is_xdart_processed=H5Viewer._is_xdart_processed,
        _remember_displayed_frames=lambda: None,
        sigUpdate=_FakeSignal(),
    )
    viewer._try_raw_detectors = MethodType(H5Viewer._try_raw_detectors, viewer)
    viewer._load_single_frame = MethodType(H5Viewer._load_single_frame, viewer)
    viewer._populate_image_viewer_rows = MethodType(
        H5Viewer._populate_image_viewer_rows, viewer,
    )
    viewer._load_image_file = MethodType(H5Viewer._load_image_file, viewer)

    viewer._load_image_file(str(path))

    assert viewer.ui.listData.count() == 1
    assert viewer.ui.listData.item(0).text() == "1"
    assert viewer.ui.listData.selectedItems()[0].text() == "1"
    assert viewer.frame_ids == ["1"]
    np.testing.assert_array_equal(viewer.data_2d[1]["map_raw"], raw)
    assert viewer.sigUpdate.emitted == [()]


def test_gi_2d_auto_ranges_freeze_from_scout_result():
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        _freeze_gi_2d_ranges_from_result,
    )

    args = {"gi_mode_2d": "qip_qoop", "x_range": None, "y_range": None}
    result = SimpleNamespace(
        radial=np.array([0.0, 10.0]),
        azimuthal=np.array([1.0, 3.0]),
    )

    assert _freeze_gi_2d_ranges_from_result(args, result) is True
    assert args["x_range"] == (-0.2, 10.2)
    assert args["y_range"] == (0.96, 3.04)


def test_gi_2d_auto_ranges_use_radial_keys_for_q_chi():
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        _freeze_gi_2d_ranges_from_result,
    )

    args = {
        "gi_mode_2d": "q_chi",
        "radial_range": None,
        "azimuth_range": (-180, 180),
    }
    result = SimpleNamespace(
        radial=np.array([1.0, 5.0]),
        azimuthal=np.array([-90.0, 90.0]),
    )

    assert _freeze_gi_2d_ranges_from_result(args, result) is True
    assert args["radial_range"] == (0.92, 5.08)
    assert args["azimuth_range"] == (-180, 180)


def test_gi_1d_auto_range_freezes_from_scout_result():
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        _freeze_gi_1d_range_from_result,
    )

    args = {"gi_mode_1d": "q_ip", "radial_range": None}
    result = SimpleNamespace(radial=np.array([2.0, 8.0]))

    assert _freeze_gi_1d_range_from_result(args, result) is True
    assert args["radial_range"] == (1.88, 8.12)


def test_raw_preview_does_not_lazy_load_on_gui_data_access():
    calls = []
    frame = SimpleNamespace(
        scan_info={},
        thumbnail=np.ones((2, 2)),
        map_raw=None,
        _lazy_load_raw=lambda: calls.append("loaded"),
    )
    host = SimpleNamespace(
        idxs_2d=[1],
        data_lock=RLock(),
        data_1d={1: frame},
        data_2d={1: {"map_raw": None, "thumbnail": np.ones((2, 2))}},
        normalize=lambda arr, _info: arr,
    )
    host._snapshot_data = MethodType(DisplayDataMixin._snapshot_data, host)
    data = DisplayDataMixin.get_frames_map_raw(host)
    np.testing.assert_array_equal(data, np.ones((2, 2)))
    assert calls == []


def test_map_raw_masks_eiger_sentinel_before_display_average():
    raw = np.array([[2.0, 4294967295.0], [4.0, 8.0]])
    host = SimpleNamespace(
        idxs_2d=[1],
        data_lock=RLock(),
        data_1d={1: SimpleNamespace(scan_info={})},
        data_2d={1: {"map_raw": raw, "bg_raw": 0, "thumbnail": None}},
        normalize=lambda arr, _info: arr,
    )
    host._snapshot_data = MethodType(DisplayDataMixin._snapshot_data, host)

    data = DisplayDataMixin.get_frames_map_raw(host)

    assert data[0, 0] == 2.0
    assert np.isnan(data[0, 1])
    assert data[1, 0] == 4.0


def test_normal_raw_display_all_sentinel_image_clears_safely():
    calls = []
    host = SimpleNamespace(
        overall=False,
        frame_ids=["1"],
        idxs_2d=[1],
        data_lock=RLock(),
        data_2d={1: {"mask": None}},
        scan=SimpleNamespace(global_mask=None),
        bkg_map_raw=0,
        get_frames_map_raw=lambda **kwargs: (
            np.full((2, 2), np.nan), "raw",
        ),
        clear_image_view=lambda: calls.append("clear"),
    )

    displayFrameWidget.update_image(host)

    assert calls == ["clear"]


def test_overall_preview_prefers_bounded_thumbnail_data():
    frame = SimpleNamespace(scan_info={}, thumbnail=np.ones((2, 2)))
    host = SimpleNamespace(
        idxs_2d=[1],
        data_lock=RLock(),
        data_1d={1: frame},
        data_2d={1: {"map_raw": np.full((8, 8), 9.0),
                     "thumbnail": np.ones((2, 2))}},
        normalize=lambda arr, _info: arr,
    )
    host._snapshot_data = MethodType(DisplayDataMixin._snapshot_data, host)
    data = DisplayDataMixin.get_frames_map_raw(host, prefer_thumbnail=True)
    np.testing.assert_array_equal(data, np.ones((2, 2)))


def test_overall_preview_requires_all_requested_frames_when_strict():
    frame = SimpleNamespace(scan_info={}, thumbnail=np.ones((2, 2)))
    host = SimpleNamespace(
        idxs_2d=[1, 2],
        data_lock=RLock(),
        data_1d={1: frame},
        data_2d={1: {"map_raw": None, "thumbnail": np.ones((2, 2))}},
        normalize=lambda arr, _info: arr,
    )
    host._snapshot_data = MethodType(DisplayDataMixin._snapshot_data, host)

    data = DisplayDataMixin.get_frames_map_raw(
        host, [1, 2], prefer_thumbnail=True, require_all=True,
    )

    assert data is None


def test_overall_2d_requires_all_requested_frames_when_strict():
    result = SimpleNamespace(
        intensity=np.ones((2, 2)),
        radial=np.array([1.0, 2.0]),
        azimuthal=np.array([0.0, 1.0]),
    )
    frame = SimpleNamespace(scan_info={})
    host = SimpleNamespace(
        idxs_2d=[1, 2],
        data_lock=RLock(),
        data_1d={1: frame},
        data_2d={1: {"int_2d": result, "gi_2d": {}}},
        get_int_2d=lambda int_2d, frame_1d, gi_2d=None: int_2d.intensity,
        get_xydata=lambda int_2d, gi_2d=None, frame=None: (
            int_2d.radial, int_2d.azimuthal,
        ),
    )
    host._snapshot_data = MethodType(DisplayDataMixin._snapshot_data, host)

    intensity, xdata, ydata = DisplayDataMixin.get_frames_int_2d(
        host, [1, 2], require_all=True,
    )

    assert intensity is None
    assert xdata is None
    assert ydata is None


def test_map_raw_reports_thumbnail_source():
    frame = SimpleNamespace(scan_info={}, thumbnail=np.ones((2, 2)))
    host = SimpleNamespace(
        idxs_2d=[1],
        data_lock=RLock(),
        data_1d={1: frame},
        data_2d={1: {"map_raw": None, "thumbnail": np.ones((2, 2))}},
        normalize=lambda arr, _info: arr,
    )
    host._snapshot_data = MethodType(DisplayDataMixin._snapshot_data, host)
    data, source = DisplayDataMixin.get_frames_map_raw(host, return_source=True)
    np.testing.assert_array_equal(data, np.ones((2, 2)))
    assert source == "thumbnail"


def test_thumbnail_image_update_skips_full_detector_flat_mask():
    calls = []
    host = SimpleNamespace(
        overall=False,
        frame_ids=[1],
        idxs_2d=[1],
        data_lock=RLock(),
        data_2d={1: {"mask": np.array([0], dtype=int)}},
        scan=SimpleNamespace(global_mask=np.array([1], dtype=int)),
        bkg_map_raw=0,
        get_frames_map_raw=lambda **kwargs: (np.ones((2, 2)), "thumbnail"),
        update_image_view=lambda: calls.append("updated"),
    )

    displayFrameWidget.update_image(host)

    assert calls == ["updated"]
    assert np.isfinite(host.image_data[0]).all()


def test_hydrated_raw_cache_evicts_only_full_resolution_payload():
    viewer = SimpleNamespace(
        _raw_cache_order=[],
        _raw_cache_limit=2,
        data_2d={
            idx: {"map_raw": np.full((4, 4), idx), "thumbnail": np.ones((2, 2))}
            for idx in (1, 2, 3)
        },
    )
    viewer._remember_hydrated_raw = MethodType(H5Viewer._remember_hydrated_raw, viewer)
    for idx in (1, 2, 3):
        viewer._remember_hydrated_raw(idx)
    assert viewer._raw_cache_order == [2, 3]
    assert viewer.data_2d[1]["map_raw"] is None
    assert viewer.data_2d[1]["thumbnail"] is not None


def test_hydrated_raw_cache_reset_clears_order():
    viewer = SimpleNamespace(_raw_cache_order=[1, 2, 3])
    viewer._clear_raw_cache = MethodType(H5Viewer._clear_raw_cache, viewer)
    viewer._clear_raw_cache()
    assert viewer._raw_cache_order == []


def _wrangler_host(mode_text, *, live=False, batch=False):
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    ui = SimpleNamespace(
        processingModeCombo=_FakeCombo(mode_text),
        liveCheckBox=_FakeControl(live),
        batchCheckBox=_FakeControl(batch),
        coresLabel=_FakeControl(),
        maxCoresSpinBox=_FakeControl(),
        startButton=_FakeControl(),
        stopButton=_FakeControl(),
        frame=_FakeControl(),
    )
    integration_calls = []
    host = SimpleNamespace(
        ui=ui,
        tree=_FakeControl(),
        live_mode=live,
        batch_mode=batch,
        xye_only=False,
        scan=SimpleNamespace(skip_2d=None),
        thread=SimpleNamespace(batch_mode=None, xye_only=None, live_mode=None),
        sigViewerModeChanged=_FakeSignal(),
        sigStart=_FakeSignal(),
        sender=lambda: None,
        _integration_calls=integration_calls,
        _set_integration_controls_enabled=lambda enabled, **kwargs: (
            integration_calls.append((enabled, kwargs)),
            setattr(host, "_integration_controls_enabled", enabled),
        ),
        _set_gi_controls_readonly=lambda readonly: setattr(
            host, "_gi_readonly", readonly,
        ),
    )
    host._on_mode_changed = MethodType(imageWrangler._on_mode_changed, host)
    host.start = MethodType(imageWrangler.start, host)
    return host


def test_wrangler_enabled_reapplies_viewer_mode_controls():
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    host = _wrangler_host("Image Viewer", live=True, batch=True)

    imageWrangler.enabled(host, True)

    assert host.ui.liveCheckBox.isChecked() is False
    assert host.ui.liveCheckBox.isEnabled() is False
    assert host.ui.batchCheckBox.isEnabled() is False
    assert host.ui.frame.isVisible() is False
    assert host._integration_controls_enabled is False
    assert host.thread.live_mode is False


def test_wrangler_enabled_reapplies_xye_mode_controls():
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    host = _wrangler_host("Int 1D (XYE)", live=True, batch=False)

    imageWrangler.enabled(host, True)

    assert host.ui.liveCheckBox.isChecked() is False
    assert host.ui.liveCheckBox.isEnabled() is False
    assert host.ui.batchCheckBox.isChecked() is True
    assert host.ui.batchCheckBox.isEnabled() is False
    assert host.xye_only is True
    assert host.thread.xye_only is True


def test_wrangler_enabled_keeps_normal_live_clickable():
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    host = _wrangler_host("Int 2D", live=True, batch=True)

    imageWrangler.enabled(host, True)

    assert host.ui.liveCheckBox.isChecked() is False
    assert host.ui.liveCheckBox.isEnabled() is True
    assert host.ui.batchCheckBox.isEnabled() is True
    assert host.ui.frame.isVisible() is True
    assert host._integration_controls_enabled is True


def test_start_click_forces_non_live_run():
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    host = _wrangler_host("Int 2D", live=True, batch=False)

    imageWrangler._on_start_clicked(host)

    assert host.ui.liveCheckBox.isChecked() is False
    assert host.live_mode is False
    assert host.thread.live_mode is False
    assert host.command == "start"
    assert host.thread.command == "start"
    assert host.ui.stopButton.isEnabled() is True
    assert host.sigStart.emitted == [()]


def test_active_live_run_disables_batch_but_keeps_live_clickable():
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    host = _wrangler_host("Int 2D", live=True, batch=False)
    host.live_mode = True

    imageWrangler.enabled(host, False)

    assert host.ui.startButton.isEnabled() is False
    assert host.ui.liveCheckBox.isEnabled() is True
    assert host.ui.batchCheckBox.isEnabled() is False
    assert host._integration_calls[-1] == (
        False, {"include_gi": False},
    )
    assert host._gi_readonly is True


def test_active_non_live_run_disables_live_and_batch():
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    host = _wrangler_host("Int 2D", live=False, batch=True)
    host.live_mode = False

    imageWrangler.enabled(host, False)

    assert host.ui.startButton.isEnabled() is False
    assert host.ui.liveCheckBox.isEnabled() is False
    assert host.ui.batchCheckBox.isEnabled() is False
    assert host._gi_readonly is True


def test_active_run_keeps_parameter_tree_enabled_for_gi_checkbox_paint():
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    host = _wrangler_host("Int 2D", live=False, batch=False)

    imageWrangler.enabled(host, False)

    assert host.tree.isEnabled() is True
    assert host._integration_calls[-1] == (
        False, {"include_gi": False},
    )
    assert host._gi_readonly is True
