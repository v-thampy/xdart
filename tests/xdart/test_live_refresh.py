"""Focused tests for throttled live-scan GUI refresh helpers."""

from __future__ import annotations

import logging
import os
from queue import Queue
from types import MethodType, SimpleNamespace
from threading import RLock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtCore

from xdart.gui.tabs.static_scan.h5viewer import H5Viewer
from xdart.gui.tabs.static_scan.display_data import (
    DisplayDataMixin,
    available_norm_channels,
)
from xdart.gui.tabs.static_scan.display_frame_widget import displayFrameWidget
from xdart.gui.tabs.static_scan.display_plot import (
    DisplayPlotMixin,
    update_plot_accumulator,
)
from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget


def test_display_wavelength_rejects_default_sentinel(tmp_path):
    host = SimpleNamespace(
        scan=SimpleNamespace(
            mg_args={"wavelength": 1.0e-10},
            data_file=str(tmp_path / "missing.nxs"),
        )
    )

    assert DisplayDataMixin._get_wavelength(host) is None


def test_display_wavelength_accepts_authoritative_one_angstrom_integrator(tmp_path):
    frame = SimpleNamespace(
        integrator=SimpleNamespace(wavelength=1.0e-10),
        poni=None,
    )
    host = SimpleNamespace(
        scan=SimpleNamespace(
            mg_args={"wavelength": 1.0e-10},
            data_file=str(tmp_path / "missing.nxs"),
        )
    )

    assert DisplayDataMixin._get_wavelength(host, frame) == pytest.approx(1.0e-10)


def test_display_wavelength_accepts_authoritative_one_angstrom_poni(tmp_path):
    frame = SimpleNamespace(
        integrator=None,
        poni=SimpleNamespace(wavelength=1.0e-10),
    )
    host = SimpleNamespace(
        scan=SimpleNamespace(
            mg_args={"wavelength": 1.0e-10},
            data_file=str(tmp_path / "missing.nxs"),
        )
    )

    assert DisplayDataMixin._get_wavelength(host, frame) == pytest.approx(1.0e-10)


def test_display_wavelength_reads_v2_instrument_source(tmp_path):
    import h5py

    path = tmp_path / "wl.nxs"
    with h5py.File(path, "w") as f:
        src = f.create_group("entry/instrument/source")
        src.create_dataset("wavelength_A", data=0.7293)
    host = SimpleNamespace(
        scan=SimpleNamespace(
            mg_args={"wavelength": 1.0e-10},
            data_file=str(path),
        )
    )

    assert DisplayDataMixin._get_wavelength(host) == pytest.approx(0.7293e-10)


def test_display_wavelength_prefers_authoritative_reloaded_one_angstrom(tmp_path):
    host = SimpleNamespace(
        scan=SimpleNamespace(
            mg_args={"wavelength": 1.0e-10},
            _persisted_wavelength_m=1.0e-10,
            data_file=str(tmp_path / "missing.nxs"),
        )
    )

    assert DisplayDataMixin._get_wavelength(host) == pytest.approx(1.0e-10)


def test_display_wavelength_accepts_authoritative_one_angstrom_stamp(tmp_path):
    import h5py

    path = tmp_path / "wl_one_angstrom.nxs"
    with h5py.File(path, "w") as f:
        src = f.create_group("entry/instrument/source")
        src.create_dataset("wavelength_A", data=1.0)
    host = SimpleNamespace(
        scan=SimpleNamespace(
            mg_args={},
            data_file=str(path),
        )
    )

    assert DisplayDataMixin._get_wavelength(host) == pytest.approx(1.0e-10)


@pytest.mark.parametrize("flags", [
    {"live_run": True, "no_nxs": False},
    {"live_run": False, "no_nxs": True},
])
def test_live_repoint_clears_stale_persisted_wavelength(tmp_path, flags):
    """T0-1: the live-run / XYE set_datafile repoint skips load_from_h5, so it
    must clear the wavelength restored from the PREVIOUS file — otherwise
    _get_wavelength keeps short-circuiting on file A's λ for every frame of
    the run now writing file B."""
    from xdart.modules.ewald.scan import LiveScan
    from xdart.gui.tabs.static_scan.scan_threads import fileHandlerThread

    # As if file A was previously loaded and its real wavelength restored
    # (the load-restore path itself is covered in test_nexus_writer_roundtrip).
    scan = LiveScan("a", data_file=str(tmp_path / "a.nxs"))
    scan._persisted_wavelength_m = 0.7293e-10
    scan.mg_args["wavelength"] = 0.7293e-10

    # Repoint to file B through the real fileHandlerThread branch.
    holder = SimpleNamespace(
        scan=scan, fname=str(tmp_path / "b.nxs"),
        file_lock=RLock(),
        sigNewFile=SimpleNamespace(emit=lambda *_: None),
        sigUpdate=SimpleNamespace(emit=lambda *_: None),
        **flags,
    )
    MethodType(fileHandlerThread.set_datafile, holder)()

    assert scan.data_file == str(tmp_path / "b.nxs")
    assert scan._persisted_wavelength_m is None
    # Display no longer sees file A's λ: mg_args is back to the rejected
    # sentinel and file B has no stamp -> unknown, never A's value.
    host = SimpleNamespace(scan=scan)
    assert DisplayDataMixin._get_wavelength(host) is None


class _FakeItem:
    def __init__(self, text):
        if hasattr(text, "text"):
            self._text = str(text.text())
            self._data = {}
            try:
                from pyqtgraph.Qt import QtCore
                value = text.data(QtCore.Qt.UserRole)
                if value is not None:
                    self._data[QtCore.Qt.UserRole] = value
            except Exception:
                pass
        else:
            self._text = str(text)
            self._data = {}
        self._selected = False

    def text(self):
        return self._text

    def setData(self, role, value):
        self._data[role] = value

    def data(self, role):
        return self._data.get(role)

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
        self._items.append(item if isinstance(item, _FakeItem) else _FakeItem(item))

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

    def currentItem(self):
        if 0 <= self._current_row < len(self._items):
            return self._items[self._current_row]
        return None

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


class _FakeImageItem:
    def __init__(self):
        self.cleared = False
        self.levels = None

    def clear(self):
        self.cleared = True

    def setLevels(self, levels):
        self.levels = tuple(levels)


class _FakeHistogram:
    def __init__(self):
        self.visible = True
        self.levels = None

    def setVisible(self, visible):
        self.visible = bool(visible)

    def setLevels(self, values=None, **kwargs):
        self.levels = tuple(values)


class _FakeImageWidget:
    def __init__(self):
        self.images = []
        self.rects = []
        self.raw_image = np.ones((2, 2))
        self.displayed_image = np.ones((2, 2))
        self.imageItem = _FakeImageItem()
        self.histogram = _FakeHistogram()

    def setImage(self, data, *args, **kwargs):
        self.images.append(np.asarray(data))

    def setRect(self, rect):
        self.rects.append(rect)

    def width(self):
        return 200

    def height(self):
        return 200


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
        self.started = 0

    def isActive(self):
        return self.active

    def stop(self):
        self.active = False

    def start(self):
        self.active = True
        self.started += 1


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

    # Phase B: the morphing action button (Start/Pause/Resume) sets text +
    # a dynamic 'runPhase' property + repaints via style().unpolish/polish.
    def setText(self, text):
        self._text = str(text)

    def text(self):
        return getattr(self, "_text", "")

    def setProperty(self, key, value):
        setattr(self, "_prop_" + key, value)

    def property(self, key):
        return getattr(self, "_prop_" + key, None)

    def style(self):
        class _S:
            def unpolish(self, *a):
                pass

            def polish(self, *a):
                pass
        return _S()


class _FakeCombo:
    def __init__(self, text):
        self._text = text
        self._enabled = True

    def currentText(self):
        return self._text

    def setEnabled(self, enabled):
        self._enabled = bool(enabled)

    def isEnabled(self):
        return self._enabled


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
            frame_ids=[],
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


def test_flush_pending_update_feeds_overlay_all_pending_frames():
    calls = []
    frame_ids = []

    def data_changed(*, show_all=False):
        calls.append(("data_changed", show_all, tuple(frame_ids)))

    widget = SimpleNamespace(
        _pending_update_idx=5,
        scan=SimpleNamespace(scan_lock=RLock(), frames=SimpleNamespace(index=[1, 2, 3, 4, 5])),
        h5viewer=SimpleNamespace(
            auto_last=True,
            frame_ids=frame_ids,
            update_data=lambda **kwargs: calls.append(("update_data", kwargs)),
            data_changed=data_changed,
        ),
        latest_frame=lambda **kwargs: calls.append(("latest_frame", kwargs)),
        displayframe=SimpleNamespace(
            ui=SimpleNamespace(plotMethod=_FakeCombo("Overlay")),
        ),
    )

    staticWidget._flush_pending_update(widget)

    assert widget._pending_update_idx is None
    assert calls == [
        ("update_data", {"emit_update": False}),
        ("latest_frame", {"emit_update": False}),
        ("data_changed", True, ("1", "2", "3", "4", "5")),
    ]


def test_update_plot_preserves_overlay_accumulator_on_failed_read():
    """Append-only invariant (regression guard): when the incremental read returns
    no data and an Overlay/Waterfall accumulator already exists, update_plot must
    PRESERVE it (redraw) and NEVER clear_plot_view -- clearing wiped the whole
    stack, which is what made a slow live scan (e.g. Share Axis) collapse the
    waterfall to ~0 and repopulate + re-stack frames as it caught up."""
    calls = []
    host = SimpleNamespace(
        scan=SimpleNamespace(name="scanA", data_file="scanA"),
        frame_ids=[0, 1, 2, 3],
        overlaid_idxs=[0, 1, 2],
        idxs_1d=[3],                       # a new 'missing' frame -> the
                                           # already-covered early-return is skipped
        _overlay_scan_key="scanA",         # same scan -> no boundary reset
        _last_plot_unit=0,
        _payload_x_axis_label="stale",
        _payload_y_axis_label="stale",
        plot_data=[np.arange(3), np.ones((3, 3))],
        frame_names=["scanA_0", "scanA_1", "scanA_2"],
        ui=SimpleNamespace(
            plotMethod=_FakeCombo("Overlay"),
            plotUnit=_FakeIndexedCombo("Q", index=0),
        ),
        collect_plot_rows=lambda: (None, None),    # failed/empty incremental read
        draw_plot_state=lambda: calls.append("draw"),
        clear_plot_view=lambda: calls.append("clear"),
    )
    host.update_plot = MethodType(DisplayPlotMixin.update_plot, host)

    host.update_plot()

    assert "clear" not in calls                # never wipe an existing accumulator
    assert "draw" in calls                     # redraw what we already have
    assert host.overlaid_idxs == [0, 1, 2]     # untouched -> monotonic


def test_update_plot_clears_when_no_overlay_accumulator_and_read_fails():
    """The genuine empty case (fresh reload, nothing accumulated yet) still
    clears -- the preserve guard is scoped to a non-empty accumulator."""
    calls = []
    host = SimpleNamespace(
        scan=SimpleNamespace(name="scanA", data_file="scanA"),
        frame_ids=[0, 1, 2, 3],
        overlaid_idxs=[],                  # nothing accumulated yet
        idxs_1d=[0, 1, 2, 3],
        _overlay_scan_key="scanA",
        _last_plot_unit=0,
        _payload_x_axis_label=None,
        _payload_y_axis_label=None,
        plot_data=[np.zeros(0), np.zeros(0)],
        frame_names=[],
        ui=SimpleNamespace(
            plotMethod=_FakeCombo("Overlay"),
            plotUnit=_FakeIndexedCombo("Q", index=0),
        ),
        collect_plot_rows=lambda: (None, None),
        draw_plot_state=lambda: calls.append("draw"),
        clear_plot_view=lambda: calls.append("clear"),
    )
    host.update_plot = MethodType(DisplayPlotMixin.update_plot, host)

    host.update_plot()

    assert "clear" in calls
    assert "draw" not in calls


def _display_host():
    image_widget = _FakeImageWidget()
    binned_widget = _FakeImageWidget()
    wf_widget = _FakeImageWidget()
    label = _FakeLabel()
    curve = _FakeCurve()
    removed = []
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
        plot=SimpleNamespace(removeItem=removed.append),
        _removed_curves=removed,
        legend=legend,
        ui=SimpleNamespace(labelCurrent=label),
    )
    host.clear_overlay = MethodType(displayFrameWidget.clear_overlay, host)
    host._clear_image_widget = displayFrameWidget._clear_image_widget
    host.clear_image_view = MethodType(displayFrameWidget.clear_image_view, host)
    host.clear_binned_view = MethodType(displayFrameWidget.clear_binned_view, host)
    host.clear_plot_view = MethodType(displayFrameWidget.clear_plot_view, host)
    host.clear_display_state = MethodType(displayFrameWidget.clear_display_state, host)
    return host, image_widget, binned_widget, wf_widget, curve, legend, label


def test_clear_display_state_resets_visible_and_cached_state():
    host, image_widget, binned_widget, wf_widget, curve, legend, label = _display_host()

    host.clear_display_state("XYE Viewer")

    assert host.image_data is None
    assert host.binned_data is None
    assert host.plot_data[0].size == 0
    assert host.plot_data[1].size == 0
    assert host.plot_data_range == [[0, 0], [0, 0]]
    assert host.frame_names == []
    assert host.overlaid_idxs == []
    assert host.curves == []
    # New contract: curves are REMOVED from the plot (removeItem), not just
    # data-cleared -- PlotDataItem.clear() left items accumulating.
    assert host._removed_curves == [curve]
    assert legend.cleared is True
    assert label.text == "XYE Viewer"
    assert image_widget.images == []
    assert binned_widget.images == []
    assert wf_widget.images == []
    assert image_widget.imageItem.cleared is True
    assert binned_widget.imageItem.cleared is True
    assert wf_widget.imageItem.cleared is True
    assert image_widget.raw_image.size == 0
    assert binned_widget.raw_image.size == 0
    assert wf_widget.raw_image.size == 0


def _persist_image_host(processing, persist=True, data=None):
    """Minimal host to exercise update_image's blank-vs-keep-last decision."""
    image_widget = _FakeImageWidget()
    cleared = []
    host = SimpleNamespace(
        PERSIST_2D_DURING_PROCESSING=persist,
        _processing_active=processing,
        _image_levels_override=None,
        overall=False,
        frame_ids=[],
        image_widget=image_widget,
        get_frames_map_raw=lambda *a, **k: (data, None),
    )
    host.clear_image_view = lambda: cleared.append(True)
    host.update_image = MethodType(displayFrameWidget.update_image, host)
    return host, cleared


def test_update_image_keeps_last_during_run_when_data_missing():
    # Panel-consistency: while a run is active, a missing in-flight frame must
    # NOT blank the raw panel — it keeps the last image (like the 1D plot).
    host, cleared = _persist_image_host(processing=True, data=None)

    host.update_image()

    assert cleared == []                 # not blanked during the run


def test_update_image_blanks_when_idle_and_data_missing():
    # Outside a run the old behavior is unchanged: missing data blanks the raw
    # panel (so mode-switch / selection still clears stale content).
    host, cleared = _persist_image_host(processing=False, data=None)

    host.update_image()

    assert cleared == [True]             # blanked when not running


def test_update_image_blanks_during_run_when_flag_off():
    # The revert switch restores blanking even during a run.
    host, cleared = _persist_image_host(processing=True, persist=False, data=None)

    host.update_image()

    assert cleared == [True]


def test_update_image_masks_uint16_sentinels_in_legacy_path():
    """Regression: the live Int-raw panel uses the LEGACY update_image path (the
    cake reads the store; the raw panel does not).  That path must mask the
    uint16-ceiling (65535) detector sentinels to NaN — for parity with the
    payload path (display_publication.raw_image) — or a TIFF whose invalid
    pixels sit at the ceiling renders them as bright bands/speckle."""
    from threading import RLock

    raw = np.full((100, 100), 500.0)
    raw[10:20, :] = 65535.0          # a masked band of uint16 sentinels (1000 px)
    captured = []
    host = SimpleNamespace(
        PERSIST_2D_DURING_PROCESSING=True, _processing_active=False,
        _image_levels_override=None, overall=False, frame_ids=[0],
        idxs_2d=[0], data_lock=RLock(), data_2d={0: {'mask': None}},
        bkg_map_raw=0.0, scan=SimpleNamespace(global_mask=None),
        get_frames_map_raw=lambda *a, **k: (raw, 'raw'),
    )
    host.clear_image_view = lambda: captured.append('cleared')
    host.update_image_view = lambda: captured.append(host.image_data)
    host.update_image = MethodType(displayFrameWidget.update_image, host)

    host.update_image()

    assert captured and captured[0] != 'cleared'
    shown = captured[0][0]                       # (data, rect) -> the drawn data
    assert np.isnan(shown).sum() == 1000         # all 65535 sentinels masked
    assert np.isfinite(shown).any()              # real pixels survive


def test_draw_delegate_cake_persists_during_run():
    # CAKE_2D: a None cake payload normally returns clear_binned_view (blank);
    # during a run it returns None so render skips the panel (keep last).
    from xdart.gui.tabs.static_scan.display_logic import Mode, PanelRole

    host = SimpleNamespace(
        PERSIST_2D_DURING_PROCESSING=True,
        _processing_active=False,
        clear_binned_view=lambda: None,
    )
    host._draw_delegate = MethodType(displayFrameWidget._draw_delegate, host)

    # Idle -> blanks the cake (old behavior).
    assert host._draw_delegate(PanelRole.CAKE_2D, Mode.INT_2D) is host.clear_binned_view

    # Running -> keep last (None delegate -> render skips this panel).
    host._processing_active = True
    assert host._draw_delegate(PanelRole.CAKE_2D, Mode.INT_2D) is None

    # Revert switch off -> blanks even during a run.
    host.PERSIST_2D_DURING_PROCESSING = False
    assert host._draw_delegate(PanelRole.CAKE_2D, Mode.INT_2D) is host.clear_binned_view


def test_draw_delegate_raw_uses_payload_not_legacy_update_image():
    # Item-2 flip: RAW_2D no longer falls back to legacy update_image -- it
    # renders solely from the raw_image payload, mirroring CAKE_2D (None during
    # a run = keep-last; clear_image_view idle).  Image/NeXus viewer modes still
    # clear (no integration raw).
    from xdart.gui.tabs.static_scan.display_logic import Mode, PanelRole

    host = SimpleNamespace(
        PERSIST_2D_DURING_PROCESSING=True,
        _processing_active=False,
        clear_image_view=lambda: None,
        update_image=lambda: None,
    )
    host._draw_delegate = MethodType(displayFrameWidget._draw_delegate, host)

    # Idle Int -> blanks (NOT the legacy update_image).
    d = host._draw_delegate(PanelRole.RAW_2D, Mode.INT_2D)
    assert d is host.clear_image_view
    assert d is not host.update_image
    # Running -> keep last (None delegate -> render skips this panel).
    host._processing_active = True
    assert host._draw_delegate(PanelRole.RAW_2D, Mode.INT_2D) is None
    # Revert switch off -> blanks even during a run.
    host.PERSIST_2D_DURING_PROCESSING = False
    assert (host._draw_delegate(PanelRole.RAW_2D, Mode.INT_2D)
            is host.clear_image_view)
    # Image Viewer still clears (no integration raw).
    assert (host._draw_delegate(PanelRole.RAW_2D, Mode.IMAGE_VIEWER)
            is host.clear_image_view)


def test_set_processing_active_sets_flag():
    host = SimpleNamespace(_processing_active=False)
    host.set_processing_active = MethodType(
        displayFrameWidget.set_processing_active, host)
    host.set_processing_active(True)
    assert host._processing_active is True
    host.set_processing_active(False)
    assert host._processing_active is False


def _empty_update_host(processing, persist=True):
    """Host that drives update()'s nothing-to-draw branch with empty caches
    (the silent-batch case): _updated() False, no cached data."""
    rendered = []
    host = SimpleNamespace(
        PERSIST_2D_DURING_PROCESSING=persist,
        _processing_active=processing,
        _display_blanked=False,
        display_generation=0,
        data_1d={},                 # empty caches == silent batch
        data_2d={},
        data_lock=RLock(),
        refresh_norm_channels=lambda: None,
        get_idxs=lambda: None,
        _note_selection_generation=lambda: None,
        _updated=lambda: False,     # nothing to draw for the selection
        _live_mode=lambda: False,
        render_display=lambda state, payload: rendered.append("render"),
    )
    host.update = MethodType(displayFrameWidget.update, host)
    host._update_impl = MethodType(displayFrameWidget._update_impl, host)
    return host, rendered


def test_update_keeps_display_during_run_when_caches_empty():
    # Silent batch: nothing cached, but a run is active -> keep the current
    # display (no empty render), so 1D + 2D persist together.
    host, rendered = _empty_update_host(processing=True)

    host.update()

    assert rendered == []                 # no empty render -> no blank
    assert host._display_blanked is False


def test_update_blanks_when_idle_and_caches_empty():
    # Not running + nothing cached -> the explicit blank still happens.
    host, rendered = _empty_update_host(processing=False)

    host.update()

    assert rendered == ["render"]         # empty_display_state rendered
    assert host._display_blanked is True


def test_update_blanks_during_run_when_flag_off():
    # Revert switch restores the old mid-run blank.
    host, rendered = _empty_update_host(processing=True, persist=False)

    host.update()

    assert rendered == ["render"]
    assert host._display_blanked is True


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
    # set_viewer_display_mode now routes panel geometry through the table-driven
    # _apply_layout; bind it too (it pokes the mocked ui / _showImageBtn only).
    host._apply_layout = MethodType(
        displayFrameWidget._apply_layout, host)
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


def test_select_last_scan_entry_picks_last_file_row():
    # End-of-(XYE)-batch auto-select: select the last data-file row in
    # listScans (most recent output), skipping '..' and directories.
    from xdart.gui.tabs.static_scan.h5viewer import H5Viewer

    class _Item:
        def __init__(self, t): self._t = t
        def text(self): return self._t

    selected = {}
    class _List:
        def __init__(self, texts): self._items = [_Item(t) for t in texts]
        def count(self): return len(self._items)
        def item(self, r): return self._items[r]
        def setCurrentRow(self, r, _mode=None): selected['row'] = r

    host = SimpleNamespace(ui=SimpleNamespace(listScans=_List(
        ['..', 'subdir/', 'iq_scan_0001.xye', 'iq_scan_0002.xye', 'iq_scan_0003.xye'])))
    host.select_last_scan_entry = MethodType(H5Viewer.select_last_scan_entry, host)

    row = host.select_last_scan_entry()
    assert row == 4                          # last .xye row, not '..' or the dir
    assert selected['row'] == 4

    # Nothing selectable -> -1, no crash.
    host2 = SimpleNamespace(ui=SimpleNamespace(listScans=_List(['..', 'd/'])))
    host2.select_last_scan_entry = MethodType(H5Viewer.select_last_scan_entry, host2)
    assert host2.select_last_scan_entry() == -1


def test_metadata_panel_populates_when_layout_reparented():
    # Regression: the host installs only metadataWidget.layout into its
    # metaFrame, so the metadataWidget QWidget itself is never shown and
    # self.isVisible() is always False — update() must gate on the tableview
    # (which IS on screen), else the metadata panel stays blank.
    import pandas as pd
    from PySide6 import QtWidgets
    from PySide6.QtCore import QModelIndex
    from xdart.gui.tabs.static_scan.metadata import metadataWidget

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    scan = SimpleNamespace(
        scan_data=pd.DataFrame({"th": [0.1, 0.2], "i0": [1e6, 1.1e6]}, index=[1, 2]))
    mw = metadataWidget(scan, None, ["1"], {})
    assert not mw.isVisible()                    # the widget itself never shows

    frame = QtWidgets.QFrame()
    frame.setLayout(mw.layout)                   # mirror static_scan_widget
    win = QtWidgets.QWidget()
    QtWidgets.QVBoxLayout(win).addWidget(frame)
    win.show()
    app.processEvents()

    assert mw.tableview.isVisible()              # the tableview IS on screen
    mw.update()
    model = mw.tableview.model()
    assert model.rowCount(QModelIndex()) == 2    # th + i0 -> populated, not blank
    assert list(model.dataFrame.columns) == [1]  # selected frame only, not whole scan
    win.close()


def test_metadata_panel_accepts_store_mapping_proxy():
    import pandas as pd
    from PySide6 import QtWidgets
    from PySide6.QtCore import QModelIndex
    from xrd_tools.core import FrameView
    from xdart.gui.tabs.static_scan.metadata import metadataWidget
    from xdart.modules.frame_publication import (
        PublicationStore,
        publication_from_frame_view,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    store = PublicationStore()
    store.upsert(
        publication_from_frame_view(
            FrameView(
                label=7,
                metadata_raw={"sample": "LaB6", "monitor": 2.0},
            )
        )
    )
    scan = SimpleNamespace(scan_data=pd.DataFrame())
    mw = metadataWidget(
        scan,
        None,
        ["7"],
        {},
        data_1d={},
        publication_store=store,
        data_lock=RLock(),
    )

    frame = QtWidgets.QFrame()
    frame.setLayout(mw.layout)
    win = QtWidgets.QWidget()
    QtWidgets.QVBoxLayout(win).addWidget(frame)
    win.show()
    app.processEvents()

    mw.update()
    model = mw.tableview.model()
    assert model.rowCount(QModelIndex()) == 2
    win.close()


def test_live_new_scan_invalidates_publication_store():
    import pandas as pd
    from xdart.modules.frame_publication import PublicationStore

    store = PublicationStore()
    old_generation = store.generation

    scan = SimpleNamespace(
        name="old",
        gi=False,
        incidence_motor="th",
        single_img=False,
        series_average=False,
        global_mask=None,
        scan_lock=RLock(),
        frames=SimpleNamespace(index=[1], _in_memory={1: object()}),
        scan_data=pd.DataFrame({"old": [1.0]}, index=[1]),
    )
    host = SimpleNamespace(
        scan=scan,
        h5viewer=SimpleNamespace(
            dirname="",
            live_run_active=True,
            scan_name="old",
            auto_last=False,
            latest_idx=9,
            set_file=lambda fname, **k: None,
            update_scans=lambda: None,
            update=lambda: None,
        ),
        wrangler=SimpleNamespace(thread=SimpleNamespace(mask=None)),
        integratorTree=SimpleNamespace(
            get_args=lambda name: None,
            set_image_units=lambda: None,
        ),
        _update_timer=SimpleNamespace(stop=lambda: None),
        _flush_pending_update=lambda: None,
        frames={1: object()},
        frame_ids=["1"],
        publication_store=store,
        displayframe=SimpleNamespace(set_axes=lambda: None),
        metawidget=SimpleNamespace(update=lambda: None),
    )
    host._sync_h5viewer_save_dir = MethodType(
        staticWidget._sync_h5viewer_save_dir, host,
    )

    staticWidget.new_scan(
        host,
        "new",
        "/tmp/new.nxs",
        False,
        "th",
        False,
        False,
    )

    assert len(store) == 0
    assert store.generation == old_generation + 1
    assert host.frame_ids == []
    assert host.dirname == "/tmp"
    assert host.h5viewer.dirname == "/tmp"
    assert list(scan.frames.index) == []
    assert scan.scan_data.empty


def _new_scan_host_with_wrangler_mask(wrangler_mask, initial_global_mask,
                                      wrangler_detector_shape=None):
    """A minimal staticWidget host for driving ``new_scan`` and observing how
    ``scan.global_mask`` / ``scan.detector_shape`` sync from the wrangler thread."""
    import pandas as pd
    from xdart.modules.frame_publication import PublicationStore

    scan = SimpleNamespace(
        name="old",
        gi=False,
        incidence_motor="th",
        single_img=False,
        series_average=False,
        global_mask=initial_global_mask,
        scan_lock=RLock(),
        frames=SimpleNamespace(index=[1], _in_memory={1: object()}),
        scan_data=pd.DataFrame({"old": [1.0]}, index=[1]),
    )
    host = SimpleNamespace(
        scan=scan,
        h5viewer=SimpleNamespace(
            dirname="", live_run_active=True, scan_name="old",
            auto_last=False, latest_idx=9,
            set_file=lambda fname, **k: None,
            update_scans=lambda: None, update=lambda: None,
        ),
        wrangler=SimpleNamespace(thread=SimpleNamespace(
            mask=wrangler_mask, detector_shape=wrangler_detector_shape)),
        integratorTree=SimpleNamespace(
            get_args=lambda name: None, set_image_units=lambda: None,
        ),
        _update_timer=SimpleNamespace(stop=lambda: None),
        _flush_pending_update=lambda: None,
        frames={1: object()},
        frame_ids=["1"],
        publication_store=PublicationStore(),
        displayframe=SimpleNamespace(set_axes=lambda: None),
        metawidget=SimpleNamespace(update=lambda: None),
    )
    host._sync_h5viewer_save_dir = MethodType(
        staticWidget._sync_h5viewer_save_dir, host,
    )
    return host, scan


def test_new_scan_clears_stale_global_mask_when_mask_file_removed():
    """Regression: removing the Mask File (so the wrangler rebuilds
    ``thread.mask = None``) must CLEAR the previous run's ``scan.global_mask``,
    not leave it stale — otherwise the dropped mask keeps rendering on the raw
    image (and in the cake payload path) on the next run."""
    stale = np.array([0, 1, 2, 3], dtype=int)   # a previous run's mask.edf
    host, scan = _new_scan_host_with_wrangler_mask(
        wrangler_mask=None, initial_global_mask=stale,
    )
    staticWidget.new_scan(host, "new", "/tmp/new.nxs", False, "th", False, False)
    assert scan.global_mask is None


def test_new_scan_propagates_wrangler_mask_when_present():
    """Companion: a mask IS drawn — ``thread.mask`` (flat indices) propagates to
    ``scan.global_mask`` so the overlay shows (the original v2 regression fix)."""
    mask_idx = np.array([5, 6, 7], dtype=int)
    host, scan = _new_scan_host_with_wrangler_mask(
        wrangler_mask=mask_idx, initial_global_mask=None,
    )
    staticWidget.new_scan(host, "new", "/tmp/new.nxs", False, "th", False, False)
    np.testing.assert_array_equal(scan.global_mask, mask_idx)


def test_new_scan_propagates_wrangler_detector_shape():
    """The full-res detector shape syncs from the wrangler thread onto the scan
    (alongside global_mask) so the display can map the gap mask without a
    resident full-res frame."""
    host, scan = _new_scan_host_with_wrangler_mask(
        wrangler_mask=np.array([5, 6, 7], dtype=int), initial_global_mask=None,
        wrangler_detector_shape=(2167, 2070),
    )
    staticWidget.new_scan(host, "new", "/tmp/new.nxs", False, "th", False, False)
    assert scan.detector_shape == (2167, 2070)


def test_save_path_sync_updates_scans_browser(tmp_path):
    calls = []
    host = SimpleNamespace(
        dirname="old",
        h5viewer=SimpleNamespace(
            dirname="old",
            update_scans=lambda: calls.append("update_scans"),
        ),
    )

    staticWidget._sync_h5viewer_save_dir(host, tmp_path)

    assert host.dirname == str(tmp_path)
    assert host.h5viewer.dirname == str(tmp_path)
    assert calls == ["update_scans"]

    staticWidget._sync_h5viewer_save_dir(host, tmp_path / "next", refresh=False)

    # New contract: a save dir that doesn't exist yet (fresh project, no run
    # has created it) falls back to the nearest existing ancestor so the
    # browser shows real contents instead of an empty nonexistent path.
    assert host.dirname == str(tmp_path)

    (tmp_path / "next").mkdir()
    staticWidget._sync_h5viewer_save_dir(host, tmp_path / "next", refresh=False)
    assert host.dirname == str(tmp_path / "next")
    assert host.h5viewer.dirname == str(tmp_path / "next")
    assert calls == ["update_scans"]


def test_gi_motor_options_default_manual_when_no_metadata():
    # GI incidence: when no motors are found (eiger / no metadata), the Theta
    # Motor must default to 'Manual' and reveal the Theta value field so the
    # incidence angle can be entered directly.  With a 'th' motor present it
    # selects 'th' and hides the manual field.
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    class _P:
        def __init__(self, value=None):
            self._v = value
            self.visible = True
        def value(self):
            return self._v
        def setValue(self, v):
            self._v = v
        def setOpts(self, **o):
            if "value" in o:
                self._v = o["value"]
            if "visible" in o:
                self.visible = o["visible"]
            self.opts = o
        def hide(self):
            self.visible = False
        def show(self):
            self.visible = True

    def _host(motors):
        gi = {"th_motor": _P("th"), "th_val": _P("0.1")}
        params = SimpleNamespace(child=lambda name: SimpleNamespace(child=lambda n: gi[n]))
        h = SimpleNamespace(motors=motors, parameters=params, incidence_motor=None)
        h.set_gi_th_motor = MethodType(imageWrangler.set_gi_th_motor, h)
        h.set_gi_motor_options = MethodType(imageWrangler.set_gi_motor_options, h)
        return h, gi

    # No metadata -> Manual default, Theta field visible, incidence = th_val.
    h, gi = _host([])
    h.set_gi_motor_options()
    assert gi["th_motor"].value() == "Manual"
    assert gi["th_val"].visible is True
    assert h.incidence_motor == "0.1"

    # 'th' present -> selects th, manual field hidden.
    h, gi = _host(["th", "i0"])
    h.set_gi_motor_options()
    assert gi["th_motor"].value() == "th"
    assert gi["th_val"].visible is False
    assert h.incidence_motor == "th"


def test_data_changed_tolerates_non_integer_labels():
    # Regression: data_changed crashed with ValueError int('..._0001.xye')
    # when listData still held xye filenames during a viewer<->scan mode
    # transition (viewer_mode not yet 'xye').  It must treat non-integer
    # labels as "nothing to load" instead of crashing.
    from PySide6 import QtCore

    class _Item:
        def __init__(self, text):
            self._text = text
        def text(self):
            return self._text
        def data(self, role):
            return None

    class _List:
        def __init__(self, items):
            self._items = items
        def selectedItems(self):
            return self._items

    loaded = []
    host = SimpleNamespace(
        viewer_mode=None,                       # NOT 'xye' — the crash condition
        frame_ids=[],
        update_2d=False,
        data_1d={}, data_2d={},
        scan=SimpleNamespace(frames=SimpleNamespace(index=[0, 1, 2])),
        ui=SimpleNamespace(listData=_List([
            _Item('iq_eiger_w2s3_test_2_scan001_0001.xye'),
            _Item('iq_eiger_w2s3_test_2_scan001_0002.xye'),
        ])),
        load_frames_data=lambda *a, **k: loaded.append(a),
        sigUpdate=SimpleNamespace(emit=lambda: None),
    )
    host.data_changed = MethodType(H5Viewer.data_changed, host)

    host.data_changed()                          # must not raise
    assert loaded == []                          # nothing loaded from xye names


def test_absorb_chunk_drops_stale_generation():
    # Stage 5: a background load worker publishes ONLY through a
    # generation-checked snapshot — a chunk whose generation no longer
    # matches the store's _load_generation (a newer load/selection has begun)
    # is dropped, never written into data_1d/data_2d.
    viewer = SimpleNamespace(
        _load_generation=7,
        data_lock=RLock(),
        data_1d={},
        data_2d={},
    )
    viewer._absorb_chunk = MethodType(H5Viewer._absorb_chunk, viewer)

    class _Frame:
        def copy_for_display(self, include_2d=False):
            return self

    # Stale chunk (gen 6 < current 7) -> dropped.
    viewer._absorb_chunk(6, 3, _Frame(), False)
    assert viewer.data_1d == {} and viewer.data_2d == {}


def test_live_gi_clip_warning_fires_once_in_live_gi_only():
    """#75: live GI emits a one-time 'reprocess in batch' clip advisory; batch
    and non-GI never warn."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread

    def _host(*, gi, batch):
        labels = []
        h = SimpleNamespace(
            gi=gi, batch_mode=batch,
            showLabel=SimpleNamespace(emit=labels.append),
        )
        h._maybe_warn_live_gi_clip = MethodType(
            imageThread._maybe_warn_live_gi_clip, h)
        return h, labels

    # Live + GI → warns exactly once.
    h, labels = _host(gi=True, batch=False)
    h._maybe_warn_live_gi_clip()
    assert len(labels) == 1 and "batch" in labels[0].lower()
    h._maybe_warn_live_gi_clip()
    assert len(labels) == 1                      # once only

    # Batch GI → never (union scout brackets all frames).
    h, labels = _host(gi=True, batch=True)
    h._maybe_warn_live_gi_clip()
    assert labels == []

    # Live non-GI → never.
    h, labels = _host(gi=False, batch=False)
    h._maybe_warn_live_gi_clip()
    assert labels == []


def test_wrangler_thread_no_longer_owns_integration_executor():
    """The GUI wrangler base should not keep an integration executor.

    Batch parallelism is owned by xrd_tools.reduction.run_reduction now;
    keeping a second GUI-side executor makes shutdown and cancellation harder
    to reason about.
    """
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import wranglerThread

    assert not hasattr(wranglerThread, "_get_executor")
    assert not hasattr(wranglerThread, "_shutdown_executor")
    assert not hasattr(wranglerThread, "_parallel_integrate")


def test_wrangler_thread_reuses_reduction_session_by_key():
    """Reducer sessions persist across chunks but reset when policy changes."""
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import wranglerThread

    class _Session:
        def __init__(self):
            self.finished = False

        def finish(self, **kw):    # kw absorbs join_timeout=
            self.finished = True
            return None

    thread = wranglerThread(Queue(), {}, "scan.nxs", None)
    first = thread._get_reduction_session(("scan", 4), _Session)
    second = thread._get_reduction_session(("scan", 4), _Session)
    third = thread._get_reduction_session(("scan", 2), _Session)

    try:
        assert first is second
        assert third is not first
        assert first.finished
        assert not third.finished
    finally:
        thread._close_reduction_session()


def _scout_host(motor):
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread
    host = SimpleNamespace(incidence_motor=motor)
    host._scout_pending_frames = MethodType(imageThread._scout_pending_frames, host)
    return host


def _scout_pending(incs, motor='th'):
    # Minimal pending entries: (img_file, img_number, img_data, meta, bg_raw, _).
    # _scout_pending_frames only reads entry[3] (meta) via the motor key.
    out = []
    for i, inc in enumerate(incs):
        meta = {} if inc is None else {motor: inc}
        out.append((f'f{i}', i + 1, None, meta, 0.0, 0.0))
    return out


def test_scout_picks_incidence_extremes_order_independent():
    host = _scout_host('th')
    asc = _scout_pending([0.1, 0.2, 0.3, 0.4])
    assert host._scout_pending_frames(asc) == [asc[0], asc[3]]
    desc = _scout_pending([0.4, 0.3, 0.2, 0.1])
    # Extremes by VALUE (idx3 lowest, idx0 highest), not position → order-independent.
    assert host._scout_pending_frames(desc) == [desc[0], desc[3]]
    unsorted_ = _scout_pending([0.3, 0.1, 0.4, 0.2])
    assert host._scout_pending_frames(unsorted_) == [unsorted_[1], unsorted_[2]]  # min@1, max@2


def test_scout_partial_metadata_adds_positional_ends():
    host = _scout_host('th')
    # idx1 unreadable; resolvable min@0 (0.1), max@2 (0.3); + positional ends 0,3.
    pending = _scout_pending([0.1, None, 0.3, 0.2])
    got = host._scout_pending_frames(pending)
    assert got == [pending[0], pending[2], pending[3]]   # {0,2} ∪ {0,3}


def test_scout_all_metadata_missing_falls_back_to_ends():
    host = _scout_host('th')
    pending = _scout_pending([None, None, None])
    assert host._scout_pending_frames(pending) == [pending[0], pending[2]]


def test_scout_manual_numeric_incidence_single_frame():
    host = _scout_host('0.2')   # Manual numeric → all frames equal
    pending = _scout_pending([None, None, None], motor='th')
    assert host._scout_pending_frames(pending) == [pending[0]]


def test_absorb_chunk_populates_publication_store_for_1d_and_2d():
    from xrd_tools.core import IntegrationResult1D, IntegrationResult2D
    from xdart.modules.frame_publication import PublicationStore

    viewer = SimpleNamespace(
        _load_generation=7,
        data_lock=RLock(),
        data_1d={},
        data_2d={},
        publication_store=PublicationStore(),
        _update_coalesce_timer=_FakeTimer(active=False),
        _raw_cache_order=[],
        _raw_cache_limit=8,
    )
    viewer._absorb_chunk = MethodType(H5Viewer._absorb_chunk, viewer)
    viewer._remember_hydrated_raw = MethodType(H5Viewer._remember_hydrated_raw, viewer)

    class _Frame:
        idx = 3
        scan_info = {"th": 0.2, "monitor": 10.0}
        source_file = "raw.tif"
        source_frame_idx = 0
        map_raw = np.ones((2, 2))
        bg_raw = 0
        mask = None
        gi_2d = {}
        thumbnail = np.ones((1, 1))
        int_1d = IntegrationResult1D(
            radial=np.arange(3), intensity=np.arange(3) + 1, unit="q_A^-1",
        )
        int_2d = IntegrationResult2D(
            radial=np.arange(2), azimuthal=np.arange(2),
            intensity=np.ones((2, 2)), unit="q_A^-1", azimuthal_unit="chi_deg",
        )

        def copy_for_display(self, include_2d=False):
            return self

    viewer._absorb_chunk(7, 3, _Frame(), False)
    publication = viewer.publication_store.get(3)
    assert publication is not None
    assert publication.view.has_1d
    assert viewer.data_1d == {}
    assert viewer.data_2d == {}

    frame = _Frame()
    frame.idx = 4
    viewer._absorb_chunk(7, 4, frame, True)
    publication = viewer.publication_store.get(4)
    assert publication is not None
    assert publication.view.has_1d
    assert publication.view.has_2d
    assert viewer.data_1d == {}
    assert viewer.data_2d == {}


def test_absorb_chunk_skips_invalid_2d_cache_but_keeps_1d_publication(caplog):
    from xrd_tools.core import IntegrationResult1D, IntegrationResult2D
    from xdart.modules.frame_publication import (
        PublicationStore,
        publication_has_2d_errors,
    )

    caplog.set_level(logging.WARNING, logger="xdart.gui.tabs.static_scan.h5viewer")
    viewer = SimpleNamespace(
        _load_generation=8,
        data_lock=RLock(),
        data_1d={},
        data_2d={12: {"int_2d": object()}},
        publication_store=PublicationStore(),
        _update_coalesce_timer=_FakeTimer(active=False),
        _raw_cache_order=[],
        _raw_cache_limit=8,
    )
    viewer._absorb_chunk = MethodType(H5Viewer._absorb_chunk, viewer)
    viewer._remember_hydrated_raw = MethodType(H5Viewer._remember_hydrated_raw, viewer)

    class _Frame:
        idx = 12
        gi = True
        scan_info = {"th": 0.2, "monitor": 10.0}
        source_file = "raw.tif"
        source_frame_idx = 0
        map_raw = np.ones((2, 2))
        bg_raw = 0
        mask = None
        gi_2d = {}
        thumbnail = np.ones((1, 1))
        int_1d = IntegrationResult1D(
            radial=np.arange(3), intensity=np.arange(3) + 1, unit="q_A^-1",
        )
        int_2d = IntegrationResult2D(
            radial=np.linspace(-1.0, 1.0, 2),
            azimuthal=np.linspace(0.0, 3.0, 2),
            intensity=np.full((2, 2), -1.0),
            unit="qip_A^-1",
            azimuthal_unit="qoop_A^-1",
        )

        def _get_incident_angle(self):
            return 0.2

        def copy_for_display(self, include_2d=False):
            return self

    viewer._absorb_chunk(8, 12, _Frame(), True)

    assert 12 not in viewer.data_1d
    assert 12 not in viewer.data_2d
    publication = viewer.publication_store.get(12)
    assert publication is not None
    assert publication.view.has_1d
    assert publication_has_2d_errors(publication)
    assert "Skipping frame 12 2D publication" in caplog.text


def test_gi_common_grid_freeze_yields_uniform_axes():
    # Stage 5 (gi_axes_uniform tie-in): the shipped GI common-grid freeze
    # turns per-frame Auto axes (which differ frame-to-frame in an
    # angle-dependence GI scan and can't stack) into ONE shared grid.
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        _freeze_gi_1d_range_from_result,
    )
    from xdart.gui.tabs.static_scan.display_logic import gi_axes_uniform

    npt = 8
    # Auto: each frame auto-ranges to its own extent -> non-uniform stack.
    auto = [
        (np.linspace(1.00, 5.00, npt),),
        (np.linspace(1.15, 5.30, npt),),   # different incident angle
    ]
    assert gi_axes_uniform(auto) is False

    # Freeze a single radial range from a scout frame, then every frame
    # integrates onto linspace(frozen_range, npt) — one shared axis.
    args = {'radial_range': None}
    scout = SimpleNamespace(radial=np.array([1.0, 5.0]))
    assert _freeze_gi_1d_range_from_result(args, scout) is True
    lo, hi = args['radial_range']
    frozen = [(np.linspace(lo, hi, npt),), (np.linspace(lo, hi, npt),)]
    assert gi_axes_uniform(frozen) is True


def test_default_controllers_registered():
    # Stage 5: importing display_controllers registers a controller for every
    # core mode, so the widget never dispatches to a missing handler.
    from xdart.gui.tabs.static_scan import display_controllers  # noqa: F401
    from xdart.gui.tabs.static_scan.display_logic import controller_for, Mode
    for mode in (
        Mode.INT_1D,
        Mode.INT_2D,
        Mode.IMAGE_VIEWER,
        Mode.XYE_VIEWER,
        Mode.NEXUS_VIEWER,
    ):
        assert controller_for(mode) is not None


def _update_smoke_host():
    """Host that drives the REAL update()->render_display orchestration
    (get_idxs, _live_display_state, compute_display_state, build_payload,
    render_plan, render_display, _draw/_clear_delegate) with only the
    pixel-push leaves stubbed to record their calls.  This exercises the
    integration path that the per-method tests don't cover."""
    from unittest.mock import MagicMock
    from xdart.gui.tabs.static_scan import display_logic as dl

    calls = []
    def rec(name):
        return lambda *a, **k: calls.append(name)

    host = SimpleNamespace(
        viewer_mode=None,
        display_generation=0,
        _last_selection_sig=None,
        frame_ids=['0', '1'],
        idxs=[], idxs_1d=[], idxs_2d=[],
        overall=False,
        overlaid_idxs=[],
        bkg_1d=None,
        plot_data=[np.zeros(0), np.zeros(0)],
        plot_data_range=[[0, 0], [0, 0]],
        frame_names=[],
        _payload_x_axis_label=None,
        _payload_y_axis_label=None,
        _using_publication_plot_payload=False,
        data_1d={0: object(), 1: object()},
        data_2d={
            0: {'map_raw': np.ones((4, 4)), 'thumbnail': None},
            1: {'map_raw': np.ones((4, 4)), 'thumbnail': None},
        },
        data_lock=RLock(),
        scan=SimpleNamespace(scan_lock=RLock(),
                             frames=SimpleNamespace(index=[0, 1]),
                             gi=False, skip_2d=False, name='scan'),
        ui=MagicMock(),
        plot=MagicMock(),
        image_widget=MagicMock(),
        binned_widget=MagicMock(),
        # pixel-push leaves (unchanged by Stage 3) — recorded, not run.
        # CAKE_2D is payload-only now (no update_binned); this host has no
        # publication_store, so its cake payload is None and CAKE_2D blanks.
        update_image=rec("draw_image"),
        update_plot=rec("draw_plot"),
        update_plot_view=rec("payload_plot"),
        # IMAGE_VIEWER / XYE_VIEWER now render through the payload path
        # (_draw_image_payload / _draw_payload -> update_plot_view), not legacy
        # _update_image_viewer / _update_xye_viewer delegates.
        _draw_image_payload=lambda *a, **k: calls.append("draw_payload_image") or True,
        _set_viewer_title=rec("viewer_title"),
        clear_image_view=rec("clear_image"),
        clear_binned_view=rec("clear_binned"),
        clear_plot_view=rec("clear_plot"),
        update_2d_label=rec("label_2d"),
        _update_image_preview=rec("preview"),
        _apply_1d_only_visibility=rec("apply_1d_only"),
    )
    host.ui.shareAxis.isChecked.return_value = False
    host.ui.imageUnit.currentIndex.return_value = 0
    host.ui.plotMethod.currentText.return_value = 'Single'
    for name in ('get_idxs', '_note_selection_generation', '_bump_display_generation',
                 '_live_mode', '_live_display_state',
                 '_current_image_axis_key', '_plot_axis_key',
                 '_share_axis_plot_index', '_set_plot_unit_index_silently',
                 '_apply_share_axis_state',
                 '_draw_delegate', '_clear_delegate', '_payload_for_role',
                 '_draw_payload',
                 'render_display',
                 '_updated', 'update', '_update_impl'):
        setattr(host, name, MethodType(getattr(displayFrameWidget, name), host))
    return host, calls, dl


def test_update_render_smoke_int_collapse_and_mode_switches():
    host, calls, dl = _update_smoke_host()

    # Int-2D: the 1D plot draws via the legacy delegate; RAW_2D and CAKE_2D are
    # both payload-only now (item-2 flip), and this host has no publication_store,
    # so their payloads are None and both 2D panels blank.
    host.update()
    assert "draw_plot" in calls
    assert "clear_image" in calls and "clear_binned" in calls
    assert "label_2d" in calls

    # Int-1D (skip_2d): 1D-only — plot drawn, the two 2D panels cleared.
    calls.clear()
    host.scan.skip_2d = True
    host.update()
    assert "draw_plot" in calls
    assert "clear_image" in calls and "clear_binned" in calls
    assert "draw_image" not in calls and "draw_binned" not in calls

    # Switch to Image Viewer (mode switch bumps generation): raw drawn,
    # the 1D plot + cake from the prior mode are cleared (no stale panels).
    calls.clear()
    host.scan.skip_2d = False
    host.viewer_mode = 'image'
    host._bump_display_generation()
    host.update()
    assert "draw_payload_image" in calls      # raw drawn via the payload path
    assert "clear_plot" in calls and "clear_binned" in calls
    assert "label_2d" not in calls           # viewer owns its own title

    # Switch to XYE Viewer: 1D drawn via the payload path, the 2D panels cleared.
    calls.clear()
    host.frame_ids = ['0']
    host.data_1d = {
        0: SimpleNamespace(
            int_1d=SimpleNamespace(radial=np.arange(3, dtype=float),
                                   intensity=np.array([1.0, 2.0, 3.0])),
            scan_info={'source_file': 'iq_a.xye'})
    }
    host.data_2d = {}
    host.viewer_mode = 'xye'
    host._bump_display_generation()
    host.update()
    assert "payload_plot" in calls           # 1D drawn via the payload path
    assert "clear_image" in calls and "clear_binned" in calls

    # Switch to NeXus Viewer: dataset previews use payload rendering, while
    # the absent image/cake panels clear. This covers the per-mode smoke path
    # that is neither a scan integration view nor an Image/XYE viewer.
    calls.clear()
    host.frame_ids = ['0']
    host.data_1d = {
        0: SimpleNamespace(nexus_preview_payload={
            "kind": "plot_1d",
            "x": np.arange(3, dtype=float),
            "y": np.array([1.0, 2.0, 3.0]),
            "label": "nexus-row",
        })
    }
    host.data_2d = {}
    host.viewer_mode = 'nexus'
    host._bump_display_generation()
    host.update()
    assert "payload_plot" in calls
    assert "clear_image" in calls and "clear_binned" in calls
    assert "draw_binned" not in calls

    # Back to normal Int-2D: full panel set again.
    calls.clear()
    host.frame_ids = ['0', '1']
    host.data_1d = {0: object(), 1: object()}
    host.data_2d = {
        0: {'map_raw': np.ones((4, 4)), 'thumbnail': None},
        1: {'map_raw': np.ones((4, 4)), 'thumbnail': None},
    }
    host.viewer_mode = None
    host._bump_display_generation()
    host.update()
    assert "draw_plot" in calls                        # 1D plot legacy delegate
    # RAW_2D + CAKE_2D are payload-only; no publication_store -> both 2D blank.
    assert "clear_image" in calls and "clear_binned" in calls


def test_update_render_smoke_gi_scan_propagates_and_dispatches():
    # A GI scan still renders through the same path; gi flag propagates into
    # the state.  RAW_2D + CAKE_2D are payload-only (item-2 flip); no
    # publication_store on this host -> both 2D panels blank; the 1D plot draws.
    host, calls, dl = _update_smoke_host()
    host.scan.gi = True
    host.update()
    assert host._live_display_state().gi is True
    assert "draw_plot" in calls
    assert "clear_image" in calls and "clear_binned" in calls


def test_update_render_smoke_stale_generation_is_dropped():
    host, calls, dl = _update_smoke_host()
    state = host._live_display_state()
    calls.clear()
    stale = dl.DisplayPayload(generation=state.generation - 1, raw_image=None,
                              cake_image=None, plot=None)
    host.render_display(state, stale)
    assert calls == []                        # nothing drawn or cleared


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
        update_plot=rec("draw_plot"),
        # IMAGE_VIEWER renders its raw panel via the payload path; the title is
        # set by render_display (no longer a side effect of a legacy draw).
        _draw_image_payload=lambda *a, **k: calls.append("draw_payload_image") or True,
        _set_viewer_title=rec("viewer_title"),
        clear_image_view=rec("clear_image"),
        clear_binned_view=rec("clear_binned"),
        clear_plot_view=rec("clear_plot"),
        _apply_1d_only_visibility=rec("apply_1d_only"),
        update_2d_label=rec("label_2d"),
        _update_image_preview=rec("preview"),
    )
    host.ui.shareAxis.isChecked.return_value = False
    host.ui.imageUnit.currentIndex.return_value = 0
    for name in ("_current_image_axis_key", "_plot_axis_key",
                 "_share_axis_plot_index", "_set_plot_unit_index_silently",
                 "_apply_share_axis_state",
                 "_draw_delegate", "_clear_delegate", "_payload_for_role",
                 "_draw_payload", "render_display"):
        setattr(host, name, MethodType(getattr(displayFrameWidget, name), host))
    return host, calls, dl


def test_update_plot_view_sum_average_never_auto_waterfalls():
    # Bug (Vivek): Sum/Average of >15 frames rendered a WATERFALL instead of the
    # collapsed curve.  update_plot_view auto-switches to a waterfall on n_curves>15
    # (or Waterfall>3), but Sum/Average collapse to ONE curve in update_1d_view, so
    # they must never auto-waterfall however many rows are stacked.
    from unittest.mock import MagicMock
    cases = (("Average", "1d"), ("Sum", "1d"), ("Overlay", "wf"), ("Single", "wf"))
    n = 20
    for method, expect in cases:
        calls = []
        host = SimpleNamespace(
            _using_publication_plot_payload=True,
            frame_ids=["0"], data_1d={}, curves=[], plot=MagicMock(),
            frame_names=["f"] * n,
            plot_data=[np.arange(5), np.ones((n, 5))],
            ui=SimpleNamespace(
                plotMethod=SimpleNamespace(currentText=lambda m=method: m),
                yOffset=MagicMock()),
            update_wf=lambda: calls.append("wf"),
            update_1d_view=lambda: calls.append("1d"),
        )
        host.update_plot_view = MethodType(displayFrameWidget.update_plot_view, host)
        host.update_plot_view()
        assert calls == [expect], f"{method} (n={n}) -> {calls}, expected {expect}"


def test_clear_binned_view_drops_stale_slice_overlay():
    # P3 (Step-5 review): a blanked cake must not keep a floating slice-band ROI.
    # The flip moved the band re-attach to the cake renderer (_draw_image_payload),
    # so clearing the cake must clear the band too.
    from unittest.mock import MagicMock
    calls = []
    host = SimpleNamespace(
        binned_widget=MagicMock(),
        binned_data=("img", "rect"),
        _clear_image_widget=lambda w: calls.append("clear_img"),
        clear_slice_overlay=lambda: calls.append("clear_overlay"),
    )
    host.clear_binned_view = MethodType(displayFrameWidget.clear_binned_view, host)
    host.clear_binned_view()
    assert host.binned_data is None
    assert "clear_overlay" in calls


def test_render_display_int2d_draws_all_panels():
    host, calls, dl = _render_host()
    state = dl.compute_display_state(
        mode=dl.Mode.INT_2D, selected_ids=(0,), all_frame_index=[0],
        loaded_1d_keys={0}, loaded_2d_keys={0}, gi=False, plot_unit='q_A^-1',
        method='Single', unit_changed=False, prev_overlaid_ids=(),
        raw_availability={0: dict(has_raw=True)}, titles={}, generation=1)
    # RAW_2D + CAKE_2D draw via the payload path (_draw_image_payload); PLOT_1D
    # has no payload here so it falls through to the legacy update_plot.
    payload = dl.DisplayPayload(
        generation=1,
        raw_image=dl.ImagePayload(image=np.ones((2, 2))),
        cake_image=dl.ImagePayload(image=np.ones((2, 2))),
        plot=None)
    host.render_display(state, payload)
    assert "draw_payload_image" in calls and "draw_plot" in calls
    assert "label_2d" in calls and "preview" in calls
    assert not any(c.startswith("clear_") for c in calls)


def test_render_display_uses_publication_plot_payload_when_present():
    host, calls, dl = _render_host()
    host.bkg_1d = 0
    host.plot_data = [np.zeros(0), np.zeros(0)]
    host.plot_data_range = [[0, 0], [0, 0]]
    host.frame_names = []
    host.overlaid_idxs = []
    host._payload_x_axis_label = None
    host._using_publication_plot_payload = False
    host.update_plot_view = lambda: calls.append("payload_plot")

    state = dl.compute_display_state(
        mode=dl.Mode.INT_1D, selected_ids=(3,), all_frame_index=[3],
        loaded_1d_keys={3}, loaded_2d_keys=set(), gi=False, plot_unit='q_A^-1',
        method='Single', unit_changed=False, prev_overlaid_ids=(),
        raw_availability={}, titles={}, generation=4)
    payload = dl.DisplayPayload(
        generation=4,
        raw_image=None,
        cake_image=None,
        plot=dl.PlotPayload(
            axis_x=dl.Axis("2θ", "°"),
            traces=(dl.Trace("frame3", np.arange(3), np.array([1.0, 2.0, 3.0])),),
        ),
    )

    host.render_display(state, payload)

    assert "payload_plot" in calls
    assert "draw_plot" not in calls
    assert host.frame_names == ["frame3"]
    assert host._payload_x_axis_label == ("2θ", "°")
    np.testing.assert_allclose(host.plot_data[0], np.arange(3))
    np.testing.assert_allclose(host.plot_data[1], [[1.0, 2.0, 3.0]])


def test_publication_plot_native_uses_payload_derived_falls_back():
    # Step 5 FLIP: native INT 1D draws through the publication payload
    # (update_plot_view).  A 2D-derived axis / slice on a 1D-ONLY frame can't
    # build a 2D projection (no intensity_2d), so the payload returns None and
    # render_display falls back to the legacy update_plot.
    from xrd_tools.core import Axis, FrameView
    from xdart.gui.tabs.static_scan.display_publication import PublicationDisplayAdapter
    from xdart.modules.frame_publication import (
        PublicationStore,
        publication_from_frame_view,
    )

    def render_with_axis(source, sliced):
        host, calls, dl = _render_host()
        host.scan = SimpleNamespace(name="scan", gi=False)
        host._plot_axis_info = [{
            "source": source,
            "slice_axis": "χ",
            "axis": "radial",
        }]
        host.normalize = lambda data, metadata: data
        # The publication 1D draw funnels into update_plot_view; give the host
        # the state it reads/writes (Step-5 flip).
        host.bkg_1d = None
        host.plot_data = [np.zeros(0), np.zeros(0)]
        host.plot_data_range = [[0, 0], [0, 0]]
        host.frame_names = []
        host.overlaid_idxs = []
        host._payload_x_axis_label = None
        host._payload_y_axis_label = None
        host._using_publication_plot_payload = False
        host.update_plot_view = lambda: calls.append("payload_plot")
        host.ui.plotUnit.currentIndex.return_value = 0
        host.ui.plotUnit.currentText.return_value = "Q (Å⁻¹)"
        host.ui.slice.isChecked.return_value = sliced

        store = PublicationStore()
        store.upsert(
            publication_from_frame_view(
                FrameView(
                    label=3,
                    axis_1d=Axis("Q", "q_A^-1", values=np.arange(3)),
                    intensity_1d=np.array([1.0, 2.0, 3.0]),
                )
            )
        )
        state = dl.compute_display_state(
            mode=dl.Mode.INT_1D,
            selected_ids=(3,),
            all_frame_index=[3],
            loaded_1d_keys={3},
            loaded_2d_keys={3},
            gi=False,
            plot_unit='q_A^-1',
            method='Single',
            unit_changed=False,
            prev_overlaid_ids=(),
            raw_availability={},
            titles={},
            generation=store.generation,
        )
        payload = dl.build_payload(state, PublicationDisplayAdapter(store, widget=host))
        host.render_display(state, payload)
        return calls, payload

    # Native 1D -> the publication payload draws it.
    calls, payload = render_with_axis("1d", False)
    assert payload.plot is not None
    assert "payload_plot" in calls
    assert "draw_plot" not in calls

    # 2D-derived axis / slice on a 1D-only frame -> payload can't build -> legacy.
    for source, sliced in (("2d", False), ("1d_2d", True)):
        calls, payload = render_with_axis(source, sliced)
        assert payload.plot is None
        assert "draw_plot" in calls
        assert "payload_plot" not in calls


def test_render_display_image_viewer_draws_raw_clears_others():
    host, calls, dl = _render_host()
    state = dl.compute_display_state(
        mode=dl.Mode.IMAGE_VIEWER, selected_ids=(0,), all_frame_index=[],
        loaded_1d_keys=set(), loaded_2d_keys={0}, gi=False, plot_unit='q_A^-1',
        method='Single', unit_changed=False, prev_overlaid_ids=(),
        raw_availability={0: dict(has_raw=True)}, titles={}, generation=1)
    payload = dl.DisplayPayload(
        generation=1,
        raw_image=dl.ImagePayload(image=np.ones((2, 2))),
        cake_image=None, plot=None)
    host.render_display(state, payload)
    assert "draw_payload_image" in calls        # RAW_2D via the payload path
    assert "clear_binned" in calls and "clear_plot" in calls  # absent panels blanked
    assert "label_2d" not in calls              # viewer sets its own title
    assert "viewer_title" in calls              # render_display set the title
    assert "draw_image" not in calls            # not the Int-mode raw delegate


def test_render_display_image_viewer_none_payload_clears_raw():
    # A None raw payload (no frame / no map_raw / all-non-finite) must blank the
    # Image Viewer's raw panel — there is no legacy fallback draw.
    host, calls, dl = _render_host()
    state = dl.compute_display_state(
        mode=dl.Mode.IMAGE_VIEWER, selected_ids=(0,), all_frame_index=[],
        loaded_1d_keys=set(), loaded_2d_keys={0}, gi=False, plot_unit='q_A^-1',
        method='Single', unit_changed=False, prev_overlaid_ids=(),
        raw_availability={0: dict(has_raw=True)}, titles={}, generation=1)
    host.render_display(state, dl.build_payload(state))   # store=None -> raw None
    assert "clear_image" in calls               # raw panel blanked, not drawn
    assert "draw_payload_image" not in calls and "draw_image" not in calls


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


def test_live_display_state_render_ids_match_legacy_idxs():
    # The controller-built DisplayState's render_ids must match the legacy
    # idxs the (delegated) draw methods consume — the two paths coexist until
    # the data-source unification removes the idxs path.
    from xdart.gui.tabs.static_scan.display_logic import Mode

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
    for name in ('_live_mode', '_live_display_state'):
        setattr(host, name, MethodType(getattr(displayFrameWidget, name), host))

    state = host._live_display_state()
    expected = (sorted(host.idxs_2d)
                if state.mode in (Mode.INT_2D, Mode.IMAGE_VIEWER)
                else sorted(host.idxs_1d))
    assert list(state.render_ids) == expected


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
    viewer._teardown_load_worker = MethodType(
        H5Viewer._teardown_load_worker, viewer,
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


def test_cancel_pending_loads_invalidates_late_chunks():
    calls = []
    timer = _FakeTimer(active=True)
    viewer = SimpleNamespace(
        _load_generation=7,
        _update_coalesce_timer=timer,
        data_lock=RLock(),
        data_1d={},
        data_2d={},
        _load_worker=None,
        _load_thread=None,
        publication_store=None,
        _teardown_load_worker=lambda: calls.append("teardown"),
    )
    viewer.cancel_pending_loads = MethodType(H5Viewer.cancel_pending_loads, viewer)
    viewer._absorb_chunk = MethodType(H5Viewer._absorb_chunk, viewer)

    viewer.cancel_pending_loads()
    viewer._absorb_chunk(7, 12, object(), True)

    assert viewer._load_generation == 8
    assert calls == ["teardown"]
    assert timer.isActive() is False
    assert viewer.data_1d == {}
    assert viewer.data_2d == {}


def test_viewer_cleanup_stress_drops_stale_chunks_across_mode_switches():
    viewer = SimpleNamespace(
        data_lock=RLock(),
        data_1d={1: object()},
        data_2d={1: {"map_raw": np.ones((2, 2))}},
        frame_ids=["1"],
        latest_idx=1,
        new_scan_loaded=True,
        _raw_cache_order=[1],
        _load_generation=0,
        _load_worker=None,
        _load_thread=None,
        _update_coalesce_timer=_FakeTimer(active=True),
        ui=SimpleNamespace(
            listData=_FakeListWidget([1]),
            listScans=_FakeListWidget(["scan.nxs"]),
        ),
        _displayed_list_count=1,
        _displayed_last_label="1",
        publication_store=None,
    )
    viewer._clear_raw_cache = MethodType(H5Viewer._clear_raw_cache, viewer)
    viewer._remember_displayed_frames = MethodType(
        H5Viewer._remember_displayed_frames, viewer,
    )
    viewer._teardown_load_worker = MethodType(
        H5Viewer._teardown_load_worker, viewer,
    )
    viewer.cancel_pending_loads = MethodType(H5Viewer.cancel_pending_loads, viewer)
    viewer.enter_viewer_mode_cleanup = MethodType(
        H5Viewer.enter_viewer_mode_cleanup, viewer,
    )
    viewer._absorb_chunk = MethodType(H5Viewer._absorb_chunk, viewer)

    for frame_id in range(10):
        stale_generation = viewer._load_generation
        viewer.enter_viewer_mode_cleanup()
        viewer._absorb_chunk(stale_generation, frame_id, object(), True)

    assert viewer._load_generation == 10
    assert viewer.data_1d == {}
    assert viewer.data_2d == {}
    assert viewer.frame_ids == []
    assert viewer.ui.listData.count() == 0


def test_viewer_mode_change_blocks_scan_list_autoload():
    calls = []
    list_scans = _FakeListWidget(["old.xye"])

    def update_scans():
        calls.append(("update_scans_blocked", list_scans._signals_blocked))
        if not list_scans._signals_blocked:
            calls.append("autoload")

    def sync_dir(path, *, refresh=True):
        calls.append(("sync_dir", path, refresh))
        widget.h5viewer.dirname = path

    widget = SimpleNamespace(
        wrangler=SimpleNamespace(h5_dir="/tmp/xdart-out", tree=_FakeControl()),
        _apply_integration_control_state=lambda: None,
        h5viewer=SimpleNamespace(
            ui=SimpleNamespace(listScans=list_scans),
            actionNewFile=_FakeAction(),
            actionSaveDataAs=_FakeAction(),
            dirname="/tmp/stale",
            viewer_mode="xye",
            _suspend_scan_selection_loads=False,
            _apply_frames_panel_width=lambda vm: None,
            enter_viewer_mode_cleanup=lambda: calls.append(
                ("cleanup_suspend", widget.h5viewer._suspend_scan_selection_loads),
            ),
            cancel_pending_loads=lambda: calls.append("cancel"),
            update_scans=update_scans,
        ),
        displayframe=SimpleNamespace(
            _wrangler=None,
            _viewer_is_xdart=True,
            set_viewer_display_mode=lambda mode: calls.append(("display", mode)),
            clear_display_state=lambda: calls.append("clear_display"),
        ),
        _sync_h5viewer_save_dir=sync_dir,
        local_path="/tmp/stale",
    )

    staticWidget._on_viewer_mode_changed(widget, "image")

    assert ("sync_dir", "/tmp/xdart-out", False) in calls
    assert widget.h5viewer.dirname == "/tmp/xdart-out"
    assert ("cleanup_suspend", True) in calls
    assert ("update_scans_blocked", True) in calls
    assert "autoload" not in calls
    assert widget.h5viewer._suspend_scan_selection_loads is False
    assert list_scans._signals_blocked is False
    # New policy: the tree stays ENABLED in viewers (Project Folder / Save
    # Path drive the file browser); processing groups disable per-group.
    assert widget.wrangler.tree.isEnabled() is True
    assert widget.displayframe._viewer_is_xdart is False


def test_viewer_mode_tree_disable_only_for_file_viewers():
    calls = []
    list_scans = _FakeListWidget(["scan.nxs"])
    widget = SimpleNamespace(
        wrangler=SimpleNamespace(h5_dir="/tmp/xdart-out", tree=_FakeControl()),
        _apply_integration_control_state=lambda: None,
        h5viewer=SimpleNamespace(
            ui=SimpleNamespace(listScans=list_scans),
            actionNewFile=_FakeAction(),
            actionSaveDataAs=_FakeAction(),
            dirname="/tmp/xdart-out",
            viewer_mode="",
            _suspend_scan_selection_loads=False,
            _apply_frames_panel_width=lambda mode: calls.append(("width", mode)),
            enter_viewer_mode_cleanup=lambda: calls.append("cleanup"),
            cancel_pending_loads=lambda: calls.append("cancel"),
            update_scans=lambda: calls.append("update_scans"),
        ),
        displayframe=SimpleNamespace(
            _wrangler=None,
            set_viewer_display_mode=lambda mode: calls.append(("display", mode)),
            clear_display_state=lambda: calls.append("clear_display"),
        ),
        _sync_h5viewer_save_dir=lambda path, *, refresh=True: None,
        local_path="/tmp/xdart-out",
    )

    # New policy: the tree stays enabled in EVERY viewer mode -- Project
    # Folder / Save Path remain usable; processing groups disable per-group
    # on the wrangler side.
    staticWidget._on_viewer_mode_changed(widget, "xye")
    assert widget.wrangler.tree.isEnabled() is True

    staticWidget._on_viewer_mode_changed(widget, "nexus")
    assert widget.wrangler.tree.isEnabled() is True

    staticWidget._on_viewer_mode_changed(widget, "")
    assert widget.wrangler.tree.isEnabled() is True


def test_leaving_viewer_mode_clears_stale_global_mask():
    calls = []
    list_scans = _FakeListWidget(["scan.nxs"])
    widget = SimpleNamespace(
        wrangler=SimpleNamespace(h5_dir="/tmp/xdart-out", tree=_FakeControl()),
        _apply_integration_control_state=lambda: None,
        h5viewer=SimpleNamespace(
            ui=SimpleNamespace(listScans=list_scans),
            actionNewFile=_FakeAction(),
            actionSaveDataAs=_FakeAction(),
            dirname="/tmp/xdart-out",
            viewer_mode="image",
            _suspend_scan_selection_loads=False,
            _apply_frames_panel_width=lambda mode: calls.append(("width", mode)),
            enter_viewer_mode_cleanup=lambda: calls.append("cleanup"),
            cancel_pending_loads=lambda: calls.append("cancel"),
            update_scans=lambda: calls.append("update_scans"),
        ),
        displayframe=SimpleNamespace(
            _wrangler=None,
            set_viewer_display_mode=lambda mode: calls.append(("display", mode)),
            clear_display_state=lambda: calls.append("clear_display"),
        ),
        scan=SimpleNamespace(global_mask=np.ones((2, 2), dtype=bool)),
        _sync_h5viewer_save_dir=lambda path, *, refresh=True: None,
        local_path="/tmp/xdart-out",
    )

    staticWidget._on_viewer_mode_changed(widget, "")

    assert "cancel" in calls
    assert "clear_display" in calls
    assert widget.scan.global_mask is None


def test_load_frames_data_skips_missing_placeholder_file(tmp_path):
    calls = []
    viewer = SimpleNamespace(
        scan=SimpleNamespace(data_file=str(tmp_path / "missing_default.nxs")),
        cancel_pending_loads=lambda: calls.append("cancel"),
    )

    H5Viewer.load_frames_data(viewer, [1, 2], load_2d=True)

    assert calls == ["cancel"]
    assert not hasattr(viewer, "_load_worker")


def test_h5viewer_update_does_not_restore_stale_session_directory(monkeypatch, tmp_path):
    from xdart.gui.tabs.static_scan import h5viewer as h5mod

    stale_dir = tmp_path / "stale"
    current_dir = tmp_path / "current"
    stale_dir.mkdir()
    current_dir.mkdir()
    calls = []
    viewer = SimpleNamespace(
        dirname=str(current_dir),
        update_data=lambda: calls.append("update_data"),
        update_scans=lambda: calls.append("update_scans"),
    )
    monkeypatch.setattr(
        h5mod,
        "load_session",
        lambda: {"data_dir": str(stale_dir)},
    )

    H5Viewer.update(viewer)

    assert calls == ["update_data"]
    assert viewer.dirname == str(current_dir)


def test_viewer_mode_keeps_explicit_open_folder():
    calls = []
    opened_dir = "/tmp/user-opened-images"
    list_scans = _FakeListWidget(["raw.h5"])

    widget = SimpleNamespace(
        wrangler=SimpleNamespace(h5_dir="/tmp/xdart-out", tree=_FakeControl()),
        _apply_integration_control_state=lambda: None,
        h5viewer=SimpleNamespace(
            ui=SimpleNamespace(listScans=list_scans),
            actionNewFile=_FakeAction(),
            actionSaveDataAs=_FakeAction(),
            dirname=opened_dir,
            viewer_mode="",
            _suspend_scan_selection_loads=False,
            _apply_frames_panel_width=lambda vm: None,
            enter_viewer_mode_cleanup=lambda: calls.append("cleanup"),
            cancel_pending_loads=lambda: calls.append("cancel"),
            update_scans=lambda: calls.append("update_scans"),
        ),
        displayframe=SimpleNamespace(
            _wrangler=None,
            set_viewer_display_mode=lambda mode: calls.append(("display", mode)),
            clear_display_state=lambda: calls.append("clear_display"),
        ),
        _sync_h5viewer_save_dir=lambda path, *, refresh=True: calls.append(
            ("sync_dir", path, refresh),
        ),
        local_path="/tmp/default-xdart",
    )

    staticWidget._on_viewer_mode_changed(widget, "image")

    assert ("sync_dir", "/tmp/xdart-out", False) not in calls
    assert widget.h5viewer.dirname == opened_dir


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


def test_reduction_only_nxs_not_loaded_as_generic_image(tmp_path):
    import h5py
    from xrd_tools.io import ImageSourceKind

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
        _load_single_frame=lambda *args, **kwargs: calls.append("load"),
    )
    # Bind the cache clear after the namespace exists.
    viewer._clear_raw_cache = MethodType(H5Viewer._clear_raw_cache, viewer)
    viewer._load_image_file = MethodType(H5Viewer._load_image_file, viewer)

    viewer._load_image_file(str(path))

    assert viewer._viewer_source_info.kind is ImageSourceKind.UNKNOWN
    assert viewer._viewer_is_xdart is False
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
    import numpy as np
    from xrd_tools.io import ImageSourceKind
    from xdart.gui.tabs.static_scan.display_controllers import (
        ImageViewerController,
    )

    # An integrated stack is an unambiguous xdart-processed marker, even when
    # the group is otherwise bare.
    for marker in ("integrated_1d", "integrated_2d"):
        path = tmp_path / f"processed_{marker}.nxs"
        with h5py.File(path, "w") as f:
            entry = f.create_group("entry")
            entry.create_group(marker)

        # Classification is the ssrl boundary; these markers must classify as a
        # processed xdart file (not a raw detector image).
        info = ImageViewerController.classify(str(path))
        assert info.kind in (
            ImageSourceKind.PROCESSED_XDART, ImageSourceKind.THUMBNAIL_ONLY,
        )

    # A ``frames`` group only marks a processed file when it carries real frame
    # content (thumbnail/source).  A *bare* ``entry/frames`` is the native eiger
    # group and must NOT be read as processed-xdart (regression: that misread
    # made the Image Viewer refuse genuine raw eiger files).
    frames_path = tmp_path / "processed_frames.nxs"
    with h5py.File(frames_path, "w") as f:
        e = f.create_group("entry")
        thumb = np.linspace(0, 100, 16 * 16).reshape(16, 16)
        q = (thumb / thumb.max() * 255.0).astype(np.uint8)
        ds = e.create_dataset("frames/frame_0000/thumbnail", data=q)
        ds.attrs["vmin"] = 0.0
        ds.attrs["vmax"] = 100.0
        ds.attrs["dtype"] = "uint8"
    info = ImageViewerController.classify(str(frames_path))
    assert info.kind in (
        ImageSourceKind.PROCESSED_XDART, ImageSourceKind.THUMBNAIL_ONLY,
    )

    bare_frames = tmp_path / "bare_frames.nxs"
    with h5py.File(bare_frames, "w") as f:
        f.create_group("entry/frames")     # native eiger group, no content
    info = ImageViewerController.classify(str(bare_frames))
    assert info.kind is not ImageSourceKind.PROCESSED_XDART

    reduction_only = tmp_path / "reduction_only.nxs"
    with h5py.File(reduction_only, "w") as f:
        f.create_group("entry/reduction")  # provenance only, no displayable data
    info = ImageViewerController.classify(str(reduction_only))
    assert info.kind is ImageSourceKind.UNKNOWN


def _bind_nexus_viewer_methods(viewer):
    for name in (
        "_load_nexus_file",
        "_refresh_nexus_selected_preview",
        "_load_nexus_preview_payload",
        "_nexus_1d_selection",
        "_nexus_2d_selection",
        "_nexus_selection_truncated",
        "_nexus_axis_values",
        "_nexus_summary_rows",
        "_nexus_xdart_rows",
        "_nexus_reduced_info",
        "_nexus_tree_rows",
        "_nexus_value",
        "data_changed",
        "_remember_displayed_frames",
    ):
        if isinstance(H5Viewer.__dict__.get(name), staticmethod):
            setattr(viewer, name, getattr(H5Viewer, name))
        else:
            setattr(viewer, name, MethodType(getattr(H5Viewer, name), viewer))


def _nexus_viewer_host(tmp_path):
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = SimpleNamespace(
        data_lock=RLock(),
        data_1d={},
        data_2d={},
        frame_ids=[],
        viewer_mode="nexus",
        ui=SimpleNamespace(
            listData=_FakeListWidget(),
            labelCurrent=_FakeLabel(),
        ),
        _raw_cache_order=[],
        publication_store=None,
        sigUpdate=_FakeSignal(),
        dirname=str(tmp_path),
        scan=SimpleNamespace(
            name="null_main",
            frames=SimpleNamespace(index=[]),
            scan_lock=RLock(),
        ),
    )
    _bind_nexus_viewer_methods(viewer)
    return viewer


def _write_viewer_nexus_file(path):
    import h5py

    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"

        g1 = entry.create_group("integrated_1d")
        g1.attrs["NX_class"] = "NXdata"
        g1.attrs["signal"] = "intensity"
        g1.attrs["axes"] = ["frame_index", "q"]
        g1.create_dataset("frame_index", data=np.array([10, 11]))
        q = g1.create_dataset("q", data=np.linspace(0.5, 2.5, 5))
        q.attrs["units"] = "q_A^-1"
        q.attrs["long_name"] = "Q"
        y = g1.create_dataset(
            "intensity",
            data=np.arange(10, dtype=float).reshape(2, 5),
        )
        y.attrs["units"] = "counts"
        y.attrs["long_name"] = "Integrated intensity"

        g2 = entry.create_group("integrated_2d")
        g2.attrs["NX_class"] = "NXdata"
        g2.attrs["signal"] = "intensity"
        g2.attrs["axes"] = ["frame_index", "chi", "q"]
        g2.create_dataset("frame_index", data=np.array([10, 11]))
        q2 = g2.create_dataset("q", data=np.linspace(-1.0, 1.0, 4))
        q2.attrs["units"] = "qip_A^-1"
        q2.attrs["long_name"] = "Qip"
        chi = g2.create_dataset("chi", data=np.linspace(0.0, 3.0, 3))
        chi.attrs["units"] = "qoop_A^-1"
        chi.attrs["long_name"] = "Qoop"
        g2.create_dataset(
            "intensity",
            data=np.arange(24, dtype=float).reshape(2, 3, 4),
        )

        generic = entry.create_dataset("generic_1d", data=np.array([1.0, 2.0, 3.0]))
        generic.attrs["units"] = "arb"
        generic.attrs["description"] = "Generic signal"

        det = entry.create_group("instrument/detector")
        raw = det.create_dataset(
            "data",
            data=np.arange(2 * 6 * 7, dtype=np.uint16).reshape(2, 6, 7),
        )
        raw.attrs["units"] = "counts"


def test_nexus_viewer_loads_rows_and_previews_1d_2d_units(tmp_path):
    path = tmp_path / "viewer.nxs"
    _write_viewer_nexus_file(path)
    viewer = _nexus_viewer_host(tmp_path)

    viewer._load_nexus_file(str(path))

    labels = [viewer.ui.listData.item(i).text() for i in range(viewer.ui.listData.count())]
    assert "Integrated 1D" in labels
    assert "Integrated 2D" in labels
    assert "Raw detector dataset" in labels
    row_1d = labels.index("Integrated 1D")
    key_1d = viewer.ui.listData.item(row_1d).data(QtCore.Qt.UserRole)
    assert viewer.frame_ids == [str(key_1d)]
    assert viewer.ui.labelCurrent.text == path.name
    assert viewer.data_1d[1].__class__.__name__ == "_ViewerRow"
    payload_1d = viewer.data_1d[key_1d].nexus_preview_payload
    assert payload_1d["kind"] == "plot_1d"
    assert payload_1d["x_unit"] == "q_A^-1"
    assert payload_1d["x_label"] == "Q"
    assert payload_1d["y_label"] == "Intensity"
    np.testing.assert_allclose(payload_1d["x"], np.linspace(0.5, 2.5, 5))
    np.testing.assert_allclose(payload_1d["y"], np.arange(5, dtype=float))
    assert viewer.data_1d[key_1d].scan_info["preview_selection"] == "[0, :]"

    row_2d = labels.index("Integrated 2D")
    viewer.ui.listData.setCurrentRow(row_2d)
    viewer.data_changed()
    key_2d = viewer.ui.listData.item(row_2d).data(QtCore.Qt.UserRole)
    payload_2d = viewer.data_1d[key_2d].nexus_preview_payload
    assert payload_2d["kind"] == "image_2d"
    assert payload_2d["image"].shape == (3, 4)
    assert payload_2d["x_label"] == "Qip"
    assert payload_2d["x_unit"] == "qip_A^-1"
    assert payload_2d["y_label"] == "Qoop"
    assert payload_2d["y_unit"] == "qoop_A^-1"

    row_raw = labels.index("Raw detector dataset")
    viewer.ui.listData.setCurrentRow(row_raw)
    viewer.data_changed()
    key_raw = viewer.ui.listData.item(row_raw).data(QtCore.Qt.UserRole)
    payload_raw = viewer.data_1d[key_raw].nexus_preview_payload
    assert payload_raw["kind"] == "image_2d"
    assert payload_raw["image"].shape == (6, 7)
    assert payload_raw["x_label"] == "column"
    assert payload_raw["y_label"] == "row"
    assert viewer.data_1d[key_raw].scan_info["_shape"] == (2, 6, 7)
    assert viewer.data_1d[key_raw].scan_info["dtype"] == "uint16"


def test_wrangler_expands_active_groups_on_startup():
    # P3 #8: GI / Threshold / Background groups default folded; if their
    # enabling toggle is on (e.g. a restored session) the group must expand.
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    class _P:
        def __init__(self, value=None, children=None):
            self._value = value
            self._children = children or {}
            self.opts = {}

        def value(self):
            return self._value

        def child(self, *path):
            node = self
            for name in path:
                node = node._children[name]
            return node

        def setOpts(self, **o):
            self.opts.update(o)

    def _tree(grazing, threshold, bg):
        gi = _P(children={"Grazing": _P(grazing)})
        mask = _P(children={"Threshold": _P(threshold)})
        bgg = _P(children={"bg_type": _P(bg)})
        root = _P(children={"GI": gi, "Mask": mask, "BG": bgg})
        return root, gi, mask, bgg

    # Toggles on / source selected → groups expand.
    root, gi, mask, bgg = _tree(True, True, "Single BG File")
    host = SimpleNamespace(parameters=root)
    host._expand_active_groups = MethodType(
        imageWrangler._expand_active_groups, host,
    )
    host._expand_active_groups()
    assert gi.opts.get("expanded") is True
    assert mask.opts.get("expanded") is True
    assert bgg.opts.get("expanded") is True

    # All off / no background → groups left collapsed (no expand opt set).
    root, gi, mask, bgg = _tree(False, False, "None")
    host = SimpleNamespace(parameters=root)
    host._expand_active_groups = MethodType(
        imageWrangler._expand_active_groups, host,
    )
    host._expand_active_groups()
    # UI-1 (#81): the GI / Intensity-Threshold groups' EXPANDED state is now the
    # on/off toggle, so "off" explicitly COLLAPSES them (expanded=False) rather
    # than leaving the opt unset.  BG still only expands-when-on (its toggle is
    # the bg_type list, not the header), so it stays untouched when off.
    assert gi.opts.get("expanded") is False
    assert mask.opts.get("expanded") is False
    assert "expanded" not in bgg.opts


def test_frames_panel_width_relaxes_in_nexus_mode_and_restores():
    # P3 #7: the Frames (listData) max width is 60 px in the .ui (right for
    # frame indices) but clips NeXus dataset labels.  It must relax in NeXus
    # viewer mode and restore for every other mode.
    class _FakeList:
        def __init__(self):
            self._maxw = 60

        def maximumWidth(self):
            return self._maxw

        def setMaximumWidth(self, w):
            self._maxw = int(w)

    lw = _FakeList()
    host = SimpleNamespace(ui=SimpleNamespace(listData=lw))
    host._apply_frames_panel_width = MethodType(
        H5Viewer._apply_frames_panel_width, host,
    )

    host._apply_frames_panel_width("nexus")
    assert lw.maximumWidth() == 16777215          # relaxed for NeXus labels
    host._apply_frames_panel_width(None)
    assert lw.maximumWidth() == 90                # normal mode gets a wider cap
    host._apply_frames_panel_width("image")
    assert lw.maximumWidth() == 90                # other modes stay bounded


def test_nexus_1d_selection_strides_large_axis_but_not_small():
    # P3 #5: short curves read whole; an oversized 1D axis is strided.
    sel_small, axis_small = H5Viewer._nexus_1d_selection((2, 5), max_points=8192)
    assert axis_small == slice(None)                 # full read for small data
    assert sel_small == (0, slice(None))

    sel_big, axis_big = H5Viewer._nexus_1d_selection((40000,), max_points=8192)
    assert isinstance(axis_big, slice) and axis_big.step and axis_big.step > 1
    assert len(range(0, 40000, axis_big.step)) <= 8192  # strided count within cap
    assert sel_big == axis_big                          # 1D dataset → axis slice


def test_nexus_1d_preview_downsamples_large_dataset(tmp_path):
    # P3 #5 end-to-end: a huge 1D dataset is bounded for the GUI preview, x
    # is strided to match y, and the truncated flag is set.
    import h5py

    path = tmp_path / "big1d.nxs"
    n = 50000
    with h5py.File(path, "w") as f:
        e = f.create_group("entry")
        e.create_dataset("big/signal", data=np.arange(n, dtype=float))
        e.create_dataset("big/x", data=np.linspace(0.0, 10.0, n))

    viewer = _nexus_viewer_host(tmp_path)
    info = {
        "dataset_path": "entry/big/signal",
        "_shape": (n,),
        "nexus_preview_kind": "plot_1d",
        "_attrs": {"units": "counts"},
        "x_axis_path": "entry/big/x",
        "x_label": "x", "x_unit": "m",
        "y_label": "sig", "y_unit": "counts",
    }
    payload, preview_info = viewer._load_nexus_preview_payload(str(path), info)

    assert payload["kind"] == "plot_1d"
    assert 0 < payload["y"].size <= 8192             # bounded
    assert payload["x"].shape == payload["y"].shape  # x strided to match
    assert preview_info["preview_truncated"] is True


def test_nexus_controller_builds_plot_and_image_payloads(tmp_path):
    from xdart.gui.tabs.static_scan.display_logic import (
        ImagePayload,
        Mode,
        PanelRole,
        controller_for,
    )

    path = tmp_path / "viewer.nxs"
    _write_viewer_nexus_file(path)
    viewer = _nexus_viewer_host(tmp_path)
    viewer.display_generation = 4
    viewer.overlaid_idxs = []
    viewer.ui.plotMethod = _FakeCombo("Single")
    viewer._load_nexus_file(str(path))
    labels = [viewer.ui.listData.item(i).text() for i in range(viewer.ui.listData.count())]

    row_1d = labels.index("Integrated 1D")
    ctrl = controller_for(Mode.NEXUS_VIEWER)
    state = ctrl.compute_state(viewer, Mode.NEXUS_VIEWER)
    payload = ctrl.build_payload(viewer, state)
    assert state.panel(PanelRole.PLOT_1D).has_data
    assert not state.panel(PanelRole.RAW_2D).has_data
    assert payload.plot.axis_x.unit == "q_A^-1"
    assert payload.plot.axis_y.label == "Intensity"

    row_2d = labels.index("Integrated 2D")
    viewer.ui.listData.setCurrentRow(row_2d)
    viewer.data_changed()
    state = ctrl.compute_state(viewer, Mode.NEXUS_VIEWER)
    payload = ctrl.build_payload(viewer, state)
    assert state.panel(PanelRole.RAW_2D).has_data
    assert not state.panel(PanelRole.PLOT_1D).has_data
    assert isinstance(payload.raw_image, ImagePayload)
    assert payload.raw_image.axis_x.unit == "qip_A^-1"
    assert payload.raw_image.axis_y.unit == "qoop_A^-1"


def test_nexus_viewer_real_xdart_processed_file_smoke():
    path = (
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        + "/test_data/xdart_processed_data/"
        + "Combi4_Angledependence_samz_4p9_03271002.nxs"
    )
    if not os.path.exists(path):
        import pytest
        pytest.skip("local xdart processed test data is not available")

    viewer = _nexus_viewer_host(os.path.dirname(path))
    viewer.display_generation = 11
    viewer.overlaid_idxs = []
    viewer.ui.plotMethod = _FakeCombo("Single")

    viewer._load_nexus_file(path)

    labels = [viewer.ui.listData.item(i).text() for i in range(viewer.ui.listData.count())]
    assert "Integrated 1D" in labels

    row_1d = labels.index("Integrated 1D")
    viewer.ui.listData.setCurrentRow(row_1d)
    viewer.data_changed()
    key_1d = viewer.ui.listData.item(row_1d).data(QtCore.Qt.UserRole)
    payload_1d = viewer.data_1d[key_1d].nexus_preview_payload
    assert payload_1d["kind"] == "plot_1d"
    assert payload_1d["x"].size == payload_1d["y"].size
    assert payload_1d["x_unit"]

    if "Integrated 2D" in labels:
        row_2d = labels.index("Integrated 2D")
        viewer.ui.listData.setCurrentRow(row_2d)
        viewer.data_changed()
        key_2d = viewer.ui.listData.item(row_2d).data(QtCore.Qt.UserRole)
        payload_2d = viewer.data_1d[key_2d].nexus_preview_payload
        assert payload_2d["kind"] == "image_2d"
        assert payload_2d["image"].ndim == 2
        assert payload_2d["image"].size > 0


def test_metadata_panel_nexus_viewer_uses_selected_row_not_scan_table():
    import pandas as pd
    from PySide6 import QtWidgets
    from PySide6.QtCore import QModelIndex
    from xdart.gui.tabs.static_scan.metadata import metadataWidget

    from xdart.modules.frame_publication import (
        PublicationStore, publication_from_frame_view)
    from xrd_tools.core import FrameView, numeric_metadata

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # Phase 3c: the NeXus viewer mirrors the selected row's metadata into the
    # publication store; the metadata panel reads it store-first.
    info = {"kind": "dataset", "path": "/entry/data", "_attrs": {"hidden": True}}
    store = PublicationStore()
    store.upsert(publication_from_frame_view(
        FrameView.from_results(
            label=3, metadata_raw=info, metadata_numeric=numeric_metadata(info)),
        generation=store.generation))
    scan = SimpleNamespace(
        scan_data=pd.DataFrame({"stale_scan_value": [1.0]}, index=[3]),
    )
    mw = metadataWidget(
        scan,
        None,
        ["3"],
        {},
        publication_store=store,
        data_lock=RLock(),
    )
    mw.viewer_mode = "nexus"

    frame_widget = QtWidgets.QFrame()
    frame_widget.setLayout(mw.layout)
    win = QtWidgets.QWidget()
    QtWidgets.QVBoxLayout(win).addWidget(frame_widget)
    win.show()
    app.processEvents()

    mw.update()
    model = mw.tableview.model()
    assert "kind" in list(model.dataFrame.index)
    assert "path" in list(model.dataFrame.index)
    assert "stale_scan_value" not in list(model.dataFrame.index)
    assert "_attrs" not in list(model.dataFrame.index)
    assert model.rowCount(QModelIndex()) == 2
    win.close()


# ── ImageViewerController.build_payload (Stage 4/5 step 2): the raw-preview
# semantics now live in the pure helper _image_viewer_raw_payload, replacing
# the deleted legacy _update_image_viewer.  A None payload tells render to clear.

def _img_state(render_ids):
    """Minimal DisplayState stand-in: the helper reads only ``render_ids``."""
    return SimpleNamespace(render_ids=tuple(render_ids), generation=1)


def test_image_viewer_missing_raw_and_thumbnail_yields_no_payload():
    from xdart.gui.tabs.static_scan.display_controllers import (
        _image_viewer_raw_payload,
    )
    host = SimpleNamespace(
        data_lock=RLock(),
        data_2d={1: {"map_raw": None, "thumbnail": None}},
        _viewer_is_xdart=False,
    )
    assert _image_viewer_raw_payload(host, _img_state([1])) is None


def test_image_viewer_no_selection_yields_no_payload():
    from xdart.gui.tabs.static_scan.display_controllers import (
        _image_viewer_raw_payload,
    )
    host = SimpleNamespace(data_lock=RLock(), data_2d={}, _viewer_is_xdart=False)
    assert _image_viewer_raw_payload(host, _img_state([])) is None


def test_image_viewer_all_sentinel_image_yields_no_payload():
    from xdart.gui.tabs.static_scan.display_controllers import (
        _image_viewer_raw_payload,
    )
    host = SimpleNamespace(
        data_lock=RLock(),
        data_2d={1: {"map_raw": np.full((2, 2), 4294967295.0),
                     "thumbnail": None}},
        _viewer_is_xdart=False,
    )
    # All sentinel -> standalone fill leaves no finite pixel -> blank (None).
    assert _image_viewer_raw_payload(host, _img_state([1])) is None


def test_image_viewer_standalone_uint16_ceiling_stays_raw_and_finite():
    from xdart.gui.tabs.static_scan.display_controllers import (
        _image_viewer_raw_payload,
    )
    raw = np.array([[10.0, 65535.0], [20.0, 30.0]])
    host = SimpleNamespace(
        data_lock=RLock(),
        data_2d={1: {"map_raw": raw, "thumbnail": None}},
        _viewer_is_xdart=False,
    )
    payload = _image_viewer_raw_payload(host, _img_state([1]))
    assert payload is not None
    # uint16 ceiling is a real count for a standalone file (not a NaN mask).
    assert np.isnan(payload.image).sum() == 0
    assert np.nanmax(payload.image) == 65535.0


def test_image_viewer_payload_applies_no_mask_background_or_normalization():
    from xdart.gui.tabs.static_scan.display_controllers import (
        _image_viewer_raw_payload,
    )
    raw = np.array([[1.0, 2.0], [3.0, 4.0]])
    host = SimpleNamespace(
        data_lock=RLock(),
        data_2d={1: {"map_raw": raw, "thumbnail": None}},
        _viewer_is_xdart=False,
        # A wrangler threshold/mask + a background + a monitor that the raw
        # browser must all IGNORE (it shows detector counts, not a processed
        # frame).  None of these may alter the payload pixels.
        _wrangler=SimpleNamespace(
            apply_threshold=True, threshold_min=2, threshold_max=3,
            mask_file="/definitely/not/a/raw-viewer-mask.edf"),
        bkg_map_raw=np.array([[10.0, 10.0], [10.0, 10.0]]),
        normalize=lambda data, metadata: np.asarray(data, dtype=float) / 250.0,
    )
    payload = _image_viewer_raw_payload(host, _img_state([1]))
    assert payload is not None
    # Values preserved exactly (only orientation flipped) — no mask applied, no
    # background subtracted, not divided by the monitor.
    np.testing.assert_allclose(np.sort(payload.image.ravel()), [1, 2, 3, 4])


def test_available_norm_channels_filters_present_case_insensitive_aliases():
    channels = available_norm_channels([
        "TEMP", "Sec", "MON", "I0", "i2", "bstop", "sample", "Second",
    ])

    assert channels == [
        ("sec", "Sec"),
        ("Monitor", "MON"),
        ("i0", "I0"),
        ("i2", "i2"),
    ]


def test_refresh_norm_channels_populates_combo_from_scan_data_aliases():
    combo = _FakeMutableCombo()
    host = SimpleNamespace(
        scan=SimpleNamespace(
            scan_data=SimpleNamespace(columns=["TEMP", "SEC", "I0", "foo"]),
        ),
        ui=SimpleNamespace(normChannel=combo),
    )
    host.get_normChannel = MethodType(DisplayDataMixin.get_normChannel, host)
    host.refresh_norm_channels = MethodType(
        DisplayDataMixin.refresh_norm_channels, host,
    )

    host.refresh_norm_channels()
    combo.setCurrentIndex(2)

    assert combo._items == ["Norm Channel", "sec", "i0"]
    assert host.get_normChannel() == "I0"


def test_empty_image_clear_hides_colorbar_without_zero_paint():
    widget = _FakeImageWidget()

    displayFrameWidget._clear_image_widget(widget)

    assert widget.imageItem.cleared is True
    assert widget.histogram.visible is False
    assert widget.images == []


def test_processed_image_viewer_keeps_baked_nan_mask_visible():
    from xdart.gui.tabs.static_scan.display_controllers import (
        _image_viewer_raw_payload,
    )
    raw = np.array([[10.0, np.nan], [20.0, 30.0]])
    host = SimpleNamespace(
        data_lock=RLock(),
        data_2d={1: {"map_raw": raw, "thumbnail": None}},
        _viewer_is_xdart=True,
    )
    payload = _image_viewer_raw_payload(host, _img_state([1]))
    assert payload is not None
    assert np.isnan(payload.image).sum() == 1       # baked mask preserved


def test_image_viewer_raw_pixel_axes_are_not_si_scaled():
    labels = []
    axes = {}

    class _Axis:
        def __init__(self):
            self.auto_si = None
            self.scale = None

        def enableAutoSIPrefix(self, enabled):
            self.auto_si = enabled

        def setScale(self, scale):
            self.scale = scale

    class _Plot:
        def setLabel(self, side, text, **kwargs):
            labels.append((side, text, kwargs))

        def getAxis(self, side):
            axes.setdefault(side, _Axis())
            return axes[side]

    widget = SimpleNamespace(image_plot=_Plot())

    displayFrameWidget._set_raw_pixel_axes(widget)

    assert labels == [
        ("bottom", "x (Pixels)", {}),
        ("left", "y (Pixels)", {}),
    ]
    assert axes["bottom"].auto_si is False
    assert axes["left"].auto_si is False


def test_display_preview_preserves_nan_masks_but_collapses_infinities():
    from xdart.gui.tabs.static_scan.display_constants import _downsample_for_display

    class _Widget:
        def width(self):
            return 200
        def height(self):
            return 200

    data = np.array([[np.nan, 10.0], [20.0, np.inf]])

    preview = _downsample_for_display(data, _Widget())

    assert np.isnan(preview[0, 0])
    assert preview[1, 1] == 10.0
    assert np.isfinite(preview[0, 1])


def test_real_eiger_preview_masks_sentinels_for_display():
    from pathlib import Path
    from xrd_tools.io.image import read_image
    from xdart.gui.tabs.static_scan.display_constants import _downsample_for_display

    path = Path("/Users/vthampy/repos/test_data/eiger/Eiger_B_ctrl_test__2000mdeg_scan001_master.h5")
    if not path.exists():
        pytest.skip("real Eiger test data not available")

    class _Widget:
        def width(self):
            return 500
        def height(self):
            return 500

    from xdart.gui.tabs.static_scan.display_logic import standalone_viewer_image
    raw = np.asarray(read_image(path, frame=0), dtype=float)
    data = standalone_viewer_image(raw).T[:, ::-1]

    preview = _downsample_for_display(data, _Widget())

    assert np.isfinite(preview).all()
    assert np.nanmax(preview) < 4294967295.0


def test_update_renders_blank_on_empty_no_data_instead_of_leaving_stale():
    # P2 #3: update() used to early-return when _updated() reported no usable
    # data (empty selection / failed load / cache miss), leaving a stale
    # plot/raw/cake on screen.  It must instead render an explicit empty
    # state that blanks every panel.
    from xdart.gui.tabs.static_scan.display_logic import Mode

    calls = []
    host = SimpleNamespace(
        display_generation=3,
        get_idxs=lambda: None,
        _note_selection_generation=lambda: None,
        _updated=lambda: False,
        _live_mode=lambda: Mode.IMAGE_VIEWER,   # skips INT-mode axis setup
        data_lock=RLock(),
        data_1d={},
        data_2d={},
        clear_plot_view=lambda: calls.append("plot"),
        clear_image_view=lambda: calls.append("image"),
        clear_binned_view=lambda: calls.append("cake"),
    )
    host._clear_delegate = MethodType(displayFrameWidget._clear_delegate, host)
    host.render_display = MethodType(displayFrameWidget.render_display, host)
    host.update = MethodType(displayFrameWidget.update, host)
    host._update_impl = MethodType(displayFrameWidget._update_impl, host)

    assert host.update() is True
    assert set(calls) == {"plot", "image", "cake"}   # every panel blanked
    assert host._display_blanked is True

    # Repeated empty update is a no-op — current content is already blank.
    calls.clear()
    assert host.update() is True
    assert calls == []


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
        _processing_active=False,
        publication_store=None,
        ui=SimpleNamespace(
            plotUnit=unit,
            plotMethod=method_combo,
            slice=slice_control,
            slice_center=SimpleNamespace(value=lambda: 0.0),
            slice_width=SimpleNamespace(value=lambda: 1.0),
        ),
        update_plot_view=lambda: None,
    )

    def get_frames_int_1d(idxs=None, rv="all", *, require_all=False,
                          allow_blocking_read=None):
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
    for name in (
        "resolve_plot_axis",
        "collect_plot_rows",
        "build_plot_names",
        "apply_plot_background",
        "compute_plot_range",
        "draw_plot_state",
        "_loaded_1d_overlay_labels",
        "_waterfall_active",
        "_uniform_waterfall_grid",
        "_reexpress_overlay_unit",
        "update_plot",
    ):
        setattr(host, name, MethodType(getattr(DisplayPlotMixin, name), host))
    return host


def test_update_plot_accumulator_replaces_for_single_sum_average():
    out, names, ids = update_plot_accumulator(
        [np.array([99.0]), np.array([[99.0]])],
        ["old"],
        [99],
        np.array([0.0, 1.0]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        ["scan_1", "scan_2"],
        [1, 2],
        "Average",
        False,
    )

    np.testing.assert_allclose(out[0], [0.0, 1.0])
    np.testing.assert_allclose(out[1], [[1.0, 2.0], [3.0, 4.0]])
    assert names == ["scan_1", "scan_2"]
    assert ids == [1, 2]


def test_overlay_live_tail_rows_keep_matching_tail_labels_after_store_cap():
    """Overlay/Waterfall live accumulation must append the resident tail.

    The selection may contain every processed frame while ``idxs_1d`` contains
    only the heavy-store resident tail.  Names and ids must both come from that
    tail; otherwise rows after the store cap get paired with old labels and the
    accumulator stops growing.
    """
    host = _plot_host("Overlay")
    host._processing_active = True
    host.idxs = list(range(1, 101))
    host.idxs_1d = list(range(37, 101))
    host.data_1d = {idx: object() for idx in host.idxs_1d}
    host.plot_data = [
        np.array([0.0, 1.0]),
        np.vstack([np.array([float(i), float(i) + 0.5]) for i in range(1, 37)]),
    ]
    host.frame_names = [f"scan_{i}" for i in range(1, 37)]
    host.overlaid_idxs = list(range(1, 37))
    host._last_plot_unit = 0

    host.update_plot()

    assert host.overlaid_idxs == list(range(1, 101))
    assert host.frame_names[-1] == "scan_100"
    assert host.plot_data[1].shape[0] == 100
    np.testing.assert_allclose(host.plot_data[1][-1], [100.0, 100.5])


def test_waterfall_grid_resamples_nonuniform_x_without_moving_peak():
    q = np.linspace(1.0, 8.0, 301)
    wavelength_m = 0.7293188143129427e-10
    tth = 2 * np.degrees(
        np.arcsin(np.clip(q * wavelength_m * 1e10 / (4 * np.pi), -1, 1))
    )
    peak = int(np.argmin(np.abs(q - 6.7)))
    data = np.zeros((5, q.size), dtype=float)
    data[:, peak] = 1.0

    x_uniform, data_uniform = DisplayPlotMixin._uniform_waterfall_grid(tth, data)

    col = int(np.nanargmax(np.nansum(data_uniform, axis=0)))
    bin_width = abs(x_uniform[1] - x_uniform[0])
    assert abs(x_uniform[col] - tth[peak]) <= bin_width
    endpoint_linear = (
        x_uniform[0]
        + (peak + 0.5)
        * (x_uniform[-1] - x_uniform[0])
        / q.size
    )
    assert abs(endpoint_linear - tth[peak]) > 2 * bin_width


def test_update_plot_accumulator_appends_skips_duplicates_and_merges_grids():
    out, names, ids = update_plot_accumulator(
        [np.array([0.0, 1.0]), np.array([[1.0, 2.0]])],
        ["scan_1"],
        [1],
        np.array([0.5, 1.5]),
        np.array([[10.0, 20.0], [30.0, 40.0]]),
        ["scan_1", "scan_2"],
        [1, 2],
        "Overlay",
        False,
    )

    np.testing.assert_allclose(out[0], [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(
        out[1],
        [
            [1.0, 1.5, 2.0, np.nan],
            [np.nan, 30.0, 35.0, 40.0],
        ],
        equal_nan=True,
    )
    assert names == ["scan_1", "scan_2"]
    assert ids == [1, 2]


def test_update_plot_accumulator_rebuilds_on_unit_change():
    out, names, ids = update_plot_accumulator(
        [np.array([0.0, 1.0]), np.array([[1.0, 2.0]])],
        ["scan_1"],
        [1],
        np.array([10.0, 11.0]),
        np.array([[5.0, 6.0]]),
        ["scan_1"],
        [1],
        "Waterfall",
        True,
    )

    np.testing.assert_allclose(out[0], [10.0, 11.0])
    np.testing.assert_allclose(out[1], [[5.0, 6.0]])
    assert names == ["scan_1"]
    assert ids == [1]


def test_update_plot_accumulator_skips_empty_incoming_grid():
    out, names, ids = update_plot_accumulator(
        [np.array([0.0, 1.0]), np.array([[1.0, 2.0]])],
        ["scan_1"],
        [1],
        np.array([]),
        np.zeros((1, 0)),
        ["scan_2"],
        [2],
        "Overlay",
        False,
    )

    np.testing.assert_allclose(out[0], [0.0, 1.0])
    np.testing.assert_allclose(out[1], [[1.0, 2.0]])
    assert names == ["scan_1"]
    assert ids == [1]


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


def test_overlay_new_scan_resets_accumulator_not_stale_or_appended():
    # REGRESSION (this session): processing a NEW scan in Overlay/Waterfall mode
    # must RESET the accumulator to the new scan's frames -- not show the previous
    # scan's stale curves (the new scan's ids are a SUBSET of the old, so the
    # end-of-scan catch-up SKIP wrongly fired) and not append across scans (a new
    # scan may have different integration params / GI / axis).  update_plot
    # self-heals on a scan-identity change.  Consistent for <15 and >15 frames
    # (the reset is method-gated, independent of the auto-waterfall threshold).
    host = _plot_host("Overlay")
    host.scan.data_file = "scanA.nxs"
    for idx in (1, 2, 3):
        host.idxs = [idx]
        host.idxs_1d = [idx]
        host.update_plot()
    assert host.overlaid_idxs == [1, 2, 3]
    assert host._overlay_scan_key == "scanA.nxs"

    # New scan: different file, ids that are a SUBSET of the previous scan's.
    host.scan.data_file = "scanB.nxs"
    host.idxs = [1]
    host.idxs_1d = [1]
    host.update_plot()

    assert host._overlay_scan_key == "scanB.nxs"
    assert host.overlaid_idxs == [1]                 # reset (not stale [1,2,3], not appended)
    assert np.atleast_2d(host.plot_data[1]).shape[0] == 1

    # Same-scan render still APPENDS within the scan (reset is scan-boundary only).
    host.idxs = [2]
    host.idxs_1d = [2]
    host.update_plot()
    assert host.overlaid_idxs == [1, 2]


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


def test_overlay_append_skips_empty_incoming_grid_without_crash():
    # Regression (P1 #1): a frame whose 1D grid is empty (cache miss mid
    # fast batch) reached the merge branch and _reinterp(np.interp on empty
    # src_x) raised, aborting the whole render so completed traces never
    # painted.  The empty-grid frame must be skipped, not crash.
    host = _plot_host("Overlay")
    host.idxs = [1]
    host.idxs_1d = [1]
    host.update_plot()
    assert host.plot_data[1].shape == (1, 2)

    # Frame 2 comes back with an empty x grid.
    host.get_frames_int_1d = lambda idxs=None, rv="all", *, require_all=False, allow_blocking_read=None: (
        np.zeros((1, 0)), np.zeros(0))
    host.idxs = [2]
    host.idxs_1d = [2]
    host.update_plot()  # must not raise

    assert host.plot_data[1].shape == (1, 2)        # frame 2 skipped
    assert "scan_2" not in host.frame_names
    assert host.overlaid_idxs == [1]


def test_overlay_append_empty_accumulator_seeds_fresh_grid():
    # Regression (P1 #1): with overlay "active" (plan_overlay → APPEND) but
    # an emptied accumulator (old_x empty), interpolating onto the empty x
    # crashed.  It must seed the grid from the incoming frame instead.
    host = _plot_host("Overlay")
    host.overlaid_idxs = [99]
    host.frame_names = ["scan_99"]
    host.plot_data = [np.zeros(0), np.zeros((0, 0))]
    host._last_plot_unit = 0          # no unit change → APPEND (not REBUILD)
    host.idxs = [1]
    host.idxs_1d = [1]
    host.update_plot()  # must not raise

    assert host.plot_data[0].size == 2              # grid seeded
    assert host.plot_data[1].shape == (1, 2)
    assert host.frame_names == ["scan_1"]
    assert host.overlaid_idxs == [1]


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
        # Switching to Single clears the accumulation then re-renders through the
        # payload path (update()), not the deleted _update_xye_viewer.
        update=lambda: calls.append("update"),
        _autorange_plot_view=lambda: None,    # R2-3 autorange (no-op here)
    )
    host.clear_overlay = MethodType(displayFrameWidget.clear_overlay, host)

    DisplayPlotMixin._on_plotMethod_changed(host)

    assert calls == ["update"]
    assert host.plot_data[0].size == 0
    assert host.plot_data[1].size == 0
    assert host.frame_names == []
    assert host.overlaid_idxs == []


def test_overlay_to_single_collapses_selection_and_refreshes_frame_ids():
    list_data = _FakeListWidget(["1", "2", "3", "4", "5"])
    list_data.selectAll()
    list_data._current_row = 2
    calls = []

    def data_changed():
        viewer.frame_ids[:] = [item.text() for item in list_data.selectedItems()]
        calls.append(tuple(viewer.frame_ids))

    viewer = SimpleNamespace(
        _plot_method="Overlay",
        ui=SimpleNamespace(listData=list_data),
        frame_ids=[],
        data_changed=data_changed,
    )

    H5Viewer.set_data_selection_mode(viewer, "Single")

    assert calls == [("3",)]
    assert viewer.frame_ids == ["3"]
    assert [item.text() for item in list_data.selectedItems()] == ["3"]


def test_xye_axis_label_uses_file_unit_not_hidden_transform_combo():
    host = SimpleNamespace(
        viewer_mode="xye",
        _viewer_x_axis_label=(u"2\u03b8", u"\u00b0"),
        ui=SimpleNamespace(plotUnit=_FakeCombo("Q (\u212b\u207b\u00b9)")),
    )

    assert DisplayPlotMixin._current_plot_axis_label(host) == (
        u"2\u03b8", u"\u00b0",
    )


def test_xye_loader_defaults_unprefixed_files_to_q(monkeypatch, tmp_path):
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
    assert viewer.data_1d[3].int_1d.unit == "q_A^-1"


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


class _FakePlot:
    def __init__(self):
        self.link = None
        self.autorange = 0
        self.xrange = None

    def setXLink(self, link):
        self.link = link

    def enableAutoRange(self, **kwargs):
        self.autorange += 1

    def autoRange(self):
        self.autorange += 1

    def setXRange(self, lo, hi, padding=0):
        self.xrange = (lo, hi)


class _FakeCakeVB:
    """ViewBox stand-in for the share-axis numeric mirror: records signal
    connections and serves a fixed x-range."""
    def __init__(self, xrange=(2.0, 6.0)):
        self._xrange = list(xrange)
        self.connected = []
        self.sigXRangeChanged = SimpleNamespace(
            connect=self.connected.append,
            disconnect=self.connected.remove,
        )

    def viewRange(self):
        return [list(self._xrange), [0.0, 1.0]]


def _share_axis_host(*, gi=False, plot_items=None, image_items=None, image_index=0):
    plot_unit = _FakeMutableCombo()
    for item in plot_items or ("Q (Å⁻¹)", "2θ (°)", "χ (°)"):
        plot_unit.addItem(item)
    plot_unit.setCurrentIndex(plot_unit.count() - 1)
    image_unit = _FakeMutableCombo()
    for item in image_items or ("Q-χ", "2θ-χ"):
        image_unit.addItem(item)
    image_unit.setCurrentIndex(image_index)
    share = _FakeControl(checked=True)
    host = SimpleNamespace(
        scan=SimpleNamespace(
            gi=gi,
            skip_2d=False,
            bai_2d_args={"gi_mode_2d": "qip_qoop"},
        ),
        ui=SimpleNamespace(
            plotUnit=plot_unit,
            imageUnit=image_unit,
            shareAxis=share,
        ),
        plot=_FakePlot(),
        wf_widget=SimpleNamespace(image_plot=_FakePlot(), image_win=SimpleNamespace()),
        plot_data=[np.array([0.0, 1.0]), np.ones((1, 2))],
        plotMethod="Single",
        binned_widget=SimpleNamespace(image_plot=SimpleNamespace(
            getViewBox=lambda _vb=_FakeCakeVB(): _vb)),
        _plot_axis_info=[
            {"source": "1d_2d", "axis": "radial"},
            {"source": "1d_2d", "axis": "radial"},
            {"source": "2d", "axis": "azimuthal"},
        ][:plot_unit.count()],
    )
    for name in (
        "_current_image_axis_key",
        "_plot_axis_key",
        "_share_axis_plot_index",
        "_set_plot_unit_index_silently",
        "_apply_share_axis_state",
        "_set_share_link",
        "_active_bottom_plot",
        "_active_bottom_window",
    ):
        setattr(host, name, MethodType(getattr(displayFrameWidget, name), host))
    host._waterfall_active = MethodType(DisplayPlotMixin._waterfall_active, host)
    return host


def test_share_axis_maps_by_unit_not_combo_index():
    host = _share_axis_host(image_index=1)

    assert host.ui.plotUnit.currentText().startswith("χ")
    assert host._apply_share_axis_state() is True

    assert host.ui.plotUnit.currentIndex() == 1
    assert host.ui.plotUnit.currentText().startswith("2")
    assert host.ui.plotUnit._enabled is False
    assert host.ui.shareAxis.isEnabled() is True
    # Share contract: a pure geometry x-RANGE mirror (cake -> 1D) that freezes the
    # 1D y-axis and scales only the x-extent; the align is deferred (needs real
    # widget geometry) and the native XLink is never engaged (it would force EQUAL
    # ranges = misaligned columns).  On a duck host the deferred align never runs.
    assert host._share_link_on is True
    assert host.plot.link is None


def test_share_axis_disables_when_no_matching_plot_unit():
    host = _share_axis_host(
        gi=True,
        plot_items=("Q (Å⁻¹)",),
        image_items=("Qᵢₚ-Qₒₒₚ",),
        image_index=0,
    )

    assert host._apply_share_axis_state() is False

    assert host.ui.shareAxis.isEnabled() is False
    assert host.ui.shareAxis.isChecked() is False
    assert host.ui.plotUnit._enabled is True
    assert host.plot.link is None


def _align_host(bottom_xrange):
    """Duck host for _align_plot_under_cake with controllable screen spans + the
    bottom plot's current x-range (identity scene->global maps, so the rect edge
    x-values ARE the screen spans)."""
    from types import SimpleNamespace, MethodType
    from xdart.gui.tabs.static_scan.display_frame_widget import displayFrameWidget

    class _Pt:
        def __init__(self, x): self._x = x
        def x(self): return self._x

    class _Rect:
        def __init__(self, lo, hi): self._lo = _Pt(lo); self._hi = _Pt(hi)
        def topLeft(self): return self._lo
        def bottomRight(self): return self._hi

    class _Win:
        def isVisible(self): return True
        def mapFromScene(self, pt): return pt
        def mapToGlobal(self, pt): return pt

    class _VB:
        def __init__(self, rect, xr): self._rect = rect; self._xr = list(xr)
        def sceneBoundingRect(self): return self._rect
        def viewRange(self): return [list(self._xr), [0.0, 1.0]]

    class _Plot:
        def __init__(self, vb): self._vb = vb; self.setx_calls = []
        def getViewBox(self): return self._vb
        def enableAutoRange(self, **kw): pass
        def setXRange(self, lo, hi, padding=0): self.setx_calls.append((lo, hi))

    cake_plot = _Plot(_VB(_Rect(886.0, 1149.0), (0.889, 8.626)))   # cx0,cx1 / cake range
    bottom_plot = _Plot(_VB(_Rect(473.0, 1217.0), bottom_xrange))  # px0,px1 / current range
    host = SimpleNamespace(
        _share_link_on=True,
        binned_widget=SimpleNamespace(image_win=_Win(), image_plot=cake_plot),
    )
    host._active_bottom_window = lambda: _Win()
    host._active_bottom_plot = lambda: bottom_plot
    host._align_plot_under_cake = MethodType(
        displayFrameWidget._align_plot_under_cake, host)
    # the geometry target the align should converge to
    scale = (8.626 - 0.889) / (1149.0 - 886.0)
    target = (0.889 - (886.0 - 473.0) * scale, 0.889 + (1217.0 - 886.0) * scale)
    return host, bottom_plot, target


def test_align_skips_setxrange_when_already_aligned():
    # FREEZE regression: _align must NOT re-setXRange when the bottom plot is
    # already at the geometry target — otherwise setXRange -> relayout -> a
    # re-scheduled align is an unbounded cascade that, with a hundreds-of-frame
    # waterfall repainting each iteration, freezes the GUI.
    _, _, target = _align_host(bottom_xrange=(0.0, 1.0))       # compute the target
    host, bottom_plot, _ = _align_host(bottom_xrange=target)   # bottom already aligned
    host._align_plot_under_cake()
    assert bottom_plot.setx_calls == []                        # no-op -> cascade can't run


def test_align_sets_geometry_xrange_when_not_aligned():
    # The aligned x-range extends into negative Q (y-axis frozen, x-extent scaled).
    host, bottom_plot, target = _align_host(bottom_xrange=(0.889, 8.626))
    host._align_plot_under_cake()
    assert len(bottom_plot.setx_calls) == 1
    lo, hi = bottom_plot.setx_calls[0]
    assert abs(lo - target[0]) < 1e-6 and abs(hi - target[1]) < 1e-6
    assert lo < 0


def test_share_axis_targets_waterfall_plot_when_waterfall_is_active():
    host = _share_axis_host()
    host.plotMethod = "Waterfall"
    host.plot_data = [np.array([0.0, 1.0]), np.ones((4, 2))]

    assert host._waterfall_active() is True
    assert host._active_bottom_plot() is host.wf_widget.image_plot
    assert host._active_bottom_window() is host.wf_widget.image_win

    host._share_link_on = True
    host.plot.link = "old"
    host.wf_widget.image_plot.link = "old"
    displayFrameWidget._set_share_link(host, False)
    assert host.plot.link is None
    assert host.wf_widget.image_plot.link is None


def test_unshare_rescales_active_waterfall_not_just_the_1d_line():
    # Vivek-reported: un-checking Share Axis didn't scale the WATERFALL back to
    # its own extent — _on_share_axis_toggled only refit self.plot (the hidden
    # 1D line), leaving the waterfall frozen at the cake range that _align had
    # pinned (x-auto off).  Un-share must refit the ACTIVE bottom plot.
    host = _share_axis_host()
    host.plotMethod = "Waterfall"
    host.plot_data = [np.array([0.0, 1.0]), np.ones((4, 2))]
    host._set_slice_range = lambda *a, **k: None
    host._on_share_axis_toggled = MethodType(
        displayFrameWidget._on_share_axis_toggled, host)

    assert host._active_bottom_plot() is host.wf_widget.image_plot
    host._on_share_axis_toggled(False)

    # both autoRange() and enableAutoRange() fire on the waterfall (each bumps
    # the counter) AND on the 1D line, so neither stays stuck.
    assert host.wf_widget.image_plot.autorange >= 2
    assert host.plot.autorange >= 2


def test_unshare_rescales_1d_line_when_no_waterfall():
    # Non-waterfall: the active bottom plot IS self.plot, so it is refit once
    # (no double-autoRange from the de-dupe).
    host = _share_axis_host()
    host.plotMethod = "Single"
    host._set_slice_range = lambda *a, **k: None
    host._on_share_axis_toggled = MethodType(
        displayFrameWidget._on_share_axis_toggled, host)

    assert host._active_bottom_plot() is host.plot
    host._on_share_axis_toggled(False)
    assert host.plot.autorange == 2          # autoRange + enableAutoRange, once


def test_processed_image_viewer_falls_back_to_thumbnail(tmp_path):
    # Stage 5: a processed .nxs whose source master is missing loads the
    # dequantized thumbnail (via the ssrl boundary), stored as a thumbnail
    # (its mask is already baked in) — _load_single_frame integration.
    import h5py

    path = tmp_path / "thumb_fallback.nxs"
    with h5py.File(path, "w") as f:
        e = f.create_group("entry")
        e.create_group("integrated_1d")
        s = e.create_group("frames/frame_0005/source")
        s.create_dataset("path", data=np.bytes_(b"missing_master.h5"))
        s.create_dataset("frame_index", data=0)
        ds = e.create_dataset(
            "frames/frame_0005/thumbnail",
            data=(np.ones((3, 4)) * 128).astype(np.uint8),
        )
        ds.attrs["vmin"] = 10.0
        ds.attrs["vmax"] = 20.0
        ds.attrs["dtype"] = "uint8"

    viewer = SimpleNamespace(
        _viewer_is_xdart=True, data_lock=RLock(), data_1d={}, data_2d={},
    )
    loaded = H5Viewer._load_single_frame(
        viewer, str(path), frame_idx=5, frame_id=5,
    )

    assert loaded is True
    assert viewer.data_2d[5]["map_raw"].shape == (3, 4)
    expected = 10.0 + (128 / 255) * 10.0   # dequantize(128, vmin=10, vmax=20)
    np.testing.assert_allclose(viewer.data_2d[5]["map_raw"], expected)
    # Stored as a thumbnail so the renderer won't re-apply a flat mask.
    np.testing.assert_allclose(viewer.data_2d[5]["thumbnail"], expected)


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


def test_scan_snapshot_ignores_legacy_mirror_when_store_is_active():
    from xdart.modules.frame_publication import PublicationStore

    stale = SimpleNamespace(scan_info={}, int_1d=object())
    host = SimpleNamespace(
        viewer_mode=None,
        publication_store=PublicationStore(),
        data_lock=RLock(),
        data_1d={1: stale},
        data_2d={1: {"int_2d": object()}},
    )
    host._snapshot_data = MethodType(DisplayDataMixin._snapshot_data, host)

    assert host._snapshot_data([1]) == {}


def test_viewer_snapshot_keeps_legacy_file_rows():
    stale = SimpleNamespace(scan_info={}, int_1d=object())
    two_d = {"map_raw": np.ones((2, 2))}
    host = SimpleNamespace(
        viewer_mode="image",
        publication_store=None,
        data_lock=RLock(),
        data_1d={1: stale},
        data_2d={1: two_d},
    )
    host._snapshot_data = MethodType(DisplayDataMixin._snapshot_data, host)

    assert host._snapshot_data([1]) == {1: (stale, two_d)}


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
    host._hydrate_frame_from_disk = MethodType(
        DisplayDataMixin._hydrate_frame_from_disk, host)

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
        _display_hydration_should_block=lambda allow_blocking_read=None: True,
    )
    host._snapshot_data = MethodType(DisplayDataMixin._snapshot_data, host)
    host._hydrate_frame_from_disk = MethodType(
        DisplayDataMixin._hydrate_frame_from_disk, host)

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
    # The thumbnail path must NEVER apply full-resolution flat detector indices
    # directly to the smaller thumbnail (they point at unrelated pixels).  When
    # the full-res shape isn't known the gap mask cannot be mapped into thumbnail
    # coordinates, so it is skipped — leaving the preview untouched rather than
    # corrupting it.  (The mapped, masked case is covered in test_aggregation_
    # wiring::test_nan_thumbnail_gaps_masks_downsampled_gap_rows.)
    calls = []
    host = SimpleNamespace(
        overall=False,
        frame_ids=[1],
        idxs_2d=[1],
        data_lock=RLock(),
        data_2d={1: {"mask": np.array([0], dtype=int)}},
        scan=SimpleNamespace(global_mask=np.array([1], dtype=int)),
        bkg_map_raw=0,
        # no _raw_full_shape -> the gap mask can't be mapped -> skipped
        get_frames_map_raw=lambda **kwargs: (np.ones((2, 2)), "thumbnail"),
        update_image_view=lambda: calls.append("updated"),
    )
    host._nan_thumbnail_gaps = MethodType(
        displayFrameWidget._nan_thumbnail_gaps, host)

    displayFrameWidget.update_image(host)

    assert calls == ["updated"]
    assert np.isfinite(host.image_data[0]).all()


def test_full_res_raw_masks_in_range_indices_despite_out_of_range():
    # Regression (review P3): the legacy full-res raw branch bounds each flat
    # index (combine_flat_masks size=data.size) instead of the old all-or-nothing
    # max()<size guard -- so an out-of-range index can't suppress masking the
    # in-range gaps (identical to the payload path).
    calls = []
    host = SimpleNamespace(
        overall=False,
        frame_ids=[1],
        idxs_2d=[1],
        data_lock=RLock(),
        data_2d={1: {"mask": np.array([0], dtype=int)}},        # in-range
        scan=SimpleNamespace(global_mask=np.array([3, 999], dtype=int)),  # 3 in, 999 OOB
        bkg_map_raw=0,
        get_frames_map_raw=lambda **kwargs: (np.ones((2, 2)), "raw"),
        update_image_view=lambda: calls.append("updated"),
    )

    displayFrameWidget.update_image(host)

    assert calls == ["updated"]
    # flat 0 + 3 are in-range -> 2 NaN; 999 dropped (not a crash, not skip-all).
    assert np.isnan(host.image_data[0]).sum() == 2


class _AttrDict(dict):
    """data_2d stand-in that, like FixSizeOrderedDict, can carry the shared
    hydrated-raw order attribute (plain dicts cannot)."""


def _hydrated_order(data_2d):
    return list(getattr(data_2d, "_hydrated_raw_order", []))


def test_hydrated_raw_cache_evicts_only_full_resolution_payload():
    viewer = SimpleNamespace(
        data_lock=RLock(),
        _raw_cache_limit=2,
        data_2d=_AttrDict({
            idx: {"map_raw": np.full((4, 4), idx), "thumbnail": np.ones((2, 2))}
            for idx in (1, 2, 3)
        }),
    )
    viewer._remember_hydrated_raw = MethodType(H5Viewer._remember_hydrated_raw, viewer)
    for idx in (1, 2, 3):
        viewer._remember_hydrated_raw(idx)
    assert _hydrated_order(viewer.data_2d) == [2, 3]
    assert viewer.data_2d[1]["map_raw"] is None
    assert viewer.data_2d[1]["thumbnail"] is not None


def test_image_viewer_raw_cache_evicts_reloadable_unselected_rows():
    viewer = SimpleNamespace(
        data_lock=RLock(),
        _raw_cache_limit=2,
        viewer_mode="image",
        frame_ids=[],
        data_1d={},
        data_2d=_AttrDict(),
    )
    viewer._remember_hydrated_raw = MethodType(H5Viewer._remember_hydrated_raw, viewer)

    for idx in (1, 2, 3, 4):
        viewer.frame_ids = [str(idx)]
        viewer.data_1d[idx] = SimpleNamespace(map_raw=np.full((2, 2), idx))
        viewer.data_2d[idx] = {"map_raw": np.full((2, 2), idx)}
        viewer._remember_hydrated_raw(idx)

    assert sorted(viewer.data_1d) == [3, 4]
    assert sorted(viewer.data_2d) == [3, 4]
    assert _hydrated_order(viewer.data_2d) == [3, 4]

    # If the user has an older row selected while a newer one hydrates, keep
    # the selected image resident and evict the oldest unselected row instead.
    viewer.frame_ids = ["3"]
    viewer.data_1d[5] = SimpleNamespace(map_raw=np.full((2, 2), 5))
    viewer.data_2d[5] = {"map_raw": np.full((2, 2), 5)}
    viewer._remember_hydrated_raw(5)
    assert sorted(viewer.data_1d) == [3, 5]
    assert sorted(viewer.data_2d) == [3, 5]
    assert _hydrated_order(viewer.data_2d) == [3, 5]


def test_hydrated_raw_cache_reset_clears_order():
    data_2d = _AttrDict()
    data_2d._hydrated_raw_order = [1, 2, 3]
    viewer = SimpleNamespace(_raw_cache_order=[1, 2, 3], data_2d=data_2d)
    viewer._clear_raw_cache = MethodType(H5Viewer._clear_raw_cache, viewer)
    viewer._clear_raw_cache()
    assert viewer._raw_cache_order == []
    assert _hydrated_order(data_2d) == []


def test_hydrated_raw_lru_is_shared_with_thread_insert_paths():
    """D5: the reintegrate-publish and full-reload worker paths trim the
    SAME LRU as the GUI — order state rides on the shared data_2d object."""
    from xdart.gui.tabs.static_scan.hydrated_raw import remember_hydrated_raw

    data_2d = _AttrDict()
    # GUI hydrates 1..2, then a worker path hydrates 3..4 (limit 3): the
    # worker's inserts must evict the GUI's oldest, not pile up past the cap.
    for idx in (1, 2):
        data_2d[idx] = {"map_raw": np.full((2, 2), idx)}
        remember_hydrated_raw(data_2d, idx, limit=3)
    for idx in (3, 4):
        data_2d[idx] = {"map_raw": np.full((2, 2), idx)}
        remember_hydrated_raw(data_2d, idx, limit=3)
    assert _hydrated_order(data_2d) == [2, 3, 4]
    assert data_2d[1]["map_raw"] is None
    hydrated = [i for i in (1, 2, 3, 4) if data_2d[i]["map_raw"] is not None]
    assert hydrated == [2, 3, 4]


def test_reintegrate_publish_updates_publication_store_not_legacy_raw_mirror():
    """Reintegration publishes through the store, not the old data_2d mirror."""
    from xdart.gui.tabs.static_scan.scan_threads import integratorThread

    data_2d = _AttrDict({
        idx: {"map_raw": np.full((2, 2), idx)} for idx in range(8)
    })
    data_2d._hydrated_raw_order = list(range(8))
    upserts = []
    updates = []
    host = SimpleNamespace(
        data_lock=RLock(),
        data_1d={},
        data_2d=data_2d,
        _upsert_publication_for_frame=lambda frame: upserts.append(frame.idx),
        update=SimpleNamespace(emit=lambda idx: updates.append(idx)),
    )
    frame = SimpleNamespace(
        idx=8, map_raw=np.full((2, 2), 8), bg_raw=None, mask=None,
        int_2d=None, gi_2d={},
        copy_for_display=lambda include_2d: {"int_1d": None},
    )
    integratorThread._publish_reintegrated_display(
        host, frame, include_2d=True,
    )
    assert upserts == [8]
    assert updates == [8]
    assert 8 not in data_2d
    assert data_2d[0]["map_raw"] is not None
    assert len(_hydrated_order(data_2d)) == 8


def _wrangler_host(mode_text, *, live=False, batch=False):
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    ui = SimpleNamespace(
        processingModeCombo=_FakeCombo(mode_text),
        liveCheckBox=_FakeControl(live),
        batchCheckBox=_FakeControl(batch),
        coresLabel=_FakeControl(),
        maxCoresSpinBox=_FakeControl(),
        advancedButton=_FakeControl(),
        startButton=_FakeControl(),
        stopButton=_FakeControl(),
        frame=_FakeControl(),
        specLabel=SimpleNamespace(setText=lambda *a, **k: None),
    )
    integration_calls = []
    host = SimpleNamespace(
        ui=ui,
        tree=_FakeControl(),
        # A loaded PONI by default so the start() input-gate (BUG-1) passes;
        # tests that exercise the gate set ``host.poni = None`` explicitly.
        poni=object(),
        live_mode=live,
        batch_mode=batch,
        xye_only=False,
        scan=SimpleNamespace(skip_2d=None),
        thread=SimpleNamespace(batch_mode=None, xye_only=None, live_mode=None,
                               command=None, command_lock=RLock()),
        sigViewerModeChanged=_FakeSignal(),
        sigStart=_FakeSignal(),
        sender=lambda: None,
        # _on_mode_changed now refreshes wrangler disclosure; this lightweight
        # host has no N1 'Project' param group, so _apply_disclosure no-ops.
        parameters=SimpleNamespace(names=[]),
        _integration_calls=integration_calls,
        _set_integration_controls_enabled=lambda enabled, **kwargs: (
            integration_calls.append((enabled, kwargs)),
            setattr(host, "_integration_controls_enabled", enabled),
        ),
    )
    host._on_mode_changed = MethodType(imageWrangler._on_mode_changed, host)
    host._apply_disclosure = MethodType(imageWrangler._apply_disclosure, host)
    host.start = MethodType(imageWrangler.start, host)
    host._inputs_valid = MethodType(imageWrangler._inputs_valid, host)
    # Phase B: the action-button morph helper + its state, used by enabled()/
    # start()/_on_start_clicked.
    host._run_phase = 'idle'
    host._set_action_button = MethodType(imageWrangler._set_action_button, host)
    host._on_start_clicked = MethodType(imageWrangler._on_start_clicked, host)
    host.pause = MethodType(imageWrangler.pause, host)
    host._on_paused = MethodType(imageWrangler._on_paused, host)
    host._on_resume = MethodType(imageWrangler._on_resume, host)
    return host


def test_wrangler_enabled_reapplies_viewer_mode_controls():
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    for mode in ("Image Viewer", "XYE Viewer", "NeXus Viewer"):
        host = _wrangler_host(mode, live=True, batch=True)

        imageWrangler.enabled(host, True)

        assert host.ui.liveCheckBox.isChecked() is False
        assert host.ui.liveCheckBox.isEnabled() is False
        assert host.ui.batchCheckBox.isEnabled() is False
        assert host.ui.frame.isVisible() is False
        assert host._integration_controls_enabled is False
        assert host.thread.live_mode is False
        assert host.tree.isEnabled() is True   # Project/Save Path stay usable


def test_file_viewer_mode_disables_processing_tree_but_not_mode_combo():
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    for mode in ("Image Viewer", "XYE Viewer"):
        host = _wrangler_host(mode, live=False, batch=False)

        imageWrangler._on_mode_changed(host)

        # Tree stays enabled (Project Folder / Save Path usable); the
        # processing groups are disabled per-group instead.
        assert host.tree.isEnabled() is True
        assert host._integration_controls_enabled is False
        assert host.ui.processingModeCombo.currentText() == mode


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


def test_wrangler_enabled_run_end_reenables_mode_toggles():
    """Phase B: at run END, both mode toggles re-enable and the action button
    morphs back to green 'Start' (idle)."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    host = _wrangler_host("Int 2D", live=True, batch=True)

    imageWrangler.enabled(host, True)

    assert host.ui.liveCheckBox.isChecked() is False
    assert host.ui.liveCheckBox.isEnabled() is True
    assert host.ui.batchCheckBox.isEnabled() is True
    assert host.ui.frame.isVisible() is True
    assert host._integration_controls_enabled is True
    # Action button reset to green Start (idle).
    assert host.ui.startButton.text() == "Start"
    assert host.ui.startButton.property("runPhase") == "idle"
    assert host._run_phase == "idle"


def test_start_click_honors_live_toggle_and_morphs_to_pause():
    """Phase B (DECIDED model): Start is the single action button and HONORS the
    Live MODE toggle (no longer force-unchecks it).  It morphs green Start ->
    orange Pause and sets _run_phase='running'."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    host = _wrangler_host("Int 2D", live=True, batch=False)

    imageWrangler._on_start_clicked(host)

    # Live is honored, NOT force-unchecked.
    assert host.ui.liveCheckBox.isChecked() is True
    assert host.command == "start"
    assert host.thread.command == "start"
    assert host.ui.stopButton.isEnabled() is True
    assert host.sigStart.emitted == [()]
    # The action button morphed to orange 'Pause'.
    assert host.ui.startButton.text() == "Pause"
    assert host.ui.startButton.property("runPhase") == "active"
    assert host._run_phase == "running"


def test_start_pause_resume_button_state_machine():
    """Phase B: clicking the single action button cycles Start -> Pause ->
    Resume -> Pause via _on_start_clicked, mirroring 'pause'/'start' commands."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    host = _wrangler_host("Int 2D", live=False, batch=True)
    host.sigResuming = _FakeSignal()

    imageWrangler._on_start_clicked(host)      # idle -> start (running, 'Pause')
    assert host._run_phase == "running" and host.command == "start"

    imageWrangler._on_start_clicked(host)      # running -> pause
    assert host.command == "pause" and host.thread.command == "pause"
    assert host.ui.startButton.text() == "Pausing…"   # transient until sigPaused

    imageWrangler._on_paused(host)             # worker confirms paused
    assert host._run_phase == "paused" and host.ui.startButton.text() == "Resume"

    imageWrangler._on_start_clicked(host)      # paused -> resume
    assert host.command == "start" and host.thread.command == "start"
    assert host._run_phase == "running" and host.ui.startButton.text() == "Pause"
    assert host.sigResuming.emitted == [()]    # guard re-engaged before resume


def test_stop_during_pausing_ignores_late_sigpaused():
    """Adversarial-review fix: Stop during the transient 'Pausing…' window must
    win.  A late sigPaused (queued from the worker) arriving after stop() must
    NOT flash the button back to orange 'Resume' -- _on_paused drops it because
    self.command is already 'stop'."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    host = _wrangler_host("Int 2D", live=False, batch=True)

    imageWrangler._on_start_clicked(host)      # running
    imageWrangler._on_start_clicked(host)      # -> pause ('Pausing…')
    imageWrangler.stop(host)                   # Stop lands during 'Pausing…'
    assert host.ui.startButton.text() == "Start"     # morphed back to green
    assert host._run_phase == "idle"

    imageWrangler._on_paused(host)             # late queued sigPaused arrives
    assert host.ui.startButton.text() == "Start"     # NOT flashed to 'Resume'
    assert host._run_phase == "idle"


def test_redispatch_during_pausing_is_noop():
    """Adversarial-review fix: _run_phase keeps 'pausing' distinct, so a stray
    re-dispatch during the disabled 'Pausing…' window does NOT fire a second
    pause()/start()."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    host = _wrangler_host("Int 2D", live=False, batch=True)
    imageWrangler._on_start_clicked(host)      # running
    imageWrangler._on_start_clicked(host)      # -> pause ('Pausing…')
    assert host._run_phase == "pausing"
    host.thread.command = "sentinel"           # detect any spurious command write
    imageWrangler._on_start_clicked(host)      # re-dispatch while 'pausing'
    assert host.thread.command == "sentinel"   # no-op: no pause()/start() fired
    assert host.ui.startButton.text() == "Pausing…"


def test_start_without_poni_is_gated():
    """BUG-1: Start/Live must refuse to run without a loaded PONI rather than
    re-running the previous scan with the stale calibration."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    host = _wrangler_host("Int 2D", live=False, batch=False)
    host.poni = None  # no calibration loaded

    imageWrangler.start(host)

    assert host.sigStart.emitted == []          # did not start
    assert getattr(host, "command", None) != "start"


def test_active_run_locks_modes_but_keeps_action_button_enabled():
    """Phase B: during a run the Live/Batch MODE toggles both lock (they no
    longer start/stop anything), the parameter tree + non-param widgets lock,
    but the single action button STAYS ENABLED (it is the Pause/Resume control
    now), and Stop stays enabled."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    host = _wrangler_host("Int 2D", live=True, batch=False)
    host.live_mode = True

    imageWrangler.enabled(host, False)

    assert host.ui.startButton.isEnabled() is True   # morphs to Pause/Resume
    assert host.ui.liveCheckBox.isEnabled() is False
    assert host.ui.batchCheckBox.isEnabled() is False
    assert host.tree.isEnabled() is False
    assert host.ui.processingModeCombo.isEnabled() is False
    assert host.ui.maxCoresSpinBox.isEnabled() is False
    assert host.ui.advancedButton.isEnabled() is False
    assert host.ui.stopButton.isEnabled() is True    # Stop never disabled


def test_active_non_live_run_locks_modes_keeps_action_button():
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    host = _wrangler_host("Int 2D", live=False, batch=True)
    host.live_mode = False

    imageWrangler.enabled(host, False)

    assert host.ui.startButton.isEnabled() is True   # Pause/Resume control
    assert host.ui.liveCheckBox.isEnabled() is False
    assert host.ui.batchCheckBox.isEnabled() is False
    assert host.tree.isEnabled() is False


def test_active_run_disables_parameter_tree():
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    host = _wrangler_host("Int 2D", live=False, batch=False)

    imageWrangler.enabled(host, False)

    assert host.tree.isEnabled() is False


def test_wrangler_enabled_restore_unlocks_tree_and_nonparam_widgets():
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    host = _wrangler_host("Int 2D", live=True, batch=True)
    imageWrangler.enabled(host, False)
    assert host.tree.isEnabled() is False
    assert host.ui.processingModeCombo.isEnabled() is False

    imageWrangler.enabled(host, True)
    assert host.tree.isEnabled() is True
    assert host.ui.processingModeCombo.isEnabled() is True
    assert host.ui.advancedButton.isEnabled() is True
    # batch=True host → _on_mode_changed re-enables the Cores widgets.
    assert host.ui.maxCoresSpinBox.isEnabled() is True
    assert host.ui.coresLabel.isEnabled() is True


def test_nexus_wrangler_enabled_locks_whole_panel_keeps_stop():
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler import nexusWrangler

    host = SimpleNamespace(
        startButton=_FakeControl(),
        stopButton=_FakeControl(),          # set in start/stop; untouched by enabled()
        processingModeCombo=_FakeControl(),
        maxCoresSpinBox=_FakeControl(),
        coresLabel=_FakeControl(),
        tree=_FakeControl(),
    )
    nexusWrangler.enabled(host, False)                    # run active
    assert host.tree.isEnabled() is False
    assert host.startButton.isEnabled() is False
    assert host.processingModeCombo.isEnabled() is False
    assert host.maxCoresSpinBox.isEnabled() is False
    assert host.stopButton.isEnabled() is True           # Stop untouched

    nexusWrangler.enabled(host, True)                     # run finished
    assert host.tree.isEnabled() is True
    assert host.startButton.isEnabled() is True
    assert host.processingModeCombo.isEnabled() is True


def _iter_params(param):
    yield param
    try:
        children = param.children()
    except AttributeError:
        children = ()
    for c in children:
        yield from _iter_params(c)


def _build_real_wrangler(cls):
    import threading
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    scan = SimpleNamespace(
        skip_2d=None, bai_1d_args={}, bai_2d_args={},
        scan_lock=threading.RLock(), gi=False, poni_file='', name='null_main')
    return cls("test", threading.Condition(), scan, {}, {})


def test_real_wrangler_run_lock_disables_tree_and_keeps_values():
    """End-to-end on REAL wrangler instances (not injected hosts): enabled(False)
    disables the processing tree and non-param widgets while keeping Stop
    enabled.  Parameter values must survive the disabled run state."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler import nexusWrangler

    for cls in (imageWrangler, nexusWrangler):
        w = _build_real_wrangler(cls)
        ui = w.ui if hasattr(w, 'ui') else w       # image uses .ui; nexus direct attrs
        ui.stopButton.setEnabled(True)             # mimic the start flow

        bool_before = [(p, p.value()) for p in _iter_params(w.parameters)
                       if p.opts.get('type') == 'bool']
        assert bool_before, f"{cls.__name__}: expected a bool param (e.g. Grazing)"

        # ── run active ─────────────────────────────────────────────────────────
        w.enabled(False)
        assert w.tree.isEnabled() is False
        assert ui.stopButton.isEnabled() is True   # Stop never disabled
        assert ui.processingModeCombo.isEnabled() is False
        assert ui.maxCoresSpinBox.isEnabled() is False
        # Param *values* survive the lock.
        for p, val in bool_before:
            assert p.value() == val, f"{cls.__name__}: bool {p.name()} value changed by lock"

        # ── run finished ───────────────────────────────────────────────────────
        w.enabled(True)
        assert w.tree.isEnabled() is True          # tree re-enabled
        assert ui.processingModeCombo.isEnabled() is True
        for p, val in bool_before:
            assert p.value() == val, f"{cls.__name__}: bool {p.name()} value changed after restore"


def test_integration_view_image_applies_mask_and_threshold():
    """The Int-1D 'Show Image' preview must show the image AS INTEGRATED:
    detector/global mask (flat indices) and the run's intensity threshold
    rendered as NaN, like the worker's _apply_threshold_inline.  (The Image
    Viewer mode deliberately bypasses this helper.)"""
    import numpy as np
    from xdart.gui.tabs.static_scan.display_frame_widget import displayFrameWidget

    thumb = np.arange(16, dtype=np.float32).reshape(4, 4)
    scan = SimpleNamespace(
        global_mask=np.array([0, 5]),          # flat indices
        apply_threshold=True, threshold_min=2.0, threshold_max=12.0,
    )
    img = displayFrameWidget.integration_view_image(thumb, scan)

    assert np.isnan(img.ravel()[0]) and np.isnan(img.ravel()[5])   # mask
    assert np.isnan(img.ravel()[1])                                # < min
    assert np.isnan(img.ravel()[15])                               # > max
    assert img.ravel()[7] == 7.0                                   # in-band kept
    assert thumb.ravel()[0] == 0.0                                 # input untouched

    # Threshold off, no mask: pass-through (Image Viewer semantics).
    img2 = displayFrameWidget.integration_view_image(
        thumb, SimpleNamespace(global_mask=None, apply_threshold=False))
    assert np.isfinite(img2).all()


def test_xye_viewer_single_click_navigates_directories():
    """XYE viewer: single click on a directory (or '..') navigates, matching
    the Image Viewer; single click on a FILE still defers to
    _scans_selection_changed (multi-select path) and must not double-fire."""
    calls = []
    viewer = SimpleNamespace(
        _suspend_scan_selection_loads=False,
        viewer_mode="xye",
        scans_clicked=lambda q: calls.append(("nav", q.text())),
    )

    H5Viewer._scans_single_clicked(viewer, _FakeItem("subdir/"))
    H5Viewer._scans_single_clicked(viewer, _FakeItem(".."))
    H5Viewer._scans_single_clicked(viewer, _FakeItem("data_0001.xye"))

    assert calls == [("nav", "subdir/"), ("nav", "..")]

    # Image viewer unchanged: single click acts on everything.
    viewer.viewer_mode = "image"
    H5Viewer._scans_single_clicked(viewer, _FakeItem("img.tiff"))
    assert calls[-1] == ("nav", "img.tiff")


def test_batch_process_scan_dispatches_each_frame_as_read():
    """Batch should feed the persistent streaming session per frame, not wait
    for the old 64/256-frame pending buffer.  The sink still batches writes and
    batch mode still refreshes the GUI only once at the end; this locks the
    read||reduce overlap fix at the collection-loop boundary."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread

    queue = [
        ("/tmp/scan_0001.tif", "scan", 1, np.ones((2, 2)), {"i0": 1.0}),
        ("/tmp/scan_0002.tif", "scan", 2, np.ones((2, 2)), {"i0": 2.0}),
        ("/tmp/scan_0003.tif", "scan", 3, np.ones((2, 2)), {"i0": 3.0}),
    ]
    dispatched = []
    final_updates = []

    def next_image():
        if queue:
            return queue.pop(0)
        return None, None, None, None, None

    def make_scan():
        return SimpleNamespace(
            name="scan",
            frames=SimpleNamespace(index=[]),
            skip_2d=False,
            data_file="/tmp/out.nxs",
        )

    host = SimpleNamespace(
        command="start",
        batch_mode=True,
        live_mode=False,
        single_img=False,
        xye_only=False,
        img_file="/tmp/scan_0001.tif",
        poni=None,
        scan_name="scan",
        _frames_since_save=0,
        _active_scan=None,
        _perf=None,
        _live_execution=lambda: "serial",
        showLabel=SimpleNamespace(emit=lambda *_: None),
        sigUpdate=SimpleNamespace(emit=lambda value: final_updates.append(value)),
        _wait_if_paused=lambda: None,
        get_next_image=next_image,
        _middle_truncate=lambda text, max_len=40: text,
        initialize_scan=make_scan,
        get_background=lambda *_: 0.0,
        _flush_xye_buffer=lambda *_args, **_kw: None,
    )

    def dispatch(scan, pending, *, force_save=False):
        dispatched.append((tuple(item[1] for item in pending), bool(force_save)))
        return len(pending)

    host._dispatch_batch = dispatch

    MethodType(imageThread.process_scan, host)()

    assert dispatched == [((1,), False), ((2,), False), ((3,), False)]
    assert final_updates == [-1]


def test_batch_single_frame_still_routes_to_streaming_when_live_policy_serial():
    """N2 submits one-frame pending chunks.  Batch must still use the streaming
    dispatcher even if the live-mode fallback env is serial; otherwise the new
    cadence would silently revert batch to the old serial path."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread

    calls = []
    host = SimpleNamespace(
        batch_mode=True,
        _maybe_warn_live_gi_clip=lambda: None,
        _dispatch_batch_streaming=lambda scan, pending: calls.append(
            ("streaming", tuple(item[1] for item in pending))) or len(pending),
        _dispatch_batch_serial=lambda scan, pending, force_save=False: calls.append(
            ("serial", tuple(item[1] for item in pending))) or len(pending),
        _live_execution=lambda: "serial",
    )
    pending = [("/tmp/scan_0001.tif", 1, np.ones((2, 2)), {}, 0.0, 0.0)]

    assert MethodType(imageThread._dispatch_batch, host)(object(), pending) == 1
    assert calls == [("streaming", (1,))]
