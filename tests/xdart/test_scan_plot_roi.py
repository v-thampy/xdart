# -*- coding: utf-8 -*-
"""Scan Plot ROI path: reachable-raw gating, the RectROI<->fields round-trip in
the ROI picker, and the end-to-end worker fill (ROI series become table columns
equal to a direct headless run_roi_signals — the mini spine).  Offscreen GUI; an
occasional pyqtgraph teardown SIGSEGV is a known flake — just rerun."""
import time

import numpy as np
import pytest


@pytest.fixture(scope="module")
def qapp():
    from pyqtgraph.Qt import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _stack():
    """3 frames: a bright 2x2 block at rows/cols 1:3 whose mean rises 10,20,30."""
    out = []
    for f in range(3):
        im = np.zeros((6, 6), dtype=float)
        im[1:3, 1:3] = (f + 1) * 10.0
        out.append(im)
    return out


def _pump(qapp, predicate, timeout=8.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    qapp.processEvents()
    return predicate()


# ── reachable-raw gating truth-table ──────────────────────────────────────


def test_raw_is_reachable_truth_table(qapp):
    from xdart.gui.tabs.static_scan.scan_plot_dialog import raw_is_reachable
    from xrd_tools.sources import MemoryFrameSource

    # Eiger/TIFF-like: the images ARE the source -> reachable.
    assert raw_is_reachable(MemoryFrameSource(_stack())) is True

    # SPEC / metadata-only: no frame source at all.
    assert raw_is_reachable(None) is False

    # empty scan -> nothing to read.
    assert raw_is_reachable(MemoryFrameSource([])) is False

    # processed NeXus whose linked raw is missing: strict load_frame raises.
    class _NoRaw(MemoryFrameSource):
        def load_frame(self, index):
            raise KeyError("raw master unresolved")

    assert raw_is_reachable(_NoRaw(_stack())) is False

    # a 1-D "frame" is not a usable raw image.
    class _OneD(MemoryFrameSource):
        def load_frame(self, index):
            return np.arange(5.0)

    assert raw_is_reachable(_OneD(_stack())) is False


_SPEC = """#F myscan
#E 1
#D today
#O0 th  chi

#S 5 ascan th 0 2 2 1
#D today
#P0 0 5
#N 3
#L th  i0  det
0 100 10
1 110 20
2 120 30

#S 6 ascan chi 0 1 1 1
#D today
#P0 7 0
#N 2
#L chi  i0
0 300
1 310
"""


def test_scan_plot_loads_spec_metadata_roi_disabled(qapp, tmp_path):
    """An extensionless SPEC file populates the table from its columns; the scan
    selector appears; Plot ROI stays disabled (metadata only — no raw)."""
    pytest.importorskip("silx")
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog

    p = tmp_path / "myscan"            # extensionless, the SSRL convention
    p.write_text(_SPEC)
    dlg = ScanPlotDialog()
    try:
        dlg.load_uri(str(p))                           # routes through the source widget
        assert {"th", "i0", "det", "chi"} <= set(dlg._columns)   # scan 5 columns
        assert dlg.roi_btn.isEnabled() is False        # no raw -> ROI off
        # the shared widget's scan selector lists both scans; switching reloads
        scan_combo = dlg.source_widget.scan_combo
        assert scan_combo.count() == 2
        scan_combo.setCurrentIndex(1)                  # -> scan 6
        assert "chi" in dlg._columns and "det" not in dlg._columns
    finally:
        dlg.close()


def test_roi_button_reflects_reachability(qapp):
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog
    from xrd_tools.sources import MemoryFrameSource

    dlg = ScanPlotDialog()
    try:
        assert dlg.roi_btn.isEnabled() is False         # blank -> disabled
        dlg._source = MemoryFrameSource(_stack())
        dlg._raw_reachable = True
        dlg._update_roi_button()
        assert dlg.roi_btn.isEnabled() is True
        dlg._source = None
        dlg._raw_reachable = False
        dlg._update_roi_button()
        assert dlg.roi_btn.isEnabled() is False
    finally:
        dlg.close()


# ── ROI picker: RectROI <-> fields two-way sync ───────────────────────────


def test_roi_select_field_round_trip(qapp):
    from xdart.gui.tabs.static_scan.roi_select_dialog import (
        RoiSelectDialog, _rect_center_size)

    img = np.zeros((100, 80), dtype=float)       # (rows, cols)
    dlg = RoiSelectDialog(img)
    try:
        assert len(dlg._rois) == 1               # starts with one ROI
        entry = dlg._rois[0]

        # fields -> rect: type a box, commit, and the RectROI + RoiSpec follow.
        dlg.f_crow.setText("40")
        dlg.f_ccol.setText("30")
        dlg.f_wrow.setText("10")
        dlg.f_wcol.setText("6")
        dlg._fields_to_rect()
        crow, ccol, wrow, wcol = _rect_center_size(entry.rect)
        assert (crow, ccol, wrow, wcol) == pytest.approx((40, 30, 10, 6), abs=0.5)

        sig = dlg.roi_signals()[0]
        assert sig.roi.center_y == pytest.approx(40, abs=0.5)   # row
        assert sig.roi.center_x == pytest.approx(30, abs=0.5)   # col
        assert sig.roi.width_y == pytest.approx(10, abs=0.5)
        assert sig.roi.width_x == pytest.approx(6, abs=0.5)
        assert sig.reducer == "mean"
        assert sig.background is None

        # rect -> fields: move the rectangle, the fields mirror it.
        entry.rect.setSize([8, 20], update=False)
        entry.rect.setPos([10, 5])               # x0=col, y0=row
        dlg._on_rect_changed(entry, bg=False)
        assert float(dlg.f_ccol.text()) == pytest.approx(10 + 8 / 2, abs=0.5)
        assert float(dlg.f_crow.text()) == pytest.approx(5 + 20 / 2, abs=0.5)
        assert float(dlg.f_wcol.text()) == pytest.approx(8, abs=0.5)
        assert float(dlg.f_wrow.text()) == pytest.approx(20, abs=0.5)
    finally:
        dlg.close()


def test_roi_select_picker_defaults_and_gating(qapp):
    """First ROI defaults to the whole frame; the saturation toggle is exposed;
    the background controls gate to mean/sum."""
    from xdart.gui.tabs.static_scan.roi_select_dialog import (
        RoiSelectDialog, _rect_center_size)

    dlg = RoiSelectDialog(np.zeros((40, 60), dtype=float))   # rows=40, cols=60
    try:
        _crow, _ccol, wrow, wcol = _rect_center_size(dlg._rois[0].rect)
        assert (wrow, wcol) == pytest.approx((40, 60), abs=0.5)   # full frame

        assert dlg.mask_saturated() is False
        dlg.mask_sat_check.setChecked(True)
        assert dlg.mask_saturated() is True

        dlg.reducer_combo.setCurrentText("sum")
        assert dlg.bg_check.isEnabled()
        dlg.bg_check.setChecked(True)
        assert dlg.roi_signals()[0].background is not None
        dlg.reducer_combo.setCurrentText("max")          # gate background off
        assert not dlg.bg_check.isEnabled()
        assert not dlg.bg_check.isChecked()
        assert dlg.roi_signals()[0].background is None
    finally:
        dlg.close()


def test_roi_select_reducer_and_background(qapp):
    from xdart.gui.tabs.static_scan.roi_select_dialog import RoiSelectDialog

    dlg = RoiSelectDialog(np.zeros((50, 50), dtype=float))
    try:
        dlg.reducer_combo.setCurrentText("sum")
        dlg._on_reducer_changed(0)
        dlg.bg_check.setChecked(True)            # -> creates a background rect
        dlg.bg_op_combo.setCurrentIndex(1)       # divide
        dlg._on_bg_op_changed(1)
        sig = dlg.roi_signals()[0]
        assert sig.reducer == "sum"
        assert sig.background is not None
        assert sig.background_op == "divide"

        # adding/removing ROIs grows/shrinks the signal list.
        dlg._add_roi()
        assert len(dlg.roi_signals()) == 2
        dlg._remove_roi()
        assert len(dlg.roi_signals()) == 1
    finally:
        dlg.close()


# ── end-to-end: compute fills table columns == direct run_roi_signals ─────


def test_scan_plot_roi_fills_columns_end_to_end(qapp):
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog
    from xrd_tools.analysis.plans import RoiSignal, run_roi_signals
    from xrd_tools.core.roi import RoiSpec
    from xrd_tools.sources import MemoryFrameSource

    src = MemoryFrameSource(_stack())
    dlg = ScanPlotDialog()
    try:
        dlg.set_table("scan", {"frame_index": np.array([0.0, 1.0, 2.0]),
                               "motor": np.array([10.0, 11.0, 12.0])})
        dlg._source = src
        dlg._raw_reachable = True

        sig = RoiSignal(
            roi=RoiSpec(center_x=1.5, center_y=1.5, width_x=2, width_y=2),
            reducer="mean", name="roiA")
        dlg._compute_roi([sig])
        assert _pump(
            qapp,
            lambda: (dlg._roi_worker is not None
                     and not dlg._roi_worker.isRunning()
                     and not dlg._roi_run_columns)), "ROI worker never finished"

        # the column was appended to the table + the selectors, and checked.
        assert "roiA" in dlg._table
        assert "roiA" in dlg._columns
        checked = {dlg.y_list.item(i).text() for i in range(dlg.y_list.count())
                   if dlg.y_list.item(i).checkState().value == 2}
        assert "roiA" in checked

        # the streamed/aligned column equals a direct headless run (mini spine),
        # and tracks the rising signal (monotonic up).
        direct = run_roi_signals([sig], src).payload.series["roiA"]
        np.testing.assert_allclose(dlg._table["roiA"], direct)
        assert np.all(np.diff(dlg._table["roiA"]) > 0)
    finally:
        dlg.close()


def test_scan_plot_roi_aligns_noncontiguous_frame_index(qapp):
    """ROI stats stream the SOURCE frame index; _frame_row_map must map them to
    the right table rows when frame_index doesn't start at 0 / isn't 0..n-1."""
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog
    from xrd_tools.analysis.plans import RoiSignal, run_roi_signals
    from xrd_tools.core.roi import RoiSpec
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.sources import MemoryFrameSource

    # frames labelled 5,6,7 (not 0,1,2) so keyed != positional alignment.
    src = MemoryFrameSource([ScanFrame(index=5 + i, image=im)
                             for i, im in enumerate(_stack())])
    assert src.frame_indices == [5, 6, 7]
    dlg = ScanPlotDialog()
    try:
        dlg.set_table("s", {"frame_index": np.array([5.0, 6.0, 7.0]),
                            "motor": np.array([1.0, 2.0, 3.0])})
        dlg._source = src
        dlg._raw_reachable = True
        sig = RoiSignal(roi=RoiSpec(center_x=1.5, center_y=1.5, width_x=2,
                                    width_y=2), reducer="mean", name="roiA")
        dlg._compute_roi([sig])
        assert _pump(qapp, lambda: (dlg._roi_worker is not None
                                    and not dlg._roi_worker.isRunning()
                                    and not dlg._roi_run_columns))
        direct = run_roi_signals([sig], src).payload.series["roiA"]
        np.testing.assert_allclose(dlg._table["roiA"], direct)   # aligned by index
        assert np.all(np.diff(dlg._table["roiA"]) > 0)
    finally:
        dlg.close()


def test_scan_plot_same_scan_param_change_keeps_columns(qapp):
    """Editing the raw/image pairing of the SAME scan refreshes the source +
    gating but must NOT wipe the table or the user's computed ROI columns; a
    DIFFERENT scan rebuilds wholesale."""
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog
    from xdart.gui.tabs.static_scan.scan_source_widget import ScanSelection
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.sources import MemoryFrameSource

    dlg = ScanPlotDialog()
    try:
        spec5 = SourceSpec("scanA", SourceKind.SPEC, options={"scan": "5"})
        src = MemoryFrameSource(_stack())
        dlg._on_source_selected(ScanSelection(
            spec=spec5, source=src, label="A5", reachable=True,
            first_image=_stack()[0]))
        dlg._append_column("roiX", np.array([1.0, 2.0, 3.0]), check=True)
        assert "roiX" in dlg._columns

        # same scan, only the source/raw pairing changed -> columns preserved
        src2 = MemoryFrameSource(_stack())
        dlg._on_source_selected(ScanSelection(
            spec=spec5, source=src2, label="A5", reachable=True,
            first_image=_stack()[0]))
        assert "roiX" in dlg._columns and dlg._source is src2

        # a different scan -> full rebuild drops the ROI column
        spec6 = SourceSpec("scanA", SourceKind.SPEC, options={"scan": "6"})
        dlg._on_source_selected(ScanSelection(
            spec=spec6, source=src2, label="A6", reachable=True,
            first_image=_stack()[0]))
        assert "roiX" not in dlg._columns
    finally:
        dlg.close()


def test_scan_plot_metadata_less_source_roi_vs_frame(qapp):
    """An images-only source (Eiger/raw burst — no motors) plots ROI vs frame
    number: the table is just frame_index, raw is reachable, ROI fills (§2.3)."""
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog
    from xdart.gui.tabs.static_scan.scan_source_widget import ScanSelection
    from xrd_tools.analysis.plans import RoiSignal
    from xrd_tools.core.roi import RoiSpec
    from xrd_tools.sources import MemoryFrameSource

    src = MemoryFrameSource(_stack())            # images, NO metadata
    dlg = ScanPlotDialog()
    try:
        dlg._on_source_selected(ScanSelection(
            spec=None, source=src, label="burst", reachable=True,
            first_image=_stack()[0]))
        assert dlg._columns == ["frame_index"]   # metadata-less → frame_index only
        assert dlg.roi_btn.isEnabled() is True   # reachable raw, no metadata needed
        sig = RoiSignal(roi=RoiSpec(center_x=1.5, center_y=1.5, width_x=2,
                                    width_y=2), reducer="mean", name="roiB")
        dlg._compute_roi([sig])
        assert _pump(qapp, lambda: (dlg._roi_worker is not None
                                    and not dlg._roi_worker.isRunning()
                                    and not dlg._roi_run_columns))
        assert "roiB" in dlg._table and np.all(np.diff(dlg._table["roiB"]) > 0)
    finally:
        dlg.close()


def test_scan_plot_roi_applies_provider_mask(qapp):
    """The dialog's mask_provider mask is threaded to the worker, so the ROI
    column matches a direct masked run (and differs from the unmasked one)."""
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog
    from xrd_tools.analysis.plans import RoiSignal, run_roi_signals
    from xrd_tools.core.roi import RoiSpec
    from xrd_tools.sources import MemoryFrameSource

    frames = [np.arange(36, dtype=float).reshape(6, 6) + 100 * f for f in range(3)]
    src = MemoryFrameSource(frames)
    mask = np.zeros((6, 6), dtype=bool)
    mask[:, :3] = True
    dlg = ScanPlotDialog(mask_provider=lambda uri: mask)
    try:
        dlg.set_table("s", {"frame_index": np.array([0.0, 1.0, 2.0])})
        dlg._source = src
        dlg._source_uri = "scan-A"
        dlg._raw_reachable = True
        sig = RoiSignal(roi=RoiSpec.full_frame(), reducer="mean", name="roiM")
        dlg._compute_roi([sig])
        assert _pump(qapp, lambda: (dlg._roi_worker is not None
                                    and not dlg._roi_worker.isRunning()
                                    and not dlg._roi_run_columns))
        direct = run_roi_signals([sig], src, mask=mask).payload.series["roiM"]
        np.testing.assert_allclose(dlg._table["roiM"], direct)
        unmasked = run_roi_signals([sig], src).payload.series["roiM"]
        assert not np.allclose(dlg._table["roiM"], unmasked)
    finally:
        dlg.close()


def test_scan_plot_roi_column_normalizes_and_csv_roundtrips(qapp, tmp_path):
    """An ROI/metadata column normalizes (y/norm) and the CSV exports the full
    assembled table incl. a non-numeric column."""
    from pyqtgraph.Qt import QtCore
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog

    dlg = ScanPlotDialog()
    try:
        dlg.set_table("s", {"frame_index": np.array([0.0, 1.0, 2.0]),
                            "i0": np.array([2.0, 4.0, 8.0]),
                            "sig": np.array([10.0, 20.0, 40.0]),
                            "label": np.array(["a", "b", "c"], dtype=object)})
        for i in range(dlg.y_list.count()):
            item = dlg.y_list.item(i)
            item.setCheckState(QtCore.Qt.CheckState.Checked if item.text() == "sig"
                               else QtCore.Qt.CheckState.Unchecked)
        dlg.norm_combo.setCurrentText("i0")
        items = dlg.plot.getPlotItem().listDataItems()
        assert len(items) == 1
        np.testing.assert_allclose(items[0].getData()[1], [5.0, 5.0, 5.0])  # sig/i0

        out = tmp_path / "scan.csv"
        dlg._write_csv(str(out))
        import csv
        with open(out) as fh:
            rows = list(csv.reader(fh))
        header = rows[0]
        assert "label" in header and "sig" in header     # non-numeric kept
        col = {h: [r[i] for r in rows[1:]] for i, h in enumerate(header)}
        assert col["label"] == ["a", "b", "c"]
        assert [float(v) for v in col["sig"]] == [10.0, 20.0, 40.0]
    finally:
        dlg.close()


def test_scan_plot_roi_abort_on_source_swap(qapp):
    """Loading a new source mid-run stops the worker + forgets the run state, so
    a stale worker can't stream into the replaced table."""
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog
    from xrd_tools.analysis.plans import RoiSignal
    from xrd_tools.core.roi import RoiSpec
    from xrd_tools.sources import MemoryFrameSource

    class _Slow(MemoryFrameSource):
        def load_frame(self, index):
            time.sleep(0.03)
            return super().load_frame(index)

    dlg = ScanPlotDialog()
    dlg.set_table("s", {"frame_index": np.arange(40.0)})
    dlg._source = _Slow([np.zeros((6, 6)) for _ in range(40)])
    dlg._raw_reachable = True
    dlg._compute_roi([RoiSignal(roi=RoiSpec.full_frame(), name="roiZ")])
    assert _pump(qapp, lambda: dlg._roi_worker.isRunning(), timeout=2.0)
    dlg._abort_roi_run()                       # what load_uri calls on a new scan
    assert not dlg._roi_worker.isRunning()
    assert dlg._roi_run_columns == [] and dlg._roi_row_of == {}
    assert dlg._roi_dialog is None


# ── cross-family right-hand axis ──────────────────────────────────────────


def test_scan_plot_right_axis_splits_series(qapp):
    """Checking a plotted column in the Right-axis list moves it from the main
    PlotItem onto the linked right ViewBox (so incommensurable columns overlay
    cleanly)."""
    from pyqtgraph.Qt import QtCore
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog

    _EX = QtCore.Qt.MatchFlag.MatchExactly
    _CHECKED = QtCore.Qt.CheckState.Checked
    dlg = ScanPlotDialog()
    try:
        dlg.set_table("s", {"frame_index": np.arange(5.0),
                            "small": np.linspace(0.0, 2.0, 5),
                            "big": np.linspace(0.0, 1e5, 5)})
        # deterministic selection: exactly small + big on Y (drop the auto-pick).
        _UNCHECKED = QtCore.Qt.CheckState.Unchecked
        for i in range(dlg.y_list.count()):
            item = dlg.y_list.item(i)
            item.setCheckState(_CHECKED if item.text() in ("small", "big")
                               else _UNCHECKED)
        # both default to the left axis: 2 curves on the main PlotItem, none right.
        assert len(dlg.plot.getPlotItem().listDataItems()) == 2
        assert len(dlg.right_vb.addedItems) == 0

        # move "big" to the right axis.
        dlg.r_list.findItems("big", _EX)[0].setCheckState(_CHECKED)
        left_names = {it.name() for it in dlg.plot.getPlotItem().listDataItems()}
        assert left_names == {"small"}
        assert len(dlg.right_vb.addedItems) == 1
        assert dlg.right_vb.addedItems[0].name() == "big"
        # the right ViewBox is not independently mouse-zoomable (no y-desync).
        assert list(dlg.right_vb.mouseEnabled()) == [False, False]
    finally:
        dlg.close()


def test_scan_plot_right_axis_plots_column_not_checked_in_y(qapp):
    """A column checked ONLY in the Right-axis list still plots (on the right) —
    the right toggle is never a silent no-op."""
    from pyqtgraph.Qt import QtCore
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog

    _EX = QtCore.Qt.MatchFlag.MatchExactly
    dlg = ScanPlotDialog()
    try:
        dlg.set_table("s", {"frame_index": np.arange(5.0),
                            "a": np.linspace(0.0, 1.0, 5),
                            "b": np.linspace(0.0, 1e5, 5)})
        for i in range(dlg.y_list.count()):
            dlg.y_list.item(i).setCheckState(QtCore.Qt.CheckState.Unchecked)
        dlg.r_list.findItems("b", _EX)[0].setCheckState(QtCore.Qt.CheckState.Checked)
        assert len(dlg.plot.getPlotItem().listDataItems()) == 0
        assert len(dlg.right_vb.addedItems) == 1
        assert dlg.right_vb.addedItems[0].name() == "b"
    finally:
        dlg.close()


def test_scan_plot_roi_close_mid_run_drops_partial_columns(qapp):
    """Closing the dialog mid-ROI-run drops the partial NaN columns (they must
    not survive into the reused single-instance dialog's next show)."""
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog
    from xrd_tools.analysis.plans import RoiSignal
    from xrd_tools.core.roi import RoiSpec
    from xrd_tools.sources import MemoryFrameSource

    class _Slow(MemoryFrameSource):
        def load_frame(self, index):
            time.sleep(0.03)
            return super().load_frame(index)

    src = _Slow([np.zeros((6, 6)) for _ in range(40)])
    dlg = ScanPlotDialog()
    dlg.set_table("s", {"frame_index": np.arange(40.0)})
    dlg._source = src
    dlg._raw_reachable = True
    dlg._compute_roi([RoiSignal(roi=RoiSpec.full_frame(), name="roiZ")])
    assert "roiZ" in dlg._table                       # column seeded
    assert _pump(qapp, lambda: dlg._roi_worker.isRunning(), timeout=2.0)
    dlg.close()                                        # -> sigRoiDone(None) queued
    assert _pump(qapp, lambda: (not dlg._roi_worker.isRunning()
                                and "roiZ" not in dlg._table), timeout=5.0), \
        "partial ROI column survived the close"
    assert "roiZ" not in dlg._columns
    assert dlg._roi_run_columns == []


def test_roi_frame_updates_are_redraw_coalesced(qapp):
    from pyqtgraph.Qt import QtCore
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog

    dlg = ScanPlotDialog()
    try:
        dlg.set_table("s", {"frame_index": np.arange(4.0), "i0": np.arange(4.0)})
        dlg._append_column("roiZ", np.full(4, np.nan), check=True)
        dlg._roi_row_of = {i: i for i in range(4)}

        calls = {"n": 0}

        def _count_redraw():
            calls["n"] += 1

        dlg._roi_redraw_timer.timeout.disconnect()
        dlg._roi_redraw_timer.timeout.connect(_count_redraw)

        for i in range(4):
            dlg._on_roi_frame(i, {"roiZ": float(i)})

        assert calls["n"] == 0
        assert dlg._roi_redraw_timer.isActive()
        assert _pump(qapp, lambda: calls["n"] == 1, timeout=1.0)
        np.testing.assert_allclose(dlg._table["roiZ"], [0.0, 1.0, 2.0, 3.0])
    finally:
        dlg.close()


def test_roi_done_flushes_final_coalesced_redraw(qapp):
    """A completing sigRoiDone cancels the pending debounce timer, so it must
    issue a final _redraw() itself — else the last frame's update stays stale."""
    import types
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog

    dlg = ScanPlotDialog()
    try:
        dlg.set_table("s", {"frame_index": np.arange(4.0), "i0": np.arange(4.0)})
        dlg._append_column("roiZ", np.full(4, np.nan), check=True)
        dlg._roi_row_of = {i: i for i in range(4)}
        dlg._roi_run_columns = ["roiZ"]

        calls = {"n": 0}
        dlg._redraw = lambda *a, **k: calls.__setitem__("n", calls["n"] + 1)

        # last frame schedules a debounced redraw (timer pending, not yet fired)
        dlg._on_roi_frame(3, {"roiZ": 3.0})
        assert dlg._roi_redraw_timer.isActive()
        assert calls["n"] == 0

        # the completing sigRoiDone must cancel the timer AND flush a final redraw
        dlg._on_roi_done(types.SimpleNamespace(
            payload=types.SimpleNamespace(diagnostics={})))
        assert not dlg._roi_redraw_timer.isActive()
        assert calls["n"] == 1
    finally:
        dlg.close()


def test_scan_plot_skips_all_nan_series_without_pyqtgraph_warning(qapp):
    from pyqtgraph.Qt import QtCore
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog

    dlg = ScanPlotDialog()
    try:
        dlg.set_table("s", {"frame_index": np.arange(3.0), "i0": np.arange(3.0)})
        dlg._append_column("empty", np.full(3, np.nan), check=True)
        for i in range(dlg.y_list.count()):
            item = dlg.y_list.item(i)
            item.setCheckState(
                QtCore.Qt.CheckState.Checked
                if item.text() == "empty"
                else QtCore.Qt.CheckState.Unchecked
            )
        dlg._redraw()
        assert len(dlg.plot.getPlotItem().listDataItems()) == 0
    finally:
        dlg.close()


# ---- §10 refinements: axis defaults, log, styling, viewer parity, Esc -------

def test_scan_plot_default_y_priority_counter(qapp):
    """10.2: default Y is the first present of the counter priority list
    (Photod>bs>mon>i2>i1>i0), and an ROI counter is never auto-selected (10.1)."""
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog
    dlg = ScanPlotDialog()
    try:
        n = 5
        table = {
            "frame_index": np.arange(n, dtype=float),
            "ROI1": np.linspace(1, 2, n),       # beamline counter — must NOT win
            "i0": np.full(n, 3.0),
            "i1": np.full(n, 4.0),
            "mon": np.full(n, 5.0),
            "samz": np.linspace(0, 4, n),       # the scanned positioner
        }
        dlg._positioner_names = ["samz"]
        dlg.set_table("s", table)
        assert dlg._checked_y() == ["mon"]      # mon precedes i1/i0; ROI1 excluded
        assert dlg.x_combo.currentText() == "samz"
    finally:
        dlg.close()


def test_scan_plot_default_y_never_roi(qapp):
    """10.1: with only ROI columns present (no priority counter), default Y falls
    back to frame_index — never an ROI column."""
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog
    dlg = ScanPlotDialog()
    try:
        n = 5
        table = {
            "frame_index": np.arange(n, dtype=float),
            "ROI1": np.linspace(1, 2, n),
            "ROI2": np.linspace(2, 3, n),
            "samz": np.linspace(0, 4, n),
        }
        dlg._positioner_names = ["samz"]
        dlg.set_table("s", table)
        assert dlg._checked_y() == ["frame_index"]   # not ROI1/ROI2
    finally:
        dlg.close()


def test_scan_plot_default_x_positioner_else_frame_index(qapp):
    """10.3: default X = the positioner that actually varies; with no positioner
    recorded, frame_index."""
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog
    n = 5
    table = {
        "frame_index": np.arange(n, dtype=float),
        "i0": np.linspace(1, 2, n),
        "samz": np.linspace(10, 14, n),     # scanned (varies)
        "samx": np.full(n, 7.0),            # parked (constant) — not the scan axis
    }
    dlg = ScanPlotDialog()
    try:
        dlg._positioner_names = ["samx", "samz"]
        dlg.set_table("s", table)
        assert dlg.x_combo.currentText() == "samz"
    finally:
        dlg.close()
    dlg2 = ScanPlotDialog()
    try:
        dlg2._positioner_names = []          # e.g. an older file with no positioners
        dlg2.set_table("s", table)
        assert dlg2.x_combo.currentText() == "frame_index"
    finally:
        dlg2.close()


def test_scan_plot_log_button_drives_left_axis(qapp):
    """10.5: Log toggles the LEFT axis only and survives a redraw; the RIGHT axis
    stays linear so its ticks match its (untransformed) curves."""
    from pyqtgraph.Qt import QtCore
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog
    dlg = ScanPlotDialog()
    try:
        n = 5
        dlg._positioner_names = ["samz"]
        dlg.set_table("s", {"frame_index": np.arange(n, dtype=float),
                            "samz": np.linspace(0, 4, n),
                            "i0": np.linspace(1, 100, n),
                            "big": np.linspace(1e4, 1e5, n)})
        for i in range(dlg.r_list.count()):     # 'big' onto the right axis
            it = dlg.r_list.item(i)
            if it.text() == "big":
                it.setCheckState(QtCore.Qt.CheckState.Checked)
        qapp.processEvents()
        assert hasattr(dlg, "log_btn")
        dlg.log_btn.setChecked(True)
        qapp.processEvents()
        assert dlg.plot.getAxis("left").logMode
        assert not dlg.right_axis.logMode        # right stays linear (matches its curve)
        dlg._redraw()                            # a later redraw must keep both
        assert dlg.plot.getAxis("left").logMode
        assert not dlg.right_axis.logMode
        dlg.log_btn.setChecked(False)
        qapp.processEvents()
        assert not dlg.plot.getAxis("left").logMode
    finally:
        dlg.close()


def test_roi_select_has_viewer_controls(qapp):
    """10.4: the ROI picker carries the intensity bar + Default/Log controls, and
    a Log re-render does not move the RectROI (the picked geometry is preserved)."""
    from xdart.gui.tabs.static_scan.roi_select_dialog import (
        RoiSelectDialog, _rect_center_size)
    img = np.abs(np.random.RandomState(0).normal(100, 20, (40, 60)))
    dlg = RoiSelectDialog(img)
    try:
        assert hasattr(dlg, "colorbar")
        assert dlg.scale_default_btn.isChecked()     # default = linear
        before = _rect_center_size(dlg._rois[0].rect)
        dlg.scale_log_btn.setChecked(True)
        qapp.processEvents()
        assert _rect_center_size(dlg._rois[0].rect) == before
        dlg.scale_default_btn.setChecked(True)
        qapp.processEvents()
    finally:
        dlg.close()


def test_dialogs_swallow_escape(qapp):
    """10.8: Esc does not dismiss the Scan Plot / ROI picker popups."""
    from pyqtgraph.Qt import QtCore, QtGui
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog
    from xdart.gui.tabs.static_scan.roi_select_dialog import RoiSelectDialog

    def press_escape(widget):
        ev = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress,
                             QtCore.Qt.Key.Key_Escape,
                             QtCore.Qt.KeyboardModifier.NoModifier)
        widget.keyPressEvent(ev)

    sp = ScanPlotDialog()
    sp.show()
    try:
        press_escape(sp)
        qapp.processEvents()
        assert sp.isVisible()                # still open after Esc
    finally:
        sp.close()

    roi = RoiSelectDialog(np.zeros((20, 20), dtype=float))
    roi.show()
    try:
        press_escape(roi)
        qapp.processEvents()
        assert roi.isVisible()
    finally:
        roi.close()


# ── close() shuts down the source-probe executor ──────────────────────────


def test_dialog_close_shuts_down_probe_executor(qapp, tmp_path):
    """Qt never delivers closeEvent to CHILD widgets, so the dialog's own
    closeEvent must stop the source widget's probe executor explicitly — a
    probe in flight at close otherwise survives as a live non-daemon thread
    that concurrent.futures joins only at interpreter exit (post-summary CI
    hang; app-exit hang live).  Real dialog, real widget, real probe on a
    real TIFF."""
    import fabio.tifimage
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog

    tif = tmp_path / "close_probe_0000.tif"
    fabio.tifimage.TifImage(data=np.ones((6, 6), dtype=np.int32)).write(
        str(tif))

    dlg = ScanPlotDialog()
    sw = dlg.source_widget
    # The dialog constructs its source widget with the async flag off today;
    # flip the constructor flag so the REAL async probe path (executor +
    # worker thread + done-callback) runs — that is the path close() guards.
    sw._async_probe = True
    sw.set_uri(str(tif))
    assert _pump(qapp, lambda: sw._probe_executor is not None, timeout=5.0), \
        "async probe never started — set_uri produced no candidate?"

    dlg.close()
    qapp.processEvents()
    assert sw._probe_executor is None, \
        "closeEvent did not shut down the probe executor"


# ── X1-6: processed-.nxs reads take the writer-coordinating lock ──────────


def _write_processed_nxs_with_scan_data(path):
    """A real minimal processed .nxs: integrated_1d (classifies the file as
    PROCESSED_XDART) + a scan_data group (what read_scan_data returns)."""
    import h5py

    labels = np.asarray([1, 2, 3], dtype=np.int64)
    q = np.linspace(0.1, 1.0, 4, dtype=np.float32)
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        g1 = entry.create_group("integrated_1d")
        g1.attrs["NX_class"] = "NXdata"
        g1.attrs["signal"] = "intensity"
        g1.attrs["axes"] = ["frame_index", "q"]
        g1.create_dataset("frame_index", data=labels)
        q_ds = g1.create_dataset("q", data=q)
        q_ds.attrs["units"] = "q_A^-1"
        g1.create_dataset(
            "intensity",
            data=np.arange(labels.size * q.size, dtype=np.float32).reshape(
                labels.size, q.size))
        sd = entry.create_group("scan_data")
        sd.attrs["NX_class"] = "NXcollection"
        sd.create_dataset("frame_index", data=labels)
        sd.create_dataset("i0", data=np.asarray([10.0, 20.0, 30.0]))


class _RecordingRLock:
    """A real re-entrant lock that counts acquisitions — the observation point
    for the X1-6 seam (the dialog code path itself stays fully production)."""

    def __init__(self):
        import threading
        self._lock = threading.RLock()
        self.enter_count = 0

    def __enter__(self):
        self._lock.acquire()
        self.enter_count += 1
        return self

    def __exit__(self, *exc):
        self._lock.release()
        return False


def test_scan_plot_processed_reads_take_writer_lock_x16(qapp, tmp_path):
    """X1-6 GUARD: the Scan Plot dialog's .nxs reads (scan_data table +
    positioners) must enter the provider's writer-coordinating lock when the
    picked source is the loaded scan's data file — the unlocked read raced a
    live writer's `r+` saves (the RN-1 family; every other display reader goes
    through _locked_scan_read)."""
    import os
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog

    p = tmp_path / "processed_scan.nxs"
    _write_processed_nxs_with_scan_data(p)

    rec = _RecordingRLock()

    def lock_provider(uri):
        same = os.path.realpath(str(uri)) == os.path.realpath(str(p))
        return rec if same else None

    dlg = ScanPlotDialog(lock_provider=lock_provider)
    try:
        dlg.load_uri(str(p))
        assert "i0" in dlg._columns, (
            "fixture did not load as a processed scan_data table — "
            f"columns: {dlg._columns}")
        # Both PROCESSED_NEXUS reads (read_scan_data + get_metadata) locked.
        assert rec.enter_count >= 2, (
            f"processed .nxs reads ran without the writer lock "
            f"(acquisitions: {rec.enter_count})")
    finally:
        dlg.close()


def test_static_widget_read_lock_provider_matches_loaded_scan_only_x16(
        qapp, tmp_path, monkeypatch):
    """X1-6 GUARD (wiring half): the analysis context hands popups the loaded
    scan's file_lock for the loaded data file ONLY — an arbitrary other file
    has no in-process writer, so it reads unlocked (None)."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xdart.gui.tabs.static_scan.display_data import DisplayDataMixin

    loaded = tmp_path / "loaded_scan.nxs"
    _write_processed_nxs_with_scan_data(loaded)
    other = tmp_path / "other_scan.nxs"
    _write_processed_nxs_with_scan_data(other)

    widget = staticWidget()
    try:
        widget.scan.data_file = str(loaded)
        ctx = widget._analysis_context()
        expected = DisplayDataMixin._scan_file_lock(widget)
        assert expected is not None, "loaded scan carries no file_lock?"
        assert ctx.read_lock_for_uri(str(loaded)) is expected
        assert ctx.read_lock_for_uri(str(other)) is None
        # uri=None means "the current scan" (the mask_for_scan_uri semantic):
        # with a loaded data file it resolves to that scan's lock.
        assert ctx.read_lock_for_uri(None) is expected
    finally:
        widget.close()
        widget.deleteLater()
