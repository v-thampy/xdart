"""End-to-end per-mode GUI smoke tests through the REAL widget wire.

These build a real ``staticWidget`` offscreen (`QT_QPA_PLATFORM=offscreen`)
and drive the actual ``H5Viewer → set_data → displayframe`` path, instead of
injecting state on a SimpleNamespace host.  That host-injection style is
exactly why P0 shipped green (nothing exercised the real wire), so these
assert each mode's panel draws through the production hand-off:

* processed-`.nxs` Image Viewer preserves its baked NaN mask (catches P0 —
  ``_viewer_is_xdart`` propagation at ``set_data``);
* a tiff/raw standalone preview fills detector sentinels (no transparent
  streaks);
* the XYE→Image and Image→Int2D transitions end in a coherent, populated
  display (catches T1);
* Int 1D / Int 2D render a plot / cake.

Real-data cells are gated on ``$XDART_TEST_DATA``.
"""
from __future__ import annotations

import os
import gc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.gui

_DATA = Path(os.environ.get("XDART_TEST_DATA",
                            Path(__file__).resolve().parents[3] / "test_data"))


@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    try:
        yield app
    finally:
        for top in list(app.topLevelWidgets()):
            try:
                top.close()
                top.deleteLater()
            except Exception:
                pass
        for _ in range(5):
            app.processEvents()
        app.quit()


@pytest.fixture
def widget(qapp):
    """A real staticWidget, torn down after each test."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    for _ in range(3):
        qapp.processEvents()
    gc.collect()
    for _ in range(2):
        qapp.processEvents()
    w = staticWidget()
    try:
        yield w
    finally:
        try:
            w.close()
        except Exception:
            pass
        try:
            w.deleteLater()
        except Exception:
            pass
        for _ in range(3):
            qapp.processEvents()
        gc.collect()
        for _ in range(2):
            qapp.processEvents()


def _set_image_frame(w, idx, raw):
    """Put one raw frame into the shared viewer dicts + select it."""
    with w.data_lock:
        w.viewer_rows_1d.clear()
        w.viewer_rows_2d.clear()
        w.viewer_rows_2d[idx] = {"map_raw": raw, "bg_raw": 0, "mask": None,
                          "int_2d": None, "gi_2d": {}, "thumbnail": None}
    w.frame_ids[:] = [str(idx)]
    w.displayframe.idxs_2d = [idx]


def test_perf_heartbeat_records_event_loop_stall(qapp, monkeypatch, caplog):
    """Item 5: the XDART_PERF main-thread heartbeat records the worst event-loop
    gap during a run and logs it at run end.  Drives the REAL widget methods on a
    REAL staticWidget (constructed with the env flag so the timer is created)."""
    import logging
    monkeypatch.setenv("XDART_PERF", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    for _ in range(3):
        qapp.processEvents()
    w = staticWidget()
    try:
        # The heartbeat timer is created under XDART_PERF; stop it so only our
        # deterministic manual ticks count (no reliance on wall-clock timer firing).
        assert w._perf_hb_timer is not None
        w._perf_hb_timer.stop()

        w._perf_hb_start_window()
        # Inject a 1.0 s stall: pretend the last tick was 1 s ago, then tick once.
        w._perf_hb_last -= 1.0
        w._perf_heartbeat_tick()
        assert w._perf_hb_max_gap_ms >= 900.0        # the stall was detected

        caplog.set_level(logging.INFO)
        w._perf_hb_end_window()
        assert any("max event-loop gap during run" in r.getMessage()
                   for r in caplog.records)          # logged at run end
        assert w._perf_hb_active is False            # window closed

        # Sanity: without XDART_PERF the timer is never created (no-op methods).
    finally:
        try:
            w.close()
            w.deleteLater()
        except Exception:
            pass
        for _ in range(3):
            qapp.processEvents()
        gc.collect()


def test_image_viewer_processed_nxs_preserves_baked_mask(widget):
    # P0: a processed-xdart frame's baked NaN mask must survive the real
    # H5Viewer→set_data→displayframe wire.  If _viewer_is_xdart is not
    # propagated at set_data, _update_image_viewer fills the NaN (mask lost).
    w = widget
    w._on_viewer_mode_changed("image")

    raw = np.arange(36, dtype=float).reshape(6, 6)
    raw[2, 3] = np.nan          # baked mask pixel
    _set_image_frame(w, 0, raw)
    # processed-xdart classification (what _load_image_file sets on a
    # processed .nxs); set ONLY on the H5Viewer — set_data must propagate it.
    w.h5viewer._viewer_is_xdart = True
    w.displayframe._viewer_is_xdart = False

    w.set_data()                       # propagates classification + renders

    assert w.displayframe._viewer_is_xdart is True       # P0 propagation
    img = w.displayframe.image_data[0]                   # rendered via the payload
    assert img is not None
    assert np.isnan(img).any()                            # mask preserved


def test_image_viewer_standalone_fills_sentinels(widget):
    # Inverse of P0: a standalone detector file (not xdart) fills the masked
    # sentinel so a raw preview isn't riddled with transparent streaks.
    w = widget
    w._on_viewer_mode_changed("image")

    raw = np.arange(36, dtype=float).reshape(6, 6)
    raw[2, 3] = np.nan
    _set_image_frame(w, 0, raw)
    w.h5viewer._viewer_is_xdart = False
    w.set_data()                       # propagates classification + renders

    assert w.displayframe._viewer_is_xdart is False
    img = w.displayframe.image_data[0]                   # rendered via the payload
    assert img is not None
    assert np.isfinite(img).all()                         # sentinel filled


def test_xye_to_image_transition_renders(widget):
    # T1: switching XYE Viewer → Image Viewer must leave a coherent Image
    # Viewer (not stuck on the XYE plot).  Render a frame after the switch.
    w = widget
    w._on_viewer_mode_changed("xye")
    assert w.displayframe.viewer_mode == "xye"

    w._on_viewer_mode_changed("image")
    assert w.displayframe.viewer_mode == "image"
    # viewer_rows_2d/frame_ids were cleared by the transition cleanup.
    assert len(w.viewer_rows_2d) == 0

    raw = np.arange(36, dtype=float).reshape(6, 6)
    _set_image_frame(w, 0, raw)
    w.h5viewer._viewer_is_xdart = False
    w.set_data()                       # propagates classification + renders
    img = w.displayframe.image_data[0]                   # rendered via the payload
    assert img is not None and img.size > 0 and np.isfinite(img).any()


def test_image_viewer_idxs_ignore_stale_scan_frames(widget):
    """Regression (Int 1D (XYE) → Image Viewer real state corruption).

    Running Int 1D (XYE) integrates a file and leaves ``scan.frames``
    populated.  Opening that SAME file in Image Viewer makes
    ``len(frame_ids) == len(scan.frames.index)``, which flips
    ``resolve_selection``'s ``overall`` heuristic True and rebases the
    selection onto the stale scan labels.  Those don't intersect the
    viewer's loaded ``viewer_rows_2d`` keys, so ``idxs_2d`` comes back empty and
    the panel renders blank — even though the image loaded fine.  Reached
    any other way (``scan.frames`` empty) it renders, which is the exact
    asymmetry reported.  ``get_idxs`` must ignore ``scan.frames`` in viewer
    mode."""
    import threading
    from types import SimpleNamespace
    w = widget
    w._on_viewer_mode_changed("image")
    df = w.displayframe
    # Stale scan from a prior Int 1D (XYE) run: same COUNT (3) as the image
    # we're viewing, but DIFFERENT labels.  (Stubbed so the test also proves
    # the fix path never consults scan.frames at all.)
    df.scan = SimpleNamespace(
        scan_lock=threading.RLock(),
        frames=SimpleNamespace(index=[10, 11, 12]),
        name="stale", skip_2d=False,
    )
    raw = np.arange(36, dtype=float).reshape(6, 6)
    with df.data_lock:
        df.viewer_rows_1d.clear()
        df.viewer_rows_2d.clear()
        for k in (1, 2, 3):
            df.viewer_rows_2d[k] = {"map_raw": raw, "bg_raw": 0, "mask": None,
                             "int_2d": None, "gi_2d": {}, "thumbnail": None}
    df.frame_ids[:] = ["1", "2", "3"]

    df.get_idxs()

    # Pre-fix: overall=True → ids rebased to [10,11,12] → idxs_2d == [] (blank).
    # Post-fix: the loaded frame_ids drive the selection.
    assert df.overall is False
    assert df.idxs_2d == [1, 2, 3]


def test_image_viewer_restores_twodwindow_height_after_1d_only(widget):
    """Regression (Int 1D (XYE) -> Image Viewer blank, root cause).

    A 1D-only processing mode (Int 1D / Int 1D (XYE)) collapses the 2D
    container ``twoDWindow`` to height 0 via ``_apply_1d_only_visibility``.
    Viewer modes return early from that method, so entering Image Viewer must
    restore the container's height itself — otherwise the raw image is drawn
    correctly into a zero-height widget and is invisible (the reported blank;
    data + sanitize + draw were all confirmed identical to the working case,
    the only difference was widget height = 0)."""
    w = widget
    df = w.displayframe
    # Reproduce the collapse an Int 1D (XYE) run leaves behind.
    df.scan.skip_2d = True
    df.viewer_mode = None
    df._apply_1d_only_visibility()
    assert df.ui.twoDWindow.maximumHeight() == 0          # collapsed

    df.set_viewer_display_mode("image")
    assert df.ui.twoDWindow.maximumHeight() > 0           # restored, not 0


def test_image_viewer_uses_percentile_not_wrangler_threshold(widget):
    """Regression: the Image Viewer color scale must be the nanpercentile
    autoscale (same as the Int 2D raw/cake panels), NOT the wrangler Intensity
    Threshold (Min/Max).  The threshold is an integration mask parameter; using
    it as vmin/vmax washed the image out (a 0-1000 threshold flattening detector
    counts that actually span ~0-60)."""
    from types import SimpleNamespace
    w = widget
    w._on_viewer_mode_changed("image")
    df = w.displayframe
    # A wrangler with an active intensity threshold, exactly as in the report.
    df._wrangler = SimpleNamespace(
        apply_threshold=True, threshold_min=0.0, threshold_max=1000.0)
    raw = np.arange(36, dtype=float).reshape(6, 6)     # data spans 0..35
    _set_image_frame(w, 0, raw)
    w.h5viewer._viewer_is_xdart = False
    w.set_data()                       # renders through the payload path
    lo, hi = df.image_widget.imageItem.levels
    # Percentile of 0..35 (~0.4..34.6), NOT the 1000 threshold ceiling.
    assert hi < 100.0


def test_image_viewer_all_nonfinite_payload_clears_panel_and_colorbar(widget):
    """An all-non-finite frame yields a None raw payload, so the Image Viewer
    must blank the image AND hide the colorbar — never a zero placeholder."""
    w = widget
    w._on_viewer_mode_changed("image")
    df = w.displayframe
    raw = np.full((6, 6), np.nan)                  # xdart frame, all masked
    _set_image_frame(w, 0, raw)
    w.h5viewer._viewer_is_xdart = True             # preserve NaN -> all-non-finite
    w.set_data()                                   # renders through the payload path

    assert df.image_data is None                   # cleared, not (zeros, rect)
    assert df.image_widget.raw_image.size == 0     # no zero placeholder painted
    assert df.image_widget.histogram.isVisible() is False    # colorbar hidden


def test_image_viewer_no_selection_clears_panel(widget):
    """With nothing selected the Image Viewer renders an explicit blank."""
    w = widget
    w._on_viewer_mode_changed("image")
    df = w.displayframe
    with w.data_lock:
        w.viewer_rows_1d.clear()
        w.viewer_rows_2d.clear()
    w.frame_ids[:] = []
    df.idxs_2d = []
    df.update()
    assert df.image_data is None


def test_f8_intensity_controls_are_mode_scoped(widget):
    """Intensity controls live in the existing second row for every 1D/viewer
    display, but stay out of the full Int 2D panel."""
    w = widget
    df = w.displayframe

    w._on_viewer_mode_changed("image")
    assert not df._intensityWidget.isHidden()
    assert df.ui.imageToolbar.isHidden()
    assert not df.ui.plotToolBar.isHidden()
    assert df.ui.verticalLayout_3.indexOf(df.ui.plotToolBar) < df.ui.verticalLayout_3.indexOf(df.ui.twoDWindow)
    assert df.ui.plotMethod.isHidden()
    assert df.ui.clear_1D.isHidden()

    w._on_viewer_mode_changed("xye")
    assert not df._intensityWidget.isHidden()
    assert not df.ui.imageToolbar.isHidden()
    assert df.ui.plotToolBar.isHidden()
    assert df.ui.plotUnit.isHidden()
    assert not df.ui.plotMethod.isHidden()
    assert not df.ui.wf_options.isHidden()
    assert not df.ui.clear_1D.isHidden()

    w._on_viewer_mode_changed("")
    df.scan.skip_2d = False
    df._apply_1d_only_visibility()
    assert df._intensityWidget.isHidden()
    assert not df.ui.plotUnit.isHidden()

    df.scan.skip_2d = True
    df._apply_1d_only_visibility()
    assert not df._intensityWidget.isHidden()
    assert not df.ui.plotMethod.isHidden()


def test_f8_image_viewer_manual_intensity_levels(widget):
    """Turning Autoscale off seeds from the current image levels, then the range
    slider applies display-only levels to the Image Viewer."""
    w = widget
    w._on_viewer_mode_changed("image")
    df = w.displayframe
    raw = np.arange(100, dtype=float).reshape(10, 10)
    _set_image_frame(w, 0, raw)
    w.h5viewer._viewer_is_xdart = False
    w.set_data()

    assert df._intensityAuto.isChecked() is True
    assert df.image_widget.imageItem.levels is not None

    df._intensityAuto.setChecked(False)
    df._intensitySlider.setValues(10.0, 20.0)
    lo, hi = df.image_widget.imageItem.levels
    assert np.isclose(lo, 10.0)
    assert np.isclose(hi, 20.0)


def test_wrangler_tree_polish(widget):
    w = widget
    tree = w.wrangler.tree
    assert tree.header().minimumSectionSize() == 40
    # Direction-A Stage 3a: the name column is widened (resizeSection(0, 120), was
    # 79) so card-row labels like "Average Scan" / "Write Mode" don't clip.  The
    # *rendered* sectionSize is environment-dependent (clamped to the realized
    # tree width — tiny offscreen, ~120 in the real 334px panel), so it's not a
    # reliable headless assertion; the widening is verified by construction.  What
    # IS deterministic is the resize MODE: pyqtgraph defaults col 0 to
    # ResizeToContents (which silently ignored resizeSection + clipped labels);
    # the fix forces Interactive so the fixed width actually applies.  Guard it.
    from PySide6.QtWidgets import QHeaderView
    assert tree.header().sectionResizeMode(0) == QHeaderView.ResizeMode.Interactive
    # The wrangler tree is themed via the GLOBAL QSS by object name (replacing
    # the old inline Dracula stylesheet), so it renders in both Dark and Light
    # and live-switches.  Assert the QSS hook + that the rules exist, not colours
    # on the widget-local stylesheet (which is now empty).
    assert tree.objectName() == "WranglerTree"
    from xdart.gui.themes import render_qss
    assert "#WranglerTree" in render_qss("dark")
    assert "#WranglerTree" in render_qss("light")
    from PySide6 import QtWidgets
    browse_buttons = [
        b for b in tree.findChildren(QtWidgets.QPushButton)
        if b.text() == "Browse"
    ]
    assert browse_buttons
    # Direction-A Stage 2: the path field + Browse button are themed through the
    # global QSS by object name (QLineEdit#BrowsePathEdit / QPushButton#
    # BrowseButton) rather than inline Dracula hex, so both Dark and Light render
    # and a live theme switch recolours them.  Assert the QSS hooks, not colours.
    assert browse_buttons[0].objectName() == "BrowseButton"
    path_edits = [
        e for e in tree.findChildren(QtWidgets.QLineEdit)
        if e.objectName() == "BrowsePathEdit"
    ]
    assert path_edits


def test_param_rows_hide_reset_glyph(qapp):
    """Stage 2 (Direction A): pyqtgraph's per-row reset-to-default glyph (the
    little curved-arrow button it shows on every row whose param hasDefault())
    is suppressed globally — the redesigned wrangler/integrator forms have none.
    Patched on the shared base, so it holds for both str_browse and plain str
    rows (``isHidden`` is independent of whether the tree has been shown)."""
    from pyqtgraph.parametertree import Parameter, ParameterTree
    p = Parameter.create(name="root", type="group", children=[
        {"name": "poni_file", "title": "Calibration",
         "type": "str_browse", "value": "/a/b/LaB6.poni"},
        {"name": "entry", "title": "Entry", "type": "str", "value": "entry"},
    ])
    tree = ParameterTree()
    tree.setParameters(p, showTop=False)
    try:
        for child in ("poni_file", "entry"):
            item = next(iter(p.child(child).items))
            assert item.defaultBtn.isHidden()
    finally:
        tree.deleteLater()
        qapp.processEvents()


def test_theme_token_and_selector_invariants():
    """Direction-A guard rails for the theme refactor:
    (1) DARK and LIGHT define the SAME token set and both render with no
        unresolved ``$token`` — ``string.Template.substitute`` raises on a missing
        key, so a one-sided token would crash ``render_qss`` (and the live app).
    (2) NO descendant selector targets the integrator's ``frame1D``/``frame2D``
        subtree, which is REPARENTED at runtime (frame_3 -> toolsFrame,
        integratorFrame.setLayout) — a descendant style rule across a reparented
        subtree segfaults Qt's style engine at teardown (the Stage-3b Mono-rule
        crash).  Only direct ``#id`` selectors are allowed there."""
    import re
    from xdart.gui.themes.dark import DARK, LIGHT, _QSS_TEMPLATE, render_qss
    assert set(DARK) == set(LIGHT)
    for name in ("dark", "light"):
        assert not re.search(r"\$\w+", render_qss(name)), f"unresolved token in {name}"
    assert "#frame1D Q" not in _QSS_TEMPLATE, "descendant selector on reparented frame1D"
    assert "#frame2D Q" not in _QSS_TEMPLATE, "descendant selector on reparented frame2D"


def test_metadata_popup_and_tools_placeholder(widget):
    """Stage 4 (Direction A): the metadata table opens as an on-demand non-modal
    popup (reparenting the live metawidget) and the vacated bottom-left hosts the
    Tools placeholder."""
    from PySide6 import QtWidgets
    w = widget
    # The Metadata button exists in the Data-Browser button row.
    assert hasattr(w.h5viewer.ui, "metadata_btn")
    # Tools placeholder occupies metaFrame (the table is no longer inline there).
    assert w.ui.metaFrame.findChild(QtWidgets.QFrame, "toolsPlaceholder") is not None
    # Each tool is now a button labelled with the tool name (the descriptive note
    # was dropped; the hover tooltip carries the description).
    btn_texts = {b.text() for b in w.ui.metaFrame.findChildren(QtWidgets.QPushButton)
                 if b.objectName() == "toolButton"}
    # Buttons carry a leading glyph (e.g. "∧   Peak Fitting"); match the label.
    assert all(any(name in t for t in btn_texts)
               for name in ("Peak Fitting", "Phase Fitting", "Plot Metadata"))
    # Opening the popup reparents the live metawidget into a non-modal dialog.
    assert w._metadata_dialog is None
    w._open_metadata_dialog()
    dlg = w._metadata_dialog
    assert dlg is not None and not dlg.isModal()
    assert dlg.isAncestorOf(w.metawidget)
    # Idempotent: a second open reuses the single instance.
    w._open_metadata_dialog()
    assert w._metadata_dialog is dlg


def test_peak_fit_dialog_fits_synthetic_pattern(qapp):
    """Tools ▸ Peak Fitting: the self-contained dialog fits the provided 1-D
    pattern via xrd_tools.analysis.fitting and fills its results table.  Uses a
    synthetic two-Gaussian pattern so the recovered centers are checkable."""
    lmfit = pytest.importorskip("lmfit")  # the xrd-tools[fitting] extra
    from xdart.gui.tabs.static_scan.peak_fit_dialog import PeakFitDialog
    x = np.linspace(1.0, 5.0, 600)

    def g(c, s, a):
        return a * np.exp(-0.5 * ((x - c) / s) ** 2)

    y = g(2.0, 0.05, 1.0e5) + g(3.5, 0.07, 6.0e4) + 2000.0 + 500.0 * x
    dlg = PeakFitDialog(lambda: (x, y, "q (Å⁻¹)"))
    try:
        dlg.auto_check.setChecked(False)       # exercise the fixed-count path
        dlg.npeaks_spin.setValue(2)
        dlg.model_combo.setCurrentText("Gaussian")
        dlg.refresh_pattern()
        dlg._do_fit()
        assert dlg.table.rowCount() == 2
        centers = sorted(float(dlg.table.item(r, 1).text())
                         for r in range(dlg.table.rowCount()))
        assert abs(centers[0] - 2.0) < 0.05
        assert abs(centers[1] - 3.5) < 0.05
    finally:
        dlg.deleteLater()
        qapp.processEvents()


def test_peak_fit_auto_detect_and_range(qapp):
    """Auto peak detection finds the peaks unaided, and the fit range restricts
    the fit to the selected window (excludes out-of-range peaks)."""
    pytest.importorskip("lmfit")
    from xdart.gui.tabs.static_scan.peak_fit_dialog import PeakFitDialog
    x = np.linspace(1.0, 6.0, 800)

    def g(c, s, a):
        return a * np.exp(-0.5 * ((x - c) / s) ** 2)

    y = (g(2.0, 0.05, 1.0e5) + g(3.5, 0.06, 7.0e4) + g(4.8, 0.07, 5.0e4)
         + 2000.0 + 300.0 * x)
    dlg = PeakFitDialog(lambda: (x, y, "q (Å⁻¹)"))
    try:
        dlg.refresh_pattern()                  # auto is on by default
        # (1) auto-detect over the whole pattern finds all three peaks
        dlg._do_fit()
        assert dlg.table.rowCount() == 3
        # (2) restrict the range -> only the two in-window peaks are fit
        dlg._fit_lo, dlg._fit_hi = 1.5, 4.0
        assert dlg._fit_range() == (1.5, 4.0)
        dlg._do_fit()
        assert dlg.table.rowCount() == 2
        centers = sorted(float(dlg.table.item(r, 1).text())
                         for r in range(dlg.table.rowCount()))
        assert abs(centers[0] - 2.0) < 0.05 and abs(centers[1] - 3.5) < 0.05
    finally:
        dlg.deleteLater()
        qapp.processEvents()


def test_peak_fit_dialog_handles_no_pattern(qapp):
    """No frame selected -> the dialog reports it and does not crash on Fit."""
    from xdart.gui.tabs.static_scan.peak_fit_dialog import PeakFitDialog
    dlg = PeakFitDialog(lambda: None)
    try:
        dlg.refresh_pattern()
        assert dlg.table.rowCount() == 0
        dlg._do_fit()                      # must not raise
        assert dlg.table.rowCount() == 0
    finally:
        dlg.deleteLater()
        qapp.processEvents()


def test_peak_fit_tool_wired_in_static_widget(widget):
    """The Tools card exposes an active 'Open' affordance for Peak Fitting and
    staticWidget can build the dialog (lazy, non-modal)."""
    from PySide6 import QtWidgets
    w = widget
    tools = [b for b in w.ui.metaFrame.findChildren(QtWidgets.QPushButton)
             if b.objectName() == "toolButton"]
    assert len(tools) == 3                 # Peak + Phase + Scan Plot active
    assert all(b.isEnabled() for b in tools)
    assert w._peak_fit_dialog is None
    w._open_peak_fit_dialog()
    assert w._peak_fit_dialog is not None and not w._peak_fit_dialog.isModal()


def test_peak_fit_live_path_draws_outcome(qapp):
    """Step 3 (live): set_live_pattern shows the data, build_fit_request + analyze
    produce an outcome, and _draw_outcome fills the table — the exact path the
    live worker drives, run synchronously here so it's lmfit-real but not flaky."""
    pytest.importorskip("lmfit")
    from xdart.gui.tabs.static_scan.peak_fit_dialog import PeakFitDialog
    x = np.linspace(1.0, 5.0, 600)

    def g(c, s, a):
        return a * np.exp(-0.5 * ((x - c) / s) ** 2)

    y = g(2.0, 0.05, 1.0e5) + g(3.5, 0.07, 6.0e4) + 2000.0 + 500.0 * x
    # The provider returns nothing — live PUSHES the pattern (set_live_pattern),
    # it doesn't pull through the provider.
    dlg = PeakFitDialog(lambda: None)
    try:
        dlg.live_check.setChecked(True)
        dlg.auto_check.setChecked(False)
        dlg.npeaks_spin.setValue(2)
        dlg.model_combo.setCurrentText("Gaussian")
        dlg.set_live_pattern(x, y, "q (Å⁻¹)")  # data shown immediately
        req = dlg.build_fit_request()          # same request manual Fit builds
        assert req is not None
        inp, analyzer = req
        outcome = analyzer.analyze(inp)
        assert outcome.ok
        dlg._draw_outcome(outcome, auto=False)
        assert dlg.table.rowCount() == 2
        centers = sorted(float(dlg.table.item(r, 1).text())
                         for r in range(dlg.table.rowCount()))
        assert abs(centers[0] - 2.0) < 0.05 and abs(centers[1] - 3.5) < 0.05
    finally:
        dlg.deleteLater()
        qapp.processEvents()


def test_live_fit_is_noop_without_open_dialog(widget):
    """set_data calls _maybe_live_fit on every frame, so with the dialog closed
    (or Live off) it must be a cheap no-op that never spins up the worker."""
    w = widget
    assert w._peak_fit_dialog is None
    w._maybe_live_fit()                        # no dialog -> no-op
    assert w._live_analysis_worker is None     # worker stays lazy until a real request


def test_param_family_helpers_are_pure():
    """split_family / group_families / accumulator_to_table parse the flat param
    keys into per-peak families + an aligned vs-frame table, with no Qt."""
    from xdart.gui.tabs.static_scan.peak_fit_util import (
        accumulator_to_table, group_families, split_family)
    assert split_family("center_0") == ("center", 0)
    assert split_family("center_err_2") == ("center_err", 2)
    assert split_family("amplitude") == ("amplitude", 0)
    fams = group_families(["center_0", "center_1", "fwhm_0"])
    assert fams["center"] == [("center_0", 0), ("center_1", 1)]
    assert fams["fwhm"] == [("fwhm_0", 0)]
    # {frame: params} -> aligned (labels, columns), sorted, missing param -> nan
    labels, cols = accumulator_to_table(
        {2: {"center_0": 2.2, "center_1": 3.5}, 0: {"center_0": 2.0}})
    assert labels == ["0", "2"]                     # sorted by frame index
    assert cols["center_0"] == [2.0, 2.2]
    assert cols["center_1"][0] != cols["center_1"][0]   # nan (frame 0 lacked it)
    assert cols["center_1"][1] == 3.5


def test_peak_fit_dialog_param_trend_row(qapp):
    """Row 3 (params vs frame): accumulating frames builds the family combo and
    plots one curve per peak; Overlay toggles all-peaks vs the first; reset
    clears it."""
    from xdart.gui.tabs.static_scan.peak_fit_dialog import PeakFitDialog
    dlg = PeakFitDialog(lambda: None)
    try:
        for idx in range(3):
            dlg._accumulate_frame_params(idx, {
                "center_0": 2.0 + 0.1 * idx, "center_1": 3.5,
                "fwhm_0": 0.1, "amplitude_0": 1e5})
        fams = [dlg.param_family_combo.itemData(i)
                for i in range(dlg.param_family_combo.count())]
        assert {"center", "fwhm", "amplitude"} <= set(fams)
        assert dlg.param_family_combo.isEnabled() and dlg.param_save_btn.isEnabled()
        # default family = center; overlay off -> one curve (center_0)
        dlg.param_family_combo.setCurrentIndex(fams.index("center"))
        dlg.param_overlay_check.setChecked(False)
        assert len(dlg.param_plot.getPlotItem().listDataItems()) == 1
        # overlay on -> both center_0 + center_1
        dlg.param_overlay_check.setChecked(True)
        assert len(dlg.param_plot.getPlotItem().listDataItems()) == 2
        # a re-fit of an existing frame updates, not duplicates
        dlg._accumulate_frame_params(1, {"center_0": 9.9, "center_1": 3.5})
        assert len(dlg._param_accumulator) == 3
        # reset clears the trend + disables the controls
        dlg.reset_param_trend()
        assert dlg.param_plot.getPlotItem().listDataItems() == []
        assert not dlg.param_family_combo.isEnabled()
        assert dlg._param_accumulator == {}
    finally:
        dlg.deleteLater()
        qapp.processEvents()


def test_peak_fit_advanced_options_feed_the_plan(qapp):
    """Advanced box: manual centers override Auto, and the optional kwargs
    (σ init/bounds, center δ, max_nfev) flow into the PeakFitPlan + fit."""
    pytest.importorskip("lmfit")
    from xdart.gui.tabs.static_scan.peak_fit_dialog import PeakFitDialog
    x = np.linspace(1.0, 5.0, 600)

    def g(c, s, a):
        return a * np.exp(-0.5 * ((x - c) / s) ** 2)

    y = g(2.0, 0.05, 1e5) + g(3.5, 0.07, 6e4) + 2000.0 + 500.0 * x
    dlg = PeakFitDialog(lambda: (x, y, "q"))
    try:
        dlg.refresh_pattern()
        dlg.auto_check.setChecked(True)        # manual centers must still win
        dlg.adv_centers.setText("2.0, 3.5")
        dlg.adv_sigma_init.setText("0.06")
        dlg.adv_sigma_min.setText("0.01")
        dlg.adv_sigma_max.setText("0.2")
        dlg.adv_center_delta.setText("0.1")
        dlg.adv_maxfev.setValue(5000)
        req = dlg.build_fit_request()
        assert req is not None
        inp, analyzer = req
        plan = analyzer.plan
        assert plan.positions == (2.0, 3.5) and plan.n_peaks == 2   # manual wins
        assert plan.sigma_init == 0.06
        assert plan.sigma_bounds == (0.01, 0.2)
        assert plan.center_bounds_delta == 0.1
        assert plan.fit_kwargs == {"max_nfev": 5000}
        out = analyzer.analyze(inp)            # and it actually fits
        assert out.ok
        centers = sorted(out.params[f"center_{i}"] for i in range(2))
        assert abs(centers[0] - 2.0) < 0.05 and abs(centers[1] - 3.5) < 0.05
    finally:
        dlg.deleteLater()
        qapp.processEvents()


def test_peak_fit_manual_centers_filter_to_range_and_toggle(qapp):
    """The Advanced toggle shows the box; manual centers outside the fit range
    are dropped (seeds must lie in the fitted window)."""
    from xdart.gui.tabs.static_scan.peak_fit_dialog import PeakFitDialog
    x = np.linspace(1.0, 5.0, 400)
    dlg = PeakFitDialog(lambda: (x, x * 0 + 1.0, "q"))
    try:
        dlg.refresh_pattern()
        # isVisibleTo (not isVisible) — the dialog itself isn't shown in the test
        assert not dlg.advanced_box.isVisibleTo(dlg)
        dlg.advanced_btn.setChecked(True)
        assert dlg.advanced_box.isVisibleTo(dlg)
        dlg._fit_lo, dlg._fit_hi = 1.5, 4.0
        dlg.adv_centers.setText("0.5, 2.1, 3.5, 9.0")   # 0.5 & 9.0 out of window
        assert dlg._manual_centers(*dlg._fit_range()) == [2.1, 3.5]
        dlg.adv_centers.setText("")
        assert dlg._manual_centers(*dlg._fit_range()) is None
    finally:
        dlg.deleteLater()
        qapp.processEvents()


def test_reload_during_live_keeps_trend(qapp):
    """Reload while Live is on must NOT wipe the accumulated vs-frame trend;
    Reload when not live starts fresh."""
    from xdart.gui.tabs.static_scan.peak_fit_dialog import PeakFitDialog
    x = np.linspace(1.0, 5.0, 200)
    dlg = PeakFitDialog(lambda: (x, x * 0 + 1.0, "q"))
    try:
        dlg._accumulate_frame_params(0, {"center_0": 2.0})
        dlg._accumulate_frame_params(1, {"center_0": 2.1})
        dlg.live_check.setChecked(True)
        dlg.refresh_pattern()                  # Reload mid-live -> trend kept
        assert set(dlg._param_accumulator) == {0, 1}
        dlg.live_check.setChecked(False)
        dlg.refresh_pattern()                  # Reload not-live -> fresh
        assert dlg._param_accumulator == {}
    finally:
        dlg.deleteLater()
        qapp.processEvents()


def test_close_guard_blocks_analysis_slots(widget):
    """Once close() set _tearing_down, the worker-signal slots must bail without
    touching the dialog — guards the close-during-run use-after-free."""
    w = widget
    w._tearing_down = True
    # None of these may raise or require a live dialog (queued signal arriving
    # mid-teardown).
    w._on_batch_progress(1, 2)
    w._on_batch_frame_fit("0", {"center_0": 2.0})
    w._on_batch_done(["0"], {"center_0": [2.0]})
    w._on_live_analyzed("0", w._live_fit_gen, None)


# ── Phase Fitting tool ────────────────────────────────────────────────────


def test_phase_fit_tool_wired_in_static_widget(widget):
    """Tools exposes Peak + Phase Fitting (both active); the Phase dialog builds
    lazily + non-modal and shares the batch machinery."""
    from PySide6 import QtWidgets
    w = widget
    tools = [b for b in w.ui.metaFrame.findChildren(QtWidgets.QPushButton)
             if b.objectName() == "toolButton"]
    assert len(tools) == 3
    assert w._phase_fit_dialog is None
    w._open_phase_fit_dialog()
    assert w._phase_fit_dialog is not None and not w._phase_fit_dialog.isModal()


def test_phase_fit_dialog_scaffolding(qapp):
    """The Phase dialog builds (CIF list + options + shared trend), and
    build_fit_request refuses without phases."""
    from xdart.gui.tabs.static_scan.phase_fit_dialog import PhaseFitDialog
    x = np.linspace(1.0, 5.0, 200)
    dlg = PhaseFitDialog(lambda: (x, x * 0 + 1.0, "q"))
    try:
        dlg.refresh_pattern()
        # core widgets present
        assert dlg.cif_list is not None and dlg.profile_combo.count() == 4
        assert dlg.batch_btn is not None and dlg.param_plot is not None
        # no phases -> can't build a request
        assert dlg.build_fit_request() is None
        assert "CIF" in dlg.status.text()
        # pymatgen-absent: Add CIF shows the install hint instead of a dialog
        try:
            import pymatgen  # noqa: F401
            has_pymatgen = True
        except Exception:
            has_pymatgen = False
        if not has_pymatgen:
            dlg._add_cif()
            assert "pymatgen" in dlg.status.text()
    finally:
        dlg.deleteLater()
        qapp.processEvents()


def test_phase_fit_fills_phase_table(qapp):
    """_fill_phase_table renders the per-phase fractions + lattice from a
    FitResultStore entry (synthetic — no pymatgen / lmfit)."""
    from xdart.gui.tabs.static_scan.phase_fit_dialog import PhaseFitDialog
    dlg = PhaseFitDialog(lambda: None)
    try:
        entry = {"phase_fractions": {"Ortho": 0.6, "Mono": 0.4},
                 "lattice_params": {"Ortho": {"a": 4.0, "b": 4.1, "c": 4.2},
                                    "Mono": {"a": 5.1}},
                 "success": True, "redchi": 1.2}

        class _Result:
            payload = [entry]

        class _Outcome:
            result = _Result()
            params = {}
            label = "x"           # non-int -> trend skipped, table still fills

        dlg._fill_phase_table(_Outcome())
        assert dlg.table.rowCount() == 2
        assert dlg.table.item(0, 0).text() == "Ortho"
        assert dlg.table.item(0, 1).text() == "0.600"
        assert dlg.table.item(0, 2).text() == "4.0000"
        assert dlg.table.item(1, 2).text() == "5.1000"   # Mono a
        assert "χ²" in dlg.status.text()
    finally:
        dlg.deleteLater()
        qapp.processEvents()


def test_phase_batch_does_not_crash_on_range(widget, qapp):
    """Regression: _run_batch_fit was peak-specific (called dlg._fit_range()),
    which PhaseFitDialog lacks -> AttributeError on Phase ▸ Batch. Both fit
    dialogs now expose batch_x_range() and the phase batch reaches the worker."""
    import time
    from types import SimpleNamespace
    pytest.importorskip("lmfit")
    w = widget

    def fake_pattern(idx):
        x = np.linspace(1.0, 5.0, 300)
        y = 1.0e4 * np.exp(-0.5 * ((x - 3.0) / 0.05) ** 2) + 1000.0
        return x, y, "q"

    w._pattern_for_frame = fake_pattern
    w.frame_ids = ["0"]
    w.scan.frames = SimpleNamespace(index=[0, 1])
    w._open_phase_fit_dialog()
    dlg = w._phase_fit_dialog
    # both dialogs honor the shared batch contract
    assert len(dlg.batch_x_range()) == 2
    assert dlg.batch_x_range()[0] == -np.inf      # phase = no x-window
    # inject fake phases so build_fit_request returns a request (no pymatgen)
    dlg.refresh_pattern()
    dlg._phases = [("a.cif", SimpleNamespace(name="A"))]
    w._run_batch_fit(dlg)                          # must NOT raise AttributeError
    worker = w._batch_analysis_worker
    if worker is not None:                         # reached the worker -> fix works
        deadline = time.monotonic() + 8
        while worker.isRunning() and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        worker.wait(2000)


# ── Scan Plot tool (metadata plotting — step 1) ───────────────────────────


def test_scan_plot_tool_wired_in_static_widget(widget):
    """Tools now exposes three active tools incl. Scan Plot; it builds lazily +
    non-modal."""
    from PySide6 import QtWidgets
    w = widget
    tools = [b for b in w.ui.metaFrame.findChildren(QtWidgets.QPushButton)
             if b.objectName() == "toolButton"]
    assert len(tools) == 3
    assert w._scan_plot_dialog is None
    w._open_scan_plot_dialog()
    assert w._scan_plot_dialog is not None and not w._scan_plot_dialog.isModal()


def test_scan_plot_dialog_columns_and_normalization(qapp):
    """set_table populates X/Y/Normalize from the numeric columns; checking Y
    plots a curve; a second Y overlays; normalization divides; non-numeric
    columns are excluded."""
    from pyqtgraph.Qt import QtCore
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog
    dlg = ScanPlotDialog()
    try:
        table = {
            "frame_index": np.arange(5, dtype=float),
            "theta": np.linspace(0.0, 2.0, 5),
            "i0": np.array([100.0, 110.0, 120.0, 130.0, 140.0]),
            "label": np.array(["a", "b", "c", "d", "e"], dtype=object),
        }
        dlg.set_table("scan.nxs", table)
        cols = [dlg.x_combo.itemText(i) for i in range(dlg.x_combo.count())]
        assert {"frame_index", "theta", "i0"} <= set(cols)
        assert "label" not in cols                  # non-numeric excluded
        # sensible defaults: X is a numeric column; exactly one Y checked -> 1 curve
        assert dlg.x_combo.currentText() in cols
        assert dlg.x_combo.currentText() != "label"
        assert len(dlg.plot.getPlotItem().listDataItems()) == 1
        # overlay: check every numeric column -> one curve per checked Y
        for c in cols:
            dlg.y_list.findItems(c, QtCore.Qt.MatchFlag.MatchExactly)[0].setCheckState(
                QtCore.Qt.CheckState.Checked)
        n_checked = sum(
            1 for i in range(dlg.y_list.count())
            if dlg.y_list.item(i).checkState() == QtCore.Qt.CheckState.Checked)
        assert n_checked == len(cols)
        assert len(dlg.plot.getPlotItem().listDataItems()) == n_checked
        # normalization is available + divides without crashing (curve count holds)
        norms = [dlg.norm_combo.itemText(i) for i in range(dlg.norm_combo.count())]
        assert "None" in norms and "i0" in norms
        dlg.norm_combo.setCurrentText("i0")
        assert len(dlg.plot.getPlotItem().listDataItems()) == n_checked
    finally:
        dlg.deleteLater()
        qapp.processEvents()


def test_scan_plot_dialog_empty_table_is_graceful(qapp):
    """No metadata -> no columns, no curves, Save disabled, no crash."""
    from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog
    dlg = ScanPlotDialog()
    try:
        dlg.set_table("empty", {})
        assert dlg.x_combo.count() == 0
        assert dlg.plot.getPlotItem().listDataItems() == []
        assert not dlg.save_btn.isEnabled()
    finally:
        dlg.deleteLater()
        qapp.processEvents()


def test_batch_fit_through_static_widget_populates_results(widget, qapp):
    """End-to-end Step 4: _run_batch_fit collects every frame's pattern, the
    worker fits them off the GUI thread, and _on_batch_done opens the populated
    vs-frame results popup — exercising the full signal wiring."""
    import time
    from types import SimpleNamespace
    pytest.importorskip("lmfit")
    w = widget

    def fake_pattern(idx):
        x = np.linspace(1.0, 5.0, 400)
        shift = 0.02 * int(idx)              # peak drifts with frame -> series varies
        y = (1.0e5 * np.exp(-0.5 * ((x - (2.0 + shift)) / 0.05) ** 2)
             + 6.0e4 * np.exp(-0.5 * ((x - 3.5) / 0.07) ** 2) + 2000.0)
        return x, y, "q (Å⁻¹)"

    w._pattern_for_frame = fake_pattern      # feed synthetic patterns to batch
    w.frame_ids = ["0"]
    w.scan.frames = SimpleNamespace(index=[0, 1, 2])

    w._open_peak_fit_dialog()
    dlg = w._peak_fit_dialog
    dlg.auto_check.setChecked(False)
    dlg.npeaks_spin.setValue(2)
    dlg.model_combo.setCurrentText("Gaussian")

    w._run_batch_fit(dlg)
    worker = w._batch_analysis_worker
    assert worker is not None
    deadline = time.monotonic() + 15
    while worker.isRunning() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    worker.wait(3000)
    qapp.processEvents()                     # deliver queued sigFrameFit/sigBatchDone

    # The trend filled the dialog's embedded row 3 incrementally (no popup).
    acc = dlg._param_accumulator
    assert set(acc.keys()) == {0, 1, 2}
    assert "center_0" in acc[0]
    # the first peak's recovered center is near 2.0 across frames
    assert all(abs(acc[i]["center_0"] - 2.0) < 0.1 for i in (0, 1, 2))
    assert len(dlg.param_plot.getPlotItem().listDataItems()) >= 1


def test_image_widget_colorbar_limits_nan_aware(qapp):
    """Regression: a NaN-masked frame must still display with percentile levels.

    pgImageWidget.update_image set the colorbar lo/hi limits with np.min/np.max,
    which return NaN on NaN-masked data (Image Viewer xdart frames).  NaN limits
    clamp the nanpercentile(1,99) levels to [NaN,NaN], so the image fell back to
    pyqtgraph autoscale (data min/max) — the "Image Viewer scaled to min/max"
    symptom.  Limits must be nan-aware (finite)."""
    from xdart.gui.widgets import pgImageWidget
    w = pgImageWidget(lockAspect=True, raw=True)
    try:
        img = np.arange(100, dtype=float).reshape(10, 10)
        img[0, 0] = np.nan                       # masked pixel
        w.setImage(img, scale="Linear", cmap="viridis")
        assert np.isfinite(w.histogram.lo_lim)
        assert np.isfinite(w.histogram.hi_lim)
    finally:
        w.deleteLater()
        qapp.processEvents()


def test_gi_1d_output_range_key_by_mode():
    """The GI 1D freeze must lock the range param that controls the *output*
    axis: q/q_ip → radial_range (ip), q_oop/exit_angle → azimuth_range (oop).
    Freezing radial_range for the oop/exit modes left their output axis drifting
    per incidence angle across an angle scan -> non-uniform 1D stack."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        _gi_1d_output_range_key,
    )
    assert _gi_1d_output_range_key('q_oop') == 'azimuth_range'
    assert _gi_1d_output_range_key('exit_angle') == 'azimuth_range'
    assert _gi_1d_output_range_key('q_ip') == 'radial_range'
    assert _gi_1d_output_range_key('q_total') == 'radial_range'
    assert _gi_1d_output_range_key('q') == 'radial_range'


def test_image_widget_linear_levels_are_2_98_percentile(qapp):
    # R2-4: the Linear autoscale clips harder than the old (1, 99) so saturated
    # tiff pixels don't wash the image out.  Shared raw/cake/waterfall widget.
    from xdart.gui.widgets import pgImageWidget
    w = pgImageWidget(lockAspect=True, raw=True)
    try:
        img = np.arange(100, dtype=float).reshape(10, 10)
        w.setImage(img, scale="Linear", cmap="viridis")
        lo, hi = w.imageItem.levels
        expected = np.nanpercentile(img, (2, 98))
        assert np.isclose(lo, expected[0]) and np.isclose(hi, expected[1])
        assert np.shares_memory(w.displayed_image, w.raw_image)
    finally:
        w.deleteLater()
        qapp.processEvents()


def test_image_widget_reuses_levels_for_short_live_flush(qapp, monkeypatch):
    from xdart.gui.widgets import pgImageWidget
    import xdart.gui.widgets.image_widget as image_widget_mod

    calls = []

    def _levels(displayed, raw, pct):
        calls.append((tuple(np.asarray(displayed).shape), tuple(pct)))
        return (0.0, 1.0)

    monkeypatch.setattr(image_widget_mod, "_ceiling_safe_levels", _levels)
    w = pgImageWidget(lockAspect=True, raw=True)
    try:
        img = np.arange(10000, dtype=float).reshape(100, 100)
        w.setImage(img, scale="Linear", cmap="viridis")
        w.update_image(scale="Linear", cmap="viridis")
        w.update_image(scale="Sqrt", cmap="viridis")

        assert calls == [((100, 100), (2, 98)), ((100, 100), (0.5, 99.9))]
    finally:
        w.deleteLater()
        qapp.processEvents()


def test_image_widget_level_cache_is_scoped_by_scan_token(qapp, monkeypatch):
    from xdart.gui.widgets import pgImageWidget
    import xdart.gui.widgets.image_widget as image_widget_mod

    calls = []

    def _levels(displayed, raw, pct):
        calls.append((float(np.asarray(raw).ravel()[0]), tuple(pct)))
        return (0.0, 1.0)

    monkeypatch.setattr(image_widget_mod, "_ceiling_safe_levels", _levels)
    w = pgImageWidget(lockAspect=True, raw=True)
    try:
        first = np.ones((20, 20), dtype=float)
        second_same_scan = np.full((20, 20), 2.0)
        third_new_scan = np.full((20, 20), 3.0)

        w.setImage(first, scale="Linear", cmap="viridis",
                   level_scan_token=("scan-a", 1))
        w.setImage(second_same_scan, scale="Linear", cmap="viridis",
                   level_scan_token=("scan-a", 1))
        w.setImage(third_new_scan, scale="Linear", cmap="viridis",
                   level_scan_token=("scan-b", 1))

        assert calls == [((1.0), (2, 98)), ((3.0), (2, 98))]

        fourth_unscoped = np.full((20, 20), 4.0)
        fifth_unscoped = np.full((20, 20), 5.0)
        w.setImage(fourth_unscoped, scale="Linear", cmap="viridis")
        w.setImage(fifth_unscoped, scale="Linear", cmap="viridis")

        assert calls == [
            (1.0, (2, 98)),
            (3.0, (2, 98)),
            (4.0, (2, 98)),
        ]
    finally:
        w.deleteLater()
        qapp.processEvents()


# ── Real-data cells: exercise the full _load_image_file (classify + load) ──

_TIFF = _DATA / "Tiff" / "Combi4_Angledependence_samz_4p9_03271002_0001.tif"
_EIGER = _DATA / "eiger" / "Eiger_B_ctrl_test__2000mdeg_scan001_master.h5"
_PROC = _DATA / "xdart_processed_data" / "Combi4_Angledependence_samz_4p9_03271002.nxs"


def _load_image_through_wire(w, path):
    """Drive the real Image-Viewer load: classify+load then set_data render."""
    w._on_viewer_mode_changed("image")
    w.h5viewer.dirname = str(Path(path).parent)
    w.h5viewer._load_image_file(str(path))
    w.set_data()                       # propagates classification + renders via payload
    return w.displayframe.image_data


@pytest.mark.skipif(not _TIFF.exists(), reason="tiff test data absent")
def test_image_viewer_real_tiff_renders(widget):
    data = _load_image_through_wire(widget, _TIFF)
    assert widget.h5viewer._viewer_is_xdart is False     # standalone detector file
    assert data is not None and data[0] is not None
    assert np.isfinite(data[0]).any()


@pytest.mark.skipif(not _EIGER.exists(), reason="eiger test data absent")
def test_image_viewer_real_eiger_master_renders(widget):
    data = _load_image_through_wire(widget, _EIGER)
    assert data is not None and data[0] is not None
    assert np.isfinite(data[0]).any()


@pytest.mark.skipif(not _PROC.exists(), reason="processed nxs test data absent")
def test_image_viewer_real_processed_nxs_is_xdart_and_renders(widget):
    # P0 on real data: a processed .nxs classifies as xdart through the wire
    # and renders a frame (its baked mask, if any, is preserved by the
    # synthetic mask test above).
    data = _load_image_through_wire(widget, _PROC)
    assert widget.h5viewer._viewer_is_xdart is True
    assert widget.displayframe._viewer_is_xdart is True   # propagated at set_data
    assert data is not None and data[0] is not None and data[0].size > 0


# ── Layout table (Stage 4/5 step 1): `_apply_layout(mode)` geometry ────────
#
# Panel geometry is now a pure, idempotent function of Mode (PANEL_LAYOUT).
# These drive the REAL staticWidget into each mode through the production entry
# points and assert every managed widget's visibility + (min,max) height/width
# matches the table — the per-mode invariant — and that the *destination*
# geometry is correct regardless of origin — the leak class that produced the
# Int 1D (XYE) -> Image Viewer blank (twoDWindow stuck at maximumHeight 0).

from xdart.gui.tabs.static_scan.display_logic import Mode, PANEL_LAYOUT

_VIEWER_STR = {
    Mode.IMAGE_VIEWER: "image",
    Mode.XYE_VIEWER: "xye",
    Mode.NEXUS_VIEWER: "nexus",
}
_ALL_MODES = [Mode.INT_1D, Mode.INT_2D, Mode.IMAGE_VIEWER,
              Mode.XYE_VIEWER, Mode.NEXUS_VIEWER]


def _enter(w, mode):
    """Drive the widget into ``mode`` through the production routing.

    Viewer modes go through ``set_viewer_display_mode``; the Int processing
    modes go through ``set_viewer_display_mode(None)`` (the normal-restore path
    a user hits when switching a viewer back to Int 1D/2D), which itself routes
    geometry through ``_apply_1d_only_visibility`` → ``_apply_layout``.
    """
    df = w.displayframe
    if mode in _VIEWER_STR:
        df.set_viewer_display_mode(_VIEWER_STR[mode])
    else:
        df.scan.skip_2d = (mode is Mode.INT_1D)
        df.set_viewer_display_mode(None)
    return df


def _geom(df):
    """Live geometry snapshot of every managed widget.

    Visibility uses ``isHidden()`` (the widget's own explicit flag, independent
    of whether the offscreen top-level was ever shown), which is exactly what
    the table sets via ``setVisible``.
    """
    ui = df.ui
    return {
        "frame_top_vis": not ui.frame_top.isHidden(),
        "twoDWindow_vis": not ui.twoDWindow.isHidden(),
        "imageToolbar_vis": not ui.imageToolbar.isHidden(),
        "frame_4_vis": not ui.frame_4.isHidden(),
        "frame_6_vis": not ui.frame_6.isHidden(),
        "plotToolBar_vis": not ui.plotToolBar.isHidden(),
        "show_image_btn_vis": not df._showImageBtn.isHidden(),
        "twoDWindow_h": (ui.twoDWindow.minimumHeight(), ui.twoDWindow.maximumHeight()),
        "imageWindow_h": (ui.imageWindow.minimumHeight(), ui.imageWindow.maximumHeight()),
        "plotWindow_h": (ui.plotWindow.minimumHeight(), ui.plotWindow.maximumHeight()),
        "imageToolbar_h": (ui.imageToolbar.minimumHeight(), ui.imageToolbar.maximumHeight()),
        "plotToolBar_h": (ui.plotToolBar.minimumHeight(), ui.plotToolBar.maximumHeight()),
        "binnedFrame_w": (ui.binnedFrame.minimumWidth(), ui.binnedFrame.maximumWidth()),
    }


def _expected(mode):
    """The geometry the table says ``mode`` must produce."""
    s = PANEL_LAYOUT[mode]
    return {
        "frame_top_vis": s.frame_top_vis,
        "twoDWindow_vis": s.twoDWindow_vis,
        "imageToolbar_vis": s.imageToolbar_vis,
        "frame_4_vis": s.frame_4_vis,
        "frame_6_vis": s.frame_6_vis,
        "plotToolBar_vis": s.plotToolBar_vis,
        "show_image_btn_vis": s.show_image_btn_vis,
        "twoDWindow_h": tuple(s.twoDWindow_h),
        "imageWindow_h": tuple(s.imageWindow_h),
        "plotWindow_h": tuple(s.plotWindow_h),
        "imageToolbar_h": tuple(s.imageToolbar_h),
        "plotToolBar_h": tuple(s.plotToolBar_h),
        "binnedFrame_w": tuple(s.binnedFrame_w),
    }


@pytest.mark.parametrize("mode", _ALL_MODES, ids=lambda m: m.name)
def test_layout_per_mode_matches_table(widget, mode):
    """Each mode drives the production entry point to the full table geometry."""
    df = _enter(widget, mode)
    assert _geom(df) == _expected(mode)


@pytest.mark.parametrize("mode", _ALL_MODES, ids=lambda m: m.name)
def test_layout_idempotent(widget, mode):
    """Applying a mode twice is a no-op vs once (no reliance on prior state)."""
    df = widget.displayframe
    df._apply_layout(mode)
    once = _geom(df)
    df._apply_layout(mode)
    twice = _geom(df)
    assert once == twice == _expected(mode)


# Origin→destination cells that matter (the leak class): the destination
# geometry must be correct no matter where we came from.
_TRANSITIONS = [
    (Mode.INT_1D, Mode.IMAGE_VIEWER),   # the original blank: twoDWindow restored
    (Mode.INT_1D, Mode.NEXUS_VIEWER),   # 2D pane also needed by nexus
    (Mode.XYE_VIEWER, Mode.IMAGE_VIEWER),
    (Mode.IMAGE_VIEWER, Mode.INT_2D),
    (Mode.INT_2D, Mode.IMAGE_VIEWER),
    (Mode.IMAGE_VIEWER, Mode.INT_1D),   # viewer → normal (1D-only)
    (Mode.NEXUS_VIEWER, Mode.INT_2D),   # viewer → normal (full 2D)
]


@pytest.mark.parametrize(
    "origin,dest", _TRANSITIONS,
    ids=[f"{o.name}->{d.name}" for o, d in _TRANSITIONS])
def test_layout_transition_destination_geometry(widget, origin, dest):
    """The destination geometry is independent of the origin mode."""
    _enter(widget, origin)
    df = _enter(widget, dest)
    assert _geom(df) == _expected(dest)
    # The exact leak that caused the blank: a 2D-pane mode reached from a
    # 1D-only origin must have a non-zero twoDWindow height.
    if dest in (Mode.IMAGE_VIEWER, Mode.NEXUS_VIEWER, Mode.INT_2D):
        assert df.ui.twoDWindow.maximumHeight() > 0


def test_processing_mode_change_uses_selected_text_not_stale_skip(widget, qapp):
    """Fresh-start mode changes must not depend on the previous skip_2d flag.

    The static-widget mode handler is connected before the wrangler's handler,
    so it must seed ``scan.skip_2d`` from the selected text before applying the
    layout.  Otherwise a fresh startup can show the Int 1D panel for Int 2D, and
    vice versa, until a scan run updates the flag through another path.
    """
    w = widget
    df = w.displayframe

    _set_processing_mode(w, "Int 1D")
    qapp.processEvents()
    assert _geom(df) == _expected(Mode.INT_1D)

    # Simulate the stale fresh-start state: the selected text is about to switch
    # to Int 2D, but all scan objects still say 1D-only.
    w.scan.skip_2d = True
    df.scan.skip_2d = True
    getattr(w.wrangler, "scan", w.scan).skip_2d = True
    _set_processing_mode(w, "Int 2D")
    qapp.processEvents()
    assert w.scan.skip_2d is False
    assert df.scan.skip_2d is False
    assert _geom(df) == _expected(Mode.INT_2D)

    # And the reverse: switching to Int 1D while the scan objects still say 2D.
    w.scan.skip_2d = False
    df.scan.skip_2d = False
    getattr(w.wrangler, "scan", w.scan).skip_2d = False
    _set_processing_mode(w, "Int 1D")
    qapp.processEvents()
    assert w.scan.skip_2d is True
    assert df.scan.skip_2d is True
    assert _geom(df) == _expected(Mode.INT_1D)


def test_layout_render_path_int_1d_only_is_self_sufficient(widget):
    """The per-frame render path (``_apply_1d_only_visibility`` alone, without
    going through ``set_viewer_display_mode``) must itself produce the FULL
    INT_1D geometry.

    This is the path that runs on every frame update, and the original blank
    proved it was *not* self-sufficient (it left plotWindow implicit and never
    reasserted a previously-collapsed twoDWindow).  Lock it: from a viewer
    origin, switching the processing mode and running only the render-path
    visibility hook lands the complete table."""
    df = widget.displayframe
    df.set_viewer_display_mode("image")        # a 2D-pane viewer origin
    df.viewer_mode = None
    df.scan.skip_2d = True
    df._apply_1d_only_visibility()             # render-path entry only
    assert _geom(df) == _expected(Mode.INT_1D)


# ── XYE Viewer payload-only rendering (Stage 4/5 step 3, Part A) ───────────
#
# XYE renders through XYEViewerController.build_payload -> PlotPayload; these
# drive the real wire (set frames -> update -> render_display) and assert the
# plot draws with the filename-derived axis and that Overlay accumulation +
# the mixed-unit warning are preserved (no legacy _update_xye_viewer).

def _set_xye_frames(w, frames):
    """Load synthetic 1D viewer frames + select them all.

    ``frames``: list of ``(idx, source_file, radial, intensity)``.
    """
    with w.data_lock:
        w.viewer_rows_1d.clear()
        w.viewer_rows_2d.clear()
        for idx, src, radial, intensity in frames:
            w.viewer_rows_1d[idx] = SimpleNamespace(
                int_1d=SimpleNamespace(
                    radial=np.asarray(radial, dtype=float),
                    intensity=np.asarray(intensity, dtype=float)),
                scan_info={"source_file": src},
            )
    w.frame_ids[:] = [str(idx) for idx, *_ in frames]
    w.displayframe.idxs_1d = [idx for idx, *_ in frames]


def test_xye_viewer_single_curve_uses_filename_axis(widget):
    w = widget
    w._on_viewer_mode_changed("xye")
    df = w.displayframe
    _set_xye_frames(w, [(0, "iq_sample.xye", np.arange(5.0),
                         np.array([1.0, 2.0, 3.0, 2.0, 1.0]))])
    from xdart.gui.tabs.static_scan.display_logic import x_axis_for_unit
    df.update()
    assert df.frame_names == ["iq_sample.xye"]
    assert df.plot_data[1].shape[0] == 1                 # one curve drawn
    # x-axis label comes from the iq prefix (Q), not the hidden transform combo.
    assert df._current_plot_axis_label() == x_axis_for_unit("q_A^-1")


def test_xye_viewer_multiselect_draws_all(widget):
    w = widget
    w._on_viewer_mode_changed("xye")
    df = w.displayframe
    _set_xye_frames(w, [
        (0, "iq_a.xye", np.arange(5.0), np.ones(5)),
        (1, "iq_b.xye", np.arange(5.0), 2.0 * np.ones(5)),
    ])
    df.update()
    assert set(df.frame_names) == {"iq_a.xye", "iq_b.xye"}
    assert df.plot_data[1].shape[0] == 2                 # both selected drawn


def test_xye_viewer_empty_selection_clears_plot(widget):
    w = widget
    w._on_viewer_mode_changed("xye")
    df = w.displayframe
    _set_xye_frames(w, [(0, "iq_a.xye", np.arange(5.0), np.ones(5))])
    df.update()
    assert df.plot_data[1].shape[0] == 1
    # Deselect everything -> the plot must clear (no stale curve).
    with w.data_lock:
        w.viewer_rows_1d.clear()
        w.viewer_rows_2d.clear()
    w.frame_ids[:] = []
    df.idxs_1d = []
    df.update()
    assert len(df.curves) == 0


def test_xye_viewer_selection_equals_shown_deselect_removes_curve(widget):
    # selection == shown: in Overlay the plot draws exactly the selected files,
    # so deselecting one removes its curve immediately (no lingering).
    w = widget
    w._on_viewer_mode_changed("xye")
    df = w.displayframe
    df.ui.plotMethod.setCurrentText("Overlay")
    _set_xye_frames(w, [
        (0, "iq_a.xye", np.arange(5.0), np.ones(5)),
        (1, "iq_b.xye", np.arange(5.0), 2.0 * np.ones(5)),
    ])
    df.update()
    assert set(df.frame_names) == {"iq_a.xye", "iq_b.xye"}
    # Deselect b: only a is selected now -> only a is drawn (b's curve is gone).
    _set_xye_frames(w, [(0, "iq_a.xye", np.arange(5.0), np.ones(5))])
    df.update()
    assert df.frame_names == ["iq_a.xye"]
    assert df.plot_data[1].shape[0] == 1


def test_xye_viewer_mode_uses_extended_selection(widget):
    # E1: ExtendedSelection restores arrow-key single-browse (the plot follows);
    # a plain-click overlay toggle is layered on via the click filter.
    from PySide6.QtWidgets import QAbstractItemView
    w = widget
    w._on_viewer_mode_changed("xye")
    assert (w.h5viewer.ui.listScans.selectionMode()
            == QAbstractItemView.ExtendedSelection)
    # Image/NeXus stay single-select.
    w._on_viewer_mode_changed("image")
    assert (w.h5viewer.ui.listScans.selectionMode()
            == QAbstractItemView.SingleSelection)


def _xye_list(tmp_path, w, names=("iq_a.xye", "iq_b.xye", "iq_c.xye"), method="Single"):
    """Set up the XYE file list in a given plotMethod; return the list widget."""
    for n in names:
        (tmp_path / n).write_text("0 1\n1 2\n")
    h5v = w.h5viewer
    h5v.dirname = str(tmp_path)
    w._on_viewer_mode_changed("xye")
    pm = w.displayframe.ui.plotMethod
    pm.blockSignals(True)
    pm.setCurrentText(method)
    pm.blockSignals(False)
    h5v.update_scans()
    return h5v.ui.listScans


def _click_item(lw, text):
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    for row in range(lw.count()):
        it = lw.item(row)
        if it.text() == text:
            QTest.mouseClick(lw.viewport(), Qt.LeftButton, Qt.NoModifier,
                             lw.visualItemRect(it).center())
            return
    pytest.skip(f"{text} not listed")


def test_xye_plain_click_toggles_overlay_in_accumulating_mode(tmp_path, widget):
    # E1/E2: in an accumulating method, a plain left-click toggles a file
    # into/out of the overlay; a third click removes it.
    w = widget
    lw = _xye_list(tmp_path, w, method="Overlay")
    _click_item(lw, "iq_a.xye")
    assert {i.text() for i in lw.selectedItems()} == {"iq_a.xye"}
    _click_item(lw, "iq_b.xye")                    # plain click ADDS
    assert {i.text() for i in lw.selectedItems()} == {"iq_a.xye", "iq_b.xye"}
    _click_item(lw, "iq_b.xye")                    # plain click REMOVES
    assert {i.text() for i in lw.selectedItems()} == {"iq_a.xye"}


def test_xye_plain_click_replaces_in_single_mode(tmp_path, widget):
    # E2: in Single mode a plain click browses one file (replace, not toggle).
    w = widget
    lw = _xye_list(tmp_path, w, method="Single")
    _click_item(lw, "iq_a.xye")
    assert {i.text() for i in lw.selectedItems()} == {"iq_a.xye"}
    _click_item(lw, "iq_b.xye")                    # replaces, not adds
    assert {i.text() for i in lw.selectedItems()} == {"iq_b.xye"}


def _arrow(lw, key):
    from PySide6.QtTest import QTest
    QTest.keyClick(lw, key)


def test_xye_arrow_extends_selection_in_accumulating_mode(tmp_path, widget):
    # E2: in an accumulating method, Up/Down arrows EXTEND the selection so
    # arrow-browsing builds the comparison set.
    from PySide6.QtCore import Qt
    w = widget
    lw = _xye_list(tmp_path, w, method="Overlay")
    # Start on the first data row.
    first = next(lw.item(r) for r in range(lw.count())
                 if lw.item(r).text() == "iq_a.xye")
    first.setSelected(True)
    lw.setCurrentItem(first)
    _arrow(lw, Qt.Key_Down)                        # -> iq_b added
    _arrow(lw, Qt.Key_Down)                        # -> iq_c added
    assert {i.text() for i in lw.selectedItems()} == {"iq_a.xye", "iq_b.xye", "iq_c.xye"}


def test_xye_arrow_browses_single_in_single_mode(tmp_path, widget):
    # E2: in Single mode arrows move one row (Qt default replace), not extend.
    from PySide6.QtCore import Qt
    w = widget
    lw = _xye_list(tmp_path, w, method="Single")
    first = next(lw.item(r) for r in range(lw.count())
                 if lw.item(r).text() == "iq_a.xye")
    first.setSelected(True)
    lw.setCurrentItem(first)
    _arrow(lw, Qt.Key_Down)
    sel = {i.text() for i in lw.selectedItems()}
    assert sel == {"iq_b.xye"}                     # moved + replaced, single row


def test_xye_entry_has_no_default_overlay(tmp_path, widget):
    # E1: entering XYE shows the current row (or nothing), not a default overlay.
    w = widget
    for n in ("iq_a.xye", "iq_b.xye"):
        (tmp_path / n).write_text("0 1\n1 2\n")
    h5v = w.h5viewer
    h5v.dirname = str(tmp_path)
    w._on_viewer_mode_changed("xye")
    h5v.update_scans()
    assert len(h5v.ui.listScans.selectedItems()) <= 1


def test_xye_live_refresh_preserves_multiselection(tmp_path, widget):
    # Real-time use case: as new .xye files arrive and the scans list
    # repopulates, the user's current multi-selection must survive the rebuild.
    w = widget
    (tmp_path / "iq_a.xye").write_text("0 1\n1 2\n")
    (tmp_path / "iq_b.xye").write_text("0 1\n1 2\n")
    h5v = w.h5viewer
    h5v.dirname = str(tmp_path)
    w._on_viewer_mode_changed("xye")
    h5v.update_scans()
    lw = h5v.ui.listScans
    # Select A and B (modifier-free multi-select).
    for row in range(lw.count()):
        if lw.item(row).text() in ("iq_a.xye", "iq_b.xye"):
            lw.item(row).setSelected(True)
    assert {i.text() for i in lw.selectedItems()} == {"iq_a.xye", "iq_b.xye"}

    # A new file C lands -> the list rebuilds; A and B must stay selected.
    (tmp_path / "iq_c.xye").write_text("0 1\n1 2\n")
    h5v.update_scans()
    assert {i.text() for i in lw.selectedItems()} == {"iq_a.xye", "iq_b.xye"}


def test_xye_viewer_mixed_units_warns_and_labels_from_first(widget, caplog):
    import logging
    w = widget
    w._on_viewer_mode_changed("xye")
    df = w.displayframe
    _set_xye_frames(w, [
        (0, "iq_a.xye", np.arange(5.0), np.ones(5)),       # Q
        (1, "itth_b.xye", np.arange(5.0), np.ones(5)),     # 2theta
    ])
    from xdart.gui.tabs.static_scan.display_logic import x_axis_for_unit
    with caplog.at_level(logging.WARNING):
        df.update()
    assert any("mixes different x-axis units" in r.message for r in caplog.records)
    # Axis labelled from the FIRST file (Q), not the mixed-in 2theta.
    assert df._current_plot_axis_label() == x_axis_for_unit("q_A^-1")


# ── NeXus Viewer payload-only rendering (Stage 4/5 step 3, Part B) ─────────

def _set_nexus_row(w, idx, payload):
    """Load one NeXus preview row + select it."""
    with w.data_lock:
        w.viewer_rows_1d.clear()
        w.viewer_rows_2d.clear()
        w.viewer_rows_1d[idx] = SimpleNamespace(
            nexus_preview_payload=payload, scan_info={})
    w.frame_ids[:] = [str(idx)]
    w.displayframe.idxs_1d = [idx]


def test_nexus_viewer_plot_row_draws_plot_clears_image(widget):
    w = widget
    w._on_viewer_mode_changed("nexus")
    df = w.displayframe
    _set_nexus_row(w, 0, {
        "kind": "plot_1d",
        "x": np.arange(4.0), "y": np.array([1.0, 2.0, 3.0, 4.0]),
        "label": "I(q)", "x_label": "Q", "x_unit": "A^-1",
    })
    df.update()
    assert df.plot_data[1].shape[0] == 1                  # 1D preview drawn
    assert df._current_plot_axis_label() == ("Q", "A^-1")  # dataset units/label
    assert df.image_data is None                          # 2D panel cleared


def test_nexus_viewer_image_row_draws_image_clears_plot(widget):
    w = widget
    w._on_viewer_mode_changed("nexus")
    df = w.displayframe
    _set_nexus_row(w, 0, {
        "kind": "image_2d",
        "image": np.arange(12.0).reshape(3, 4),
    })
    df.update()
    assert df.image_data is not None                      # 2D preview drawn
    assert df.image_data[0].size > 0
    assert len(df.curves) == 0                            # plot cleared


def test_nexus_viewer_metadata_row_clears_both(widget):
    w = widget
    w._on_viewer_mode_changed("nexus")
    df = w.displayframe
    # A metadata-only row (neither plot_1d nor image_2d) blanks both panels.
    _set_nexus_row(w, 0, {"kind": "dataset", "path": "/entry/instrument"})
    df.update()
    assert df.image_data is None                          # image blanked
    assert df.image_widget.histogram.isVisible() is False  # colorbar hidden
    assert len(df.curves) == 0                            # plot blanked


# ── Int 2D cake: payload-authoritative + the imageUnit Q↔2θ toggle (step 4-1) ─
#
# CAKE_2D renders solely from cake_image now (update_binned deleted), and
# cake_image owns the 2D-unit (imageUnit) Q↔2θ conversion, so the cake unit is
# consistent on every render and the toggle re-renders through the payload.

def _set_int_scan(w, *, n=1, wavelength_m=0.7293e-10):
    """Populate the real widget for an Int 2D scan: viewer_rows_2d + viewer_rows_1d +
    publications + a scan stub, selected, in Int 2D mode (non-GI, q-integrated)."""
    import threading
    from xrd_tools.core import IntegrationResult1D, IntegrationResult2D
    from xdart.modules.frame_publication import publication_from_live_frame
    df = w.displayframe
    q = np.linspace(0.5, 3.0, 5)
    chi = np.linspace(-90.0, 90.0, 4)
    w.publication_store.clear()
    with w.data_lock:
        w.viewer_rows_1d.clear()
        w.viewer_rows_2d.clear()
        for i in range(n):
            f = SimpleNamespace()
            f.idx = i
            f.gi = False
            f.scan_info = {}
            f.source_file = f"scan_{i}.tif"
            f.source_frame_idx = 0
            f.map_raw = (np.arange(36, dtype=float).reshape(6, 6) + i)
            f.bg_raw = 0
            f.mask = None
            f.gi_2d = {}
            f.thumbnail = None
            f.int_1d = IntegrationResult1D(
                radial=q, intensity=np.ones_like(q) + i,
                sigma=np.ones_like(q), unit="q_A^-1")
            f.int_2d = IntegrationResult2D(
                radial=q, azimuthal=chi,
                intensity=np.ones((q.size, chi.size)) + i,
                unit="q_A^-1", azimuthal_unit="chi_deg")
            f._get_incident_angle = lambda: 0.2
            w.viewer_rows_2d[i] = {"map_raw": f.map_raw, "bg_raw": 0, "mask": None,
                            "int_2d": f.int_2d, "gi_2d": {}, "thumbnail": None}
            w.viewer_rows_1d[i] = f
            w.publication_store.upsert(publication_from_live_frame(f))
    w.frame_ids[:] = [str(i) for i in range(n)]
    df.idxs_2d = list(range(n))
    df.idxs_1d = list(range(n))
    df.scan = SimpleNamespace(
        scan_lock=threading.RLock(),
        frames=SimpleNamespace(index=list(range(n))),
        gi=False, skip_2d=False, name="scan", global_mask=None,
        # Both bai_*_args must be present: _apply_1d_only_visibility reads
        # scan.bai_1d_args even on the non-GI (else) path.  Omitting it made the
        # display crash with AttributeError whenever a prior test left the
        # display in 1d-only mode (green in the full suite only because other
        # tests happened to keep it out of that path; red running this file
        # alone, as CI does via `pytest tests/xdart`).  A real scan always has
        # both, so this completes the mock rather than guarding production.
        scan_data=SimpleNamespace(columns=[]), bai_1d_args={}, bai_2d_args={},
        series_average=False, single_img=False,
        mg_args={"wavelength": wavelength_m})
    df.viewer_mode = None
    # Populate the 2D-unit combo with the non-GI Q-χ / 2θ-χ entries (the real
    # non-GI scan-load flow does this; the stub above bypasses it).
    from xdart.gui.tabs.static_scan.display_constants import imageUnits
    df.ui.imageUnit.blockSignals(True)
    df.ui.imageUnit.clear()
    for u in imageUnits:
        df.ui.imageUnit.addItem(u)
    df.ui.imageUnit.setCurrentIndex(0)
    df.ui.imageUnit.blockSignals(False)
    return df


def test_int2d_cake_draws_via_payload(widget):
    w = widget
    df = _set_int_scan(w, n=1)
    df.update()
    assert df.binned_data is not None              # cake drawn via cake_image
    assert df.binned_data[0].size > 0


def test_int2d_imageunit_toggle_reexpresses_cake(widget):
    w = widget
    df = _set_int_scan(w, n=1)                      # q-integrated cake
    df.ui.imageUnit.setCurrentIndex(0)             # Q-χ
    df.update()
    q_rect = df.binned_data[1]
    df.ui.imageUnit.setCurrentIndex(1)             # 2θ-χ
    df.update()
    tth_rect = df.binned_data[1]
    assert q_rect is not None and tth_rect is not None
    # The radial (x) extent changes: q (0.5..3.0) -> 2θ via the wavelength.
    assert abs(q_rect.width() - tth_rect.width()) > 1e-6


def test_int2d_cake_clears_without_publication(widget):
    # The deletion's only behavior change: a 2D panel with data but no
    # publication blanks (payload-only, no legacy update_binned fallback).
    # The store-sync guarantee makes this unreachable for clean frames; this
    # forces it to lock the clear-on-None contract.
    w = widget
    df = _set_int_scan(w, n=1)
    df.update()
    assert df.binned_data is not None
    w.publication_store.clear()                     # viewer_rows_2d kept, publications gone
    df.update()
    assert df.binned_data is None                   # cake blanked, not stale


def test_int_scan_resolves_loaded_rows_from_publication_store(widget):
    # Phase A: the old viewer_rows_1d/viewer_rows_2d mirrors are bounded recent-row caches.
    # A frame still resident in PublicationStore must look loaded even when the
    # legacy mirrors have evicted it, otherwise the update gate would falsely
    # blank a valid scan frame.
    w = widget
    df = _set_int_scan(w, n=2)
    with w.data_lock:
        w.viewer_rows_1d.clear()
        w.viewer_rows_2d.clear()

    w.frame_ids[:] = ["0", "1"]
    df.frame_ids = ["0", "1"]
    df.get_idxs()

    assert df.idxs_1d == [0, 1]
    assert df.idxs_2d == [0, 1]

    df.update()

    assert len(df.curves) == 2
    assert df.binned_data is not None


def test_viewer_rows_are_bounded_and_mode_stable(widget):
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _VIEWER_ROWS_1D_CACHE_MAX,
    )

    w = widget
    assert getattr(w.viewer_rows_1d, "_max", None) == _VIEWER_ROWS_1D_CACHE_MAX

    for i in range(_VIEWER_ROWS_1D_CACHE_MAX + 2):
        w.viewer_rows_1d[i] = SimpleNamespace(idx=i)
    assert len(w.viewer_rows_1d) == _VIEWER_ROWS_1D_CACHE_MAX
    assert 0 not in w.viewer_rows_1d

    w._on_viewer_mode_changed("xye")
    assert getattr(w.viewer_rows_1d, "_max", None) == _VIEWER_ROWS_1D_CACHE_MAX

    w._on_viewer_mode_changed("")
    assert getattr(w.viewer_rows_1d, "_max", None) == _VIEWER_ROWS_1D_CACHE_MAX


# ── INT 1D plot characterization before update_plot cleanup ────────────────
#
# The normal INT plot is update_plot-canonical.  These tests lock the computed
# render state (arrays/names/ranges/labels), not pixels, so the cleanup remains
# behavior-preserving.

def _plot_state(df):
    x, y = df.plot_data
    y = np.asarray(y)
    return {
        "x": np.asarray(x, dtype=float).copy(),
        "y": np.asarray(y, dtype=float).copy(),
        "names": tuple(df.frame_names),
        "range": tuple(tuple(float(v) for v in row)
                       for row in df.plot_data_range),
        "x_axis": df._current_plot_axis_label(),
        "curves": len(df.curves),
        "waterfall": bool(df.plotMethod == "Waterfall"
                          and (y.shape[0] if y.ndim > 1 else 1) > 3),
    }


@pytest.mark.parametrize("method, frame_ids", [
    ("Single", ["0"]),
    ("Single", ["0", "1", "2"]),
    ("Sum", ["0", "1", "2"]),
    ("Average", ["0", "1", "2"]),
])
def test_int_plot_native_update_plot_state(widget, method, frame_ids):
    from xdart.gui.tabs.static_scan.display_constants import AA_inv
    w = widget
    df = _set_int_scan(w, n=3)
    df.ui.plotMethod.setCurrentText(method)
    df.ui.plotUnit.setCurrentIndex(0)  # native Q axis
    w.frame_ids[:] = list(frame_ids)
    df.frame_ids = list(frame_ids)
    df.idxs_1d = [int(i) for i in frame_ids]
    df.idxs_2d = [int(i) for i in frame_ids]

    df.update()

    state = _plot_state(df)
    expected_ids = [int(i) for i in frame_ids]
    expected_x = np.linspace(0.5, 3.0, 5)
    expected_y = np.vstack([np.ones_like(expected_x) + i for i in expected_ids])
    expected_names = tuple(f"scan_{i}" for i in expected_ids)
    np.testing.assert_allclose(state["x"], expected_x)
    np.testing.assert_allclose(state["y"], expected_y)
    assert state["names"] == expected_names
    assert state["x_axis"] == ("Q", AA_inv)
    assert state["curves"] == (1 if method in ("Sum", "Average") else len(expected_ids))


@pytest.mark.parametrize("method", ["Average", "Sum"])
def test_evicted_whole_scan_aggregate_without_disk_blanks_not_subset(widget, method):
    # Final Role-A cleanup: when the bounded store evicts old frames but no
    # on-disk aggregate exists (this fixture has no data_file), Sum/Average must
    # NOT fall back to stale viewer_rows_1d rows or the one store-resident row.  The
    # production long-scan path is covered by test_aggregation_wiring.py and
    # routes through _whole_scan_aggregate; this stub-scan path blanks/defer.
    from xdart.modules.frame_publication import _semilight_publication
    w = widget
    df = _set_int_scan(w, n=3)
    # Evict frames 0,1 from the STORE (drop their 1D arrays) but leave viewer_rows_1d
    # intact.  viewer_rows_1d is now a transitional/viewer mirror, not an integration
    # display authority.
    store = df.publication_store
    with store._lock:
        for i in (0, 1):
            store._items[i] = _semilight_publication(store._items[i])
    assert not store.get(0).view.has_1d and not store.get(1).view.has_1d
    assert df.viewer_rows_1d[0].int_1d is not None       # stale mirror still has it

    df.ui.plotMethod.setCurrentText(method)
    df.ui.plotUnit.setCurrentIndex(0)             # native Q
    w.frame_ids[:] = ["0", "1", "2"]
    df.frame_ids = ["0", "1", "2"]
    df.idxs_1d = [0, 1, 2]
    df.idxs_2d = [0, 1, 2]

    df.update()

    # No wrong subset curve from frame 2 and no stale mirror aggregate.
    assert len(df.curves) == 0
    assert df.plot_data is not None
    assert np.size(df.plot_data[0]) == 0
    assert np.size(df.plot_data[1]) == 0


def test_evicted_overall_cake_blanks_not_subset(widget):
    # §2.C (review_2026-06-15, P1): an Average/Sum Overall cake must NOT silently
    # average only the store-resident subset when an intended frame was evicted.
    # CAKE_2D has no legacy fallback — a None payload BLANKS — and the on-disk
    # aggregate is unavailable here (stub scan, no data_file), so it blanks rather
    # than draw a wrong subset average.  (Overlay/Single show the current frame,
    # not the aggregate — that path is exercised elsewhere; this is the Average
    # path where the §2.C guard + aggregate fallback apply.)
    from xdart.modules.frame_publication import _semilight_publication
    w = widget
    df = _set_int_scan(w, n=3)            # all 3 selected -> Overall
    df.ui.plotMethod.setCurrentText("Average")   # aggregate path (not current-frame)
    df.update()
    assert df.binned_data is not None     # all resident -> cake drawn (average)
    store = df.publication_store
    with store._lock:                     # evict ONE intended frame from the store
        store._items[0] = _semilight_publication(store._items[0])
    assert not store.get(0).view.has_2d

    df.update()

    assert df.binned_data is None         # blanked, not a 2-frame subset average


def test_average_all_nan_column_does_not_warn(widget):
    # Vivek: Average/Sum over GI frames with an all-NaN q-bin (empty/padded bins)
    # warned "Mean of empty slice" at the update_1d_view collapse.  The collapse
    # must use nanmean_slice (no warning; the NaN gap is kept).
    import warnings as _w
    from xrd_tools.core import IntegrationResult1D
    from xdart.modules.frame_publication import publication_from_live_frame
    w = widget
    df = _set_int_scan(w, n=3)
    q = np.linspace(0.5, 3.0, 5)
    w.publication_store.clear()
    with w.data_lock:
        for i in range(3):
            inten = np.ones_like(q) + i
            inten[2] = np.nan                       # q-bin 2 all-NaN across frames
            f = w.viewer_rows_1d[i]
            f.int_1d = IntegrationResult1D(
                radial=q, intensity=inten, sigma=np.ones_like(q), unit="q_A^-1")
            w.publication_store.upsert(publication_from_live_frame(f))
    df.ui.plotMethod.setCurrentText("Average")
    df.ui.plotUnit.setCurrentIndex(0)
    w.frame_ids[:] = ["0", "1", "2"]
    df.frame_ids = ["0", "1", "2"]
    df.idxs_1d = [0, 1, 2]
    df.idxs_2d = [0, 1, 2]
    with _w.catch_warnings():
        _w.filterwarnings("error", message="Mean of empty slice")
        df.update()
    _cx, cy = df.curves[0].getData()
    cy = np.asarray(cy, dtype=float)
    assert np.isnan(cy[2])                           # empty bin stays a gap
    np.testing.assert_allclose(cy[[0, 1, 3, 4]], 2.0)  # others = mean(1,2,3)


@pytest.mark.parametrize("method", ["Overlay", "Waterfall"])
def test_int_plot_accumulating_modes_characterize_update_plot_state(widget, method):
    # Overlay/Waterfall are intentionally update_plot-live today because they
    # own cross-render history.  Lock append + duplicate-skip state.
    w = widget
    df = _set_int_scan(w, n=3)
    df.ui.plotMethod.setCurrentText(method)
    df.ui.plotUnit.setCurrentIndex(0)
    for frame_id in ("0", "1", "1", "2"):
        w.frame_ids[:] = [frame_id]
        df.frame_ids = [frame_id]
        df.idxs_1d = [int(frame_id)]
        df.idxs_2d = [int(frame_id)]
        df.update()

    state = _plot_state(df)
    assert state["names"] == ("scan_0", "scan_1", "scan_2")
    assert state["y"].shape == (3, 5)
    assert df.overlaid_idxs == [("scan", 0), ("scan", 1), ("scan", 2)]

    df.ui.plotUnit.setCurrentIndex(1)   # unit-switch rebuild, not last-only
    w.frame_ids[:] = ["2"]
    df.frame_ids = ["2"]
    df.idxs_1d = [2]
    df.update()
    rebuilt = _plot_state(df)
    assert rebuilt["names"] == ("scan_0", "scan_1", "scan_2")
    assert rebuilt["y"].shape == (3, 5)
    assert df.overlaid_idxs == [("scan", 0), ("scan", 1), ("scan", 2)]
    assert rebuilt["x_axis"][0].startswith("2")


def test_directory_overlay_accumulates_reused_zero_index_at_live_cadence(
        widget, qapp):
    """Sixteen one-frame directory scans survive boundaries and auto-waterfall.

    Arrivals are 40 ms apart, faster than both the 150 ms heavy-flush timer and
    the 500 ms full-overlay paint cadence.  This drives the production handoff,
    frame-driven rescope, publication store, controller, accumulator and Qt
    coalescers, so it covers both the scan-qualified O(new) filter and the
    cadence-sensitive outgoing-frame flush.
    """
    from PySide6 import QtTest
    from xrd_tools.core import IntegrationResult1D

    w = widget
    df = w.displayframe
    _set_processing_mode(w, "Int 1D")
    df.ui.plotMethod.setCurrentText("Overlay")
    df.ui.plotUnit.setCurrentIndex(0)
    w.h5viewer.auto_last = True
    w.h5viewer.live_run_active = True
    w.h5viewer.file_thread.live_run = True
    w._enter_run_state()

    thread = w.wrangler.thread
    thread.batch_mode = False
    thread.mask = None
    q = np.linspace(0.5, 3.0, 32)

    def arrive(scan_number):
        scan_name = f"SLP_position00_scan{scan_number:04d}"
        source = f"/data/{scan_name}_0000.tif"
        frame = SimpleNamespace(
            idx=0,
            gi=False,
            scan_info={"mon": float(scan_number)},
            source_file=source,
            source_frame_idx=0,
            map_raw=None,
            bg_raw=0,
            mask=None,
            thumbnail=None,
            int_1d=IntegrationResult1D(
                radial=q,
                intensity=np.full_like(q, float(scan_number)),
                sigma=np.ones_like(q),
                unit="q_A^-1",
            ),
            int_2d=None,
            gi_1d={},
            gi_2d={},
        )
        w.new_scan(
            scan_name,
            f"/tmp/{scan_name}.nxs",
            False,
            "th",
            False,
            False,
        )
        thread._published_frames[0] = frame
        w.update_data(0)
        QtTest.QTest.qWait(40)

    try:
        for scan_number in range(1, 16):
            arrive(scan_number)
        QtTest.QTest.qWait(600)

        history = df._waterfall_history
        assert history.count == 15
        assert history.ids == tuple(
            (f"SLP_position00_scan{scan_number:04d}", 0)
            for scan_number in range(1, 16)
        )
        assert len(df.curves) == 15
        assert not df._waterfall_active()

        arrive(16)
        QtTest.QTest.qWait(600)

        history = df._waterfall_history
        assert history.count == 16
        assert history.ids[-1] == ("SLP_position00_scan0016", 0)
        assert df._waterfall_active()
        assert df.wf_widget.isVisible()

        # Normal acquisition completion shares the run-state finalizer with
        # reintegration, but must not inherit reintegration's history reset.
        type(w)._finalize_processing_run(
            w,
            reset_overlay=False,
            origin="wrangler",
        )
        QtTest.QTest.qWait(600)
        assert df._waterfall_history.count == 16
        assert df._waterfall_history.ids[-1] == ("SLP_position00_scan0016", 0)
    finally:
        w._update_timer.stop()
        w._list_timer.stop()
        w.h5viewer.live_run_active = False
        w.h5viewer.file_thread.live_run = False
        w._exit_run_state()


def test_int_plot_slice_characterizes_update_plot_state(widget):
    # Slice-derived 1D is update_plot-live: it projects from the cake and
    # includes the slice parameters in the trace name.
    w = widget
    df = _set_int_scan(w, n=1)
    df.ui.plotMethod.setCurrentText("Single")
    df.update()                                # populate the plot-unit axes first
    chi_index = df.ui.plotUnit.findText("χ (°)")
    assert chi_index >= 0
    df.ui.plotUnit.setCurrentIndex(chi_index)
    df._on_plotUnit_changed()
    w.frame_ids[:] = ["0"]
    df.frame_ids = ["0"]
    df.idxs = [0]
    df.idxs_1d = [0]
    df.idxs_2d = [0]
    df.ui.slice.setChecked(True)
    df.ui.slice_center.setValue(0)
    df.ui.slice_width.setValue(10)

    df.update()

    state = _plot_state(df)
    assert state["names"] == ("scan_0 · χ@q=0.00±10.00",)
    assert state["y"].shape == (1, 4)
    assert state["x_axis"][0] == "χ"


# ── plotUnit combo: Int 1D must not offer 2D-derived axes (step 4-2 fold-in) ──
#
# set_axes is skip_2d-aware: Int 1D offers only axes computable from the 1D
# result (source '1d' / '1d_2d'); the 2D-derived axes (χ non-GI; the GI
# Q_ip/Q_oop reciprocal axes) appear only in Int 2D, where they slice the cake.

def _plotunit_sources(df):
    return [info["source"] for info in df._plot_axis_info]


def test_int1d_plotunit_excludes_chi(widget):
    df = widget.displayframe
    df.scan = SimpleNamespace(gi=False, skip_2d=False,
                              bai_1d_args={"unit": "q_A^-1"}, bai_2d_args={})
    df.set_axes()                                   # Int 2D
    assert "2d" in _plotunit_sources(df)            # χ is 2D-derived
    n_int2d = df.ui.plotUnit.count()                # Q, 2θ, χ
    assert n_int2d == 3

    df.scan.skip_2d = True
    df.set_axes()                                   # Int 1D
    assert "2d" not in _plotunit_sources(df)        # χ dropped
    assert df.ui.plotUnit.count() == 2              # Q, 2θ only


def test_int1d_gi_plotunit_excludes_reciprocal_axes(widget):
    df = widget.displayframe
    df.scan = SimpleNamespace(
        gi=True, skip_2d=False,
        bai_1d_args={"gi_mode_1d": "q_total", "unit": "q_A^-1"},
        bai_2d_args={"gi_mode_2d": "qip_qoop"})
    df.set_axes()                                   # Int 2D (GI)
    assert "2d" in _plotunit_sources(df)            # Q_ip / Q_oop present

    df.scan.skip_2d = True
    df.set_axes()                                   # Int 1D (GI)
    assert "2d" not in _plotunit_sources(df)        # reciprocal axes dropped
    assert df.ui.plotUnit.count() == 2              # q_total, 2θ only


def test_int_mode_transition_rebuilds_plotunit(widget):
    df = widget.displayframe
    df.viewer_mode = None
    df.scan = SimpleNamespace(gi=False, skip_2d=False,
                              bai_1d_args={"unit": "q_A^-1"}, bai_2d_args={})
    df._was_skip_2d = False
    df.set_axes()
    assert df.ui.plotUnit.count() == 3              # Int 2D: Q, 2θ, χ

    df.scan.skip_2d = True
    df._apply_1d_only_visibility()                  # -> Int 1D
    assert df.ui.plotUnit.count() == 2              # χ dropped on transition
    assert "2d" not in _plotunit_sources(df)

    df.scan.skip_2d = False
    df._apply_1d_only_visibility()                  # -> Int 2D
    assert df.ui.plotUnit.count() == 3              # restored
    assert "2d" in _plotunit_sources(df)


# ── A1: the DISPLAY unit toggle must never change the INTEGRATION unit ──────
#
# The plotUnit / imageUnit combos above the plots are *view* settings; toggling
# them must not write scan.bai_1d_args / bai_2d_args (the integration unit comes
# only from the right-hand integrator panel).  Locks the invariant — the leak
# reported pre-refactor is gone now that the 2D toggle renders via cake_image
# and the display layer only *reads* bai_args.

def test_display_unit_toggle_does_not_change_integration_unit(widget):
    w = widget
    sc = w.scan
    sc.bai_1d_args["unit"] = "q_A^-1"
    sc.bai_2d_args["unit"] = "q_A^-1"
    df = w.displayframe
    df.scan = sc
    df.set_axes()                                  # Q / 2θ display combos
    # Toggle the DISPLAY 1D unit to 2θ and fire the production signal.
    df.ui.plotUnit.setCurrentIndex(1)
    df.ui.plotUnit.activated.emit(1)
    # Toggle the DISPLAY 2D unit to 2θ-χ.
    if df.ui.imageUnit.count() > 1:
        df.ui.imageUnit.setCurrentIndex(1)
        df.ui.imageUnit.activated.emit(1)
    # Integration units are untouched — a later Int 1D (XYE) run still saves Q.
    assert sc.bai_1d_args["unit"] == "q_A^-1"
    assert sc.bai_2d_args["unit"] == "q_A^-1"


# ── C2: Int 1D (XYE) is a processing mode — keep wrangler inputs enabled ────

def test_int1d_xye_keeps_wrangler_inputs_enabled(widget):
    w = widget
    tree = getattr(w.wrangler, "tree", None)
    assert tree is not None
    combo = w.wrangler.ui.processingModeCombo

    def _set_mode(text):
        i = combo.findText(text)
        if i < 0:
            pytest.skip(f"processing mode {text!r} not available")
        combo.blockSignals(True)
        combo.setCurrentIndex(i)
        combo.blockSignals(False)

    # Int 1D (XYE): display auto-switches to XYE to list outputs, but it's a
    # processing mode — inputs stay enabled.
    _set_mode("Int 1D (XYE)")
    w._on_viewer_mode_changed("xye")
    assert tree.isEnabled() is True

    # XYE Viewer: a file browser — the TREE stays enabled (Project Folder /
    # Save Path drive the browser) while the processing groups disable.
    _set_mode("XYE Viewer")
    w._on_viewer_mode_changed("xye")
    assert tree.isEnabled() is True
    # (Per-group disables are the WRANGLER's _on_mode_changed job -- covered
    # by test_file_viewer_mode_disables_processing_tree_but_not_mode_combo;
    # this helper drives the widget-side handler with combo signals blocked.)
    assert w.wrangler.parameters.child('Project').child('h5_dir').opts.get(
        'enabled', True) is True


# ── C3 / C4: per-mode integration control enable/dim ───────────────────────

def test_integration_controls_enabled_per_mode(widget):
    w = widget
    combo = w.wrangler.ui.processingModeCombo
    iu = w.integratorTree.ui

    def _mode(text):
        i = combo.findText(text)
        if i < 0:
            pytest.skip(f"processing mode {text!r} not available")
        combo.blockSignals(True)
        combo.setCurrentIndex(i)
        combo.blockSignals(False)
        w._apply_integration_control_state()

    # Int 2D: both integration panels enabled.
    _mode("Int 2D")
    assert iu.frame1D.isEnabled() and iu.frame2D.isEnabled()

    # Int 1D: the 2-D panel is disabled (no cake); 1-D stays.
    _mode("Int 1D")
    assert iu.frame1D.isEnabled() and not iu.frame2D.isEnabled()

    # Int 1D (XYE): also 1D-only -> 2-D panel disabled.
    i = combo.findText("Int 1D (XYE)")
    if i >= 0:
        _mode("Int 1D (XYE)")
        assert not iu.frame2D.isEnabled()

    # XYE Viewer: both integration panels disabled, but Calibrate / Make Mask
    # stay enabled.  The GI (Fiber) + Threshold rows (relocated into the
    # integrator this cycle) must dim with the rest -- they used to stay bright.
    _mode("XYE Viewer")
    assert not iu.frame1D.isEnabled() and not iu.frame2D.isEnabled()
    assert not iu.gi_frame.isEnabled() and not iu.frame_pixreject.isEnabled()
    assert iu.pyfai_calib.isEnabled() and iu.get_mask.isEnabled()

    # Back to Int 2D restores everything.
    _mode("Int 2D")
    assert iu.frame1D.isEnabled() and iu.frame2D.isEnabled()
    assert iu.gi_frame.isEnabled() and iu.frame_pixreject.isEnabled()


# ── Reintegrate row (resurfaced buttons) ───────────────────────────────────

def _fake_processed_scan(*, n_frames=3, raw_reachable=True, skip_2d=False,
                         cached=True):
    """Minimal stand-in carrying just the attrs the reintegrate enable-state
    and bai_* guards read: frames.index length, has_reload_only_frames(),
    skip_2d, and the cached integrator a finished/loaded scan carries (so the
    calibration guard at bai_* click time is satisfied)."""
    from types import SimpleNamespace
    return SimpleNamespace(
        name="reint_test",
        frames=SimpleNamespace(index=list(range(n_frames))),
        has_reload_only_frames=(lambda: not raw_reachable),
        skip_2d=skip_2d,
        # A just-run (or already-bridged) scan has a cached integrator; a
        # bare-reloaded scan does not (cached=False) — see the calibration guard.
        _cached_integrator=(object() if cached else None),
        _cached_poni=(object() if cached else None),
        _cached_fiber_integrator=None,
    )


def test_reintegrate_row_present_and_advanced_rehomed(widget):
    """One row of three buttons (Reintegrate 1D | Reintegrate 2D | Advanced)
    sits directly above Calibrate/Make Mask, and Advanced is re-homed here so
    there is exactly ONE (the wrangler's old advancedButton is hidden)."""
    w = widget
    iu = w.integratorTree.ui
    assert iu.reintegrate1D.text() == "Reintegrate 1D"
    assert iu.reintegrate2D.text() == "Reintegrate 2D"
    assert iu.advanced_int.text() == "Advanced"
    # frame_3 (Calibrate / Make Mask) moved to the top tools bar, so it's no
    # longer in the integrator layout; the Reintegrate row is present, below the
    # Threshold row.
    vlay = iu.verticalLayout
    assert vlay.indexOf(iu.frame_reint) >= 0                 # present in the panel
    assert vlay.indexOf(iu.frame_3) == -1                    # moved to tools bar
    assert vlay.indexOf(iu.frame_pixreject) < vlay.indexOf(iu.frame_reint)
    # Exactly one Advanced: the wrangler's old button is explicitly hidden.
    if hasattr(w.wrangler, "ui") and hasattr(w.wrangler.ui, "advancedButton"):
        assert w.wrangler.ui.advancedButton.isHidden()
    # Advanced opens the combined 1D+2D advanced-settings dialog (real wiring).
    iu.advanced_int.clicked.emit()
    assert hasattr(w, "_integ_adv_combined_dlg")


def test_reintegrate_enable_state(widget):
    """Reintegrate / Advanced track the integration panels: enabled in any
    non-viewer Int mode; Reintegrate 2D follows frame2D (off in Int-1D-only
    modes); viewers disable all.  Enable does NOT depend on scan.frames / raw
    reachability — bai_1d/bai_2d enforce those at CLICK time."""
    w = widget
    iu = w.integratorTree.ui
    combo = w.wrangler.ui.processingModeCombo

    def _mode(text):
        i = combo.findText(text)
        if i < 0:
            pytest.skip(f"processing mode {text!r} not available")
        combo.blockSignals(True)
        combo.setCurrentIndex(i)
        combo.blockSignals(False)
        w._apply_integration_control_state()

    # Int 2D → Reintegrate 1D/2D + Advanced all enabled (mirrors frame1D/frame2D).
    _mode("Int 2D")
    assert iu.reintegrate1D.isEnabled()
    assert iu.reintegrate2D.isEnabled()
    assert iu.advanced_int.isEnabled()
    # And frame2D agrees (same flag) — the buttons mirror the panel.
    assert iu.reintegrate2D.isEnabled() == iu.frame2D.isEnabled()

    # Int 1D → Reintegrate 2D disabled (no cake), 1D + Advanced still enabled.
    _mode("Int 1D")
    assert iu.reintegrate1D.isEnabled()
    assert not iu.reintegrate2D.isEnabled()
    assert iu.advanced_int.isEnabled()
    assert iu.reintegrate2D.isEnabled() == iu.frame2D.isEnabled()

    # Viewer → whole row disabled (like the integration panels).
    _mode("XYE Viewer")
    assert not iu.reintegrate1D.isEnabled()
    assert not iu.reintegrate2D.isEnabled()
    assert not iu.advanced_int.isEnabled()

    # Back to Int 2D → re-enabled.
    _mode("Int 2D")
    assert iu.reintegrate1D.isEnabled() and iu.reintegrate2D.isEnabled()
    assert iu.advanced_int.isEnabled()


def test_reintegrate_buttons_wired_to_bai(widget):
    """The new visible buttons drive the same reintegrate path as the retained
    hidden stubs: Reintegrate 1D → bai_1d, Reintegrate 2D → bai_2d."""
    w = widget
    it = w.integratorTree
    iu = it.ui
    started = []
    it.integrator_thread.isRunning = lambda: False
    it.integrator_thread.start = lambda: started.append(it.integrator_thread.method)
    it.scan = _fake_processed_scan(raw_reachable=True)

    iu.reintegrate1D.clicked.emit()
    assert started == ["bai_1d_all"]

    started.clear()
    iu.reintegrate2D.clicked.emit()
    assert started == ["bai_2d_all"]


def test_reintegrate_no_processed_data_pops_message_not_thread(widget, monkeypatch):
    """The buttons mirror the panels, so they're clickable with no scan loaded
    (and in directory mode with no processed .nxs).  Clicking then surfaces a
    'nothing to re-integrate' message and must NOT start the integrator thread."""
    from types import SimpleNamespace
    from pyqtgraph.Qt import QtWidgets as _qtw
    w = widget
    it = w.integratorTree
    iu = it.ui
    started, msgs = [], []
    it.integrator_thread.isRunning = lambda: False
    it.integrator_thread.start = lambda: started.append("start")
    it.scan = SimpleNamespace(frames=SimpleNamespace(index=[]))   # no processed data
    monkeypatch.setattr(_qtw.QMessageBox, "information",
                        lambda *a, **k: msgs.append(a))

    iu.reintegrate1D.clicked.emit()
    iu.reintegrate2D.clicked.emit()
    assert started == []                 # blocked — thread never started
    assert len(msgs) == 2                # a message popped for each click


def test_reintegrate_uses_scans_own_cached_calibration(widget, monkeypatch):
    """The geometry a re-integration uses comes entirely from the scan's OWN
    calibration (a live run caches it; a .nxs reload restores it) — never the
    GUI's configured PONI File.  A scan that carries a cached integrator
    re-integrates with exactly that object, untouched."""
    w = widget
    it = w.integratorTree
    iu = it.ui
    started = []
    it.integrator_thread.isRunning = lambda: False
    it.integrator_thread.start = lambda: started.append(it.integrator_thread.method)
    scan = _fake_processed_scan(raw_reachable=True, cached=True)
    it.scan = scan
    own_integrator = scan._cached_integrator

    # The guard must not rebuild calibration from any GUI source: poni_to_integrator
    # must never be called when the scan already carries one.
    monkeypatch.setattr('xrd_tools.integrate.calibration.poni_to_integrator',
                        lambda p: (_ for _ in ()).throw(
                            AssertionError("reintegrate must not rebuild "
                                           "calibration from the GUI")))

    iu.reintegrate1D.clicked.emit()

    assert started == ["bai_1d_all"]                 # thread started — not blocked
    assert scan._cached_integrator is own_integrator  # the scan's own, untouched


def test_reintegrate_no_stored_calibration_pops_message(widget, monkeypatch):
    """Reloaded scan with NO stored calibration (a .nxs written before
    calibration round-trip): surface a clear 're-process once' message and do
    NOT start the thread — never hand the reduction a pixel-less integrator."""
    from pyqtgraph.Qt import QtWidgets as _qtw
    w = widget
    it = w.integratorTree
    iu = it.ui
    started, msgs = [], []
    it.integrator_thread.isRunning = lambda: False
    it.integrator_thread.start = lambda: started.append("start")
    it.scan = _fake_processed_scan(raw_reachable=True, cached=False)  # no calibration
    monkeypatch.setattr(_qtw.QMessageBox, "information",
                        lambda *a, **k: msgs.append(a))

    iu.reintegrate1D.clicked.emit()
    iu.reintegrate2D.clicked.emit()

    assert started == []                 # blocked — no stored calibration
    assert len(msgs) == 2                # one message per click


def test_reintegrate_apply_never_probes_raw(widget):
    """Regression (crash 2026-06-18): _apply_integration_control_state must NEVER
    call scan.has_reload_only_frames() — neither during a run (the probe opens the
    .nxs READ-ONLY, colliding with the streaming writer's r+ open → aborted scan)
    nor when idle (it false-disabled freshly written scans).  Reachability is
    enforced only at bai_1d/bai_2d CLICK time.  A scan whose probe RAISES proves
    the enable-state never touches it; the buttons still reach the right state."""
    from types import SimpleNamespace
    w = widget
    iu = w.integratorTree.ui
    combo = w.wrangler.ui.processingModeCombo

    def _boom():
        raise AssertionError("_apply must never probe raw reachability")

    w.scan = SimpleNamespace(
        name="probe_test", skip_2d=False,
        frames=SimpleNamespace(index=[0, 1, 2]),
        has_reload_only_frames=_boom,
    )

    # _apply must complete in EVERY run-state + mode without invoking the probe;
    # the _boom scan raises if it ever is.  (Per-mode/run button states are
    # covered by test_reintegrate_enable_state.)
    for mode in ("Int 2D", "Int 1D", "XYE Viewer"):
        j = combo.findText(mode)
        if j >= 0:
            combo.blockSignals(True)
            combo.setCurrentIndex(j)
            combo.blockSignals(False)
        for active in (True, False):
            w._run_active = active
            try:
                w._apply_integration_control_state()   # must not raise (no probe)
            finally:
                w._run_active = False


def test_cake_radial_entry_converts_q_to_2theta(widget):
    """A 2D-DERIVED radial plot entry (get_int_1d's source='2d', axis='radial')
    must return the SELECTED plotUnit's unit — converting the cake's stored Q
    radial to 2θ when the entry is labeled 2θ.  Covers the chi-integration mode's
    Q/2θ cake entries (and the standard sliced-2θ entry); without conversion the
    '2θ' entry plotted raw Å⁻¹ values under a degree label (review finding)."""
    w = widget
    df = _set_int_scan(w, n=1)                 # cake radial = Q, 0.5..3.0 Å⁻¹
    df.ui.slice.setChecked(False)
    # A cake-derived radial entry, labeled 2θ.
    df._plot_axis_info = [{"source": "2d", "slice_axis": "χ (°)", "axis": "radial"}]
    df.ui.plotUnit.blockSignals(True)
    df.ui.plotUnit.clear()
    df.ui.plotUnit.addItem("2θ (°)")
    df.ui.plotUnit.setCurrentIndex(0)
    df.ui.plotUnit.blockSignals(False)

    frame, frame_2d = w.viewer_rows_1d[0], w.viewer_rows_2d[0]
    xdata, _ = df.get_int_1d(frame, frame_2d, 0)
    xdata = np.asarray(xdata, dtype=float)
    # Q 3.0 Å⁻¹ at λ=0.7293 Å → 2θ ≈ 20°, NOT the raw 3.0; degrees range is sane.
    assert xdata.max() > 5.0, "2θ entry returned raw Q (no conversion)"
    assert xdata.max() < 90.0

    # The Q entry returns Q unchanged (no spurious conversion).
    df.ui.plotUnit.setItemText(0, "Q (Å⁻¹)")
    xq, _ = df.get_int_1d(frame, frame_2d, 0)
    assert np.isclose(np.asarray(xq, dtype=float).max(), 3.0, atol=0.2)


# ── C1: Grazing toggle defers the DISPLAY combo rebuild until a run ─────────

def test_grazing_toggle_defers_display_combo_rebuild(widget):
    w = widget
    df = w.displayframe
    df.scan = w.scan
    w.scan.gi = False
    df.set_axes()                                  # non-GI display combos
    before = [df.ui.plotUnit.itemText(i) for i in range(df.ui.plotUnit.count())]
    assert before                                  # has Q / 2θ / χ

    # Toggle Grazing on: integration state + integrator panel update now, but
    # the display plotUnit combo must NOT switch to GI axes yet (the plot is
    # still old-mode data).
    w.update_scattering_geometry(True)
    after = [df.ui.plotUnit.itemText(i) for i in range(df.ui.plotUnit.count())]
    assert after == before                         # display combo unchanged
    assert w.scan.gi is True                        # integration state updated


# ── B1 / B2: Share Axis re-expresses the 1D curve; unit change autoranges ───

def test_share_axis_reexpresses_1d_curve(widget):
    # B1: Share Axis links the 1D plot to the cake unit AND converts the
    # x-values (not just the label).  Non-GI; requires a wavelength.
    w = widget
    df = _set_int_scan(w, n=1)                      # INT 2D, Q-integrated
    df.update()
    q_x = np.asarray(df.plot_data[0], dtype=float)
    assert q_x.max() < 4.0                          # Q (~0.5..3)

    df.ui.imageUnit.setCurrentIndex(1)             # cake -> 2θ-χ
    df.update()
    assert df.ui.shareAxis.isEnabled()
    df.ui.shareAxis.setChecked(True)
    df.update()

    tth_x = np.asarray(df.plot_data[0], dtype=float)
    assert df.ui.plotUnit.currentText().strip().startswith("2")   # 2θ
    assert tth_x.max() > 4.0                        # re-expressed to 2θ
    assert not np.allclose(tth_x, q_x)              # values changed, not relabel


def test_plot_unit_change_autoranges_view(widget):
    # B2: a user 1D-unit change refits the plot view to the new data range.
    w = widget
    df = _set_int_scan(w, n=1)                      # INT 2D, Q
    df.ui.plotUnit.setCurrentIndex(0)              # Q
    df.update()
    # Switch to 2θ via the user-interaction signal.
    df.ui.plotUnit.setCurrentIndex(1)
    df.ui.plotUnit.activated.emit(1)

    xmax = float(np.nanmax(df.plot_data[0]))
    assert xmax > 10.0                              # 2θ values (~4..28)
    view_xmax = df.plot.viewRange()[0][1]
    assert view_xmax > 10.0                         # view refit to 2θ, not Q (~3)


def test_share_axis_converts_in_a_single_toggle_render(widget):
    # R2-2 (reopen B1): toggling Share Axis ONCE must re-express the 1D values
    # AND set the matching label in one render — no intermittent label-2θ-over-
    # Q-data flash.  Drives the toggle via the production signal only (no extra
    # update()), which is what the previous race failed.
    w = widget
    df = _set_int_scan(w, n=1)                      # INT 2D, Q
    df.update()
    df.ui.imageUnit.setCurrentIndex(1)             # cake -> 2θ-χ
    df.update()
    assert df.ui.shareAxis.isEnabled()

    # Single user toggle: shareAxis.toggled -> update() fires exactly once.
    df.ui.shareAxis.setChecked(True)

    tth_x = np.asarray(df.plot_data[0], dtype=float)
    assert tth_x.max() > 4.0                        # values converted in 1 render
    label, unit = df._current_plot_axis_label()
    assert label.strip().startswith("2")           # …and the label is 2θ, atomically


def test_plot_method_change_autoranges_view(widget):
    # R2-3: a plotMethod change (and other option/mode changes) refits the 1D
    # plot view instead of leaving it frozen at the old range.
    w = widget
    df = _set_int_scan(w, n=2)                      # INT 2D, Q
    df.update()
    df.plot.setXRange(100.0, 200.0, padding=0)     # freeze at a wrong range
    assert df.plot.viewRange()[0][0] > 50.0
    df._on_plotMethod_changed()                    # triggers re-render + autorange
    vx = df.plot.viewRange()[0]
    assert vx[0] < 50.0                            # refit to the Q data (~0.5..3)


# ── R2-1: slice c/w label = complementary 2D axis, refreshed on change ──

def test_slice_range_label_tracks_plotunit_and_mode(widget):
    from xdart.gui.tabs.static_scan.display_constants import Chi
    df = widget.displayframe
    df.viewer_mode = None

    # GI qip/qoop: a 2D-derived plotUnit slices over the *complementary* axis
    # (Q_ip -> Q_oop), so the label is a GI reciprocal axis, never χ.
    df.scan = SimpleNamespace(
        gi=True, skip_2d=False,
        bai_1d_args={"gi_mode_1d": "q_total", "unit": "q_A^-1"},
        bai_2d_args={"gi_mode_2d": "qip_qoop"})
    df.set_axes()
    gi_idx = next(i for i, info in enumerate(df._plot_axis_info)
                  if info["source"] == "2d")
    df.ui.plotUnit.setCurrentIndex(gi_idx)
    df._on_plotUnit_changed()
    assert df.ui.slice.text().endswith("(c/w)")
    assert Chi not in df.ui.slice.text()           # GI axis, not χ

    # GI -> non-GI: set_axes rebuilds (plotUnit Q over a Q-χ cake) and the label
    # must refresh to "χ (c/w)" immediately — not stay the stale GI label.
    df.scan.gi = False
    df.set_axes()
    assert Chi in df.ui.slice.text()               # refreshed, no click needed


# ---------------------------------------------------------------------------
# Run-state owner (task #68) — single source of truth for "a run is active".
#
# These drive the REAL staticWidget's run-state owner (_enter_run_state /
# _exit_run_state) and assert it flips displayframe._processing_active on
# start/finish, that the finished slot (which fires on success AND Stop/abort)
# reaches the exit, and that re-entry is idempotent.  (Part 2 adds the
# control-disable tests on top of this owner.)
# ---------------------------------------------------------------------------

def test_run_state_toggles_processing_active(widget):
    w = widget
    w._exit_run_state()                      # idle baseline
    assert w._run_active is False

    w._enter_run_state()
    assert w._run_active is True
    assert w.displayframe._processing_active is True

    w._exit_run_state()
    assert w._run_active is False
    assert w.displayframe._processing_active is False


def test_run_state_disables_whole_integrator_and_mode_row(widget):
    """During a run the WHOLE integrator greys -- including the GI (Fiber) +
    Threshold rows relocated into the integrator this cycle (the regression: they
    stayed bright while the rest dimmed) -- and the mode row (mode combo / Batch /
    Cores) locks; the action row (Pause/Resume/Stop) stays usable."""
    w = widget
    iu = w.integratorTree.ui
    w._exit_run_state()                          # idle baseline
    combo = w.wrangler.ui.processingModeCombo
    i = combo.findText("Int 2D")                 # a mode where the rows start on
    if i >= 0:
        combo.blockSignals(True)
        combo.setCurrentIndex(i)
        combo.blockSignals(False)
        w._apply_integration_control_state()
    assert iu.gi_frame.isEnabled() and iu.frame_pixreject.isEnabled()

    w._enter_run_state()
    assert not iu.frame1D.isEnabled() and not iu.frame2D.isEnabled()
    assert not iu.gi_frame.isEnabled() and not iu.frame_pixreject.isEnabled()
    assert not w.controls.modeCombo.isEnabled()
    assert not w.controls.batchButton.isEnabled()
    assert w.controls.actionRow.isEnabled()      # action row stays live

    w._exit_run_state()
    assert iu.gi_frame.isEnabled() and iu.frame_pixreject.isEnabled()
    assert w.controls.modeCombo.isEnabled()


def test_run_state_is_idempotent(widget):
    w = widget
    w._exit_run_state()                      # idle baseline (exit-when-idle is a no-op)
    assert w._run_active is False

    calls = []
    real = type(w.displayframe).set_processing_active
    w.displayframe.set_processing_active = lambda active: (
        calls.append(active), real(w.displayframe, active))[1]

    w._enter_run_state()
    w._enter_run_state()                     # re-entry: no-op (guard)
    w._exit_run_state()
    w._exit_run_state()                      # re-exit: no-op (guard)

    # The setter fired exactly once True then once False — no double-toggle.
    assert calls == [True, False], calls
    assert w._run_active is False


def test_finished_slot_reaches_exit_state(widget):
    """QThread.finished fires on success AND Stop/exception, so the finished
    slot must drive _exit_run_state.  Calling the real integrator_thread_finished
    slot while a run is active must land in the idle end-state — the same path a
    Stop/abort takes (finished always emits)."""
    w = widget
    w._enter_run_state()
    assert w.displayframe._processing_active is True and w._run_active is True

    w.integrator_thread_finished()           # the finished slot (success/Stop/abort)

    assert w.displayframe._processing_active is False
    assert w._run_active is False


# ---------------------------------------------------------------------------
# Processing-control disable during a run (task #71) — layered on the owner.
#
# During a run the integratorTree processing controls (1D/2D range fields,
# point counts, Auto toggles, unit + GI-mode combos, the Re-Integrate buttons,
# all children of frame1D/frame2D) plus Calibrate / Make Mask must disable, and
# re-enable MODE-CORRECTLY after.  The integratorTree controls are plain Qt
# widgets, so disabling keeps a checkable Auto toggle's checked look (no
# pyqtgraph repaint-to-unchecked).
# ---------------------------------------------------------------------------

def _set_processing_mode(w, mode_text):
    """Set the wrangler processing-mode combo the per-mode control state keys
    off.  Skips if the active wrangler has no such combo."""
    combo = getattr(getattr(w.wrangler, 'ui', None), 'processingModeCombo', None)
    if combo is None:
        pytest.skip("active wrangler has no processingModeCombo")
    combo.setCurrentText(mode_text)


def _proc_controls(w):
    ui = w.integratorTree.ui
    d = {name: getattr(ui, name).isEnabled() for name in (
        'frame1D', 'frame2D', 'integrate1D', 'integrate2D',
        'pyfai_calib', 'get_mask')}
    # Advanced 1D/2D dialogs are integratorTree attrs (pyqtgraph trees), not ui.
    for name in ('advancedWidget1D', 'advancedWidget2D'):
        adv = getattr(w.integratorTree, name, None)
        if adv is not None:
            d[name] = adv.isEnabled()
    return d


def test_run_disables_processing_controls(widget):
    w = widget
    _set_processing_mode(w, 'Int 2D')
    w._exit_run_state()                      # idle baseline
    assert all(_proc_controls(w).values())   # all enabled when idle (Int 2D)

    w._enter_run_state()
    ctl = _proc_controls(w)
    # Every processing-affecting control is locked, incl. the Re-Integrate
    # buttons (children of frame1D/frame2D) and Calibrate / Make Mask.
    assert not any(ctl.values()), f"controls not all disabled during run: {ctl}"


def test_run_keeps_stop_and_browsing_alone(widget):
    # Stop lives on the wrangler; the run-control disable must not touch it,
    # nor the h5viewer browsing list.
    w = widget
    _set_processing_mode(w, 'Int 2D')
    stop = getattr(w.wrangler.ui, 'stopButton', None)
    list_data = w.h5viewer.ui.listData
    if stop is not None:
        stop.setEnabled(True)
    list_enabled_before = list_data.isEnabled()
    w._enter_run_state()
    if stop is not None:
        assert stop.isEnabled(), "Stop must stay enabled during a run"
    assert list_data.isEnabled() == list_enabled_before


def test_exit_restores_mode_correct_int_2d(widget):
    w = widget
    _set_processing_mode(w, 'Int 2D')
    w._enter_run_state()
    w._exit_run_state()

    ui = w.integratorTree.ui
    # Int 2D: both panels back on; Calibrate / Make Mask back on.
    assert ui.frame1D.isEnabled() and ui.frame2D.isEnabled()
    assert ui.pyfai_calib.isEnabled() and ui.get_mask.isEnabled()


def test_exit_restores_mode_correct_int_1d_keeps_2d_off(widget):
    w = widget
    _set_processing_mode(w, 'Int 1D')
    w._enter_run_state()
    w._exit_run_state()

    ui = w.integratorTree.ui
    # Mode-correct restore — NOT a blanket enable: Int 1D has no cake, so the
    # 2D panel stays disabled after the run.
    assert ui.frame1D.isEnabled()
    assert not ui.frame2D.isEnabled()
    assert ui.pyfai_calib.isEnabled() and ui.get_mask.isEnabled()


def test_run_preserves_auto_toggle_checked_look(widget):
    """The repaint-to-unchecked bug must NOT bite the integratorTree: its Auto
    toggles are checkable QPushButtons, whose checked state (== visual look)
    survives setEnabled(False).  Also the GI-mode combo keeps its selection."""
    w = widget
    _set_processing_mode(w, 'Int 2D')
    ui = w.integratorTree.ui
    ui.radial_autoRange_1D.setChecked(True)
    ui.azim_autoRange_2D.setChecked(True)
    axis_before = ui.axis1D.currentText()

    w._enter_run_state()
    # Disabled, but the checked look + combo selection are preserved.
    assert not ui.radial_autoRange_1D.isEnabled()
    assert ui.radial_autoRange_1D.isChecked()
    assert ui.azim_autoRange_2D.isChecked()
    assert ui.axis1D.currentText() == axis_before

    w._exit_run_state()
    assert ui.radial_autoRange_1D.isChecked()
    assert ui.azim_autoRange_2D.isChecked()


def test_finished_slot_restores_controls(widget):
    # The finished slot (Stop/abort path) must also restore the controls.
    w = widget
    _set_processing_mode(w, 'Int 2D')
    w._enter_run_state()
    assert not any(_proc_controls(w).values())

    w.integrator_thread_finished()
    ui = w.integratorTree.ui
    assert ui.frame1D.isEnabled() and ui.frame2D.isEnabled()
    assert ui.pyfai_calib.isEnabled() and ui.get_mask.isEnabled()


def test_run_disables_advanced_param_dialogs(widget):
    """The Advanced 1D/2D dialogs feed bai_*_args too (a leak vector if left
    open mid-run); they must disable during a run and re-enable after."""
    w = widget
    _set_processing_mode(w, 'Int 2D')
    adv1d = getattr(w.integratorTree, 'advancedWidget1D', None)
    adv2d = getattr(w.integratorTree, 'advancedWidget2D', None)
    if adv1d is None or adv2d is None:
        pytest.skip("advancedWidget1D/2D not present")
    w._exit_run_state()                      # idle baseline
    assert adv1d.isEnabled() and adv2d.isEnabled()

    w._enter_run_state()
    assert not adv1d.isEnabled() and not adv2d.isEnabled()

    w._exit_run_state()
    assert adv1d.isEnabled() and adv2d.isEnabled()


def test_exit_restores_mode_correct_viewer(widget):
    """After a run in a Viewer mode, the 1D/2D panels stay disabled (file-browser
    mode) but Calibrate / Make Mask re-enable — the run overlay must not leave
    them stuck off."""
    w = widget
    ui = w.integratorTree.ui
    for viewer_mode in ('Image Viewer', 'XYE Viewer'):
        _set_processing_mode(w, viewer_mode)
        w._enter_run_state()
        w._exit_run_state()
        assert not ui.frame1D.isEnabled(), f"{viewer_mode}: frame1D should stay disabled"
        assert not ui.frame2D.isEnabled(), f"{viewer_mode}: frame2D should stay disabled"
        assert ui.pyfai_calib.isEnabled(), f"{viewer_mode}: Calibrate should re-enable"
        assert ui.get_mask.isEnabled(), f"{viewer_mode}: Make Mask should re-enable"


def test_integrator_finish_while_wrangler_running_keeps_controls_locked(widget, monkeypatch):
    """Overlap guard: a reintegrate can finish while a wrangler run is still
    active (a wrangler can be started mid-reintegrate).  The integrator finished
    slot must NOT exit the shared run-state while the wrangler still runs —
    controls stay locked until the wrangler's own finished handler exits."""
    w = widget
    _set_processing_mode(w, 'Int 2D')
    w._enter_run_state()
    assert w._run_active is True

    # Simulate the wrangler thread still running while the integrator finishes.
    monkeypatch.setattr(w.wrangler.thread, 'isRunning', lambda: True)
    monkeypatch.setattr(w.wrangler, '_run_phase', 'running', raising=False)
    w.integrator_thread_finished()
    assert w._run_active is True, "run-state exited while wrangler still running"
    assert not any(_proc_controls(w).values()), "controls re-enabled mid-wrangler-run"

    # Now the wrangler finishes too → the integrator finished slot exits cleanly.
    monkeypatch.setattr(w.wrangler.thread, 'isRunning', lambda: False)
    w.integrator_thread_finished()
    assert w._run_active is False
    ui = w.integratorTree.ui
    assert ui.frame1D.isEnabled() and ui.frame2D.isEnabled()


def test_wrangler_finish_while_reintegrate_running_keeps_controls_locked(widget, monkeypatch):
    """Mirror overlap guard: a wrangler started mid-reintegrate can FINISH first.
    wrangler_finished must NOT run the shared finalizer while the reintegrate is
    still running — otherwise the controls re-enable mid-reintegrate."""
    w = widget
    _set_processing_mode(w, 'Int 2D')
    w._enter_run_state()
    assert w._run_active is True and not any(_proc_controls(w).values())

    # Reintegrate still running; this wrangler's own thread has finished.
    monkeypatch.setattr(w.integratorTree.integrator_thread, 'isRunning', lambda: True)
    monkeypatch.setattr(w.wrangler.thread, 'isRunning', lambda: False)
    monkeypatch.setattr(w.wrangler.thread, 'batch_mode', False, raising=False)
    monkeypatch.setattr(w.wrangler.thread, 'xye_only', False, raising=False)
    # Force the scan-matches branch so the overlap guard (not the name check)
    # is what prevents the exit.
    monkeypatch.setattr(w.wrangler, 'scan_name', w.scan.name, raising=False)
    # Stub the heavy collaborators wrangler_finished would otherwise drive.
    monkeypatch.setattr(w, '_flush_pending_update', lambda: None)
    monkeypatch.setattr(w.wrangler, 'stop', lambda: None)
    finalized = []
    monkeypatch.setattr(
        w,
        '_finalize_processing_run',
        lambda **kwargs: finalized.append(kwargs),
    )

    w.wrangler_finished()

    assert w._run_active is True, "run-state exited while reintegrate still running"
    assert finalized == [], "finalized the wrangler while reintegrate still ran"
    assert not any(_proc_controls(w).values()), "controls re-enabled mid-reintegrate"


def _stub_wrangler_finish_collaborators(w, monkeypatch, nxs):
    monkeypatch.setattr(w.wrangler.thread, 'batch_mode', False, raising=False)
    monkeypatch.setattr(w.wrangler.thread, 'xye_only', False, raising=False)
    monkeypatch.setattr(w.wrangler.thread, 'fname', str(nxs), raising=False)
    monkeypatch.setattr(w.integratorTree.integrator_thread, 'isRunning', lambda: False)
    monkeypatch.setattr(w, '_flush_pending_update', lambda: None)
    monkeypatch.setattr(w.wrangler, 'stop', lambda: None)
    monkeypatch.setattr(w, '_finalize_processing_run', lambda **kwargs: None)
    w.h5viewer.dirname = str(nxs.parent)         # skip the update_scans branch
    loaded = []
    monkeypatch.setattr(w.h5viewer, 'set_file',
                        lambda f, **k: loaded.append(f))
    return loaded


def test_wrangler_finished_append_zero_new_frames_shows_last_frame(
        widget, monkeypatch, tmp_path):
    """Append-mode feedback: a NON-batch run that processed 0 new frames still
    loads the existing scan file and auto-selects its LAST frame, so the user
    sees that the run ran."""
    w = widget
    _set_processing_mode(w, 'Int 2D')
    w._enter_run_state()                         # resets _run_saw_frame=False
    assert w._run_saw_frame is False
    nxs = tmp_path / "scan.nxs"
    nxs.write_text("x")
    loaded = _stub_wrangler_finish_collaborators(w, monkeypatch, nxs)

    w.wrangler_finished()

    assert loaded == [str(nxs)]                  # existing scan reloaded
    assert w.h5viewer._auto_select_last_on_finish is True
    # The 0-new-frames append fix: the Scans panel must now FOLLOW to the finished
    # scan (previously only the last FRAME was selected, not the scan ROW -- the
    # scans_select for a live run is gated on _run_saw_frame, False here).
    assert w.h5viewer.scan_name == "scan"        # pointed at the finished .nxs stem
    selected = w.h5viewer.ui.listScans.currentItem()
    assert selected is not None and selected.text() == "scan.nxs"


def test_wrangler_finished_with_frames_does_not_reload(
        widget, monkeypatch, tmp_path):
    """When frames WERE shown live, the run must NOT reload from disk — the live
    display stands (no redundant disk read / cursor jump)."""
    w = widget
    _set_processing_mode(w, 'Int 2D')
    w._enter_run_state()
    w._run_saw_frame = True                      # frames displayed live
    nxs = tmp_path / "scan.nxs"
    nxs.write_text("x")
    loaded = _stub_wrangler_finish_collaborators(w, monkeypatch, nxs)

    w.wrangler_finished()

    assert loaded == []                          # no append-feedback reload


def test_current_pattern_for_fit_reads_drawn_1d_frames(widget, monkeypatch):
    """The fit provider must read the DRAWN 1-D frames (displayframe.idxs_1d --
    what the plot shows and what get_frames_int_1d defaults to), because the
    staticWidget/h5viewer frame_ids are NOT populated on a manual frame click,
    which left the popup stuck on "No frame selected"."""
    w = widget
    seen = {}
    monkeypatch.setattr(
        w, '_pattern_for_frame',
        lambda idx: (seen.setdefault('idx', idx), ([1.0], [2.0], 'q'))[1])
    w.frame_ids = []
    try:
        w.h5viewer.frame_ids[:] = []
    except Exception:
        pass
    w.displayframe.idxs_1d[:] = [5]           # the frame the 1-D plot draws
    assert w._current_pattern_for_fit() == ([1.0], [2.0], 'q')
    assert seen['idx'] == 5
    # nothing drawn + no selection -> None (the correct "No frame selected" case)
    w.displayframe.idxs_1d[:] = []
    assert w._current_pattern_for_fit() is None


def test_pattern_for_frame_uses_idle_gated_blocking_read(widget, monkeypatch):
    """A non-resident frame's per-frame read must be a store-MISS BLOCKING read on
    an idle scan (Set-Bkg precedent) so the 1-D actually loads -- else the popup
    lies "No frame selected". During a run it must NOT block (live push feeds it)."""
    w = widget
    seen = {}

    def _spy(idxs, rv='all', allow_blocking_read=None):
        seen['idxs'] = list(idxs)
        seen['blocking'] = allow_blocking_read
        return ([[10.0, 20.0, 30.0]], [0.1, 0.2, 0.3])   # (ydata, xdata)

    monkeypatch.setattr(w.displayframe, 'get_frames_int_1d', _spy)
    w.displayframe._processing_active = False
    assert w._pattern_for_frame(4) is not None
    assert seen == {'idxs': [4], 'blocking': True}       # idle -> block-and-read
    w.displayframe._processing_active = True
    w._pattern_for_frame(4)
    assert seen['blocking'] is False                     # mid-run -> no contention


def test_run_end_preserves_overlay_then_arms_optional_catchup(
        widget, monkeypatch, tmp_path):
    """Normal run end preserves history before arming optional disk catch-up."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    w = widget
    _set_processing_mode(w, 'Int 2D')
    w.displayframe.ui.plotMethod.setCurrentText("Overlay")
    w.h5viewer.auto_last = True
    w._enter_run_state()
    w._run_saw_frame = True                       # LIVE branch (frames displayed)
    nxs = tmp_path / "scan.nxs"
    nxs.write_text("x")
    _stub_wrangler_finish_collaborators(w, monkeypatch, nxs)
    w.scan.name = w.wrangler.scan_name            # take the delegate branch

    order = []
    monkeypatch.setattr(
        w,
        '_finalize_processing_run',
        lambda **kwargs: order.append(('finalize', kwargs)),
    )
    monkeypatch.setattr(
        staticWidget, '_arm_runend_overlay_catchup',
        lambda self: order.append('overlay_catchup_armed'))

    w.wrangler_finished()

    expected = ('finalize', {'reset_overlay': False, 'origin': 'wrangler'})
    assert expected in order, order
    assert 'overlay_catchup_armed' in order[order.index(expected) + 1:], \
        f"overlay catch-up not armed after wrangler finalization: {order}"


def test_update_data_selection_restore_is_linear_not_quadratic(widget):
    """The full-rebuild selection restore must be O(N), not O(N^2).  After Show
    All on a long scan EVERY id is selected, so the old findItems()-per-id restore
    was O(N^2) (3621 x 3621 ~ 13M main-thread ops) -- a multi-second freeze when a
    large, all-selected scan is rebuilt/reloaded.  Now a text->item map makes it
    O(N)."""
    import time
    w = widget
    lw = w.h5viewer.ui.listData
    N = 3000
    lw.clear()
    lw.addItems([str(i) for i in range(1, N + 1)])
    for r in range(N):
        lw.item(r).setSelected(True)
    assert len(lw.selectedItems()) == N          # Show-All state
    idx = w.scan.frames.index
    idx.clear()
    for i in range(1, N + 1):
        idx.append(i)
    w.h5viewer.new_scan_loaded = False
    w.h5viewer.auto_last = False

    t0 = time.perf_counter()
    w.h5viewer.update_data(emit_update=False, force_rebuild=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert lw.count() == N
    assert len(lw.selectedItems()) == N          # selection restored correctly
    # O(N) map restore is single-digit ms; the old O(N^2) findItems loop measured
    # ~150 ms+ at N=3000 -> a generous 100 ms ceiling catches a regression.
    assert elapsed_ms < 100.0, \
        f"selection restore too slow ({elapsed_ms:.0f}ms) -- O(N^2) regression?"


def test_cores_spin_visible_in_processing_hidden_in_viewers(widget, monkeypatch):
    """Item 4 (browse_1d_cap_raise): the Cores spinbox drives the LIVE streaming
    reduction pool AND batch reprocess, so it is visible+enabled in ANY processing
    mode -- hidden ONLY in the file viewers (was cosmetically batch-only)."""
    wr = widget.wrangler
    vis = {}
    monkeypatch.setattr(wr.ui.maxCoresSpinBox, 'setVisible',
                        lambda v: vis.__setitem__('cores', v))

    def _mode(text, live=False, batch=False):
        wr.ui.processingModeCombo.setCurrentText(text)
        for cb, val in ((wr.ui.liveCheckBox, live), (wr.ui.batchCheckBox, batch)):
            cb.blockSignals(True); cb.setChecked(val); cb.blockSignals(False)
        vis.clear()
        wr._on_mode_changed()

    # live processing -> enabled + visible
    _mode('Int 2D', live=True)
    assert wr.ui.maxCoresSpinBox.isEnabled() and vis.get('cores') is True
    # processing mode, NEITHER checked -> STILL visible + enabled (the ask)
    _mode('Int 2D', live=False, batch=False)
    assert vis.get('cores') is True, "Cores hidden in a processing mode"
    assert wr.ui.maxCoresSpinBox.isEnabled()
    # a file viewer -> hidden (whole processing row is hidden)
    _mode('Image Viewer')
    assert vis.get('cores') is False, "Cores visible in a viewer mode"


def test_date_sort_toggle_orders_scans_by_mtime(widget, tmp_path):
    """date_sort_scans: the checkable Date toggle sorts listScans by mtime
    (newest first); unchecked restores the natural-name order."""
    import os
    w = widget
    hv = w.h5viewer
    hv.viewer_mode = 'normal'
    hv.scan_name = None                           # avoid the follow-select path
    # ascending by NAME, ascending by mtime -> the two orders are reverses
    for i, name in enumerate(["a_scan.nxs", "b_scan.nxs", "c_scan.nxs"]):
        p = tmp_path / name
        p.write_text("x")
        os.utime(p, (1000 + i, 1000 + i))         # a oldest .. c newest
    hv.dirname = str(tmp_path)
    assert hv.ui.dateSort.isCheckable()

    def _nxs_order():
        return [hv.ui.listScans.item(r).text()
                for r in range(hv.ui.listScans.count())
                if hv.ui.listScans.item(r).text().endswith('.nxs')]

    hv.ui.dateSort.blockSignals(True)
    hv.ui.dateSort.setChecked(False)
    hv.ui.dateSort.blockSignals(False)
    hv.update_scans()
    assert _nxs_order() == ["a_scan.nxs", "b_scan.nxs", "c_scan.nxs"]  # name

    hv.ui.dateSort.blockSignals(True)
    hv.ui.dateSort.setChecked(True)
    hv.ui.dateSort.blockSignals(False)
    hv.update_scans()
    assert _nxs_order() == ["c_scan.nxs", "b_scan.nxs", "a_scan.nxs"]  # mtime desc


def test_run_end_reselects_scans_panel(widget, monkeypatch, tmp_path):
    """scans_select_after_run: wrangler_finished must re-run update_scans in the
    LIVE saw-frames branch (post-write), so the Scans panel follows to the newly-
    processed scan.  update_scans at scan START ran before the writer created
    <name>.nxs -> nothing to select, so the panel kept the prior scan."""
    w = widget
    _set_processing_mode(w, 'Int 2D')
    w._enter_run_state()
    w._run_saw_frame = True                       # LIVE saw-frames branch
    nxs = tmp_path / "scan.nxs"
    nxs.write_text("x")
    _stub_wrangler_finish_collaborators(w, monkeypatch, nxs)
    calls = []
    monkeypatch.setattr(w.h5viewer, 'update_scans', lambda: calls.append(True))

    w.wrangler_finished()

    assert calls, "wrangler_finished did not re-select the Scans panel (update_scans)"


def test_update_scans_follows_current_scan_in_normal_mode(widget, tmp_path):
    """The Scans panel must highlight the CURRENTLY-loaded scan after a rebuild.
    A new-scan boundary repopulates listScans (clear drops the old selection), so
    the panel used to show the stale prior scan while the display had moved to the
    newly-processed one."""
    w = widget
    hv = w.h5viewer
    (tmp_path / "Eiger_long.nxs").write_text("x")
    (tmp_path / "Combi4_new.nxs").write_text("x")
    hv.dirname = str(tmp_path)
    hv.viewer_mode = 'normal'
    # simulate the old stale state: Eiger selected
    hv.scan_name = "Eiger_long"
    hv.update_scans()
    assert hv.ui.listScans.currentItem().text() == "Eiger_long.nxs"
    # process/load a new scan -> scan_name moves; the panel must follow
    hv.scan_name = "Combi4_new"
    hv.update_scans()
    cur = hv.ui.listScans.currentItem()
    assert cur is not None and cur.text() == "Combi4_new.nxs", \
        f"Scans panel did not follow the new scan: {cur.text() if cur else None}"
    # scan_name may carry a frame-count/display suffix the file stem does not
    # ("<scan>_5" vs "<scan>.nxs") -- must still follow.
    hv.scan_name = "Combi4_new_5"
    hv.update_scans()
    cur = hv.ui.listScans.currentItem()
    assert cur is not None and cur.text() == "Combi4_new.nxs", \
        f"Scans panel did not follow a suffixed scan_name: {cur.text() if cur else None}"


def test_shutdown_threads_stops_file_thread(widget):
    """Production teardown must stop the persistent fileHandlerThread so it is
    not 'destroyed while running' on tab/app close.  Idempotent."""
    w = widget
    ft = w.h5viewer.file_thread
    assert not ft.isRunning(), "file_thread should start lazily"
    w.h5viewer._ensure_file_thread_running()
    assert ft.isRunning(), "file_thread should run once file work is queued"

    w.h5viewer.shutdown_threads()
    assert ft.wait(2000)
    assert not ft.isRunning(), "file_thread still running after shutdown_threads"

    # Idempotent — a second call (and the close() path that also calls it) is safe.
    w.h5viewer.shutdown_threads()
    assert not ft.isRunning()


def test_integrator_panel_session_roundtrip(widget, monkeypatch):
    """Session persistence for the integration panel: units/pts/ranges/Auto
    flags + Advanced params survive a save/restore cycle (saved at app close,
    restored at startup -- they previously weren't persisted at all)."""
    w = widget

    w._on_controls_v2_field_changed(("Int1D", "points"), "1234")
    w._on_controls_v2_field_changed(("Int1D", "radial_auto"), False)
    w._on_controls_v2_field_changed(("Int1D", "radial_low"), "0.5")
    w._on_controls_v2_field_changed(("Int1D", "radial_high"), "4.5")
    w._on_controls_v2_field_changed(("Int1D", "chi_offset"), "45.0")
    state = w._controls_v2_int_session_state()
    import json
    json.dumps(state)                       # must be JSON-serializable

    # Scramble, then restore.
    w._on_controls_v2_field_changed(("Int1D", "points"), "999")
    w._on_controls_v2_field_changed(("Int1D", "radial_auto"), True)
    w._on_controls_v2_field_changed(("Int1D", "chi_offset"), "90.0")
    monkeypatch.setattr(
        "xdart.utils.session.load_session",
        lambda: {"controls_v2_int": state},
    )
    w._restore_controls_v2_int_session_state()

    args = w.scan.bai_1d_args
    assert args["numpoints"] == 1234
    assert args["radial_range"] == (0.5, 4.5)
    assert args["chi_offset"] == 45.0
    assert str(w._controls_v2_field_values()[("Int1D", "points")]) == "1234"


def test_cake_view_trim_rearms_after_own_trim_but_respects_user_zoom(widget):
    """The display trim disables pyqtgraph auto-range via its own setRange, so
    it must recognize its OWN previous range and re-trim on the next render
    (axis-kind switches stranded a stale window otherwise) -- while a range
    the USER set stays untouched."""
    import numpy as np
    from xdart.gui.tabs.static_scan.display_frame_widget import displayFrameWidget

    w = widget.displayframe.binned_widget
    vb = w.image_plot.getViewBox()

    img = np.zeros((100, 100)); img[10:30, 40:60] = 5.0      # (x, y) block
    x = np.linspace(0.0, 10.0, 100); y = np.linspace(-180.0, 180.0, 100)
    displayFrameWidget._trim_view_to_data(w, img, x, y)
    first = vb.viewRange()
    assert first[0][1] < 6.0                                 # trimmed in x

    # Same auto-range-off state, new data elsewhere -> must RE-trim.
    img2 = np.zeros((100, 100)); img2[70:90, 70:90] = 5.0
    displayFrameWidget._trim_view_to_data(w, img2, x, y)
    second = vb.viewRange()
    assert second != first and second[0][0] > 5.0            # followed the data

    # User zoom (a range we did NOT set) -> respected, no trim.
    vb.setRange(xRange=(2.0, 3.0), yRange=(0.0, 10.0), padding=0)
    user = vb.viewRange()
    displayFrameWidget._trim_view_to_data(w, img, x, y)
    assert vb.viewRange() == user


def test_scale_switch_without_2d_panels_does_not_crash(widget):
    """Switching Linear/Sqrt/Log re-renders all views; in Int 1D (or before
    anything is drawn) image_data/binned_data are None and the unpack raised
    TypeError.  XYE Viewer never routes here, which is why it was immune."""
    df = widget.displayframe
    df.image_data = None
    df.binned_data = None
    df.update_image_view()          # must no-op, not raise
    df.update_binned_view()


def test_slice_controls_wait_for_binned_data(widget):
    from PySide6 import QtCore

    df = widget.displayframe
    df.scan.skip_2d = False
    df.scan.gi = False
    df.binned_data = None
    df.clear_slice_overlay()
    df.ui.plotMethod.setCurrentText("Single")
    df.ui.plotUnit.setCurrentIndex(2)       # χ: slice along the cake radial axis
    df._on_plotUnit_changed()

    assert not df.ui.slice.isEnabled()
    assert "no 2D data yet" in df.ui.slice.toolTip()
    assert not df.ui.slice_center.isEnabled()
    assert not df.ui.slice_width.isEnabled()
    assert not df.ui.pinSlice.isEnabled()

    # A restored/session-driven checked state before the cake exists must no-op,
    # not unpack binned_data=None.
    df.ui.slice.setChecked(True)
    df.show_slice_overlay()
    assert df.overlay is None

    df.binned_data = (np.ones((16, 16)), QtCore.QRectF(0.0, -180.0, 25.0, 360.0))
    df._on_plotUnit_changed()
    assert df.ui.slice.isEnabled()
    assert df.ui.slice_center.isEnabled()
    assert df.ui.slice_width.isEnabled()
    assert not df.ui.pinSlice.isEnabled()
    assert "Overlay or Waterfall" in df.ui.pinSlice.toolTip()

    df.ui.plotMethod.setCurrentText("Overlay")
    df._on_plotMethod_changed()
    assert df.ui.pinSlice.isEnabled()

    df.show_slice_overlay()
    assert df.overlay is not None

    df.binned_data = None
    df.update_binned_view()
    assert not df.ui.slice.isEnabled()
    assert not df.ui.slice_center.isEnabled()
    assert not df.ui.slice_width.isEnabled()
    assert not df.ui.pinSlice.isEnabled()


def test_gi_1d_npts_defaults_and_per_axis_memory(widget):
    """1-D Pts: fiber axes (Qip/Qoop/Exit) default to 1000/1000, q_total
    ('Q'/'Chi', plain pyFAI) and standard mode to 2000; a user-chosen value
    persists per axis across switches.  (Was: stale session text -- e.g.
    1234 -- landed in whatever axis was active.)"""
    tree = widget.integratorTree
    tree._npts_memory_1d = {}          # independent of restored session state
    tree._npts_key_1d = None
    tree.scan.gi = True
    tree.set_image_units()

    labels = [tree.ui.axis1D.itemText(i) for i in range(tree.ui.axis1D.count())]
    qip_idx = next(i for i, t in enumerate(labels) if "Qip" in t)

    tree.ui.axis1D.setCurrentIndex(qip_idx)            # -> q_ip
    assert tree.ui.npts_1D.text() == "1000"
    assert tree.ui.npts_oop_1D.text() == "1000"

    tree.ui.npts_1D.setText("800")                     # user override
    tree.ui.axis1D.setCurrentIndex(0)                  # -> q_total ('Q')
    assert tree.ui.npts_1D.text() == "2000"
    tree.ui.axis1D.setCurrentIndex(qip_idx)            # back -> q_ip
    assert tree.ui.npts_1D.text() == "800"             # remembered
    assert tree.ui.npts_oop_1D.text() == "1000"

    # Session roundtrip carries the per-axis memory
    state = tree.session_state()
    import json
    json.dumps(state)
    assert state['npts_1d']['q_ip'][0] == "800"

    # new_scan re-runs set_image_units: it must NOT clobber the fiber-axis
    # value (the legacy force-to-2000 snippet did exactly that and poisoned
    # the per-axis memory via the trailing stash).
    tree.set_image_units()
    assert tree.ui.npts_1D.text() == "800"
    assert tree._npts_memory_1d.get('q_ip', ('800',))[0] == "800"

    tree.scan.gi = False                               # -> standard mode
    tree.set_image_units()
    tree._update_gi_mode_1d(tree.ui.axis1D.currentIndex())
    assert tree.ui.npts_1D.text() == "2000"


def test_gi_entry_forces_q_unit(widget):
    """Switching to GI with 2th selected in standard mode must force the
    unit (combo AND bai args) back to Q -- GI has no 2th option, and the
    retained '2th_deg' integrated GI under a Q-labelled axis (wrong
    results, user-reported)."""
    tree = widget.integratorTree
    tree._npts_memory_1d = {}
    tree._npts_key_1d = None

    tree.scan.gi = False
    tree.set_image_units()
    tree.ui.unit_1D.setCurrentIndex(1)                 # 2th
    tree.ui.unit_2D.setCurrentIndex(1)
    assert tree.scan.bai_1d_args['unit'] == '2th_deg'
    assert tree.scan.bai_2d_args['unit'] == '2th_deg'

    tree.scan.gi = True
    tree.set_image_units()
    assert tree.ui.unit_1D.currentIndex() == 0          # forced to Q
    assert tree.ui.unit_2D.currentIndex() == 0
    assert tree.scan.bai_1d_args['unit'] == 'q_A^-1'
    assert tree.scan.bai_2d_args['unit'] == 'q_A^-1'
    assert tree.ui.unit_1D.count() == 1                 # 2th removed in GI
    assert tree.ui.unit_2D.count() == 1

    # Back to standard: both units restored as valid choices.
    tree.scan.gi = False
    tree.set_image_units()
    assert tree.scan.bai_1d_args['unit'] == 'q_A^-1'    # stays Q, no surprise
    assert tree.ui.unit_1D.count() == 3                 # 2θ + χ (azimuthal) back
    assert tree.ui.unit_2D.count() == 2                 # 2D has no χ profile


def test_reintegrate_applies_current_threshold_config(widget):
    """A reintegrate snapshots the wrangler's CURRENT Intensity-Threshold /
    Mask-Saturated policy onto the integrator thread (GUI thread, read fresh) and
    bakes it into the reintegrate plan — so saturation/threshold rejection that a
    live run applies also applies on reintegrate."""
    from xdart.modules.reduction import ThresholdSaturationConfig
    w = widget
    it = w.integratorTree
    cfg = ThresholdSaturationConfig(apply_threshold=True, threshold_min=2,
                                    threshold_max=50, mask_saturation=True)
    it.get_threshold_config = lambda: cfg

    it._apply_threshold_config_to_thread()
    assert it.integrator_thread.threshold_config is cfg          # snapshot on GUI thread

    plan = it.integrator_thread._plan_for_reintegration(integrate_2d=False)
    assert plan.threshold_min == 2 and plan.threshold_max == 50  # baked into the plan
    assert plan.mask_saturation is True


def test_integrator_owns_threshold_config(widget):
    """The integrator panel OWNS the pixel-rejection policy: get_threshold_config
    reads its row widgets fresh (default: threshold OFF, Mask Saturated ON)."""
    w = widget
    it = w.integratorTree
    cfg = it.get_threshold_config()
    assert cfg.apply_threshold is False
    assert cfg.mask_saturation is True          # default-on, mirrors old wrangler

    w._on_controls_v2_field_changed(("Mask", "Threshold"), True)
    w._on_controls_v2_field_changed(("Mask", "min"), "7")
    w._on_controls_v2_field_changed(("Mask", "max"), "9000")
    w._on_controls_v2_field_changed(("MaskSat", "mask_sentinel"), False)
    cfg2 = it.get_threshold_config()
    assert cfg2.apply_threshold is True
    assert cfg2.threshold_min == 7 and cfg2.threshold_max == 9000
    assert cfg2.mask_saturation is False
    assert cfg2 == w._controls_v2_threshold_config()


def test_live_run_injects_integrator_threshold_into_wrangler(widget):
    """SPINE: a live run must apply the SAME pixel rejection as Reintegrate.
    _push_threshold_to_wrangler copies the integrator's policy into the wrangler's
    hidden Mask/MaskSat carrier params before setup() — so live == reintegrate by
    construction (both source from the integrator)."""
    w = widget
    w._on_controls_v2_field_changed(("Mask", "Threshold"), True)
    w._on_controls_v2_field_changed(("Mask", "min"), "3")
    w._on_controls_v2_field_changed(("Mask", "max"), "5000")
    w._on_controls_v2_field_changed(("MaskSat", "mask_sentinel"), False)

    w._push_threshold_to_wrangler()

    p = w.wrangler.parameters
    try:
        mask = p.child('Mask')
    except Exception:
        mask = None
    if mask is not None:                         # image-series wrangler
        assert mask.child('Threshold').value() is True
        assert mask.child('min').value() == 3
        assert mask.child('max').value() == 5000
    # Mask Saturated exists on both image + nexus wranglers.
    assert p.child('MaskSat').child('mask_sentinel').value() is False


def test_layout_wrangler_above_integrator(widget):
    """The processing controls live in the right controls area.

    In the legacy layout the wrangler pane sat above the integrator pane in the
    right splitter.  Controls Panel V2 embeds the existing integrator widget
    inside its Processing card instead.
    """
    w = widget
    rs = w.ui.rightSplitter
    if getattr(w, "controls_v2", None) is not None:
        assert rs.indexOf(w.ui.wranglerFrame) >= 0
        assert rs.indexOf(w.ui.integratorFrame) == -1
        assert w.controls_v2.processing_card.isAncestorOf(w.ui.integratorFrame)
    else:
        assert rs.indexOf(w.ui.wranglerFrame) < rs.indexOf(w.ui.integratorFrame)


def test_tools_bar_hosts_calibrate_and_make_mask(widget):
    """Sections refactor Stage 1: the whole Calibrate / Make Mask row (frame_3)
    is reparented into the top tools bar (toolsFrame, index 0 of the right
    splitter); the buttons stay inside it (same objects, wiring + enable refs
    intact) and visible."""
    w = widget
    tools = w.ui.toolsFrame
    f3 = w.integratorTree.ui.frame_3
    assert f3.parentWidget() is tools                   # whole row reparented
    assert w.integratorTree.ui.pyfai_calib.parentWidget() is f3
    assert not w.integratorTree.ui.pyfai_calib.isHidden()
    assert not w.integratorTree.ui.get_mask.isHidden()
    rs = w.ui.rightSplitter
    assert rs.indexOf(tools) == 0                       # very top pane
    if getattr(w, "controls_v2", None) is not None:
        assert rs.indexOf(w.ui.integratorFrame) == -1
        assert w.controls_v2.processing_card.isAncestorOf(w.ui.integratorFrame)
    else:
        assert rs.indexOf(w.ui.wranglerFrame) < rs.indexOf(w.ui.integratorFrame)


def test_calibrate_offers_recent_poni_from_project_folder(widget, tmp_path, monkeypatch):
    """After pyFAI-calib2 (an external program that can't report its save path)
    closes, ``_autofill_poni_after_calibrate`` rglobs the project folder for a
    ``.poni`` written during calibration and offers it via a confirmation popup
    (newest wins).  Nothing is offered if none are recent.  (The Yes-path reuses
    the existing poni-browse plumbing, so we only assert the offer here.)"""
    import os
    import time
    from xdart.gui.tabs.static_scan import static_scan_widget as ssw

    w = widget
    proj = tmp_path / "proj"
    proj.mkdir()
    w.wrangler.parameters.child("Project", "project_folder").setValue(str(proj))

    asked = []
    monkeypatch.setattr(
        ssw.QMessageBox, "question",
        lambda *a, **k: asked.append(a[2]) or ssw.QMessageBox.No)

    # Nothing written yet -> no popup.
    w._autofill_poni_after_calibrate(time.time())
    assert asked == []

    # An OLD .poni (predating this calibration) is ignored.
    old = proj / "old.poni"
    old.write_text("x")
    os.utime(old, (1.0, 1.0))                       # far in the past
    w._autofill_poni_after_calibrate(time.time())
    assert asked == []

    # A .poni written during this calibration is offered, by path.
    t0 = time.time()
    fresh = proj / "fresh.poni"
    fresh.write_text("x")
    w._autofill_poni_after_calibrate(t0)
    assert asked and str(fresh) in asked[-1]

    # When several are recent, the newest is the one offered.
    asked.clear()
    newest = proj / "newest.poni"
    newest.write_text("x")
    os.utime(newest, (time.time() + 5, time.time() + 5))
    w._autofill_poni_after_calibrate(t0 - 1)
    assert str(newest) in asked[-1]


def test_gi_manual_incidence_reintegrate_uses_numeric_not_motor_name(widget):
    """Regression: GI **Manual** incidence (eiger / no metadata, no baked
    per-frame angle) must reintegrate at the entered theta value.  The integrator
    writes the NUMERIC theta to ``scan.incidence_motor`` — writing the literal
    'Manual' made the reduction raise "cannot resolve GI incident angle from
    metadata motor 'Manual'"; writing the panel-default 0.1 reintegrated at the
    wrong incidence."""
    from xdart.modules.reduction import plan_from_live_scan

    w = widget
    w._on_controls_v2_field_changed(("GI", "Grazing"), True)
    w._on_controls_v2_field_changed(("GI", "th_motor"), "Manual")
    w._on_controls_v2_field_changed(("GI", "th_val"), "2.0")
    w._controls_v2_apply_gi_config_to_scan()

    # Numeric theta, NOT the literal 'Manual' (which would crash the reduction).
    assert w.scan.incidence_motor == '2.0'
    assert w.scan.gi_config.get('th_val') == 2.0

    # The plan resolves a real incident angle (not None -> no per-frame crash).
    w.scan.bai_1d_args = {'gi_mode_1d': 'q_total'}
    w.scan.bai_2d_args = {'gi_mode_2d': 'qip_qoop'}
    plan = plan_from_live_scan(w.scan, integrate_1d=True, integrate_2d=False)
    assert plan.gi is not None
    assert plan.gi.incident_angle == 2.0


def test_hydrate_from_scan_populates_panel_from_saved_settings(widget):
    """Stage C (2-way sync): loading a scan hydrates the integration panel from
    its saved settings — GI section (incl. the Manual theta value, which fixes
    reload-Manual reintegrate), npts, and ranges — WITHOUT clobbering the saved
    ranges reintegrate relies on (set_image_units writes mode defaults, which
    hydrate restores over)."""
    w = widget
    it = w.integratorTree
    w.scan.gi = False  # hydrate must turn GI on from gi_config
    w.scan.gi_config = {
        'gi_mode_1d': 'q_total', 'gi_mode_2d': 'qip_qoop',
        'incidence_motor': 'Manual', 'th_val': 2.0,
        'sample_orientation': 4, 'tilt_angle': 0.0,
    }
    w.scan.bai_1d_args = {'gi_mode_1d': 'q_total', 'unit': 'q_A^-1',
                          'numpoints': 1234, 'radial_range': (0.5, 7.5)}
    w.scan.bai_2d_args = {'gi_mode_2d': 'qip_qoop', 'unit': 'q_A^-1',
                          'npt_rad': 800, 'npt_azim': 900}

    it.hydrate_from_scan()

    assert it.ui.gi_enable.isChecked() is True
    assert w.scan.gi is True
    assert it.ui.gi_motor.currentText() == 'Manual'
    assert it.ui.gi_motor_value.text() == '2.0'         # reload-Manual recovered
    assert it.ui.gi_sample_orientation.value() == 4
    assert it.ui.npts_1D.text() == '1234'
    # The saved ranges survive set_image_units' mode-default clobber.
    assert w.scan.bai_1d_args.get('radial_range') == (0.5, 7.5)
    assert it.ui.radial_low_1D.text() == '0.5'
    assert it.ui.radial_high_1D.text() == '7.5'


def test_gi_detail_widgets_live_in_hidden_holder(widget):
    """The GI detail fields (Motor / Value / Orient / Tilt) are no longer in a
    floating popup — the popup was removed in favour of inline Controls Panel V2
    rows.  The widgets are re-parented into a hidden holder so they stay ALIVE
    (get_gi_config / session / hydrate read these same objects) and round-trip;
    there is no popup window and no separate More button."""
    w = widget
    it = w.integratorTree

    # The floating popup is gone; the detail widgets live in a hidden holder.
    assert getattr(it, 'gi_more_popup', None) is None
    assert getattr(it.ui, 'gi_more', None) is None
    holder = it.ui.gi_hidden_holder
    assert not holder.isVisible()
    for wdg in (it.ui.gi_motor, it.ui.gi_motor_value,
                it.ui.gi_sample_orientation, it.ui.gi_tilt):
        # Re-parented into the hidden holder and kept alive (never destroyed).
        assert wdg.parent() is holder

    # Motor + Orient + Tilt still round-trip through the native V2 state; the
    # hidden widgets stay alive for the advanced inspector only.
    w._on_controls_v2_field_changed(("GI", "Grazing"), True)
    w._on_controls_v2_field_changed(("GI", "th_motor"), "Manual")
    w._on_controls_v2_field_changed(("GI", "th_val"), "0.3")
    w._on_controls_v2_field_changed(("GI", "sample_orientation"), "6")
    w._on_controls_v2_field_changed(("GI", "tilt_angle"), "1.5")
    cfg = it.get_gi_config()
    assert cfg['incidence_motor'] == 'Manual'
    assert cfg['th_val'] == 0.3
    assert cfg['sample_orientation'] == 6
    assert cfg['tilt_angle'] == 1.5
    assert cfg == w._controls_v2_gi_config()
    w._controls_v2_apply_gi_config_to_scan()
    assert w.scan.gi_config['incidence_motor'] == 'Manual'
    assert w.scan.gi_config['th_val'] == 0.3
    assert w.scan.gi_config['sample_orientation'] == 6
    assert w.scan.gi_config['tilt_angle'] == 1.5


def test_stitch_modes_present_and_mode_flags(widget):
    """Stitch 1D / Stitch 2D appear in the image wrangler's mode list and set the
    stitch_mode flag (+ the 1D/2D skip_2d) without being viewer modes."""
    w = widget
    wr = w.wrangler
    assert {'Stitch 1D', 'Stitch 2D'} <= set(wr.controls_profile()['modes'])
    combo = w.controls.modeCombo
    combo.setCurrentIndex(combo.findText('Stitch 1D'))
    assert wr.stitch_mode is True and w.scan.skip_2d is True
    assert wr.viewer_mode is None
    combo.setCurrentIndex(combo.findText('Stitch 2D'))
    assert wr.stitch_mode is True and w.scan.skip_2d is False
    combo.setCurrentIndex(combo.findText('Int 2D'))
    assert wr.stitch_mode is False


def test_run_in_stitch_mode_routes_to_stitch_not_wrangler(widget):
    """A Run click while a Stitch mode is active emits sigStitchRequested (→
    start_stitch) and does NOT emit sigStart (no wrangler run) nor morph the
    action button to Pause — it stays a green 'Run' (stitch is Start/Stop only)."""
    w = widget
    wr = w.wrangler
    combo = w.controls.modeCombo
    combo.setCurrentIndex(combo.findText('Stitch 1D'))
    got = {'stitch': None, 'start': False}
    wr.sigStitchRequested.connect(lambda m: got.__setitem__('stitch', m))
    wr.sigStart.connect(lambda: got.__setitem__('start', True))
    wr.poni = object()                 # satisfy _inputs_valid (PONI loaded)
    wr.start()
    assert got['stitch'] == '1d'
    assert got['start'] is False
    assert w.controls.startButton.text() == '▶ Run'      # no Pause morph


def test_build_stitch_params_reads_integrator_args(widget):
    """_build_stitch_params pulls npts/unit/ranges/method from the scan's
    bai_*_args, leaves mask None, and passes no `backend` kwarg (Phase 1a)."""
    w = widget
    w.scan.bai_1d_args.update(numpoints=1234, unit='q_A^-1', method='csr',
                              radial_range=(0.0, 5.0), azimuth_range=None)
    p1 = w._build_stitch_params('1d')
    assert p1['npt_1d'] == 1234 and p1['unit'] == 'q_A^-1'
    assert p1['method'] == 'csr' and p1['radial_range'] == (0.0, 5.0)
    assert p1['mask'] is None and 'backend' not in p1
    w.scan.bai_2d_args.update(npt_rad=600, npt_azim=360, unit='2th_deg')
    p2 = w._build_stitch_params('2d')
    assert p2['npt_rad_2d'] == 600 and p2['npt_azim_2d'] == 360
    assert p2['unit'] == '2th_deg' and 'npt_1d' not in p2


def test_reintegrate_live_default_and_stop_wiring(widget, monkeypatch):
    """Reintegrate runs LIVE by default and switches to the fast batch path via
    the shared Batch toggle (no separate Live button); the shared Stop dispatches
    to the active run — a running reintegrate takes priority, else the wrangler."""
    w = widget
    it = w.integratorTree

    # Batch is off by default → live; checking Batch → fast multicore.  Driven
    # by the shared controls.batchButton through the host-installed provider.
    assert w.controls.batchButton.isChecked() is False        # live default
    assert it._reintegrate_is_live() is True
    w.controls.batchButton.setChecked(True)
    assert it._reintegrate_is_live() is False
    w.controls.batchButton.setChecked(False)
    assert it._reintegrate_is_live() is True

    # No reintegrate running → Stop dispatches to the wrangler, not the
    # integrator (its stop flag is untouched).
    wrangler_stops = []
    monkeypatch.setattr(w.wrangler, 'stop', lambda: wrangler_stops.append(1))
    it.integrator_thread.stop_requested = False
    w._on_stop_clicked()
    assert it.integrator_thread.stop_requested is False
    assert wrangler_stops == [1]

    # Reintegrate running → Stop asks before discarding the staged shadow write.
    # If confirmed, it aborts the reintegrate and does NOT also trip the
    # wrangler (no idle-stop side effects).
    monkeypatch.setattr(it.integrator_thread, 'isRunning', lambda: True)
    monkeypatch.setattr(w, '_confirm_discard_reintegrate', lambda: True)
    w._on_stop_clicked()
    assert it.integrator_thread.stop_requested is True
    assert wrangler_stops == [1]

    # Entering run-state for a reintegrate LOCKS Start (so a scan can't rebuild
    # scan.frames out from under the reintegrate loop); _exit returns to the
    # normal idle readiness gate (this fresh widget is still not ready).
    w._run_active = False
    w._enter_run_state()
    assert not w.controls.startButton.isEnabled()
    w._exit_run_state()
    assert w.controls.actionRow.isEnabled() is False

    # Per-frame reintegrate updates are THROTTLED (coalesced), not rendered
    # synchronously — integrator_thread_update just stashes the latest index;
    # the timer's _flush_reintegrate_update does the actual refresh.
    w._pending_reint_idx = None
    w.integrator_thread_update(7)
    assert w._pending_reint_idx == 7

    # Streaming reintegrate only swaps the shadow stack into place when the full
    # requested pass finishes. Stop therefore always warns first: "let it
    # finish" cancels the stop; "stop & discard" proceeds.
    it.integrator_thread.stop_requested = False
    monkeypatch.setattr(w, '_confirm_discard_reintegrate', lambda: False)
    w._on_stop_clicked()
    assert it.integrator_thread.stop_requested is False        # cancelled
    monkeypatch.setattr(w, '_confirm_discard_reintegrate', lambda: True)
    w._on_stop_clicked()
    assert it.integrator_thread.stop_requested is True         # discarded


def test_threshold_autoenables_on_value_entry(widget):
    """Entering a non-default Threshold Min/Max auto-enables the Threshold toggle
    so the clip actually applies (the "I set Max=1000 but it didn't clip" trap).
    Default 0/0 never auto-enables (Max=0 would mask everything)."""
    w = widget

    w._on_controls_v2_field_changed(("Mask", "Threshold"), False)
    w._on_controls_v2_field_changed(("Mask", "max"), "1000")
    cfg = w._controls_v2_threshold_config()
    assert cfg.apply_threshold is True and cfg.threshold_max == 1000.0

    # Default 0/0 must NOT auto-enable (Max=0 masks everything).
    w._on_controls_v2_field_changed(("Mask", "max"), "0")
    w._on_controls_v2_field_changed(("Mask", "Threshold"), False)
    w._on_controls_v2_field_changed(("Mask", "min"), "0")
    w._on_controls_v2_field_changed(("Mask", "max"), "0")
    cfg = w._controls_v2_threshold_config()
    assert cfg.apply_threshold is False
    assert cfg.threshold_min == 0.0 and cfg.threshold_max == 0.0


def test_shared_controls_reroute_on_wrangler_swap(widget):
    """Stage 2b: the ONE StaticControls bar follows the active wrangler.  On
    swap, apply_profile repopulates the mode items + shows/hides Live/Batch for
    the new wrangler, the new wrangler's control refs ALIAS onto the shared
    widgets, and the old wrangler's signal connections are dropped (no stale
    double-dispatch)."""
    w = widget
    stack = w.ui.wranglerStack
    ctrls = w.controls

    # Resolve the image / nexus stack indices by wrangler class name.
    idx = {type(stack.widget(i)).__name__: i for i in range(stack.count())}
    img_i = idx['imageWrangler']
    nx_i = idx['nexusWrangler']

    # Start on the image wrangler: Live + Batch present, full image mode list,
    # and the image wrangler's ``self.ui.*`` refs alias the shared widgets.
    stack.setCurrentIndex(img_i)
    assert not ctrls.liveButton.isHidden()
    assert not ctrls.batchButton.isHidden()
    img_modes = [ctrls.modeCombo.itemText(i) for i in range(ctrls.modeCombo.count())]
    assert 'Int 2D' in img_modes and 'NeXus Viewer' not in img_modes
    assert w.wrangler.ui.processingModeCombo is ctrls.modeCombo
    assert w.wrangler.ui.startButton is ctrls.startButton
    assert w.wrangler._control_conns                     # image owns the wiring

    # Swap to the NeXus wrangler: Live + Batch hidden, the 3 NeXus modes, and
    # the nexus wrangler aliases the SAME shared widgets.
    stack.setCurrentIndex(nx_i)
    assert ctrls.liveButton.isHidden()
    assert ctrls.batchButton.isHidden()
    nx_modes = [ctrls.modeCombo.itemText(i) for i in range(ctrls.modeCombo.count())]
    assert nx_modes == ['Int 1D + 2D', 'Int 1D', 'Int 1D (XYE)']
    assert w.wrangler.processingModeCombo is ctrls.modeCombo
    assert w.wrangler.startButton is ctrls.startButton
    assert w.wrangler._control_conns                     # nexus now owns wiring
    # The image wrangler released its shared-control connections on swap.
    img_w = stack.widget(img_i)
    assert not img_w._control_conns

    # Swap back to image: Live/Batch return, image modes restored.
    stack.setCurrentIndex(img_i)
    assert not ctrls.liveButton.isHidden()
    assert not ctrls.batchButton.isHidden()
    back_modes = [ctrls.modeCombo.itemText(i) for i in range(ctrls.modeCombo.count())]
    assert 'Int 2D' in back_modes


def test_stitch_display_persists_across_updates(widget):
    """The persistent StitchDisplayController keeps a merged result on screen
    across subsequent update() calls (the regression the one-shot render had),
    and returns to the per-frame view when the Stitch mode is left or the result
    is cleared."""
    from xdart.gui.tabs.static_scan.display_logic import Mode
    from xrd_tools.core.containers import IntegrationResult1D
    w = widget
    df = w.displayframe
    # No frames selected → the per-frame view has nothing; the stitch result is
    # the only thing to draw.  get_idxs early-returns on empty frame_ids and
    # _updated() returns True because a stitch is active.
    df.frame_ids[:] = []
    radial = np.linspace(1.0, 5.0, 40)
    df.scan.stitched_2d = None
    df.scan.stitched_1d = IntegrationResult1D(
        radial=radial, intensity=np.linspace(10.0, 50.0, 40), unit="q_A^-1")
    df.stitch_display_mode = '1d'

    assert df._active_stitch_mode() == '1d'
    assert df._live_mode() is Mode.STITCH_1D

    df.update()
    items = df.plot.listDataItems()
    assert items, "stitch curve should be drawn"
    x0 = np.asarray(items[0].getData()[0], dtype=float)
    assert x0.size == 40
    np.testing.assert_allclose([x0.min(), x0.max()], [1.0, 5.0], atol=1e-6)

    # Persistence: the NEXT update() (the call that used to overwrite the one-shot
    # paint with the per-frame view) must keep the stitch.
    df.update()
    assert df._live_mode() is Mode.STITCH_1D
    items2 = df.plot.listDataItems()
    assert items2
    np.testing.assert_allclose(
        np.asarray(items2[0].getData()[0], dtype=float), x0)

    # Leaving the Stitch mode (dropdown → a per-frame mode) returns to per-frame.
    df.stitch_display_mode = None
    assert df._active_stitch_mode() is None
    assert df._live_mode() in (Mode.INT_1D, Mode.INT_2D)

    # Result-existence guard: a Stitch mode selected with no result also falls
    # back to per-frame (no premature blank before a run / after new_scan clears).
    df.stitch_display_mode = '1d'
    df.scan.stitched_1d = None
    assert df._active_stitch_mode() is None
    assert df._live_mode() in (Mode.INT_1D, Mode.INT_2D)


def test_build_stitch_params_converts_flat_mask_to_2d_bool(widget):
    """_build_stitch_params turns the scan's flat-index detector mask + detector_
    shape into the 2D boolean (True = exclude) run_stitch/pyFAI want; it degrades
    to None (mask dropped, never crashes) when the shape is unknown."""
    w = widget
    w.scan.detector_shape = (4, 5)
    w.scan.global_mask = np.array([0, 7])          # flat indices 0 and 7 masked
    w.scan.bai_1d_args = {'unit': 'q_A^-1', 'method': 'BBox', 'numpoints': 1000}
    m = w._build_stitch_params('1d')['mask']
    assert m is not None and m.shape == (4, 5) and m.dtype == bool
    assert m.ravel()[0] and m.ravel()[7] and m.sum() == 2
    # Unknown detector shape → converter returns None (graceful, no crash).
    w.scan.detector_shape = None
    assert w._build_stitch_params('1d')['mask'] is None
    # No mask at all → None.
    w.scan.detector_shape = (4, 5)
    w.scan.global_mask = None
    assert w._build_stitch_params('1d')['mask'] is None


def test_gi_mode_blocks_stitch_with_message(widget):
    """A Stitch launched while GI (Fiber) mode is ON is blocked with a clear
    message: the GUI stitch is the multigeometry (non-GI) backend, and the
    GI-corrected path is gated on real-data validation, so it must not silently
    run a non-GI merge.  With GI off the guard passes and the worker starts."""
    w = widget
    msgs, started = [], []
    w._stitch_status = lambda m: msgs.append(m)
    w.stitch_thread.start = lambda *a, **k: started.append(True)
    # frames + geometry present so start_stitch reaches the GI guard.
    w.scan.frames = [object()]
    w.scan.geometry = object()
    w.scan.bai_1d_args = {'unit': 'q_A^-1', 'method': 'BBox', 'numpoints': 1000}
    w.scan.global_mask = None
    w.scan.detector_shape = None

    w.scan.gi = True
    w.start_stitch('1d')
    assert not started, "stitch must not start while GI mode is active"
    assert msgs and 'GI' in msgs[-1]

    msgs.clear(); started.clear()
    w.scan.gi = False
    w.start_stitch('1d')
    assert started, "stitch should start when GI is off"
