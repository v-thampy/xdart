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
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.gui

_DATA = Path(os.environ.get("XDART_TEST_DATA",
                            Path(__file__).resolve().parents[2] / "test_data"))


@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def widget(qapp):
    """A real staticWidget, torn down after each test."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    w = staticWidget()
    try:
        yield w
    finally:
        # Stop the long-running background threads the widget started before the
        # module-scoped QApplication GCs it.  Otherwise the still-running
        # fileHandlerThread (started in H5Viewer._init_file_thread) triggers
        # "QThread: Destroyed while thread is still running" -> abort at
        # interpreter shutdown.
        try:
            h5v = w.h5viewer
            h5v.cancel_pending_loads()            # quit+wait the load worker
            ft = getattr(h5v, "file_thread", None)
            if ft is not None:
                ft.queue.put(None)                # sentinel: clean run() exit
                ft.wait(2000)
            pool = getattr(h5v, "_h5pool", None)
            if pool is not None:
                pool.close_all()
        except Exception:
            pass
        try:
            w.close()
        except Exception:
            pass
        qapp.processEvents()


def _set_image_frame(w, idx, raw):
    """Put one raw frame into the shared viewer dicts + select it."""
    with w.data_lock:
        w.data_1d.clear()
        w.data_2d.clear()
        w.data_2d[idx] = {"map_raw": raw, "bg_raw": 0, "mask": None,
                          "int_2d": None, "gi_2d": {}, "thumbnail": None}
    w.frame_ids[:] = [str(idx)]
    w.displayframe.idxs_2d = [idx]


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
    # data_2d/frame_ids were cleared by the transition cleanup.
    assert len(w.data_2d) == 0

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
    viewer's loaded ``data_2d`` keys, so ``idxs_2d`` comes back empty and
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
        df.data_1d.clear()
        df.data_2d.clear()
        for k in (1, 2, 3):
            df.data_2d[k] = {"map_raw": raw, "bg_raw": 0, "mask": None,
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
        w.data_1d.clear()
        w.data_2d.clear()
    w.frame_ids[:] = []
    df.idxs_2d = []
    df.update()
    assert df.image_data is None


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
        w.data_1d.clear()
        w.data_2d.clear()
        for idx, src, radial, intensity in frames:
            w.data_1d[idx] = SimpleNamespace(
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
        w.data_1d.clear()
        w.data_2d.clear()
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
        w.data_1d.clear()
        w.data_2d.clear()
        w.data_1d[idx] = SimpleNamespace(
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

def _set_int_scan(w, *, n=1, wavelength_m=1.0e-10):
    """Populate the real widget for an Int 2D scan: data_2d + data_1d +
    publications + a scan stub, selected, in Int 2D mode (non-GI, q-integrated)."""
    import threading
    from ssrl_xrd_tools.core import IntegrationResult1D, IntegrationResult2D
    from xdart.modules.frame_publication import publication_from_live_frame
    df = w.displayframe
    q = np.linspace(0.5, 3.0, 5)
    chi = np.linspace(-90.0, 90.0, 4)
    w.publication_store.clear()
    with w.data_lock:
        w.data_1d.clear()
        w.data_2d.clear()
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
            w.data_2d[i] = {"map_raw": f.map_raw, "bg_raw": 0, "mask": None,
                            "int_2d": f.int_2d, "gi_2d": {}, "thumbnail": None}
            w.data_1d[i] = f
            w.publication_store.upsert(publication_from_live_frame(f))
    w.frame_ids[:] = [str(i) for i in range(n)]
    df.idxs_2d = list(range(n))
    df.idxs_1d = list(range(n))
    df.scan = SimpleNamespace(
        scan_lock=threading.RLock(),
        frames=SimpleNamespace(index=list(range(n))),
        gi=False, skip_2d=False, name="scan", global_mask=None,
        scan_data=SimpleNamespace(columns=[]), bai_2d_args={},
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
    w.publication_store.clear()                     # data_2d kept, publications gone
    df.update()
    assert df.binned_data is None                   # cake blanked, not stale


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

    # XYE Viewer: a file browser — inputs disabled.
    _set_mode("XYE Viewer")
    w._on_viewer_mode_changed("xye")
    assert tree.isEnabled() is False


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
    # stay enabled.
    _mode("XYE Viewer")
    assert not iu.frame1D.isEnabled() and not iu.frame2D.isEnabled()
    assert iu.pyfai_calib.isEnabled() and iu.get_mask.isEnabled()

    # Back to Int 2D restores everything.
    _mode("Int 2D")
    assert iu.frame1D.isEnabled() and iu.frame2D.isEnabled()


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


# ── R2-1: slice X-Range label = complementary 2D axis, refreshed on change ──

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
    assert df.ui.slice.text().endswith("Range")
    assert Chi not in df.ui.slice.text()           # GI axis, not χ

    # GI -> non-GI: set_axes rebuilds (plotUnit Q over a Q-χ cake) and the label
    # must refresh to "χ Range" immediately — not stay the stale GI label.
    df.scan.gi = False
    df.set_axes()
    assert Chi in df.ui.slice.text()               # refreshed, no click needed
