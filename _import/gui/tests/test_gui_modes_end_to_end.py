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
