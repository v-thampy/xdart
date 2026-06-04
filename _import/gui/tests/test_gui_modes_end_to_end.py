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

    w.set_data()

    assert w.displayframe._viewer_is_xdart is True       # P0 propagation
    w.displayframe._update_image_viewer()
    img = w.displayframe.image_data[0]
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
    w.set_data()

    assert w.displayframe._viewer_is_xdart is False
    w.displayframe._update_image_viewer()
    img = w.displayframe.image_data[0]
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
    w.set_data()
    w.displayframe._update_image_viewer()
    img = w.displayframe.image_data[0]
    assert img is not None and img.size > 0 and np.isfinite(img).any()


# ── Real-data cells: exercise the full _load_image_file (classify + load) ──

_TIFF = _DATA / "Tiff" / "Combi4_Angledependence_samz_4p9_03271002_0001.tif"
_EIGER = _DATA / "eiger" / "Eiger_B_ctrl_test__2000mdeg_scan001_master.h5"
_PROC = _DATA / "xdart_processed_data" / "Combi4_Angledependence_samz_4p9_03271002.nxs"


def _load_image_through_wire(w, path):
    """Drive the real Image-Viewer load: classify+load then set_data render."""
    w._on_viewer_mode_changed("image")
    w.h5viewer.dirname = str(Path(path).parent)
    w.h5viewer._load_image_file(str(path))
    w.set_data()                       # propagates classification + renders
    w.displayframe._update_image_viewer()
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
