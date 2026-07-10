# -*- coding: utf-8 -*-
"""H5 — Stage-6 readiness PARITY TEST: freeze the two gating truth sources together.

The controls-panel gating logic currently lives TWICE:

* **GUI-inline** — ``staticWidget._controls_v2_state`` builds ``SourceCaps`` via
  ``staticWidget._controls_v2_source_caps`` (the collapsed
  ``has_frames = has_raw = raw_reachable = source_ready`` simplification, with
  ``source_ready = bool(source_label) and (live_unknown or frame_count > 0)``)
  and builds ``ResultCaps`` inline (``has_raw = raw_reachable = source_ready or
  loaded_scan_available``) — ``src/xdart/gui/tabs/static_scan/static_scan_widget.py``.
* **Headless** — ``describe_source_readiness`` / ``capabilities_for_processed`` —
  ``src/xrd_tools/sources/readiness.py``.

Both sides project into the SAME frozen dataclasses
(``xrd_tools.session.readiness.SourceCaps`` / ``ResultCaps``), so fields compare
directly.  These tests drive a REAL ``staticWidget`` (real wrangler parameters,
real fixture files on disk, no fakes on the seam) and the real headless calls
over one fixture matrix, and assert field-by-field parity.  Where the two sides
legitimately DIVERGE today, the divergence is pinned on BOTH sides with a
comment recording WHICH SIDE WINS — that record is the H18 migration contract:
H18 delegates the GUI gating to the headless core, and every pinned divergence
below is an explicit H18 decision to resolve, not a behavior to flip silently.

Matrix (SourceCaps side; ResultCaps parity applies to the processed rows only —
``capabilities_for_processed`` is defined over processed metadata, so a
source-only GUI ``ResultCaps`` has no headless counterpart until H18):

    case                       tri-fields (has_frames/has_raw/raw_reachable)
    SPEC source                GUI F/F/F   vs headless T/T/F   (headless wins)
    Eiger master               GUI T/T/T   ==  headless T/T/T  (full agreement)
    processed + reachable raw  GUI F/F/F   vs headless T/T/T   (headless wins)
    processed, raw missing     GUI F/F/F   vs headless T/T/F   (headless wins)
    live / unknown length      GUI T/T/T   ==  headless T/T/T  (agree, different
                               mechanisms: inline live_unknown collapse vs the
                               headless true-live escape hatch)
    nonexistent path           GUI F/F/F   vs headless T/T/F   (GUI wins)

READ-ONLY chunk: this file adds tests only; pinning a divergence documents it,
it does not endorse it.
"""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gc

import h5py
import numpy as np
import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtWidgets

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io.read import get_metadata
from xrd_tools.sources.readiness import (
    capabilities_for_processed,
    describe_source_readiness,
)
from xdart.gui.tabs.static_scan.controls_logic import RunTarget

SOURCE_FIELDS = (
    "has_frames", "has_raw", "raw_reachable", "has_metadata",
    "has_motors", "has_energy", "has_geometry", "has_psi_metadata",
)
RESULT_FIELDS = (
    "has_1d", "has_2d", "has_raw", "raw_reachable", "has_scan_metadata",
    "has_rsm", "has_phase_result", "has_psi_metadata",
)

_SPEC = """#F myscan
#E 1
#O0 th  chi

#S 5 ascan th 0 2 2 1
#P0 0 5
#N 3
#L th  i0  det
0 100 10
1 110 20
2 120 30
"""


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _controls_panel_session_isolation():
    """``staticWidget.close()`` persists the integrator session; keep it from
    leaking between tests (same guard as test_controls_panel_v2)."""
    path = os.environ.get("XDART_SESSION_FILE")

    def _unlink_session():
        if not path:
            return
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass

    _unlink_session()
    yield
    _unlink_session()


@pytest.fixture(autouse=True)
def _drain_qt_events_after_test(qapp):
    yield
    for _ in range(3):
        qapp.processEvents()
    gc.collect()
    for _ in range(2):
        qapp.processEvents()


# ── fixture builders (real files, mirroring test_scan_source_widget /
#    test_image_source) ────────────────────────────────────────────────


def _spec_with_images(tmp_path):
    """SPEC file + sibling scan-5 .raw frames (needs shape/dtype params to read)."""
    spec = tmp_path / "myscan"
    spec.write_text(_SPEC)
    for i in range(3):
        np.full((6, 6), i + 1, dtype="int32").tofile(
            tmp_path / f"myscan_scan5_{i:04d}.raw")
    return spec


def _eiger_master(tmp_path):
    """Minimal Eiger-style raw master: ``*_master.h5`` with entry/data/data."""
    master = tmp_path / "scan_master.h5"
    raw = np.arange(2 * 8 * 8, dtype=np.uint32).reshape(2, 8, 8)
    with h5py.File(master, "w") as f:
        f.create_dataset("entry/data/data", data=raw)
    return master


def _write_thumbnail(group, name, data):
    vmin, vmax = float(data.min()), float(data.max())
    span = (vmax - vmin) or 1.0
    q = np.clip((data - vmin) / span, 0, 1) * 255.0
    ds = group.create_dataset(name, data=q.astype(np.uint8))
    ds.attrs["vmin"] = vmin
    ds.attrs["vmax"] = vmax
    ds.attrs["dtype"] = "uint8"


def _processed_nxs(tmp_path, *, raw_reachable):
    """Processed v2 ``.nxs`` (1 frame, integrated_1d + frames record).

    ``raw_reachable=True`` writes a resolvable sibling raw master;
    ``raw_reachable=False`` points the frame record at a missing master.
    """
    if raw_reachable:
        master_name, nxs = b"scan_master.h5", tmp_path / "scan.nxs"
        _eiger_master(tmp_path)
    else:
        master_name, nxs = b"does_not_exist.h5", tmp_path / "thumb_only.nxs"
    thumb = np.linspace(0, 100, 16 * 16).reshape(16, 16)
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")
        g = e.create_group("integrated_1d")
        g.create_dataset("intensity", data=np.zeros((1, 5)))
        g.create_dataset("frame_index", data=np.array([0], dtype=np.int64))
        s = e.create_group("frames/frame_0000/source")
        s.create_dataset("path", data=np.bytes_(master_name))
        s.create_dataset("frame_index", data=1 if raw_reachable else 0)
        _write_thumbnail(e["frames/frame_0000"], "thumbnail", thumb)
    return nxs


def _gui_state(configure):
    """Real widget → real inline state: build a ``staticWidget``, apply the
    production parameter edits, snapshot ``_controls_v2_state()``."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        configure(widget)
        return widget._controls_v2_state()
    finally:
        widget.close()
        widget.deleteLater()


def _assert_agree(gui, headless, fields):
    for field in fields:
        assert getattr(gui, field) is getattr(headless, field), (
            f"{field}: inline={getattr(gui, field)!r} "
            f"headless={getattr(headless, field)!r}")


# ── the parity matrix ─────────────────────────────────────────────────


def test_spec_source_parity(qapp, tmp_path):
    pytest.importorskip("silx")
    spec = _spec_with_images(tmp_path)

    headless = describe_source_readiness(str(spec))
    state = _gui_state(
        lambda w: w._controls_v2_param(("Signal", "File")).setValue(str(spec)))
    gui = state.source_caps

    # AGREE — raw_reachable: neither side can prove frame 0 loads (the .raw
    # sidecar images need explicit shape/dtype read params neither side has
    # here), though they arrive there differently: the headless frame-0 probe
    # fails; the inline side collapses from source_ready=False.
    assert gui.raw_reachable is False
    assert headless.raw_reachable is False

    # DIVERGE (headless wins) — has_frames / has_raw: the inline panel counts
    # frames with image_io.count_frames, which cannot parse a SPEC scan table,
    # so the collapsed source_ready gate reports NO frames for a perfectly
    # valid 3-point SPEC scan.  describe_source_readiness opens a real
    # SpecSource and sees the scan rows.  H18: delegating to the headless core
    # FIXES this under-gating — keep the headless answer.
    assert gui.has_frames is False        # inline: count_frames(spec) == 0
    assert gui.has_raw is False
    assert headless.has_frames is True    # headless: SpecSource scan rows
    assert headless.has_raw is True

    # DIVERGE (headless wins) — metadata family: SpecSource serves the scan
    # table (#L columns), motors (#O/#P) and the psi-family "chi" column at
    # gate time; the panel only learns metadata once a run hydrates
    # scan.scan_data, so before any run it reports False.
    assert gui.has_metadata is False and headless.has_metadata is True
    assert gui.has_motors is False and headless.has_motors is True
    assert gui.has_psi_metadata is False and headless.has_psi_metadata is True

    # AGREE — no energy record in the SPEC table, no calibration on either side.
    _assert_agree(gui, headless, ("has_energy", "has_geometry"))


def test_eiger_master_parity(qapp, tmp_path):
    master = _eiger_master(tmp_path)

    headless = describe_source_readiness(str(master))

    def cfg(w):
        w._controls_v2_param(("Signal", "inp_type")).setValue("Image Series")
        w._controls_v2_param(("Signal", "File")).setValue(str(master))
        w._controls_v2_param(("Signal", "img_ext")).setValue("h5")

    state = _gui_state(cfg)
    gui = state.source_caps

    # FULL AGREEMENT — the one matrix row where the inline collapse and the
    # headless probe coincide on every field: both count 2 frames, both load
    # frame 0, and neither claims metadata/motors/energy/geometry/psi for a
    # bare detector master.  This row is the H18 no-op case.
    assert state.frame_count == 2
    assert gui.has_frames is True and headless.has_frames is True
    assert gui.has_raw is True and headless.has_raw is True
    assert gui.raw_reachable is True and headless.raw_reachable is True
    _assert_agree(gui, headless, SOURCE_FIELDS)
    assert gui == headless
    assert state.run_target is RunTarget.SOURCE


def test_processed_with_reachable_raw_parity(qapp, tmp_path):
    nxs = _processed_nxs(tmp_path, raw_reachable=True)

    headless_src = describe_source_readiness(str(nxs))
    headless_res = capabilities_for_processed(get_metadata(nxs))

    def cfg(w):
        w._controls_v2_param(("Signal", "File")).setValue(str(nxs))
        w._controls_v2_param(("Signal", "img_ext")).setValue("nxs")
        # The GUI's loaded-scan handle — same setup as the loaded-scan cases
        # in test_controls_panel_v2 (a full h5viewer load additionally
        # hydrates scan_data / viewer rows; see the ResultCaps pins below).
        w.scan.data_file = str(nxs)

    state = _gui_state(cfg)
    gui_src, gui_res = state.source_caps, state.result_caps

    # SourceCaps tri-fields — DIVERGE (headless wins): count_frames cannot
    # count a processed record (no raw image dataset), so the inline collapse
    # reports F/F/F; ProcessedNexusSource sees the frame record, and the
    # frame-0 probe resolves the sibling raw master → T/T/T.
    assert gui_src.has_frames is False
    assert gui_src.has_raw is False
    assert gui_src.raw_reachable is False
    assert headless_src.has_frames is True
    assert headless_src.has_raw is True
    assert headless_src.raw_reachable is True

    # DIVERGE (headless wins on source truth) — the record carries metadata
    # and geometry; the panel derives both from PANEL state (no PONI picked,
    # scan_data not hydrated), so it reports False.  H18 note: for
    # has_geometry the panel's answer stays authoritative for RUN gating (the
    # run needs the panel's calibration, not the record's), while the
    # headless answer is the record truth for viewer/launcher gating.
    assert gui_src.has_metadata is False and headless_src.has_metadata is True
    assert gui_src.has_geometry is False and headless_src.has_geometry is True
    _assert_agree(gui_src, headless_src, ("has_motors", "has_energy"))

    # The inline compensation for its all-False SourceCaps: the run gate
    # falls back to the loaded scan, so the panel does not dead-end.
    assert state.loaded_scan_available is True
    assert state.run_target is RunTarget.LOADED_SCAN

    # ResultCaps — AGREE on raw: both True, with different provenance
    # (inline: source_ready OR loaded_scan_available; headless: the
    # frames_record capability written by get_metadata).
    assert gui_res.has_raw is True and headless_res.has_raw is True
    assert gui_res.raw_reachable is True and headless_res.raw_reachable is True
    _assert_agree(gui_res, headless_res,
                  ("has_2d", "has_rsm", "has_phase_result"))

    # DIVERGE (headless wins — record truth vs hydration state): the file
    # contains 1D results and a scan table, but the inline panel only reports
    # them after the load worker hydrates viewer rows / scan_data.  For
    # gating a freshly opened processed scan the headless record answer is
    # the truth; H18 should read the record, not the hydration mirrors.
    assert gui_res.has_1d is False and headless_res.has_1d is True
    assert gui_res.has_scan_metadata is False \
        and headless_res.has_scan_metadata is True
    assert gui_res.has_psi_metadata is False \
        and headless_res.has_psi_metadata is True


def test_processed_without_reachable_raw_parity(qapp, tmp_path):
    nxs = _processed_nxs(tmp_path, raw_reachable=False)

    headless_src = describe_source_readiness(str(nxs))
    headless_res = capabilities_for_processed(get_metadata(nxs))

    def cfg(w):
        w._controls_v2_param(("Signal", "File")).setValue(str(nxs))
        w._controls_v2_param(("Signal", "img_ext")).setValue("nxs")
        w.scan.data_file = str(nxs)

    state = _gui_state(cfg)
    gui_src, gui_res = state.source_caps, state.result_caps

    # SourceCaps — the headless probe is the ONLY truth-teller in the whole
    # readiness surface for a moved/deleted raw master: has_frames/has_raw
    # stay True (the record exists) but the frame-0 probe fails.
    assert headless_src.has_frames is True
    assert headless_src.has_raw is True
    assert headless_src.raw_reachable is False   # ← probe truth

    # DIVERGE (headless wins) — same inline collapse as the reachable-raw
    # case: count_frames sees no frames, so F/F/F; the collapse cannot even
    # EXPRESS "record present, raw missing" (its three fields are one bit).
    assert gui_src.has_frames is False
    assert gui_src.has_raw is False
    assert gui_src.raw_reachable is False        # ← right value, wrong reason

    # ResultCaps — JOINT overstatement (documented, not endorsed): BOTH sides
    # claim raw_reachable=True while the master is gone.  The inline side
    # collapses to loaded_scan_available; capabilities_for_processed mirrors
    # the frames_record capability WITHOUT probing (by design — it must not
    # reopen HDF5).  H18 decision: raw-dependent launcher gates must consult
    # the describe_source_readiness probe (above, False) instead of trusting
    # ResultCaps.raw_reachable.
    assert gui_res.has_raw is True and gui_res.raw_reachable is True
    assert headless_res.has_raw is True and headless_res.raw_reachable is True
    assert headless_src.raw_reachable is False   # the probe contradicts both

    # Same record-vs-hydration divergences as the reachable-raw case.
    assert gui_res.has_1d is False and headless_res.has_1d is True
    _assert_agree(gui_res, headless_res,
                  ("has_2d", "has_rsm", "has_phase_result"))


def test_live_unknown_length_parity(qapp, tmp_path):
    master = _eiger_master(tmp_path)

    def cfg(w):
        w._controls_v2_param(("Signal", "File")).setValue(str(master))
        # The real live toggle: the checkbox drives wrangler.live_mode, which
        # _controls_v2_live_source_active reads.
        w.wrangler.ui.liveCheckBox.setChecked(True)

    state = _gui_state(cfg)
    gui = state.source_caps
    headless = describe_source_readiness(SourceSpec(str(master), SourceKind.LIVE))

    # THE H5 reconcile point (master table): both sides land T/T/T for a live
    # run with a configured source, but by DIFFERENT mechanisms — the inline
    # collapse treats live_unknown as ready (frame count None short-circuits
    # source_ready), while the headless side applies the true-live escape
    # hatch (a live acquisition may legitimately have no frame 0 yet, so
    # raw_reachable stays True even when the probe cannot load an image).
    # H18 must preserve BOTH behaviors when delegating: the escape hatch is
    # the principled home for the inline shortcut.
    assert gui.has_frames is True and headless.has_frames is True
    assert gui.has_raw is True and headless.has_raw is True
    assert gui.raw_reachable is True and headless.raw_reachable is True
    # Unknown length renders as frame_count 0 while source_ready stays True.
    assert state.frame_count == 0
    assert state.run_target is RunTarget.SOURCE

    # DIVERGE (panel wins for run gating) — LiveFrameSource optimistically
    # advertises metadata/geometry capabilities; the panel gates on ACTUAL
    # panel state (no PONI picked, nothing hydrated).  A live run without a
    # calibration must stay blocked, so H18 keeps the panel's answer for the
    # run gate and treats the source's claim as advisory.
    assert gui.has_metadata is False and headless.has_metadata is True
    assert gui.has_geometry is False and headless.has_geometry is True
    _assert_agree(gui, headless,
                  ("has_motors", "has_energy", "has_psi_metadata"))


def test_live_without_configured_source_parity(qapp):
    # Live with NO configured source label — the sides genuinely disagree.
    # DIVERGE (GUI wins): the inline gate requires a source label even in
    # live mode (a live run with nothing configured must not enable Run); the
    # headless escape hatch trusts SourceKind.LIVE alone.  H18: the label
    # requirement stays an outer GUI-side gate; the escape hatch governs only
    # once a live source is actually configured.
    # (Own test function: staticWidget.close() persists the session, so a
    # second widget inside the previous test would restore its source paths.)
    def cfg_empty(w):
        w._controls_v2_param(("Signal", "File")).setValue("")
        w.wrangler.img_file = ""
        w.wrangler.ui.liveCheckBox.setChecked(True)

    state_empty = _gui_state(cfg_empty)
    gui_empty = state_empty.source_caps
    headless_empty = describe_source_readiness(SourceSpec("", SourceKind.LIVE))
    assert state_empty.source_label == ""
    assert gui_empty.has_frames is False          # inline: no label → not ready
    assert gui_empty.has_raw is False
    assert gui_empty.raw_reachable is False
    assert headless_empty.has_frames is True      # headless: LIVE escape hatch
    assert headless_empty.has_raw is True
    assert headless_empty.raw_reachable is True


def test_unreachable_source_parity(qapp, tmp_path):
    missing = tmp_path / "nope" / "gone_0001.tif"   # parent dir absent too

    headless = describe_source_readiness(str(missing))
    state = _gui_state(
        lambda w: w._controls_v2_param(("Signal", "File")).setValue(str(missing)))
    gui = state.source_caps

    # AGREE — raw_reachable: the headless frame-0 probe fails on the missing
    # file, matching the inline collapse.
    assert gui.raw_reachable is False
    assert headless.raw_reachable is False

    # DIVERGE (GUI wins) — has_frames / has_raw: open_source builds an
    # ImageFileSource for a NONEXISTENT path without stat-ing it, so the
    # headless side claims a phantom frame_indices == [0] (has_frames=True,
    # has_raw=True); the inline count's path.is_file() gate correctly reports
    # no frames.  H18 HAZARD: naive delegation would flip has_frames
    # False→True for a typo'd path and enable Run on nothing — the H18 gate
    # must AND has_frames with raw_reachable for non-live sources (or teach
    # describe_source_readiness to stat non-live URIs first).
    assert gui.has_frames is False
    assert gui.has_raw is False
    assert headless.has_frames is True    # ← phantom frame on a missing file
    assert headless.has_raw is True

    _assert_agree(gui, headless, ("has_metadata", "has_motors", "has_energy",
                                  "has_geometry", "has_psi_metadata"))
    assert state.run_target is not RunTarget.SOURCE


def test_fresh_widget_reports_phantom_loaded_scan(qapp):
    """PRE-EXISTING inline quirk, pinned (not fixed — H18 decision).

    A fresh ``staticWidget`` constructs its LiveScan with
    ``data_file=<scratch>/default.nxs`` (static_scan_widget._init_data_objects),
    so ``loaded_scan_available`` — and with it the inline
    ``ResultCaps.has_raw/raw_reachable`` and ``run_target=LOADED_SCAN`` — are
    True before anything is loaded (the default file need not even exist).
    There is no headless counterpart for "nothing loaded"; when H18 delegates
    ResultCaps to ``capabilities_for_processed`` it should gate
    loaded_scan_available on an actually-loaded record, not a default filename.
    """
    def cfg(w):
        w._controls_v2_param(("Signal", "File")).setValue("")
        w.wrangler.img_file = ""

    state = _gui_state(cfg)

    assert state.source_label == ""
    assert state.source_caps.has_frames is False
    assert state.loaded_scan_available is True           # ← phantom
    assert state.result_caps.has_raw is True             # ← phantom
    assert state.result_caps.raw_reachable is True       # ← phantom
    assert state.run_target is RunTarget.LOADED_SCAN     # ← phantom
