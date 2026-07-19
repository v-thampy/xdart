# -*- coding: utf-8 -*-
"""H5→H18 — Stage-6 readiness UNIFICATION PIN: one gating truth source.

Under H5 this file froze TWO gating truth sources together — the GUI-inline
``has_frames = has_raw = raw_reachable = source_ready`` collapse in
``staticWidget`` vs the headless ``describe_source_readiness`` /
``capabilities_for_processed`` — and pinned every divergence with a note
recording which side wins.  H18 landed that contract: the inline constructors
are GONE (``_controls_v2_source_caps`` / the ResultCaps literal now DELEGATE to
``xrd_tools.sources.readiness``), and each pinned divergence below flipped to
an agreement assertion or an explicit headless fix.  This file is now the
regression pin that the two sides can never diverge again.

Post-H18 matrix (tri-fields = has_frames/has_raw/raw_reachable):

    case                       tri-fields         resolution
    SPEC source                BOTH T/T/F         was headless-wins; GUI inherits
                                                  the SpecSource answer
    Eiger master               BOTH T/T/T         the H5 no-op row, unchanged
    processed + reachable raw  BOTH T/T/T         was headless-wins; record truth
                                                  adopted (run stays Reintegrate)
    processed, raw missing     BOTH T/T/F         was headless-wins; plus the
                                                  hazard-2 fix: GUI ResultCaps
                                                  consume the frame-0 probe
    live / unknown length      BOTH T/T/T         the true-live escape hatch is
                                                  now the ONLY mechanism
    live, no source label      BOTH F/F/F         was GUI-wins; the label rule
                                                  moved INTO the core (empty
                                                  location is never ready)
    nonexistent path           BOTH F/F/F         was GUI-wins; the core now
                                                  stats non-live local URIs
                                                  (phantom-frame fix)

The one REMAINING policy divergence (pinned, deliberate): a configured LIVE
source's optimistic metadata/geometry claims are advisory — the panel state
overlays them in the GUI's consumption, because a live run without real
metadata/calibration must stay gated.
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
    # here); post-H18 the GUI's False IS the headless frame-0 probe answer,
    # not a collapse artifact.
    assert gui.raw_reachable is False
    assert headless.raw_reachable is False

    # UNIFIED on the headless answer (was: GUI F/F/F vs headless T/T/T-F,
    # headless wins) — the inline count_frames collapse could not parse a SPEC
    # scan table and reported NO frames for a perfectly valid 3-point scan.
    # The delegated tri-fields adopt the SpecSource scan rows.
    assert headless.has_frames is True    # SpecSource scan rows
    assert headless.has_raw is True
    assert gui.has_frames is True         # ← was False (inline collapse)
    assert gui.has_raw is True            # ← was False

    # UNIFIED on the headless answer (was: headless wins) — metadata family:
    # SpecSource serves the scan table (#L columns), motors (#O/#P) and the
    # psi-family "chi" column AT GATE TIME; the panel used to learn them only
    # after a run hydrated scan.scan_data.  Non-live panel truth is now
    # OR-composed with what the source itself serves.
    assert gui.has_metadata is True and headless.has_metadata is True
    assert gui.has_motors is True and headless.has_motors is True
    assert gui.has_psi_metadata is True and headless.has_psi_metadata is True

    # Full 8-field agreement — the SPEC row joins the no-op set.
    _assert_agree(gui, headless, SOURCE_FIELDS)
    assert gui == headless

    # The RUN gate is unchanged: readiness truth no longer hides that frames
    # exist, but the wrangler cannot run a SPEC scan (no spec wrangler; its
    # frame count is 0), so run_target must not become SOURCE.
    assert state.run_target is not RunTarget.SOURCE


def test_eiger_master_parity(qapp, tmp_path):
    master = _eiger_master(tmp_path)

    headless = describe_source_readiness(str(master))

    def cfg(w):
        w._controls_v2_param(("Signal", "inp_type")).setValue("Image Series")
        w._controls_v2_param(("Signal", "File")).setValue(str(master))
        w._controls_v2_param(("Signal", "img_ext")).setValue("h5")

    state = _gui_state(cfg)
    gui = state.source_caps

    # FULL AGREEMENT — the H5 no-op row, preserved verbatim through the H18
    # delegation: both sides count 2 frames, both load frame 0, and neither
    # claims metadata/motors/energy/geometry/psi for a bare detector master.
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

    # SourceCaps tri-fields — UNIFIED on record truth (was: GUI F/F/F vs
    # headless T/T/T, headless wins): ProcessedNexusSource sees the frame
    # record and the frame-0 probe resolves the sibling raw master; the
    # retired inline collapse could not count a processed record at all.
    assert headless_src.has_frames is True
    assert headless_src.has_raw is True
    assert headless_src.raw_reachable is True
    _assert_agree(gui_src, headless_src, SOURCE_FIELDS)
    assert gui_src == headless_src

    # UNIFIED on record truth (was: headless wins) — the record carries
    # metadata and geometry the panel had not hydrated.  RUN gating is NOT
    # loosened by this: CALIBRATION_PONI / BEAM_ENERGY key on the panel's own
    # geom state (geom.calibrated / energy resolution), so the record's
    # geometry informs rows and launchers without unblocking a run.
    assert gui_src.has_metadata is True and gui_src.has_geometry is True

    # Unchanged compensation: record-truth caps do not make a processed .nxs
    # wrangler-runnable — the run gate still points at the loaded scan
    # (Reintegrate), never a fresh SOURCE run over a processed record.
    assert state.loaded_scan_available is True
    assert state.run_target is RunTarget.LOADED_SCAN

    # ResultCaps — UNIFIED on the record (was: headless wins on has_1d /
    # scan_metadata / psi): the GUI now reads capabilities_for_processed over
    # the loaded record instead of its hydration mirrors, so a freshly opened
    # processed scan reports its 1D results and scan table BEFORE the load
    # worker hydrates a single viewer row.
    assert headless_res.has_1d is True
    assert headless_res.has_scan_metadata is True
    assert headless_res.has_psi_metadata is True
    assert gui_res.has_raw is True and gui_res.raw_reachable is True
    _assert_agree(gui_res, headless_res, RESULT_FIELDS)
    assert gui_res == headless_res


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

    # UNIFIED (was: GUI F/F/F for the wrong reason — the one-bit collapse
    # could not even EXPRESS "record present, raw missing").  The delegated
    # GUI answer is now T/T/F: right value, right reason.
    assert gui_src.has_frames is True            # ← was False
    assert gui_src.has_raw is True               # ← was False
    assert gui_src.raw_reachable is False        # probe truth on both sides
    _assert_agree(gui_src, headless_src, SOURCE_FIELDS)

    # ResultCaps — the H18 hazard-2 resolution of the JOINT overstatement.
    # The bare record mirror still reports raw_reachable=True (documented and
    # deliberate: capabilities_for_processed must not reopen HDF5)...
    assert headless_res.has_raw is True and headless_res.raw_reachable is True
    # ...but the core now accepts the probe answer, and the GUI consumes that
    # probe-informed form, so raw-dependent launcher gating (ROI) sees the
    # truth for an orphaned record:
    probed_res = capabilities_for_processed(
        get_metadata(nxs), raw_reachable=headless_src.raw_reachable)
    assert probed_res.raw_reachable is False
    assert gui_res.has_raw is True               # the record exists...
    assert gui_res.raw_reachable is False        # ← was True (overstatement)
    assert gui_res == probed_res

    # Record-vs-hydration fields, unified like the reachable-raw case.
    assert gui_res.has_1d is True and headless_res.has_1d is True
    _assert_agree(gui_res, headless_res,
                  ("has_1d", "has_2d", "has_rsm", "has_phase_result"))


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

    # THE H5 reconcile point, RESOLVED as designed: both sides land T/T/T for
    # a live run with a configured source, and the mechanism is now ONE — the
    # GUI describes its live source as SourceSpec(label, LIVE) and inherits
    # the true-live escape hatch (a live acquisition may legitimately have no
    # frame 0 yet, so raw_reachable stays True even when nothing probes).
    # The retired inline live_unknown collapse no longer exists.
    assert gui.has_frames is True and headless.has_frames is True
    assert gui.has_raw is True and headless.has_raw is True
    assert gui.raw_reachable is True and headless.raw_reachable is True
    # Unknown length renders as frame_count 0 while source_ready stays True.
    assert state.frame_count == 0
    assert state.run_target is RunTarget.SOURCE

    # THE ONE REMAINING POLICY DIVERGENCE (pinned, deliberate — the explicit
    # H18 decision): LiveFrameSource optimistically advertises
    # metadata/geometry; those claims are ADVISORY.  For LIVE sources the
    # panel state overlays the whole metadata family in the GUI's consumption
    # (a live run without a calibration must stay blocked), so the panel's
    # False wins here while the headless description keeps its claim.
    assert gui.has_metadata is False and headless.has_metadata is True
    assert gui.has_geometry is False and headless.has_geometry is True
    _assert_agree(gui, headless,
                  ("has_motors", "has_energy", "has_psi_metadata"))


def test_live_without_configured_source_parity(qapp):
    # Live with NO configured source label — UNIFIED (was: GUI wins, with the
    # headless escape hatch trusting SourceKind.LIVE alone).  H18 moved the
    # label requirement INTO the core: describe_source_readiness refuses the
    # escape hatch for an EMPTY location (nothing configured is never ready,
    # live or not), so every consumer gets the safe answer — not just the
    # wrapped GUI.
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
    assert headless_empty.has_frames is False     # ← was True (escape hatch)
    assert headless_empty.has_raw is False
    assert headless_empty.raw_reachable is False
    _assert_agree(gui_empty, headless_empty, SOURCE_FIELDS)
    assert state_empty.run_target is RunTarget.NONE


def test_unreachable_source_parity(qapp, tmp_path):
    missing = tmp_path / "nope" / "gone_0001.tif"   # parent dir absent too

    headless = describe_source_readiness(str(missing))
    state = _gui_state(
        lambda w: w._controls_v2_param(("Signal", "File")).setValue(str(missing)))
    gui = state.source_caps

    # UNIFIED (was: GUI wins) — the H18 hazard-1 fix landed in the CORE:
    # describe_source_readiness now stats non-live local URIs before trusting
    # open_source, which used to build an ImageFileSource around the typo'd
    # path and claim a phantom frame_indices == [0].  A nonexistent path is
    # all-False for every consumer (Tiled-style scheme:// URIs pass through
    # un-stat-ed; LIVE specs keep the escape hatch — a live file may simply
    # not exist yet).
    assert headless.has_frames is False   # ← was True (phantom frame)
    assert headless.has_raw is False
    assert headless.raw_reachable is False
    _assert_agree(gui, headless, SOURCE_FIELDS)
    assert gui == headless

    # And Run stays gated on nothing-to-run.
    assert state.run_target is not RunTarget.SOURCE


def test_fresh_widget_reports_no_loaded_scan(qapp):
    """H5 finding 3, FIXED by H18 (was pinned as a phantom).

    A fresh ``staticWidget`` constructs its LiveScan with
    ``data_file=<scratch>/default.nxs`` (static_scan_widget._init_data_objects);
    that placeholder used to leak into ``loaded_scan_available`` — and with it
    the inline ``ResultCaps.has_raw/raw_reachable`` and
    ``run_target=LOADED_SCAN`` — before anything was loaded.  The readiness
    gate now recognizes the pristine scratch default: until frames hydrate,
    publications exist, or ``data_file`` is re-pointed at a real record, an
    empty widget reports NO loaded scan and run_target=NONE.
    """
    def cfg(w):
        w._controls_v2_param(("Signal", "File")).setValue("")
        w.wrangler.img_file = ""

    state = _gui_state(cfg)

    assert state.source_label == ""
    assert state.source_caps.has_frames is False
    assert state.loaded_scan_available is False          # ← was phantom True
    assert state.result_caps.has_raw is False
    assert state.result_caps.raw_reachable is False
    assert state.run_target is RunTarget.NONE            # ← was LOADED_SCAN
