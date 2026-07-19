# -*- coding: utf-8 -*-
"""H18 round-two corrections (handoff §12): identity-aware ResultCaps.

H18-R1: a selected processed record owns the result raw facts — the frozen
acquisition source must not contaminate them (paused browse of an orphaned
record must not advertise ROI Statistics because the acquisition source is
reachable); an uncached record during Run/Pause is conservatively
unavailable unless identity-qualified resident browse evidence proves the
capability; a stale nonexistent ``scan.data_file`` is not a loaded scan.

H18-R2: processed raw reachability honors the explicit source-root through
the existing ``SourceSpec(PROCESSED_NEXUS, options={"source_root": ...})``
seam (N1 precedence: source_root > @source_base > scan directory), and the
result cache is keyed on it.

H18-R3: a transient processed-record read failure is never cached as stable
truth — a previous valid same-identity snapshot is preserved where safe and
the read retries after a bounded debounce.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph import Qt

from tests.xdart.test_h5_readiness_parity import (
    _eiger_master,
    _processed_nxs,
    _write_thumbnail,
)

QtWidgets = Qt.QtWidgets


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _controls_panel_session_isolation():
    path = os.environ.get("XDART_SESSION_FILE")

    def _unlink():
        if not path:
            return
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass

    _unlink()
    yield
    _unlink()


@pytest.fixture(autouse=True)
def _drain_qt_events_after_test(qapp):
    yield
    for _ in range(3):
        qapp.processEvents()
    gc.collect()
    for _ in range(2):
        qapp.processEvents()


@pytest.fixture()
def widget(qapp):
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    w = staticWidget()
    try:
        yield w
    finally:
        w.close()
        w.deleteLater()


def _configure_source(w, master):
    w._controls_v2_param(("Signal", "inp_type")).setValue("Image Series")
    w._controls_v2_param(("Signal", "File")).setValue(str(master))
    w._controls_v2_param(("Signal", "img_ext")).setValue("h5")


def _roi_enabled(result_caps) -> bool:
    from xdart.gui.tabs.static_scan.controls_logic import (
        AnalysisTool,
        build_analysis_launchers,
    )

    for spec in build_analysis_launchers(result_caps):
        if spec.tool is AnalysisTool.ROI_STATS:
            return bool(spec.enabled)
    raise AssertionError("ROI_STATS launcher missing")


# ── H18-R1: selected-record identity owns result raw facts ────────────────

def test_selected_orphaned_record_does_not_inherit_source_raw(
        widget, tmp_path, monkeypatch):
    """Reachable acquisition source A + selected processed record B whose raw
    master is missing: the RESULT raw facts are B's record truth."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    master = src_dir / "scan_master.h5"
    raw = np.arange(2 * 8 * 8, dtype=np.uint32).reshape(2, 8, 8)
    with h5py.File(master, "w") as f:
        f.create_dataset("entry/data/data", data=raw)
    _configure_source(widget, master)
    state = widget._controls_v2_state()
    assert state.source_caps.raw_reachable is True   # source A is reachable

    orphaned = _processed_nxs(tmp_path, raw_reachable=False)
    widget.scan.data_file = str(orphaned)            # the selected browse target

    state = widget._controls_v2_state()
    assert state.result_caps.has_raw is True, "record truth: raw refs exist"
    assert state.result_caps.raw_reachable is False, \
        "the acquisition source must not contaminate the selected record's " \
        "raw reachability (H18-R1)"
    assert _roi_enabled(state.result_caps) is False, \
        "ROI Statistics must not appear available for an orphaned record"


def test_paused_uncached_selected_record_is_conservative(
        widget, tmp_path, monkeypatch):
    """During Run/Pause an UNCACHED selected record is pending — 'not probed'
    must never become raw_reachable=True."""
    master = _eiger_master(tmp_path)
    _configure_source(widget, master)
    widget._controls_v2_state()                      # cache source truth

    processed = _processed_nxs(tmp_path, raw_reachable=True)
    monkeypatch.setattr(widget, "_controls_v2_run_active", lambda: True)
    widget.scan.data_file = str(processed)           # selected while paused

    state = widget._controls_v2_state()
    assert state.result_caps.raw_reachable is False, \
        "'not probed' must not be reported as reachable (H18-R1)"
    assert state.result_caps.has_raw is False
    assert _roi_enabled(state.result_caps) is False


def test_paused_cached_selected_record_serves_its_snapshot(
        widget, tmp_path, monkeypatch):
    """A record probed BEFORE the run keeps serving its identity-qualified
    snapshot during Pause."""
    processed = _processed_nxs(tmp_path, raw_reachable=True)
    widget.scan.data_file = str(processed)
    state = widget._controls_v2_state()              # probes + caches
    assert state.result_caps.has_raw is True
    assert state.result_caps.raw_reachable is True

    monkeypatch.setattr(widget, "_controls_v2_run_active", lambda: True)
    state = widget._controls_v2_state()
    assert state.result_caps.raw_reachable is True   # cached snapshot answers
    assert _roi_enabled(state.result_caps) is True


def test_resident_publication_raw_proves_capability_while_uncached(
        widget, tmp_path, monkeypatch):
    """Identity-qualified resident browse evidence (a publication of the
    selected scan carrying a raw payload) proves the capability even when the
    record cannot be probed during the run."""
    from xrd_tools.core.frame_view import FrameView
    from xdart.modules.frame_publication import publication_from_frame_view

    processed = _processed_nxs(tmp_path, raw_reachable=True)
    monkeypatch.setattr(widget, "_controls_v2_run_active", lambda: True)
    widget.scan.data_file = str(processed)

    view = FrameView(label=0, raw=np.ones((4, 4), dtype=np.uint16))
    widget.publication_store.upsert(publication_from_frame_view(
        view, generation=widget.publication_store.generation))

    state = widget._controls_v2_state()
    assert state.result_caps.has_raw is True
    assert state.result_caps.raw_reachable is True, \
        "resident raw evidence must prove the capability (H18-R1)"


def test_stale_missing_data_file_is_not_a_loaded_scan(widget, tmp_path):
    """A nonexistent non-scratch ``scan.data_file`` with no loaded
    frames/publications is not a loaded scan and no LOADED_SCAN target."""
    from xdart.gui.tabs.static_scan.controls_logic import RunTarget

    widget.scan.data_file = str(tmp_path / "vanished" / "old_session.nxs")
    state = widget._controls_v2_state()
    assert state.loaded_scan_available is False, \
        "a vanished session path must not count as a loaded scan (H18-R1)"
    assert state.run_target is RunTarget.NONE
    assert state.result_caps.has_raw is False
    assert state.result_caps.raw_reachable is False


# ── H18-R2: source_root-aware processed reachability ──────────────────────

def _relative_source_nxs(nxs_dir, root, *, source_rel="raw/scan_master.h5"):
    """Processed record whose frame source is RELATIVE and whose raw master
    lives under ``root`` (not under the record's own directory), with a stale
    absolute ``@source_base`` — reachable only via explicit source_root."""
    nxs = nxs_dir / "browsed.nxs"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _eiger_master(raw_dir)
    thumb = np.linspace(0, 100, 16 * 16).reshape(16, 16)
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")
        e.attrs["source_base"] = "/stale/absolute/base"     # N1: loses to root
        g = e.create_group("integrated_1d")
        g.create_dataset("intensity", data=np.zeros((1, 5)))
        g.create_dataset("frame_index", data=np.array([0], dtype=np.int64))
        s = e.create_group("frames/frame_0000/source")
        s.create_dataset("path", data=np.bytes_(source_rel))
        s.create_dataset("frame_index", data=1)
        _write_thumbnail(e["frames/frame_0000"], "thumbnail", thumb)
    return nxs


def test_processed_reachability_honors_explicit_source_root(
        widget, tmp_path):
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.sources.readiness import describe_source_readiness

    nxs_dir = tmp_path / "records"
    nxs_dir.mkdir()
    moved_root = tmp_path / "moved_tree"
    nxs = _relative_source_nxs(nxs_dir, moved_root)

    # headless control: bare path unreachable; explicit root reachable
    assert describe_source_readiness(str(nxs)).raw_reachable is False
    spec = SourceSpec(str(nxs), SourceKind.PROCESSED_NEXUS,
                      options={"source_root": str(moved_root)})
    assert describe_source_readiness(spec).raw_reachable is True

    # GUI: the configured project folder is the explicit source-root owner
    widget.wrangler.project_folder = str(moved_root)
    widget.scan.data_file = str(nxs)
    state = widget._controls_v2_state()
    assert state.result_caps.has_raw is True
    assert state.result_caps.raw_reachable is True, \
        "the explicit project/source root must repoint the moved raw tree " \
        "(H18-R2, N1 precedence)"


def test_source_root_change_invalidates_result_cache(widget, tmp_path):
    nxs_dir = tmp_path / "records"
    nxs_dir.mkdir()
    moved_root = tmp_path / "moved_tree"
    nxs = _relative_source_nxs(nxs_dir, moved_root)

    widget.scan.data_file = str(nxs)
    state = widget._controls_v2_state()
    assert state.result_caps.raw_reachable is False  # no root configured

    widget.wrangler.project_folder = str(moved_root)
    state = widget._controls_v2_state()
    assert state.result_caps.raw_reachable is True, \
        "changing the source root must invalidate the cached result caps " \
        "(H18-R2)"


# ── H18-R3: transient read failures never stick ───────────────────────────

def test_transient_metadata_failure_retries_after_debounce(
        widget, tmp_path, monkeypatch):
    import xrd_tools.io.read as read_module

    processed = _processed_nxs(tmp_path, raw_reachable=True)
    real = read_module.get_metadata
    calls = {"n": 0}

    def once_failing(path, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("transient sharing violation")
        return real(path, *a, **k)

    monkeypatch.setattr(read_module, "get_metadata", once_failing)
    widget._v2_result_caps_retry_delay = 0.0         # bounded debounce -> 0

    first = widget._controls_v2_loaded_result_caps(str(processed))
    assert first is None                             # transient failure
    second = widget._controls_v2_loaded_result_caps(str(processed))
    assert second is not None, \
        "a transient read failure must not be cached forever (H18-R3)"
    assert second.has_1d is True
    assert calls["n"] == 2


def test_transient_failure_preserves_previous_valid_snapshot(
        widget, tmp_path, monkeypatch):
    import time

    import xrd_tools.io.read as read_module

    processed = _processed_nxs(tmp_path, raw_reachable=True)
    good = widget._controls_v2_loaded_result_caps(str(processed))
    assert good is not None and good.raw_reachable is True

    # stamp change invalidates the key; the next read transiently fails
    stamp = os.stat(processed).st_mtime_ns + 1_000_000
    os.utime(processed, ns=(stamp, stamp))
    real = read_module.get_metadata
    calls = {"n": 0}

    def once_failing(path, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient share hiccup")
        return real(path, *a, **k)

    monkeypatch.setattr(read_module, "get_metadata", once_failing)
    widget._v2_result_caps_retry_delay = 30.0        # long debounce window

    during = widget._controls_v2_loaded_result_caps(str(processed))
    assert during is not None and during.raw_reachable is True, \
        "the previous valid same-identity snapshot must be preserved (H18-R3)"
    # within the debounce window the share is not hammered
    widget._controls_v2_loaded_result_caps(str(processed))
    assert calls["n"] == 1

    widget._v2_result_caps_retry_delay = 0.0
    widget._v2_result_caps_retry = None              # window elapsed
    after = widget._controls_v2_loaded_result_caps(str(processed))
    assert after is not None and after.has_1d is True
    assert calls["n"] == 2
