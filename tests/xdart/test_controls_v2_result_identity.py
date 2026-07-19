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


def _raw_publication(store, *, source_identity, raw=None, raw_ref=None):
    from xrd_tools.core.frame_view import FrameView
    from xdart.modules.frame_publication import publication_from_frame_view

    view = FrameView(label=0, raw=raw)
    return store.upsert(publication_from_frame_view(
        view, generation=store.generation,
        source_identity=str(source_identity), raw_ref=raw_ref))


def test_resident_publication_raw_proves_capability_while_uncached(
        widget, tmp_path, monkeypatch):
    """Identity-qualified resident browse evidence — a publication OF THE
    SELECTED SCAN carrying actual nonempty raw pixels — proves the capability
    even when the record cannot be probed during the run (H18-R1/H18-R4)."""
    processed = _processed_nxs(tmp_path, raw_reachable=True)
    monkeypatch.setattr(widget, "_controls_v2_run_active", lambda: True)
    widget.scan.data_file = str(processed)

    _raw_publication(widget.publication_store, source_identity=processed,
                     raw=np.ones((4, 4), dtype=np.uint16))

    state = widget._controls_v2_state()
    assert state.result_caps.has_raw is True
    assert state.result_caps.raw_reachable is True, \
        "selected-scan-qualified resident raw evidence must prove the capability"
    assert _roi_enabled(state.result_caps) is True


def test_outgoing_acquisition_publication_does_not_prove_selected_record(
        widget, tmp_path, monkeypatch):
    """H18-R4: the Pause transition — the store still holds acquisition A's
    resident raw publication while the selected record is orphaned B.  A's
    evidence must not make B's raw look reachable."""
    src_dir = tmp_path / "acq"
    src_dir.mkdir()
    master_a = _eiger_master(src_dir)
    orphaned_b = _processed_nxs(tmp_path, raw_reachable=False)

    monkeypatch.setattr(widget, "_controls_v2_run_active", lambda: True)
    widget.scan.data_file = str(orphaned_b)
    _raw_publication(widget.publication_store, source_identity=master_a,
                     raw=np.ones((4, 4), dtype=np.uint16))

    state = widget._controls_v2_state()
    assert state.result_caps.raw_reachable is False, \
        "acquisition A's resident raw must not prove orphaned B (H18-R4)"
    assert _roi_enabled(state.result_caps) is False


def test_dead_raw_ref_does_not_prove_reachability(
        widget, tmp_path, monkeypatch):
    """H18-R4: a bare/lazy/dead ``raw_ref`` (no resident pixels, no source)
    is a reference, not a successful reachability probe."""
    from types import SimpleNamespace

    orphaned = _processed_nxs(tmp_path, raw_reachable=False)
    monkeypatch.setattr(widget, "_controls_v2_run_active", lambda: True)
    widget.scan.data_file = str(orphaned)
    dead = SimpleNamespace(map_raw=None, image=None, source_file=None)
    _raw_publication(widget.publication_store, source_identity=orphaned,
                     raw=None, raw_ref=dead)

    state = widget._controls_v2_state()
    assert state.result_caps.raw_reachable is False, \
        "an object reference alone must not manufacture reachability (H18-R4)"
    assert _roi_enabled(state.result_caps) is False


def test_pending_browser_rescope_fails_closed(widget, tmp_path, monkeypatch):
    """H18-R4: while a manual browser rescope is pending, the outgoing
    store's evidence is not borrowed even when identities match."""
    processed = _processed_nxs(tmp_path, raw_reachable=True)
    monkeypatch.setattr(widget, "_controls_v2_run_active", lambda: True)
    widget.scan.data_file = str(processed)
    _raw_publication(widget.publication_store, source_identity=processed,
                     raw=np.ones((4, 4), dtype=np.uint16))
    widget.h5viewer._browser_scan_reset_pending = True
    try:
        state = widget._controls_v2_state()
        assert state.result_caps.raw_reachable is False, \
            "a pending rescope must fail closed (H18-R4)"
    finally:
        widget.h5viewer._browser_scan_reset_pending = False


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

def _relative_source_nxs(nxs_dir, root, *, source_rel="raw/scan_master.h5",
                         first_label=0):
    """Processed record whose frame source is RELATIVE and whose raw master
    lives under ``root`` (not under the record's own directory), with a stale
    absolute ``@source_base`` — reachable only via explicit source_root.
    ``first_label`` writes a zero- or ONE-based record (real Eiger records
    are commonly one-based, H18-R13)."""
    nxs = nxs_dir / "browsed.nxs"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _eiger_master(raw_dir)
    thumb = np.linspace(0, 100, 16 * 16).reshape(16, 16)
    frame_name = f"frame_{int(first_label):04d}"
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")
        e.attrs["source_base"] = "/stale/absolute/base"     # N1: loses to root
        g = e.create_group("integrated_1d")
        g.create_dataset("intensity", data=np.zeros((1, 5)))
        g.create_dataset(
            "frame_index", data=np.array([int(first_label)], dtype=np.int64))
        s = e.create_group(f"frames/{frame_name}/source")
        s.create_dataset("path", data=np.bytes_(source_rel))
        s.create_dataset("frame_index", data=1)
        _write_thumbnail(e[f"frames/{frame_name}"], "thumbnail", thumb)
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


def test_transient_failure_on_changed_identity_is_conservative(
        widget, tmp_path, monkeypatch):
    """H18-R6: a failed observation may NOT reuse a snapshot whose full
    identity key (path + stamp + adapter + root) no longer matches — a
    same-path stamp change plus a transient failure answers conservatively,
    never with the old record's capabilities; within the debounce window the
    share is not hammered; after it, the new identity is read."""
    import xrd_tools.io.read as read_module

    processed = _processed_nxs(tmp_path, raw_reachable=True)
    good = widget._controls_v2_loaded_result_caps(str(processed))
    assert good is not None and good.raw_reachable is True

    # same-path stamp change = a DIFFERENT identity at the same pathname
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
    assert during is None, \
        "a changed identity must not serve the old snapshot on failure (H18-R6)"
    widget._controls_v2_loaded_result_caps(str(processed))
    assert calls["n"] == 1                           # window: no hammering

    widget._v2_result_caps_retry_delay = 0.0
    widget._v2_result_caps_retry = None              # window elapsed
    after = widget._controls_v2_loaded_result_caps(str(processed))
    assert after is not None and after.has_1d is True
    assert calls["n"] == 2


def test_same_path_replacement_never_serves_old_record_on_failure(
        widget, tmp_path, monkeypatch):
    """H18-R6 repro: replace a reachable record at the same pathname with a
    DIFFERENT (orphaned) record; a transient lock on the first read must not
    resurrect the old record's raw_reachable=True."""
    import shutil

    import xrd_tools.io.read as read_module

    reachable_dir = tmp_path / "v1"
    reachable_dir.mkdir()
    v1 = _processed_nxs(reachable_dir, raw_reachable=True)
    orphan_dir = tmp_path / "v2"
    orphan_dir.mkdir()
    v2 = _processed_nxs(orphan_dir, raw_reachable=False)

    target = tmp_path / "record.nxs"
    _eiger_master(tmp_path)      # v1's relative master resolves at the target
    shutil.copy2(v1, target)
    good = widget._controls_v2_loaded_result_caps(str(target))
    assert good is not None and good.raw_reachable is True

    shutil.copy2(v2, target)                         # replacement record
    stamp = os.stat(target).st_mtime_ns + 1_000_000
    os.utime(target, ns=(stamp, stamp))
    real = read_module.get_metadata
    calls = {"n": 0}

    def once_failing(path, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise BlockingIOError("transient lock")
        return real(path, *a, **k)

    monkeypatch.setattr(read_module, "get_metadata", once_failing)
    widget._v2_result_caps_retry_delay = 0.0

    during = widget._controls_v2_loaded_result_caps(str(target))
    assert during is None, \
        "the old record's snapshot must not answer for its replacement (H18-R6)"
    after = widget._controls_v2_loaded_result_caps(str(target))
    assert after is not None
    assert after.raw_reachable is False              # the REPLACEMENT's truth


def test_root_change_with_failure_is_conservative(widget, tmp_path, monkeypatch):
    """H18-R6: a source-root change plus a transient failure must not leak
    the previous root's snapshot."""
    import xrd_tools.io.read as read_module

    nxs_dir = tmp_path / "records"
    nxs_dir.mkdir()
    moved_root = tmp_path / "moved_tree"
    nxs = _relative_source_nxs(nxs_dir, moved_root)
    widget.wrangler.project_folder = str(moved_root)
    good = widget._controls_v2_loaded_result_caps(str(nxs))
    assert good is not None and good.raw_reachable is True

    other_root = tmp_path / "other_root"
    other_root.mkdir()
    widget.wrangler.project_folder = str(other_root)  # key changes
    real = read_module.get_metadata
    calls = {"n": 0}

    def once_failing(path, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient")
        return real(path, *a, **k)

    monkeypatch.setattr(read_module, "get_metadata", once_failing)
    widget._v2_result_caps_retry_delay = 0.0

    during = widget._controls_v2_loaded_result_caps(str(nxs))
    assert during is None, \
        "the previous root's snapshot must not leak across a root change " \
        "(H18-R6)"
    after = widget._controls_v2_loaded_result_caps(str(nxs))
    assert after is not None
    assert after.raw_reachable is False              # unreachable under other_root


def test_transient_raw_probe_failure_debounces_and_retries(
        widget, tmp_path, monkeypatch):
    """H18-R5: a transient error from the frame-0 raw probe is NOT a
    definitive unreachable observation — it debounces and retries exactly
    like a transient metadata failure."""
    import xrd_tools.sources.probe as probe_module

    processed = _processed_nxs(tmp_path, raw_reachable=True)
    real = probe_module.observe_frame
    calls = {"n": 0}

    def once_transient(source, index):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("transient sharing denial")
        return real(source, index)

    monkeypatch.setattr(probe_module, "observe_frame", once_transient)
    widget._v2_result_caps_retry_delay = 0.0

    first = widget._controls_v2_loaded_result_caps(str(processed))
    assert first is None, \
        "a transient probe error must not be cached as unreachable (H18-R5)"
    second = widget._controls_v2_loaded_result_caps(str(processed))
    assert second is not None
    assert second.raw_reachable is True
    assert calls["n"] == 2


# ── H18-R9: cross-root identity ───────────────────────────────────────────

def test_same_stem_cross_root_publication_is_rejected(
        widget, tmp_path, monkeypatch):
    """H18-R9: two unrelated records sharing a basename stem must not share
    resident raw evidence — basename identity is not record identity."""
    proc_dir = tmp_path / "processed" / "B"
    proc_dir.mkdir(parents=True)
    selected = _processed_nxs(proc_dir, raw_reachable=False)
    renamed = proc_dir / "scan_0001.nxs"
    os.rename(selected, renamed)

    monkeypatch.setattr(widget, "_controls_v2_run_active", lambda: True)
    widget.scan.data_file = str(renamed)
    _raw_publication(
        widget.publication_store,
        source_identity=str(tmp_path / "acquisition" / "A" / "scan_0001.nxs"),
        raw=np.ones((4, 4), dtype=np.uint16))

    state = widget._controls_v2_state()
    assert state.result_caps.raw_reachable is False, \
        "a same-stem publication from a different root must not prove the " \
        "selected record (H18-R9)"
    assert _roi_enabled(state.result_caps) is False


def test_acquisition_master_publication_proves_its_own_processed_output(
        widget, tmp_path, monkeypatch):
    """H18-R9 legitimate case: the raw source and the processed output live
    in DIFFERENT directories; the record's stored provenance links them, so
    the acquisition master's resident raw pixels prove the selected output."""
    raw_root = tmp_path / "raw_tree"
    out_dir = tmp_path / "processed_out"
    out_dir.mkdir()
    nxs = _relative_source_nxs(out_dir, raw_root,
                               source_rel="raw/scan_master.h5")
    widget.wrangler.project_folder = str(raw_root)
    widget.scan.data_file = str(nxs)
    good = widget._controls_v2_loaded_result_caps(str(nxs))
    assert good is not None and good.raw_reachable is True   # provenance cached

    monkeypatch.setattr(widget, "_controls_v2_run_active", lambda: True)
    master = raw_root / "raw" / "scan_master.h5"
    _raw_publication(widget.publication_store, source_identity=str(master),
                     raw=np.ones((4, 4), dtype=np.uint16))

    state = widget._controls_v2_state()
    assert state.result_caps.raw_reachable is True, \
        "the record's own acquisition master must prove it across " \
        "directories (H18-R9)"


# ── H18-R13: one-based records through the natural lifecycle ──────────────

def test_one_based_record_provenance_through_natural_lifecycle(
        widget, tmp_path, monkeypatch):
    """H18-R13: a ONE-based processed record (frames start at frame_0001 —
    the real Eiger shape) resolves its stored provenance by default, driven
    only through the natural Controls V2 lifecycle: an idle refresh caches
    the record identity, then the paused acquisition master's resident raw
    publication proves ITS OWN output."""
    raw_root = tmp_path / "raw_tree"
    out_dir = tmp_path / "processed_out"
    out_dir.mkdir()
    nxs = _relative_source_nxs(out_dir, raw_root, first_label=1)
    widget.wrangler.project_folder = str(raw_root)
    widget.scan.data_file = str(nxs)

    idle = widget._controls_v2_state()        # NATURAL idle refresh: probes+caches
    assert idle.result_caps.raw_reachable is True

    monkeypatch.setattr(widget, "_controls_v2_run_active", lambda: True)
    master = raw_root / "raw" / "scan_master.h5"
    _raw_publication(widget.publication_store, source_identity=str(master),
                     raw=np.ones((4, 4), dtype=np.uint16))

    state = widget._controls_v2_state()
    assert state.result_caps.raw_reachable is True, \
        "a one-based record's provenance must resolve by default (H18-R13)"


def test_resolved_raw_source_default_uses_first_stored_frame(tmp_path):
    """H18-R13 core: the accessor's DEFAULT means the first actual stored
    source-bearing frame; explicit labels stay exact and fail closed."""
    from xrd_tools.io.read import resolved_raw_source

    raw_root = tmp_path / "tree"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    nxs = _relative_source_nxs(out_dir, raw_root, first_label=1)

    assert resolved_raw_source(
        nxs, source_root=str(raw_root)) is not None, \
        "default must find the first stored frame of a one-based record"
    assert resolved_raw_source(
        nxs, frame=1, source_root=str(raw_root)) is not None
    assert resolved_raw_source(nxs, frame=0, source_root=str(raw_root)) is None, \
        "an explicit missing label must fail closed"
