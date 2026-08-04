# -*- coding: utf-8 -*-
"""P4/OUT-1 — frozen acceptance oracle for the canonical GUI output-path consumers.

The GUI (PP-C1) half of the finite oracle.  The headless owner and its
headless consumers are pinned in ``tests/core/test_output_path_policy.py``.

Pinned here (handoff §5): the Image and NeXus wranglers resolve generated
targets through the shared owner, New/Overwrite and absent-target Append
generate ``.nexus``, an explicit existing legacy ``.nxs`` Append is retained,
H5Viewer lists/loads/restores/follows both suffixes, the scratch/default and
run-end fallback names come from the shared policy, and a function-scoped
census proves every authorized GUI generated-output constructor consumes the
owner.

Production-wired: real ``imageThread`` objects built by the real constructor,
the real unbound ``H5Viewer``/``staticWidget`` methods, and a real
``QListWidget`` — no fake stands on the seam under test (HARD RULE 2).
"""

from __future__ import annotations

import ast
import os
import threading
from pathlib import Path
from queue import Queue
from types import MethodType, SimpleNamespace

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtWidgets

from xrd_tools.io import LEGACY_OUTPUT_SUFFIX, NEW_OUTPUT_SUFFIX
from xdart.modules.ewald.scan import LiveScan
from xrd_tools.core.containers import PONI
from xdart.gui.tabs.static_scan.h5viewer import H5Viewer
from xdart.gui.tabs.static_scan.static_scan_widget import (
    _finished_output_file, staticWidget,
)
from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread
from tests.xdart._accepted_run import (
    accepted_run,
    admitted_worker,
    directory_source,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

_LIVE_THREADS: list = []


@pytest.fixture(autouse=True)
def _teardown_live_threads():
    yield
    for t in _LIVE_THREADS:
        try:
            t.command = "stop"
            t._prefetch_stop_prior()
            t._eiger_close_master()
        except Exception:
            pass
    _LIVE_THREADS.clear()


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _make_thread(watch_dir, out_dir, *, scan_name="scan", write_mode="Overwrite",
                 include_subdir=False):
    """A REAL imageThread watching *watch_dir* and writing into *out_dir*."""
    scan = LiveScan("scan", data_file=str(Path(out_dir) / "scan.nexus"),
                    static=True)
    # R4-G retired the policy constructor arguments: the watched directory,
    # extension, recursion and output mode reach the worker only through the
    # accepted configuration admitted below — the same seam production
    # admission uses (E6/PM reconciliation of this canonical oracle).
    t = imageThread(
        Queue(), threading.RLock(), "",
        scan_name,                   # scan_name
        PONI(dist=0.2, poni1=0.1, poni2=0.1, wavelength=1e-10),
        "",                          # img_file
        "None",                      # bg_type
        "", "", None, "", "",        # bg_*
        1.0, None,                   # bg_scale, bg_norm_channel
        "q_total", "qip_qoop",       # gi modes
        "start", scan,
    )
    admitted_worker(
        t,
        frozen=accepted_run(
            processing_mode="Int 2D",
            output_mode=write_mode,
            save_path=str(out_dir),
            source_spec=directory_source(
                watch_dir, ext="nxs", recursive=include_subdir),
        ),
    )
    _LIVE_THREADS.append(t)
    return t


def _write_processed(path: Path, *, n_frames: int = 2) -> Path:
    q = np.linspace(1.0, 5.0, 9, dtype=np.float32)
    frames = np.arange(n_frames, dtype=np.int64)
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["ssrl_schema_version"] = 2
        g = entry.create_group("integrated_1d")
        g.create_dataset("frame_index", data=frames)
        g.create_dataset("q", data=q)
        g.create_dataset("intensity",
                         data=np.ones((n_frames, len(q)), dtype=np.float32))
        fg = entry.create_group("frames")
        for frame in frames:
            grp = fg.create_group(f"frame_{frame:04d}")
            thumb = grp.create_dataset(
                "thumbnail", data=np.arange(12, dtype=np.uint8).reshape(3, 4))
            thumb.attrs["vmin"] = 10.0
            thumb.attrs["vmax"] = 265.0
            thumb.attrs["dtype"] = "uint8"
    return path


def _list_host(tmp_path, *, scan_name=None, viewer_mode="normal",
               date_sort=False):
    """A host driving the REAL update_scans over a REAL QListWidget.

    ``date_sort`` engages the panel's real Date-sort ordering, where a newer
    legacy sibling is listed BEFORE an older ``.nexus`` — the presentation order
    that must not be allowed to decide policy (C2 rows 5 and 6).
    """
    lw = QtWidgets.QListWidget()
    host = SimpleNamespace(
        dirname=str(tmp_path),
        viewer_mode=viewer_mode,
        scan_name=scan_name,
        ui=SimpleNamespace(
            listScans=lw,
            dateSort=(SimpleNamespace(isChecked=lambda: True)
                      if date_sort else None)),
        _update_scans_header=lambda: None,
        _date_sort_dir_cache={},
        _IMAGE_EXTS=H5Viewer._IMAGE_EXTS,
        _XYE_EXTS=H5Viewer._XYE_EXTS,
        _NEXUS_EXTS=H5Viewer._NEXUS_EXTS,
    )
    host._natural_sort_key = H5Viewer._natural_sort_key
    host._date_sort_mtime = MethodType(H5Viewer._date_sort_mtime, host)
    host.update_scans = MethodType(H5Viewer.update_scans, host)
    return host, lw


def _entries(lw):
    return [lw.item(i).text() for i in range(lw.count())]


# ---------------------------------------------------------------------------
# Row 2 (GUI) — New / Overwrite and absent-target Append generate ``.nexus``
# ---------------------------------------------------------------------------

def test_row2_gui_overwrite_generates_nexus(tmp_path):
    watch, out = tmp_path / "raw", tmp_path / "out"
    watch.mkdir()
    out.mkdir()
    t = _make_thread(watch, out, scan_name="scan_042", write_mode="Overwrite")

    assert t._append_output_path(t.run_configuration, "scan_042") == os.fspath(
        out / "scan_042.nexus")


def test_row2_gui_append_with_no_existing_target_generates_nexus(tmp_path):
    watch, out = tmp_path / "raw", tmp_path / "out"
    watch.mkdir()
    out.mkdir()
    t = _make_thread(watch, out, scan_name="scan_042", write_mode="Append")

    assert t._append_output_path(t.run_configuration, "scan_042") == os.fspath(
        out / "scan_042.nexus")


def test_row2_gui_initialize_scan_targets_nexus(tmp_path):
    """The real ``initialize_scan`` writes its output to ``<scan>.nexus``."""
    watch, out = tmp_path / "raw", tmp_path / "out"
    watch.mkdir()
    out.mkdir()
    t = _make_thread(watch, out, scan_name="scan_042", write_mode="Overwrite")

    t.initialize_scan()

    assert Path(t.fname) == out / "scan_042.nexus"
    assert (out / "scan_042.nexus").exists()
    assert not (out / "scan_042.nxs").exists()


def test_row2_gui_overwrite_never_replaces_a_sibling_legacy_file(tmp_path):
    """§2 rule 2 at the GUI seam: an existing ``.nxs`` is left untouched."""
    watch, out = tmp_path / "raw", tmp_path / "out"
    watch.mkdir()
    out.mkdir()
    legacy = _write_processed(out / "scan_042.nxs")
    before = legacy.read_bytes()

    t = _make_thread(watch, out, scan_name="scan_042", write_mode="Overwrite")
    t.initialize_scan()

    assert Path(t.fname) == out / "scan_042.nexus"
    assert legacy.read_bytes() == before


# ---------------------------------------------------------------------------
# Row 9 (GUI) — an explicit existing legacy ``.nxs`` Append is retained
# ---------------------------------------------------------------------------

def test_row9_gui_append_reuses_an_existing_legacy_target(tmp_path):
    watch, out = tmp_path / "raw", tmp_path / "out"
    watch.mkdir()
    out.mkdir()
    legacy = _write_processed(out / "scan_042.nxs")

    t = _make_thread(watch, out, scan_name="scan_042", write_mode="Append")

    assert t._append_output_path(t.run_configuration, "scan_042") == os.fspath(legacy)


def test_row9_gui_append_run_end_fallback_finds_the_legacy_target(tmp_path):
    """``_finished_output_file`` reports the legacy file an Append reused."""
    legacy = _write_processed(tmp_path / "scan_042.nxs")
    thread = admitted_worker(
        SimpleNamespace(
            _append_skip_frames_by_scan={"scan_042": {1, 2}},
            _append_output_path=None,
        ),
        processing_mode="Int 2D",
        save_path=str(tmp_path),
        source_spec=directory_source(tmp_path),
    )
    wrangler = SimpleNamespace(thread=thread, fname=None)

    found = _finished_output_file(thread, wrangler, all_skipped_append=True)

    assert found is not None
    assert Path(found) == legacy


# ---------------------------------------------------------------------------
# Row 10 (GUI) — both siblings exist: ``.nexus`` wins
# ---------------------------------------------------------------------------

def test_row10_gui_append_prefers_nexus_when_both_siblings_exist(tmp_path):
    watch, out = tmp_path / "raw", tmp_path / "out"
    watch.mkdir()
    out.mkdir()
    _write_processed(out / "scan_042.nxs")
    _write_processed(out / "scan_042.nexus")

    t = _make_thread(watch, out, scan_name="scan_042", write_mode="Append")

    assert t._append_output_path(t.run_configuration, "scan_042") == os.fspath(
        out / "scan_042.nexus")


# ---------------------------------------------------------------------------
# Row 6 (GUI) — H5Viewer explicitly opens ``.nexus``
# ---------------------------------------------------------------------------

def test_row6_gui_viewer_extension_sets_accept_nexus():
    assert NEW_OUTPUT_SUFFIX in H5Viewer._NEXUS_EXTS
    assert NEW_OUTPUT_SUFFIX in H5Viewer._IMAGE_EXTS
    assert LEGACY_OUTPUT_SUFFIX in H5Viewer._NEXUS_EXTS
    assert LEGACY_OUTPUT_SUFFIX in H5Viewer._IMAGE_EXTS


def test_row6_gui_nexus_viewer_mode_lists_both_suffixes(qapp, tmp_path):
    _write_processed(tmp_path / "a.nexus")
    _write_processed(tmp_path / "b.nxs")
    (tmp_path / "c.txt").write_text("x")

    host, lw = _list_host(tmp_path, viewer_mode="nexus")
    host.update_scans()

    assert _entries(lw) == ["..", "a.nexus", "b.nxs"]


def test_row6_gui_image_viewer_mode_lists_both_suffixes(qapp, tmp_path):
    _write_processed(tmp_path / "a.nexus")
    _write_processed(tmp_path / "b.nxs")

    host, lw = _list_host(tmp_path, viewer_mode="image")
    host.update_scans()

    assert _entries(lw) == ["..", "a.nexus", "b.nxs"]


# ---------------------------------------------------------------------------
# Row 7 — GUI run-end listing / follow / restore works for ``.nexus``
# ---------------------------------------------------------------------------

def test_row7_normal_mode_lists_nexus_and_legacy(qapp, tmp_path):
    _write_processed(tmp_path / "scan_1.nexus")
    _write_processed(tmp_path / "scan_2.nxs")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "sub").mkdir()

    host, lw = _list_host(tmp_path)
    host.update_scans()

    assert _entries(lw) == ["..", "scan_1.nexus", "scan_2.nxs", "sub/"]


def test_row7_follow_selects_the_current_nexus_scan(qapp, tmp_path):
    _write_processed(tmp_path / "scan_1.nexus")
    _write_processed(tmp_path / "scan_2.nexus")

    host, lw = _list_host(tmp_path, scan_name="scan_2")
    host.update_scans()

    assert lw.currentItem() is not None
    assert lw.currentItem().text() == "scan_2.nexus"


def test_row7_follow_still_selects_a_legacy_scan(qapp, tmp_path):
    _write_processed(tmp_path / "scan_1.nxs")
    _write_processed(tmp_path / "scan_2.nxs")

    host, lw = _list_host(tmp_path, scan_name="scan_2")
    host.update_scans()

    assert lw.currentItem() is not None
    assert lw.currentItem().text() == "scan_2.nxs"


def test_row7_follow_tolerates_a_frame_count_suffix_on_a_nexus_scan(
        qapp, tmp_path):
    """The fuzzy ``<scan>_N`` form must still resolve to ``<scan>.nexus``."""
    _write_processed(tmp_path / "scan.nexus")

    host, lw = _list_host(tmp_path, scan_name="scan_5")
    host.update_scans()

    assert lw.currentItem() is not None
    assert lw.currentItem().text() == "scan.nexus"


def test_row7_restore_selection_targets_a_nexus_scan(qapp, tmp_path):
    _write_processed(tmp_path / "scan_1.nexus")
    _write_processed(tmp_path / "scan_2.nexus")

    host, lw = _list_host(tmp_path)
    host.update_scans()
    host.scan_name = "scan_2"
    host.file_thread = None
    host._restore_loaded_scan_selection = MethodType(
        H5Viewer._restore_loaded_scan_selection, host)

    assert host._restore_loaded_scan_selection() is True
    assert lw.currentItem().text() == "scan_2.nexus"


def test_row7_restore_selection_still_targets_a_legacy_scan(qapp, tmp_path):
    _write_processed(tmp_path / "scan_1.nxs")

    host, lw = _list_host(tmp_path)
    host.update_scans()
    host.scan_name = "scan_1"
    host.file_thread = None
    host._restore_loaded_scan_selection = MethodType(
        H5Viewer._restore_loaded_scan_selection, host)

    assert host._restore_loaded_scan_selection() is True
    assert lw.currentItem().text() == "scan_1.nxs"


def test_row7_run_end_fallback_finds_a_nexus_output(tmp_path):
    output = _write_processed(tmp_path / "scan_042.nexus")
    thread = admitted_worker(
        SimpleNamespace(
            _append_skip_frames_by_scan={"scan_042": {1}},
            _append_output_path=None,
        ),
        processing_mode="Int 2D",
        save_path=str(tmp_path),
        source_spec=directory_source(tmp_path),
    )
    wrangler = SimpleNamespace(thread=thread, fname=None)

    found = _finished_output_file(thread, wrangler, all_skipped_append=True)

    assert found is not None
    assert Path(found) == output


def test_row7_canonical_run_token_strips_the_new_suffix():
    assert staticWidget._canonical_run_token("scan_042.nexus") == "scan_042"
    assert staticWidget._canonical_run_token("scan_042.nxs") == "scan_042"


# ---------------------------------------------------------------------------
# Correction round 2 — case variants and suffix PRIORITY over row order
#
# Blocker A: the viewer compared an unlowercased suffix, and follow/restore let
# presentation order decide which sibling won.  With Date sorting a newer
# ``.nxs`` is visited first, so the frozen "``.nexus`` wins" rule silently
# inverted.  Rank, not row position, must decide.
# ---------------------------------------------------------------------------

def _make_older(path: Path, *, seconds: int = 120) -> Path:
    """Backdate *path* so Date sort lists it AFTER its newer sibling."""
    stamp = os.path.getmtime(path) - seconds
    os.utime(path, (stamp, stamp))
    return path


def test_c2row4_normal_mode_lists_uppercase_variants(qapp, tmp_path):
    """C2 row 4 — ``.NEXUS``/``.NXS`` are the same files to a reader (§2 r5)."""
    _write_processed(tmp_path / "scan_1.NEXUS")
    _write_processed(tmp_path / "scan_2.NXS")
    _write_processed(tmp_path / "scan_3.nexus")
    (tmp_path / "notes.txt").write_text("x")

    host, lw = _list_host(tmp_path)
    host.update_scans()

    assert _entries(lw) == ["..", "scan_1.NEXUS", "scan_2.NXS", "scan_3.nexus"]


def test_c2row5_date_sorted_exact_follow_still_selects_the_nexus_sibling(
        qapp, tmp_path):
    """C2 row 5 — a NEWER ``.nxs`` listed first must not beat ``.nexus``."""
    _make_older(_write_processed(tmp_path / "scan.nexus"))
    _write_processed(tmp_path / "scan.nxs")

    host, lw = _list_host(tmp_path, scan_name="scan", date_sort=True)
    host.update_scans()

    assert _entries(lw) == ["..", "scan.nxs", "scan.nexus"]      # order proof
    assert lw.currentItem() is not None
    assert lw.currentItem().text() == "scan.nexus"


def test_c2row6_fuzzy_follow_prefers_nexus_independent_of_row_order(
        qapp, tmp_path):
    """C2 row 6a — the ``<scan>_N`` fuzzy form obeys rank, not last-match."""
    _write_processed(tmp_path / "scan.nexus")
    _make_older(_write_processed(tmp_path / "scan.nxs"))

    host, lw = _list_host(tmp_path, scan_name="scan_7", date_sort=True)
    host.update_scans()

    assert _entries(lw) == ["..", "scan.nexus", "scan.nxs"]      # order proof
    assert lw.currentItem() is not None
    assert lw.currentItem().text() == "scan.nexus"


def test_c2row6_restore_without_a_filename_prefers_the_nexus_sibling(
        qapp, tmp_path):
    """C2 row 6b — the no-``fname`` restore path has the same rank rule."""
    _make_older(_write_processed(tmp_path / "scan.nexus"))
    _write_processed(tmp_path / "scan.nxs")

    host, lw = _list_host(tmp_path, date_sort=True)
    host.update_scans()
    assert _entries(lw) == ["..", "scan.nxs", "scan.nexus"]      # order proof

    host.scan_name = "scan"
    host.file_thread = None
    host._restore_loaded_scan_selection = MethodType(
        H5Viewer._restore_loaded_scan_selection, host)

    assert host._restore_loaded_scan_selection() is True
    assert lw.currentItem().text() == "scan.nexus"


def test_c2row6_restore_still_honours_an_explicit_loaded_filename(
        qapp, tmp_path):
    """Retained: a real loaded ``fname`` still wins over the rank rule."""
    _write_processed(tmp_path / "scan.nexus")
    _write_processed(tmp_path / "scan.nxs")

    host, lw = _list_host(tmp_path)
    host.update_scans()
    host.scan_name = "scan"
    host.file_thread = SimpleNamespace(fname=str(tmp_path / "scan.nxs"))
    host._restore_loaded_scan_selection = MethodType(
        H5Viewer._restore_loaded_scan_selection, host)

    assert host._restore_loaded_scan_selection() is True
    assert lw.currentItem().text() == "scan.nxs"


def test_c2row7_run_end_fallback_preserves_a_case_variant_output(tmp_path):
    """C2 row 7a — the fallback returns the REAL spelling it found."""
    output = _write_processed(tmp_path / "scan_042.NEXUS")
    thread = admitted_worker(
        SimpleNamespace(
            _append_skip_frames_by_scan={"scan_042": {1}},
            _append_output_path=None,
        ),
        processing_mode="Int 2D",
        save_path=str(tmp_path),
        source_spec=directory_source(tmp_path),
    )
    wrangler = SimpleNamespace(thread=thread, fname=None)

    found = _finished_output_file(thread, wrangler, all_skipped_append=True)

    assert found is not None
    assert Path(found).name == "scan_042.NEXUS"
    assert Path(found) == output


def test_c2row7_run_end_fallback_prefers_the_new_suffix(tmp_path):
    """Retained through the shared resolver: ``.nexus`` outranks ``.nxs``."""
    _write_processed(tmp_path / "scan_042.nexus")
    _write_processed(tmp_path / "scan_042.nxs")
    thread = admitted_worker(
        SimpleNamespace(
            _append_skip_frames_by_scan={"scan_042": {1}},
            _append_output_path=None,
        ),
        processing_mode="Int 2D",
        save_path=str(tmp_path),
        source_spec=directory_source(tmp_path),
    )
    wrangler = SimpleNamespace(thread=thread, fname=None)

    found = _finished_output_file(thread, wrangler, all_skipped_append=True)

    assert Path(found).name == "scan_042.nexus"


# ---------------------------------------------------------------------------
# Correction round 2 — the NeXus wrangler must BIND the target it advertises,
# and must run the collision preflight before it mutates anything
#
# Blocker B: ``setup()`` assigned ``self.fname`` while ``nexusThread`` saves
# through ``scan.data_file``, so a reused LiveScan kept the previous run's path.
# Blocker C1: no suffix-independent guard ran before the writer was built.
# ---------------------------------------------------------------------------

def _nexus_wrangler(tmp_path, monkeypatch, *, source, out_dir, prior):
    """A REAL nexusWrangler over a REAL reused LiveScan (HARD RULE 2/10)."""
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler import nexusWrangler

    monkeypatch.setenv("XDART_SESSION_FILE",
                       str(tmp_path / "p4_c2_session.json"))
    scan = LiveScan("prior", data_file=str(prior), static=True)
    w = nexusWrangler(str(prior), threading.RLock(), scan)
    w.parameters.child('NeXus File').child('nexus_file').setValue(str(source))
    w.parameters.child('Output').child('h5_dir').setValue(str(out_dir))
    return w, scan


def test_c2row8_nexus_setup_binds_scan_and_frames_to_the_resolved_target(
        qapp, tmp_path, monkeypatch):
    """C2 row 8 — the advertised target is the one the worker actually saves."""
    source = _write_processed(tmp_path / "raw_source.h5")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    prior = tmp_path / "prior.nexus"
    _write_processed(prior)
    prior_bytes = prior.read_bytes()

    w, scan = _nexus_wrangler(tmp_path, monkeypatch, source=source,
                              out_dir=out_dir, prior=prior)
    w.setup()

    target = out_dir / "raw_source.nexus"
    assert Path(w.fname) == target
    assert Path(scan.data_file) == target
    assert Path(scan.frames.data_file) == target

    # One minimal REAL save through the production writer must land on that
    # target and leave the previous file and the source untouched.
    scan.save_to_nexus(replace=True)

    assert target.exists()
    assert prior.read_bytes() == prior_bytes


def test_c2row9_nexus_setup_rejects_a_same_inode_source_target(
        qapp, tmp_path, monkeypatch):
    """C2 row 9 — refusal happens BEFORE any scan mutation (§2 rule 8)."""
    from xrd_tools.io.output_safety import OutputCollisionError

    source = _write_processed(tmp_path / "raw_source.h5")
    source_bytes = source.read_bytes()
    alias = tmp_path / "raw_source.nexus"       # the target setup() will resolve
    os.link(source, alias)
    prior = tmp_path / "prior.nexus"
    _write_processed(prior)

    w, scan = _nexus_wrangler(tmp_path, monkeypatch, source=source,
                              out_dir=tmp_path, prior=prior)

    with pytest.raises(OutputCollisionError):
        w.setup()

    assert Path(scan.data_file) == prior        # scan identity preserved
    assert Path(scan.frames.data_file) == prior
    assert source.read_bytes() == source_bytes


def test_c2row9_nexus_setup_accepts_a_separate_output_directory(
        qapp, tmp_path, monkeypatch):
    """Retained: the preflight must not reject an ordinary separate target."""
    source = _write_processed(tmp_path / "raw_source.h5")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    prior = tmp_path / "prior.nexus"

    w, scan = _nexus_wrangler(tmp_path, monkeypatch, source=source,
                              out_dir=out_dir, prior=prior)
    w.setup()

    assert Path(w.fname) == out_dir / "raw_source.nexus"


# ---------------------------------------------------------------------------
# Row 15 (GUI) — function-scoped owner census
# ---------------------------------------------------------------------------

#: (relative source path, qualified function) for every authorized GUI
#: generated-output constructor.
GUI_PRODUCERS = (
    ("xdart/gui/tabs/static_scan/h5viewer.py", "_readable_output_rank"),
    ("xdart/gui/tabs/static_scan/h5viewer.py", "H5Viewer.update_scans"),
    ("xdart/gui/tabs/static_scan/h5viewer.py",
     "H5Viewer._restore_loaded_scan_selection"),
    ("xdart/gui/tabs/static_scan/static_scan_widget.py",
     "_finished_output_file"),
    ("xdart/gui/tabs/static_scan/static_scan_widget.py",
     "staticWidget._init_data_objects"),
    ("xdart/gui/tabs/static_scan/wranglers/image_wrangler.py",
     "imageWrangler._candidate_append_target_file"),
    ("xdart/gui/tabs/static_scan/wranglers/image_wrangler.py",
     "imageWrangler.setup"),
    ("xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py",
     "imageThread._append_output_path"),
    ("xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py",
     "imageThread.initialize_scan"),
    ("xdart/gui/tabs/static_scan/wranglers/nexus_wrangler.py",
     "nexusWrangler.setup"),
)

#: Only the owner's real API names count as "consumes the owner".  A bare
#: ``output_path`` substring is deliberately NOT here: it matches unrelated
#: local names such as ``_append_output_path``, which would pass the census
#: without consuming anything.  ``_readable_output_rank`` is the viewer's one
#: delegating helper: it is itself a censused producer above, so a consumer that
#: goes through it still terminates at the shared owner — and there is exactly
#: one such helper, not a second policy owner.
OWNER_NAMES = ("resolve_output_target", "default_output_path",
               "NEW_OUTPUT_SUFFIX", "READABLE_OUTPUT_SUFFIXES",
               "is_readable_output_path", "_readable_output_rank")


def _function_node(path: Path, qualname: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node: ast.AST = tree
    for part in qualname.split("."):
        for child in ast.iter_child_nodes(node):
            if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef))
                    and child.name == part):
                node = child
                break
        else:                                       # pragma: no cover
            raise AssertionError(f"{qualname} not found in {path}")
    return node


def _legacy_literals(path: Path, qualname: str) -> list[str]:
    """Non-docstring string constants inside *qualname* naming ``.nxs``.

    AST-based, so a comment or docstring mentioning the legacy suffix is not a
    false positive while every shape of hard-coded legacy target is caught.
    """
    node = _function_node(path, qualname)
    docstrings = set()
    for sub in ast.walk(node):
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef,
                            ast.ClassDef, ast.Module)):
            body = getattr(sub, "body", None) or []
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))
    return [sub.value for sub in ast.walk(node)
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
            and id(sub) not in docstrings and ".nxs" in sub.value.lower()]


def _owner_references(path: Path, qualname: str) -> set[str]:
    """Owner API names actually LOADED or CALLED inside *qualname*.

    C2 row 12.  Real ``ast.Name``/``ast.Attribute`` loads only — which covers
    every call target, since ``Call.func`` is one of those nodes — so a comment,
    a docstring, or a dead string naming ``resolve_output_target`` can never
    satisfy the census.  The superseded check scanned the function's source text
    for a substring and accepted exactly that (C2 mutation 9).
    """
    node = _function_node(path, qualname)
    seen: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            seen.add(sub.id)
        elif isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Load):
            seen.add(sub.attr)
    return seen & set(OWNER_NAMES)


@pytest.mark.parametrize("rel_path,qualname", GUI_PRODUCERS)
def test_row15_gui_producer_consumes_the_shared_owner(rel_path, qualname):
    assert _owner_references(SRC / rel_path, qualname), (
        f"{rel_path}::{qualname} does not consume the shared path owner")


@pytest.mark.parametrize("rel_path,qualname", GUI_PRODUCERS)
def test_row15_gui_producer_has_no_hard_coded_legacy_target(rel_path, qualname):
    found = _legacy_literals(SRC / rel_path, qualname)
    assert found == [], (
        f"{rel_path}::{qualname} still builds a hard-coded legacy target: {found}")


def test_row15_raw_source_pickers_keep_their_literal_nxs():
    """``.nexus`` must NOT leak into a RAW source picker or a raw watch config.

    These are raw *source* checks, explicitly allowed to remain literal (§5
    row 15) and required to stay literal by §2 rule 7 — ``.nexus`` is
    output-only and excluded from raw discovery.
    """
    picker = (SRC / "xdart/gui/tabs/static_scan/scan_source_widget.py"
              ).read_text(encoding="utf-8")
    assert "*.nxs" in picker
    # The glob filter specifically — the file legitimately mentions the word
    # "nexus" (source-kind labels, the ``xrd_tools.io.nexus`` import).
    assert "*.nexus" not in picker

    ssw = SRC / "xdart/gui/tabs/static_scan/static_scan_widget.py"
    watch = ast.get_source_segment(
        ssw.read_text(encoding="utf-8"),
        _function_node(ssw, "staticWidget._controls_v2_container_index_config"),
    ) or ""
    assert 'suffixes = (".nxs",)' in watch
    assert NEW_OUTPUT_SUFFIX not in watch
